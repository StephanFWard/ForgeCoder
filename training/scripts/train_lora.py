"""Train a ForgeCoder LoRA on top of Qwen2.5-Coder-1.5B-Instruct.

QLoRA by default (4-bit base + PEFT adapter). This script is designed to run
on a GPU host — the 6 GB laptop is the *inference* target, not the trainer.

Usage:
    pip install -e ".[train]"
    python training/scripts/train_lora.py --config training/configs/lora.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger("forge_train")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/configs/lora.yaml")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # graceful: training is optional at runtime
        log.error(
            "Training dependencies missing (%s). Install with: pip install -e '.[train]'",
            exc,
        )
        return 2

    base_model = cfg["base_model"]
    lora = cfg["lora"]
    training = cfg["training"]
    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model %s (4-bit QLoRA)...", base_model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora.get("r", 8),
        lora_alpha=lora.get("alpha", 16),
        lora_dropout=lora.get("dropout", 0.05),
        bias=lora.get("bias", "none"),
        task_type=lora.get("task_type", "CAUSAL_LM"),
        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_path = Path(args.train_data or "training/lora/train.jsonl")
    eval_path = Path(args.eval_data or "training/lora/eval.jsonl")

    def load_dataset(path: Path):
        from datasets import load_dataset

        return load_dataset("json", data_files=str(path))["train"]

    train_ds = load_dataset(train_path)
    eval_ds = load_dataset(eval_path) if eval_path.exists() else None

    max_len = training.get("max_seq_length", 2048)

    def tokenize(example):
        parts = [example.get("instruction", "")]
        if example.get("input"):
            parts.append(example["input"])
        text = ("\n\n".join(parts) + "\n\n" + example["output"]).strip()
        prompt = "\n\n".join(parts).strip()
        out = tokenizer(text, truncation=True, max_length=max_len)
        prompt_ids = tokenizer(prompt + "\n\n", truncation=True, max_length=max_len)["input_ids"]
        out["labels"] = [
            -100 if i < len(prompt_ids) else out["input_ids"][i]
            for i in range(len(out["input_ids"]))
        ]
        return out

    train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
    if eval_ds is not None:
        eval_ds = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training.get("epochs", 2),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        per_device_train_batch_size=training.get("batch_size", 1),
        gradient_accumulation_steps=training.get("gradient_accumulation_steps", 16),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        fp16=training.get("fp16", True),
        logging_steps=25,
        save_strategy="steps",
        save_steps=training.get("save", {}).get("every_steps", 250),
        save_total_limit=training.get("save", {}).get("keep_last", 2),
        eval_steps=training.get("evaluation", {}).get("every_steps", 250),
        evaluation_strategy="steps" if eval_ds is not None else "no",
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds is not None else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info("Adapter saved to %s", output_dir)
    return 0


def _has_bnb() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
