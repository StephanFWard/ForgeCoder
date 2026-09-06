# training/datasets

Raw task records, one folder per behavior:

```
training/datasets/
├── chat/           instruction → context → response        (ForgeChat)
├── bugfix/         bug → context → patch → verification    (ForgeFix)
├── editing/        selection → requested transform → replacement (ForgeEdit)
├── testing/        code → unit tests                       (ForgeTest)
├── completion/     prefix/suffix/target                    (ForgeComplete)
├── debugging/      error → logs → diagnosis → fix
├── architecture/   class/file → architecture explanation
└── security/       vulnerable code → hardened reponse
```

Records are JSON objects or JSONL lines. Run:

```powershell
python training\scripts\prepare_dataset.py
python training\scripts\validate_dataset.py training\lora\train.jsonl
```

## Licensing

Only include data you are licensed to redistribute. Preferred sources:
Apache-2.0 / MIT / your own synthetic data. Do **not** commit scraped
copyrighted content.