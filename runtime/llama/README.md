# runtime/llama

Place the llama.cpp binaries here (or add them to `PATH`):

| File                      | Source                                  |
| ------------------------- | --------------------------------------- |
| `llama-server.exe`        | llama.cpp release bundle                |
| `llama-quantize.exe`      | llama.cpp release bundle (training path)|
| `convert_hf_to_gguf.py`   | llama.cpp `convert_hf_to_gguf.py`       |

The launcher at `runtime/scripts/launch_llama.py` looks here first, then on
`PATH`. Everything in this directory is gitignored.

## Model download

```powershell
# e.g. from bartowski's Qwen2.5-Coder-1.5B-Instruct-GGUF release:
#   Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
Copy-Item Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf models\gguf\
```

Then:

```powershell
python runtime\scripts\launch_llama.py
# -> 127.0.0.1:8080
```