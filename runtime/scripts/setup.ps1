# One-shot development setup for Windows.

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

Write-Host "[1/5] Creating virtualenv..." -ForegroundColor Cyan
python -m venv (Join-Path $Repo ".venv")
$Py = Join-Path $Repo ".venv\Scripts\python.exe"

Write-Host "[2/5] Installing ForgeCoder runtime + dev deps..." -ForegroundColor Cyan
& $Py -m pip install --upgrade pip
& $Py -m pip install -e "$Repo[dev]"

Write-Host "[3/5] Detecting hardware profile..." -ForegroundColor Cyan
& $Py (Join-Path $Repo "runtime\scripts\detect_hardware.py")

Write-Host "[4/5] Installing VS Code extension dependencies..." -ForegroundColor Cyan
Push-Location (Join-Path $Repo "apps\vscode")
npm install
npm run compile
Pop-Location

Write-Host "[5/5] Running Python test suite..." -ForegroundColor Cyan
& $Py -m pytest (Join-Path $Repo "tests") -q

Write-Host "`nDone. Next steps:" -ForegroundColor Green
Write-Host "  1. Place a Qwen2.5-Coder-1.5B-Instruct Q4_K_M GGUF in models\gguf\"
Write-Host "  2. python runtime\scripts\launch_llama.py"
Write-Host "  3. python -m uvicorn forge_server.main:app --host 127.0.0.1 --port 8787"