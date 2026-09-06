# Installer build helper.

Builds, in order:
1. The VS Code extension VSIX   (apps/vscode -> forgecoder-0.1.0.vsix)
2. The ForgeCoder Setup.exe     (installer/ForgeCoder.iss via iscc)

Prerequisites:
- Node + npm
- Inno Setup 6 (iscc.exe on PATH) — https://jrsoftware.org/isdl.php
- A Q4_K_M GGUF at models/gguf/forgecoder-1.5b-q4_k_m.gguf (optional, warns)

Usage:
    powershell -ExecutionPolicy Bypass -File installer\scripts\build.ps1
"""

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VsCode = Join-Path $Repo "apps\vscode"

Write-Host "[1/3] Building VS Code extension VSIX..." -ForegroundColor Cyan
Push-Location $VsCode
npm install --no-audit --no-fund
npm run package
$Vsix = Get-ChildItem -Path . -Filter "*.vsix" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $Vsix) { throw "VSIX build produced no artifact." }
Write-Host "      -> $($Vsix.Name)"
Pop-Location

Write-Host "[2/3] Preparing dist..." -ForegroundColor Cyan
$Dist = Join-Path $Repo "dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null

Write-Host "[3/3] Running Inno Setup..." -ForegroundColor Cyan
$Iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $Iscc) { throw "iscc (Inno Setup) not found on PATH." }
& $Iscc.Source (Join-Path $Repo "installer\ForgeCoder.iss")

Write-Host "`nDone. Installer(s) in dist\:" -ForegroundColor Green
Get-ChildItem $Dist | Select-Object Name, Length