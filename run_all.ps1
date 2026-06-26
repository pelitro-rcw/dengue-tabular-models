# SparseFE - run all three scales and build the figures (PowerShell)
# Usage:  right-click this file > "Run with PowerShell"
#    or:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Use the ascending-limb target (the default). Clear any stale override.
Remove-Item Env:LABEL_MODE -ErrorAction SilentlyContinue

# Real TabPFN backend.
$env:AUTOTABPFN_BACKEND = "tabpfn"

# TabPFN token. Kept OUT of this file on purpose so the code folder stays safe
# to deposit publicly. Uses the token already set in this session, else asks.
if (-not $env:TABPFN_TOKEN) {
    $env:TABPFN_TOKEN = Read-Host "Paste your TabPFN token"
}

py code\run.py --scale qc --fast
py code\run.py --scale regional --fast
py code\run.py --scale country --fast
py code\make_figures.py

Write-Host ""
Write-Host "Done. Figures: outputs\figures   Metrics: outputs\<scale>\metrics.json" -ForegroundColor Green
