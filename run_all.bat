@echo off
REM SparseFE - run all three scales and build the figures (Command Prompt)
REM Just double-click this file.
cd /d "%~dp0"

REM Use the ascending-limb target (default); clear any stale override.
set "LABEL_MODE="

REM Real TabPFN backend.
set "AUTOTABPFN_BACKEND=tabpfn"

REM TabPFN token. Kept OUT of this file on purpose (safe to deposit).
REM Uses the token already set, otherwise asks for it.
if "%TABPFN_TOKEN%"=="" set /p TABPFN_TOKEN=Paste your TabPFN token: 

py code\run.py --scale qc --fast
py code\run.py --scale regional --fast
py code\run.py --scale country --fast
py code\make_figures.py

echo.
echo Done. Figures: outputs\figures   Metrics: outputs\^<scale^>\metrics.json
pause
