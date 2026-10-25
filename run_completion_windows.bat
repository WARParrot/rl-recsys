@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
if "%~1"=="" (
  echo Usage: run_completion_windows.bat "C:\path\to\original-research-root" ["artifacts\completion-new-run"]
  echo The original root must contain sample_42\run and sample_42\data.
  exit /b 1
)
set "PARENT=%~f1"
set "OUT=%~2"
if defined OUT goto output_ready
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss-fff"') do set "OUT=artifacts\completion-%%T"
:output_ready
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Existing research Python environment required: %PY%
  exit /b 1
)
if not exist "%PARENT%\sample_42\run\SFT_READY.json" (
  echo ERROR: Not an original research suite root: %PARENT%
  exit /b 1
)
echo Completion-only MSE study. Reusing original data and SFT; no download.
echo Historical joint reward remains frozen. Gate failure retains continued-SFT only.
echo All training and selection precede test. Test is exploratory reuse, not fresh.
echo Do not edit code during execution. See output logs for progress.
"%PY%" -u -m rl_recsys.completion_study --parent "%PARENT%" --output "%OUT%"
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
  echo FAILED: inspect %OUT%\study.json and logs. Do not treat failure ZIP as success.
  exit /b %CODE%
)
echo COMPLETE: upload %OUT%.zip and %OUT%.zip.sha256.json
exit /b 0
