@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
)
if "%~1"=="" (
  echo Usage: run_windows.bat reanalyze "C:\path\to\uploads" [new-output-directory]
  echo Upload directory must contain pilot-seed42-submission.zip, prepared.npz and all nine .pt files.
  echo This command runs inference only. It does not retrain or alter historical manifests.
  exit /b 1
)
set "ASSETS=%~f1"
set "OUT=%~2"
if not defined OUT set "OUT=runs\reanalysis-seed42"
if exist "%OUT%" (
  echo ERROR: Output already exists. Choose a fresh output directory.
  exit /b 1
)
if not exist "%ASSETS%\pilot-seed42-submission.zip" (
  echo ERROR: Missing pilot-seed42-submission.zip in upload directory.
  exit /b 1
)
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
"%PY%" -m pip install -e ".[test]"
if errorlevel 1 exit /b 1
"%PY%" -m pytest -q
if errorlevel 1 exit /b 1
"%PY%" -m rl_recsys.cli import-run --bundle "%ASSETS%\pilot-seed42-submission.zip" --assets "%ASSETS%" --output "%OUT%\historical"
if errorlevel 1 exit /b 1
"%PY%" -m rl_recsys.cli verify-run --run "%OUT%\historical\runs\pilot-seed42" --data "%OUT%\historical\data\prepared" --output "%OUT%\historical-verification.json"
if errorlevel 1 exit /b 1
"%PY%" -m rl_recsys.cli reanalyze --run "%OUT%\historical\runs\pilot-seed42" --data "%OUT%\historical\data\prepared" --output "%OUT%\analysis" --batch-size 128 --max-minutes 240
if errorlevel 1 exit /b 1
"%PY%" -m rl_recsys.cli bundle-reanalysis --imported "%OUT%\historical" --analysis "%OUT%\analysis" --output "%OUT%\reanalysis-complete.zip"
if errorlevel 1 exit /b 1
echo COMPLETE. Full reproducibility bundle: %OUT%\reanalysis-complete.zip
echo Includes checkpoints and prepared data; it is substantially larger than the historical ZIP.
exit /b 0
