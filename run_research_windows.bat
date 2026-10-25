@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "OUT=%~1"
if defined OUT goto output_ready
for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss-fff"') do set "OUT=artifacts\research-%%T"
:output_ready
if exist "%OUT%" (
  echo ERROR: Use a fresh output directory: %OUT%
  exit /b 1
)
if exist "%OUT%.zip" (
  echo ERROR: ZIP already exists: %OUT%.zip
  exit /b 1
)
set "PY=%CD%\.venv\Scripts\python.exe"
if exist "%PY%" goto environment_ready
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Install Python 3.11 from https://www.python.org/downloads/windows/ then rerun.
  exit /b 1
)
py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
:environment_ready
set "SETUP=%OUT%-setup"
if exist "%SETUP%" (
  echo ERROR: Setup directory exists: %SETUP%
  exit /b 1
)
mkdir "%SETUP%"
if errorlevel 1 exit /b 1
nvidia-smi > "%SETUP%\nvidia-smi.log" 2>&1
if errorlevel 1 (
  type "%SETUP%\nvidia-smi.log"
  exit /b 1
)
echo Installing dependencies. Logs: %SETUP%
"%PY%" -m pip install --upgrade pip > "%SETUP%\install.log" 2>&1
if errorlevel 1 exit /b 1
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 >> "%SETUP%\install.log" 2>&1
if errorlevel 1 exit /b 1
"%PY%" -m pip install -e ".[test]" >> "%SETUP%\install.log" 2>&1
if errorlevel 1 exit /b 1
"%PY%" -m rl_recsys.cli preflight --config configs/pilot.json --device cuda > "%SETUP%\preflight.json" 2> "%SETUP%\preflight.stderr.log"
if errorlevel 1 exit /b 1
"%PY%" -m pytest -q > "%SETUP%\tests.log" 2>&1
if errorlevel 1 (
  type "%SETUP%\tests.log"
  exit /b 1
)
echo Starting Experiment A: ONE fixed sample/SFT/RM, policy optimization seeds 42/43/44.
echo Arms: continued-SFT, completion RL, RL+CE, selection-tuned rerank. No ensemble.
echo This may take several days. Do not edit source/config during execution.
echo Old calendar test remains exploratory reuse. No automatic upload occurs.
"%PY%" -u -m rl_recsys.suite --output "%OUT%" --seeds 42 43 44 --steps 1000 --setup-logs "%SETUP%"
set "CODE=%ERRORLEVEL%"
echo Suite exit code: %CODE%
if not "%CODE%"=="0" (
  echo FAILED. Inspect %OUT%\suite.json and per-stage logs. Failure bundle is not a successful experiment.
  exit /b %CODE%
)
echo COMPLETE. Upload %OUT%.zip and %OUT%.zip.sha256.json
exit /b 0
