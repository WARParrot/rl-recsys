@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "VENV=%CD%\.venv"
set "PYTHON_EXE="

echo [1/8] Checking Python 3.11...
py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=py -3.11"

if not defined PYTHON_EXE (
  echo Python 3.11 was not found. Installing it with winget...
  where winget >nul 2>&1
  if errorlevel 1 (
    echo ERROR: winget is unavailable. Install Python 3.11 from https://www.python.org/downloads/windows/
    exit /b 1
  )
  winget install --id Python.Python.3.11 --exact --scope user --accept-source-agreements --accept-package-agreements
  if errorlevel 1 (
    echo ERROR: Python installation failed.
    exit /b 1
  )
  if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
  if not defined PYTHON_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
  if not defined PYTHON_EXE set "PYTHON_EXE=py -3.11"
)

if not exist "%VENV%\Scripts\python.exe" (
  echo [2/8] Creating virtual environment...
  %PYTHON_EXE% -m venv "%VENV%"
  if errorlevel 1 (
    echo ERROR: Could not create the virtual environment.
    exit /b 1
  )
)
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
  echo ERROR: Virtual environment Python was not created: %PY%
  exit /b 1
)

if not exist logs mkdir logs
if not exist data mkdir data
if not exist runs mkdir runs

echo [3/8] Checking NVIDIA driver...
nvidia-smi
if errorlevel 1 (
  echo ERROR: nvidia-smi failed. Install/update the NVIDIA driver before continuing.
  exit /b 1
)

echo [4/8] Updating pip and installing CUDA PyTorch...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo ERROR: CUDA PyTorch installation failed. Check the official PyTorch CUDA selector.
  exit /b 1
)

"%PY%" -m pip install -e ".[test]"
if errorlevel 1 exit /b 1

echo [5/8] Running CUDA preflight...
"%PY%" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); x=torch.randn(2048,2048,device='cuda',requires_grad=True); (x@x).mean().backward(); torch.cuda.synchronize(); print('CUDA forward/backward OK'); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,2))"
if errorlevel 1 (
  echo ERROR: CUDA preflight failed. No training was started.
  exit /b 1
)

 echo [6/8] Running tests...
"%PY%" -m pytest -q
if errorlevel 1 (
  echo ERROR: Tests failed. No training was started.
  exit /b 1
)

echo [7/8] Downloading and preparing Yambda-50M...
"%PY%" -m rl_recsys.cli download --config configs/pilot.json --output data/listens.parquet > logs\download.log 2>&1
if errorlevel 1 (
  type logs\download.log
  exit /b 1
)
"%PY%" -m rl_recsys.cli prepare --config configs/pilot.json --input data/listens.parquet --output data/prepared > logs\prepare.log 2>&1
if errorlevel 1 (
  type logs\prepare.log
  exit /b 1
)

echo [8/8] Starting pilot training. This is the only training command in the script.
"%PY%" -m rl_recsys.cli train --config configs/pilot.json --data data/prepared --run runs\pilot-seed42 > logs\training.log 2>&1
if errorlevel 1 (
  type logs\training.log
  echo Training failed. The full log is logs\training.log
  exit /b 1
)

echo Running final evaluation...
"%PY%" -m rl_recsys.cli evaluate --data data/prepared --run runs\pilot-seed42 > runs\pilot-seed42\final-evaluation.json 2> logs\evaluation.log
if errorlevel 1 (
  type logs\evaluation.log
  exit /b 1
)

echo.
echo COMPLETE. Send these files for analysis:
echo   data\prepared\audit.json
echo   runs\pilot-seed42\provenance.json
echo   runs\pilot-seed42\selection.json
echo   runs\pilot-seed42\candidates.json
echo   runs\pilot-seed42\results.json
echo   logs\training.log
echo   logs\evaluation.log
endlocal
