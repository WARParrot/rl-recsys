@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "VENV=%CD%\.venv"
set "PYTHON_EXE="
set "RUN_DIR=runs\pilot-seed42"
set "BUNDLE_DIR=artifacts\pilot-seed42-submission"
set "BUNDLE_ZIP=artifacts\pilot-seed42-submission.zip"

if exist "%RUN_DIR%\provenance.json" (
  echo ERROR: %RUN_DIR% already contains a run. Choose a fresh run directory or remove it deliberately.
  exit /b 1
)

call :step "Checking Python 3.11"
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
  if errorlevel 1 exit /b 1
  if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
  if not defined PYTHON_EXE if exist "%ProgramFiles%\Python311\python.exe" set "PYTHON_EXE=%ProgramFiles%\Python311\python.exe"
  if not defined PYTHON_EXE set "PYTHON_EXE=py -3.11"
)

if not exist "%VENV%\Scripts\python.exe" (
  call :step "Creating virtual environment"
  %PYTHON_EXE% -m venv "%VENV%"
  if errorlevel 1 exit /b 1
)
set "PY=%VENV%\Scripts\python.exe"
if not exist "%PY%" (
  echo ERROR: Virtual environment Python was not created: %PY%
  exit /b 1
)

if not exist logs mkdir logs
if not exist data mkdir data
if not exist runs mkdir runs
if not exist artifacts mkdir artifacts

call :step "Checking NVIDIA driver"
nvidia-smi > logs\nvidia-smi.log 2>&1
if errorlevel 1 (
  type logs\nvidia-smi.log
  echo ERROR: nvidia-smi failed. Install/update the NVIDIA driver before continuing.
  exit /b 1
)
type logs\nvidia-smi.log

call :step "Installing dependencies"
"%PY%" -m pip install --upgrade pip > logs\pip-install.log 2>&1
if errorlevel 1 ( type logs\pip-install.log & exit /b 1 )
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 >> logs\pip-install.log 2>&1
if errorlevel 1 ( type logs\pip-install.log & exit /b 1 )
"%PY%" -m pip install -e ".[test]" >> logs\pip-install.log 2>&1
if errorlevel 1 ( type logs\pip-install.log & exit /b 1 )

call :step "Running CUDA preflight"
"%PY%" -m rl_recsys.cli preflight --config configs/pilot.json --device cuda > logs\preflight.json 2> logs\preflight.stderr.log
if errorlevel 1 (
  type logs\preflight.stderr.log
  exit /b 1
)
"%PY%" -c "import json; d=json.load(open('logs/preflight.json')); assert d.get('status') == 'ok' and d.get('device') == 'cuda'; print('CUDA preflight JSON verified')"
if errorlevel 1 (
  echo ERROR: Preflight did not report a usable CUDA device.
  exit /b 1
)

call :step "Running tests"
"%PY%" -m pytest -q > logs\tests.log 2>&1
if errorlevel 1 ( type logs\tests.log & exit /b 1 )
type logs\tests.log

call :step "Downloading Yambda-50M"
"%PY%" -m rl_recsys.cli download --config configs/pilot.json --output data/listens.parquet > logs\download.log 2>&1
if errorlevel 1 ( type logs\download.log & exit /b 1 )

call :step "Preparing frozen temporal split"
"%PY%" -m rl_recsys.cli prepare --config configs/pilot.json --input data/listens.parquet --output data/prepared > logs\prepare.log 2>&1
if errorlevel 1 ( type logs\prepare.log & exit /b 1 )

call :step "Training SFT, reward models, and gated RL grid"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '%PY%' -m rl_recsys.cli train --config configs/pilot.json --data data/prepared --run '%RUN_DIR%' 2>&1 | Tee-Object -FilePath 'logs/training.log'; $code=$LASTEXITCODE; if (Test-Path 'logs/training.log') { $text=Get-Content 'logs/training.log' -Raw; [IO.File]::WriteAllText((Resolve-Path 'logs/training.log'), $text, (New-Object Text.UTF8Encoding($false))) }; exit $code }"
if errorlevel 1 ( type logs\training.log & exit /b 1 )

call :step "Running final evaluation"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { & '%PY%' -m rl_recsys.cli evaluate --data data/prepared --run '%RUN_DIR%' 1> 'runs/pilot-seed42/final-evaluation.json' 2> 'logs/evaluation.stderr.log'; $code=$LASTEXITCODE; foreach($f in @('runs/pilot-seed42/final-evaluation.json','logs/evaluation.stderr.log')) { if (Test-Path $f) { $text=Get-Content $f -Raw; [IO.File]::WriteAllText((Resolve-Path $f), $text, (New-Object Text.UTF8Encoding($false))) } }; if ($code -ne 0) { Get-Content 'logs/evaluation.stderr.log' }; exit $code }"
if errorlevel 1 ( type logs\evaluation.stderr.log & exit /b 1 )

call :step "Creating submission bundle"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference='Stop'; $d='%BUNDLE_DIR%'; if (Test-Path $d) { Remove-Item $d -Recurse -Force }; New-Item -ItemType Directory -Path $d -Force | Out-Null; $files=@('configs/pilot.json','logs/nvidia-smi.log','logs/pip-install.log','logs/preflight.json','logs/preflight.stderr.log','logs/tests.log','logs/download.log','logs/prepare.log','logs/training.log','logs/evaluation.stderr.log','data/prepared/audit.json','runs/pilot-seed42/provenance.json','runs/pilot-seed42/reward_audit.json','runs/pilot-seed42/selection.json','runs/pilot-seed42/candidates.json','runs/pilot-seed42/training.json','runs/pilot-seed42/READY_FOR_TEST.json','runs/pilot-seed42/results.json','runs/pilot-seed42/final-evaluation.json','runs/pilot-seed42/per_example.npz','runs/pilot-seed42/evaluation-provenance.json','runs/pilot-seed42/EVALUATION_COMPLETE.json','runs/pilot-seed42/progress.jsonl'); foreach($f in $files) { if(Test-Path $f) { $target=Join-Path $d $f; New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null; Copy-Item $f $target -Force } else { throw ('Missing expected artifact: '+$f) } }; $root=(Resolve-Path $d).Path; $checks=Get-ChildItem $d -File -Recurse | Where-Object { $_.Name -ne 'SHA256SUMS.json' } | ForEach-Object { $h=Get-FileHash $_.FullName -Algorithm SHA256; [pscustomobject]@{sha256=$h.Hash.ToLower(); path=$_.FullName.Substring($root.Length+1).Replace('\\','/')} }; $json=$checks | ConvertTo-Json -Depth 3; [IO.File]::WriteAllText((Join-Path $d 'SHA256SUMS.json'), $json, (New-Object Text.UTF8Encoding($false))); if(Test-Path '%BUNDLE_ZIP%'){Remove-Item '%BUNDLE_ZIP%' -Force}; Compress-Archive -Path (Join-Path $d '*') -DestinationPath '%BUNDLE_ZIP%' -CompressionLevel Optimal }"
if errorlevel 1 (
  echo ERROR: Submission bundle creation failed.
  exit /b 1
)

for %%F in ("%BUNDLE_ZIP%") do set "BUNDLE_SIZE=%%~zF"
echo.
echo COMPLETE. Upload this single file:
echo   %BUNDLE_ZIP% (%BUNDLE_SIZE% bytes)
echo It contains audit, provenance, reward gate, metrics, per-example results, manifests, logs, config, and SHA256SUMS.json.
exit /b 0

:step
echo.
echo ============================================================
echo %~1
echo ============================================================
exit /b 0
