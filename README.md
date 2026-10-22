# RL RecSys Lab — separate experiment

This repository is intentionally separate from the existing feature-search artifact. It implements an auditable **model-based policy optimization** experiment:

1. Yambda-50M `listens` audit and strict temporal contexts;
2. GPT-2-style decoder over item IDs for next-item prediction;
3. independent history-conditioned reward model;
4. simulator-on-policy REINFORCE with KL to frozen SFT reference;
5. full restricted train-known catalog evaluation, reranking control, reward-model auditor, and five-step synthetic rollout diagnostics.

It is **not** causal offline policy evaluation and does not establish user satisfaction or retention. Generated items are not observed user actions. The primary report must remain qualified unless returned metrics satisfy the declared protocol in `PLAN.md` and the generated `audit.json`, `provenance.json`, `selection.json`, and `results.json` agree.

## One-command Windows setup

On Windows, open **Command Prompt** in the cloned repository and run:

```bat
run_windows.bat
```

The script installs Python 3.11 with `winget` if needed, creates `.venv`, installs the CUDA PyTorch wheel, runs the CUDA forward/backward preflight and tests, downloads/prepares Yambda, then starts exactly one pilot training job and its final evaluation. It does not use PowerShell activation, so it can be started directly from `cmd.exe`. Training output is visible in the console and simultaneously written to `logs\\training.log`; structured progress is written to `runs\\pilot-seed42\\progress.jsonl`.

If the script stops at the PyTorch step, check the official CUDA wheel selector at [pytorch.org](https://pytorch.org/get-started/locally/) and update the wheel URL in `run_windows.bat`. Do not continue if the preflight reports `CUDA is unavailable`.

## GPU setup (manual alternative)

Use Python 3.11 or 3.12 and a fresh environment. Install a PyTorch wheel appropriate for your CUDA driver from [pytorch.org](https://pytorch.org/get-started/locally/), then install this project without replacing torch:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# Choose the current CUDA command from pytorch.org, e.g. cu128/cu126; do not blindly copy this example:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'  # does not replace your CUDA torch wheel
python - <<'PY'
import torch
assert torch.cuda.is_available(), torch.cuda.get_device_name(0)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x=torch.randn(256,256,device='cuda',requires_grad=True); (x@x).mean().backward(); torch.cuda.synchronize()
print('CUDA forward/backward OK; VRAM GB:', torch.cuda.get_device_properties(0).total_memory/2**30)
PY
pytest -q
```

The RTX 5060 Ti has to pass that real forward/backward check. If the wheel rejects the GPU architecture, install the PyTorch build recommended for the installed driver; do not proceed on a CPU fallback. The project pins the audited non-torch dependencies; the CUDA PyTorch wheel is intentionally installed separately because its index and version must match the driver/GPU.

## Dependency and security status

The dependency set was checked with `pip-audit` on 2026-10-23. The updated environment (`transformers 5.20.2`, `pyarrow 26.0.0`, `numpy 2.2.6`, `pytest 9.1.1`) reported **no known vulnerabilities**. `pip-audit` cannot resolve the local package or the CUDA-index PyTorch wheel through its advisory database; this is not evidence that PyTorch is vulnerability-free. Install PyTorch only from the official PyTorch index, keep NVIDIA drivers updated, and review the installed versions with `pip list` and `pip-audit --local` before training. Do not execute untrusted model files; this project only loads its own checkpoints with `weights_only=True`.

## Data and bounded run

The CLI pins the Hugging Face dataset revision and SHA-256 in `configs/pilot.json`. It downloads about 369 MB and refuses mismatches:

```bash
rl-recsys download --config configs/pilot.json --output data/listens.parquet
rl-recsys prepare --config configs/pilot.json --input data/listens.parquet --output data/prepared
rl-recsys train --config configs/pilot.json --data data/prepared --run runs/pilot-seed42
# Only after training/selection completes:
rl-recsys evaluate --data data/prepared --run runs/pilot-seed42 > runs/pilot-seed42/final-evaluation.json
```

The first run is deliberately restricted to 1,000 deterministic users and 20,000 train-known items. It is a feasibility/pilot estimand, not a full Yambda benchmark. `train` refuses CPU unless `allow_cpu` is explicitly enabled, refuses overwriting a run, and saves environment/code/data provenance. It selects checkpoints only on the selection interval; `evaluate` is a one-way final test step and refuses changed checkpoints/data.

For the RTX run, first use the config as-is. The pilot uses batch size 512 rather than 64 to avoid leaving a 16 GB GPU underutilized; reduce it to 256 if VRAM is too high. Progress records every 50 optimizer steps with phase, step/total, steps/sec, ETA, loss, KL/reward where relevant, and GPU allocated/reserved memory. Record wall time, peak VRAM, and failures. Do not increase catalog or steps until the pilot is finite and memory-safe. Full 934k-item softmax is a separate scaling experiment, not implied by pilot results.

Low CPU utilization is not by itself a fault: this model is intentionally GPU-bound and uses small fixed tensors. Check `nvidia-smi` for GPU utilization and VRAM. If GPU utilization is also low, inspect `progress.jsonl` and the phase/ETA before increasing CPU threads.

## Data caveats

- Timestamps are 5-second bins; rows tied within a user/bin are not ordered as true behavior. Ambiguous timestamp groups are excluded from target construction.
- `played_ratio_pct > 100` is valid rewind/replay behavior. The completion reward is bounded at 1; the seconds reward is a capped proxy, not wall-clock listening time.
- Logged listens are conditional on exposure and user choice. Reward-model scores do not identify causal recommendation effects.
- Repeats are allowed for every policy and reported separately.
- User bootstrap intervals are conditional on a training seed; repeated seeds are required for robustness.

## Git handoff

Do not commit `.venv`, downloaded data, model checkpoints, or results. After review:

```bash
git add .
git commit -m 'Add auditable GPT-2 recommendation RL experiment'
git remote add origin <REMOTE_URL>
git push -u origin main
```

The author will provide the remote URL after local tests and independent code audit are complete.
