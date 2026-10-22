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

The script installs Python 3.11 with `winget` if needed, creates `.venv`, installs the CUDA PyTorch wheel, runs the CUDA forward/backward preflight and tests, downloads/prepares Yambda, then starts exactly one pilot training job and its final evaluation. It does not use PowerShell activation, so it can be started directly from `cmd.exe`. Logs are written to `logs\\`.

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

The RTX 5060 Ti has to pass that real forward/backward check. If the wheel rejects the GPU architecture, install the PyTorch build recommended for the installed driver; do not proceed on a CPU fallback.

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

For the RTX run, first use the config as-is. Record wall time, peak VRAM, and failures. Do not increase catalog or steps until the pilot is finite and memory-safe. Full 934k-item softmax is a separate scaling experiment, not implied by pilot results.

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
