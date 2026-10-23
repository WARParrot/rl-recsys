# RL RecSys Lab — separate experiment

This repository is intentionally separate from the existing feature-search artifact. It implements an auditable **model-based policy optimization** experiment:

1. Yambda-50M `listens` audit and strict temporal contexts;
2. GPT-2-style decoder over item IDs for next-item prediction;
3. independent history-conditioned reward model;
4. simulator-on-policy REINFORCE with KL to frozen SFT reference;
5. full restricted train-known catalog evaluation, reranking control, reward-model auditor, and five-step synthetic rollout diagnostics.

It is **not** causal offline policy evaluation and does not establish user satisfaction or retention. Generated items are not observed user actions. The primary report must remain qualified unless returned metrics satisfy the declared protocol in `PLAN.md` and the generated `audit.json`, `provenance.json`, `selection.json`, and `results.json` agree.

## Results and revised measurement

Read [REPORT.md](REPORT.md) for the **historical negative-result pilot** and [RESEARCH_UPGRADE.md](RESEARCH_UPGRADE.md) for the post-audit protocol amendment. The current code has not produced new GPU research results. The historical run has an unresolved source-byte hash discrepancy; its internal metadata consistency is not proof of exact source identity.

Current evaluation uses 16 action samples per context, four synthetic rollout replicates, held-out validation for both reward heads, and consistent eligible-target reward selection/comparison. Rollouts additionally report initial-list overlap and policy entropy; these measure recommendation change, not user-state OOD or satisfaction. The higher sampling budget needs a fresh runtime preflight. New training provenance embeds exact Python source bytes and installed package versions; final evaluation rejects code-hash changes and saves `evaluation-provenance.json`. Do not alter historical manifests to bypass this guard.

## One-command Windows setup

On Windows, open **Command Prompt** in the cloned repository and run:

```bat
run_windows.bat
```

The script installs Python 3.11 with `winget` if needed, creates `.venv`, installs the CUDA PyTorch wheel, runs the CUDA forward/backward preflight and tests, downloads/prepares Yambda, then starts exactly one pilot training job and its final evaluation. It does not use PowerShell activation, so it can be started directly from `cmd.exe`. Training output is visible in the console and simultaneously written to `logs\\training.log`; structured progress is written to `runs\\pilot-seed42\\progress.jsonl`.

If the script stops at the PyTorch step, check the official CUDA wheel selector at [pytorch.org](https://pytorch.org/get-started/locally/) and update the wheel URL in `run_windows.bat`. Do not continue if the preflight reports `CUDA is unavailable`.

It creates `artifacts\\pilot-seed42-submission.zip` after a successful evaluation. This bundle contains the frozen config, dataset audit, reward audit/gate, provenance, selection and training manifests, final metrics, per-example arrays, completion manifests, progress and all stage logs, plus `SHA256SUMS.json`. Upload that single ZIP for result review; the raw parquet dataset and model checkpoints are intentionally not copied into the bundle because they are large and are protected by hashes in the manifests.

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

The primary test period has already been inspected and protocol changes were made after that inspection. The existing sample is therefore descriptive and exploratory, not a fresh confirmatory set. The checked-in `configs/pilot.json` retains this restricted pilot setup for smoke/readiness work; before any full GPU run, the operator must freeze a genuinely later untouched evaluation period or explicitly accept an exploratory reuse analysis. The new generated `results.json` labels reused-test outcomes accordingly.

Do not launch the full historical grid directly after preflight. Training now persists `reward_audit.json` after supervised fitting and before RL: both reward heads are compared with fit-only constant and item-mean baselines on eligible calibration targets. Calibration is time-forward from fit but also selects checkpoints, so this is not an independent audit period. A head must beat both baselines in user-macro MAE; `reward_audit.min_relative_improvement` additionally requires the declared relative improvement over item-mean (2% in the pilot; strict improvement if omitted). Failed heads are skipped, and if no requested head passes, training raises before RL and emits no `READY_FOR_TEST.json`. Audit, checkpoint, and provenance hashes are bound into the READY manifest. The architecture auditor remains a disagreement diagnostic, not independent truth.

Exact catalog-integrated `exact_one_step_expected_reward` is the primary H=1 model-based diagnostic, including eligible-target candidate selection and paired comparisons; sampled rewards/rollouts remain supplemental. Per-context exact values are retained in `per_example.npz`. This is predicted completion utility, not measured user impact. A genuinely separate time-forward reward-audit period is still preferable before a confirmatory experiment. The current CLI preflight checks a small representative policy and reward forward/backward pass, not end-to-end runtime or peak memory for the full grid. Only proceed to a reduced H=1 experiment if measured GPU throughput/memory fits the budget; retain H=5 as a later extension.

Low CPU utilization is not by itself a fault: this model is intentionally GPU-bound and uses small fixed tensors. Check `nvidia-smi` for GPU utilization and VRAM. If GPU utilization is also low, inspect `progress.jsonl` and the phase/ETA before increasing CPU threads.

## Frozen temporal protocol and audit metadata

`data.cutoffs` contains exactly three strictly increasing nonnegative int64 values in native dataset timestamp units (5-second bins), not Unix seconds. Roles are `fit < t0`, calibration `[t0,t1)`, selection `[t1,t2)`, and test `>= t2`; equality goes to the later role. The pilot freezes the historical reported values `[19853890, 22072765, 24049600]` for compatibility. Their historical quantile origin is not fresh preregistration; new confirmatory periods must be frozen independently before inspecting future events. Optional `quantiles` alongside cutoffs are descriptive only and never used to form boundaries.

`audit.json` preserves `cutoffs`, `cutoff_source`, role counts, and `audit_metadata.future_information_used_for_cutoff_formation`. Legacy quantile-only configs must explicitly set `data.allow_derived_cutoffs=true`; their audit labels the source `legacy_sample_quantiles_future_inclusive`. This escape hatch reproduces exploratory behavior, not a leakage-free protocol. Frozen cutoffs use `frozen_calendar_config`; that label describes how this preparation obtained boundaries, not independent proof of preregistration. Histories remain rolling observed, strictly earlier singleton events.

## Data caveats

- Timestamps are 5-second bins; rows tied within a user/bin are not ordered as true behavior. Ambiguous timestamp groups are excluded from both targets and histories. Production configs freeze three calendar cutoffs; legacy quantile cutoffs require `allow_derived_cutoffs=true` and are audited as future-inclusive exploratory splits.
- Evaluation reports empirical action-support/OOD diagnostics: recommendation actions absent from fit targets, their rate, and reward-model score means by supported/unsupported group. These are model/data-support diagnostics, not propensities, causal OPE, or proof of reward-model validity.
- `played_ratio_pct > 100` is valid rewind/replay behavior. The completion reward is bounded at 1; the seconds reward is a capped proxy, not wall-clock listening time.
- Logged listens are conditional on exposure and user choice. Reward-model scores do not identify causal recommendation effects.
- Repeats are allowed for every policy and reported separately.
- User bootstrap intervals are conditional on a training seed; repeated seeds are required for robustness.

## Research artifact handoff

Keep `.venv` and downloaded data out of Git. Publish checkpoints, `per_example.npz`, metadata, logs, checksums and the resolved environment through a release/artifact store, not only prose tables. Source bytes are embedded in new provenance to preserve platform-specific byte identity. Seeds do not promise bitwise cross-device reproducibility. The generic CUDA install command above is a supported fresh setup, not an exact reconstruction of the historical Windows environment.
