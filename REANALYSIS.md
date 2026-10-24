# Reproducibility and controlled follow-up

## What is closed, and what still needs execution

The supplied nine checkpoints and prepared data match the historical manifests. CPU verification now reconstructs Recall/NDCG directly from target IDs, verifies every saved relevance array, and reconstructs eligible populations, expected-reward aggregates and paired user intervals. This passed on all nine historical policies. It does not replace model inference.

Implemented, not yet established as new empirical results:

- complete ZIP64 export of historical checkpoints/data plus new evaluation contexts, targets, masks, row IDs and vocabulary;
- separate current-source reanalysis provenance, immutable historical manifests;
- reranking weights `[0, .1, 1, 10]`, selected before test inference with the same eligible relevance floor;
- original/all-UNK/cross-user-shuffled history sensitivity on observed target labels;
- matched policy/random/popularity append and observed-context replay controls for SFT and selected historical RL;
- exact two-step policy-gradient regression tests, including future KL and leave-one-out baseline;
- opt-in new item-only/attention reward models, independent late-calibration audit, earlier checkpoint evaluation, continued-SFT and completion-minus-repeat controls;
- explicit separate inference for the trained follow-up arms, with per-seed results and descriptive conditional seed dispersion.

The reused test remains exploratory. No software change makes it untouched. No acceptance/return simulator or causal identification was added. A genuinely later raw-data period must be established separately before confirmatory claims.

## Windows: historical frozen-checkpoint inference only

Put these files in one directory, e.g. `C:\recsys-uploads`:

```text
pilot-seed42-submission.zip
prepared.npz
sft.pt
reward.pt
auditor.pt
rl_completion_h1_b0.01.pt
rl_completion_h1_b0.1.pt
rl_completion_h1_b1.pt
rl_completion_h5_b0.01.pt
rl_completion_h5_b0.1.pt
rl_completion_h5_b1.pt
```

From cmd.exe in this updated checkout:

```bat
run_windows.bat reanalyze "C:\recsys-uploads"
```

An optional third argument chooses a fresh output:

```bat
run_windows.bat reanalyze "C:\recsys-uploads" "runs\reanalysis-02"
```

There is no `run_windows.dll`. The `.bat` mode installs dependencies, tests, verifies/imports historical assets, independently reconstructs saved metrics, performs new CUDA inference and packages all reproducibility inputs. It never starts training or invokes the old final-evaluation guard on modified source. Without `reanalyze`, the original script still launches the historical training workflow; do not omit the argument accidentally.

Review the official PyTorch wheel selector for your GPU/driver if cu128 installation fails. CUDA must be usable; this workflow does not silently fall back to CPU. Default inference batch size is 128 and its time budget is 240 minutes; neither is a measured runtime guarantee. The old 512 evaluation batch can be too expensive for full-catalog reranking. All models are loaded with `weights_only=True`.

Output:

```text
runs/reanalysis-seed42/
  historical/                     # verified original metadata, all weights/data
  historical-verification.json    # target-based reconstruction
  analysis/
    provenance.json               # new analysis source bytes + historical hashes
    selection.json                # weight search before test inference
    evaluation-inputs.npz          # masks/targets/contexts/labels/row IDs/vocab
    per_example.npz                # inference and diagnostic arrays, no pickle
    results.json                   # policies, RM, controls, paired intervals/resources
    REPORT.md                      # summary and historical learning curves
    *.svg                          # reward/relevance, learning and rollout plots
    COMPLETE.json                  # emitted last, hashes all analysis outputs
  reanalysis-complete.zip          # full inputs/checkpoints and output checksums
```

Files are intentionally larger than the former metadata-only ZIP. Keep ample free disk for uploads, an imported copy and the archive. Partial output is not a valid completed analysis; inspect it and choose a fresh directory for an explicit retry. Never edit hashes or remove historical completion markers to run the old `evaluate` command again.

## Manual commands / smaller inference batch

The CLI uses separate import, CPU verification, GPU inference and packaging stages:

```bash
rl-recsys import-run --bundle uploads/pilot-seed42-submission.zip --assets uploads --output runs/imported
rl-recsys verify-run --run runs/imported/runs/pilot-seed42 --data runs/imported/data/prepared --output runs/verified.json
rl-recsys reanalyze --run runs/imported/runs/pilot-seed42 --data runs/imported/data/prepared --output runs/reanalysis --batch-size 64 --max-minutes 360
rl-recsys bundle-reanalysis --imported runs/imported --analysis runs/reanalysis --output artifacts/reanalysis-complete.zip
```

`--skip-diagnostics` is an explicit reduced inference mode; do not report its output as if append/history controls ran. Historical sampled estimates can differ from new draws/batching; exact rewards and deterministic rankings should be compared with numerical tolerance, not bitwise CUDA equality.

### What the controls establish

History permutations are a sensitivity test on the fixed fitted model. All-UNK is not a trained item-only model and may be out of distribution. Cross-user histories preserve global whole-history marginals; donor rows are exported. Labels remain associated with the originally logged targets, not hypothetical actions.

Matched rollout cohorts contain users with at least five retained rows. Observed-context replay uses actual successive **retained** contexts and exports elapsed timestamp gaps. Because preparation subsamples targets, these are not necessarily consecutive listens or one session. Every append mode scores policy proposals; observed-target error is separately evaluated only at recorded contexts. Controls do not manufacture a ground-truth reward for synthetic actions. This cohort differs from all-test contexts and is labeled separately.

## Optional new training: not part of the Windows inference command

```bash
rl-recsys followup --data runs/imported/data/prepared --run runs/imported/runs/pilot-seed42 --output runs/followup --seeds 42 43 44 --steps 1000 --repeat-penalty .1
```

This is an explicit GPU training job, potentially expensive. Freeze options before running. Each seed:

1. splits eligible calibration at its median unique timestamp, keeping timestamp ties together;
2. trains fresh attention RM and learned item-only RM on fit, selecting checkpoints on early calibration every 50 updates;
3. evaluates RM on the later calibration half not used for those checkpoint selections; gate failures are retained and stop policy training for that seed;
4. if completion passes its item-mean/constant gate, trains continued-SFT, H=1 completion RL and H=1 completion-minus-repeat RL from the same historical SFT, with equal optimizer-step counts and the same learning rate (`rl_lr`);
5. samples users uniformly then their events for all three policy arms; selects checkpoints on selection only under the relevance floor and unpenalized completion objective; repeat statistics remain separate;
6. records curves, parameters, hashes and all failed gates/floors without evaluating test.

The repeat penalty is a declared `.1` intervention, not selected using test. This first follow-up does not sweep lambda. The reward gate still uses predictive MAE versus fit-only means; a pass is not validation of counterfactual recommendations. Comparison against learned item-only error is reported separately.

Then, explicitly run exploratory test inference **after all seeds are frozen**:

```bash
rl-recsys evaluate-followup --data runs/imported/data/prepared --run runs/imported/runs/pilot-seed42 --followup runs/followup --output runs/followup-evaluation --batch-size 128 --max-minutes 240
```

This exports target-complete evaluation inputs and per-example predictions, evaluates every selected arm, reports gate/floor failures and within-seed paired comparisons. Seed-specific RM reward scales are not common truth: summaries retain per-seed baselines and aggregate relative effects only. Do not pick a favorable seed/arm after this command. Preserve the entire follow-up directory (weights and manifests) alongside inference outputs for sharing; the historical-only bundle command does not automatically include separately trained follow-up weights.

**Important:** these seeds vary newly trained reward/control arms conditional on one historical SFT and prepared user sample. They are not full-pipeline baseline retraining and cannot close end-to-end seed robustness. The historical test remains reused, and the late-calibration labels have a history of prior inspection. This split is independent of the **new** RM checkpoint selector, not independently untouched research data.

## Existing overfitting evidence

Historical supervised selection (from training.json): SFT reaches its best recorded calibration loss around step 2000; RM's best is the first recorded evaluation at step 500, then worsens. Best checkpoints were correctly restored. `analysis/REPORT.md`, SVG plots and results.json now expose selected steps and full numeric curves rather than implying the last training checkpoint was used. Earlier follow-up evaluation helps resolve whether RM improves before step 500.

## Verification and remaining limitations

Tests cover imported-manifest corruption/traversal, target-based relevance reconstruction, pickle-free input/diagnostic exports, exact policy gradients, control semantics and tiny real-model CPU inference. GPU execution, Windows batch execution and new training results must be validated on the target machine. Current conclusions stay negative/diagnostic until those outputs are available.
