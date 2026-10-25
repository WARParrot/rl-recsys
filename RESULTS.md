# Pilot seed 42 — uploaded run results

## Verified frozen-checkpoint reanalysis (analysis revision e3cf359)

The updated Windows/GPU inference completed in **155.75 seconds**, with **535.5 MiB peak allocated CUDA memory** (not total process VRAM; excludes setup). All 12 completion hashes, embedded source identity, exported historical input arrays and all ten policies' target-based Recall/NDCG arrays were verified. No new training occurred; the test remains exploratory reuse.

- **History sensitivity:** original completion MAE 0.339790; cross-user whole-history replacement 0.363707 (**+7.04%, CI [+3.43%, +10.46%]**); all-UNK 0.378121 (**+11.28%, CI [+8.40%, +14.10%]**). This establishes useful dependence within the fitted RM, not superiority over a newly trained item-only model or causal personalization.
- **Reranking:** selection-only search over `[0, .1, 1, 10]` selects **1**, identical to the historical weight. Weight 10 fails the relevance floor. The main negative RL conclusion is unchanged.
- **Matched step-five RM scores (579 users):** SFT policy/random/popularity/replay = **0.759157 / 0.761059 / 0.755318 / 0.723782**; selected RL = **0.756589 / 0.761714 / 0.755798 / 0.723666**. Score growth under random appends is not distinguishable from policy append in paired saved-trajectory intervals. No RL-specific hacking mechanism is established.
- **Replay limitation:** actual retained contexts are gapped (mean 133,147.9 native units), not consecutive listens or time-matched sessions. Observed labels are scored only at their actual contexts/targets.

[Results, plots, selection and review](evidence/reanalysis-e3cf359/README.md). This is a compact excerpt; large arrays/weights were available for review but are not committed. Item-only training, new independent RM audit, alternative reward, continued SFT and training-seed repetitions remain pending execution.

## Historical training run

**Artifact:** `pilot-seed42-submission-2.zip`  
**Code revision:** `68578e4ae753214b506d146e04f8207686b3c852`  
**Status:** `exploratory_reuse` — the historical test period was inspected before the amended protocol and is not a fresh confirmatory evaluation.

## Execution

- GPU: NVIDIA GeForce RTX 5060 Ti
- Device: CUDA
- Environment: PyTorch 2.11.0+cu128, Transformers 5.20.2, NumPy 2.2.6, PyArrow 26.0.0
- Tests on the run machine: **105 passed**
- Frozen temporal cutoffs: `[19853890, 22072765, 24049600]`
- Catalog: 20,000 train-known items
- Test retained examples: 10,000
- Test eligible target coverage: see `results.json` in the release artifact

The bundle contains valid SHA-256 entries for all 23 included files. The evaluation completion manifest validates `results.json`, `per_example.npz`, and `evaluation-provenance.json`. Model checkpoints and the raw parquet dataset are not included; checkpoint hashes are retained in `READY_FOR_TEST.json`.

## Reward audit and gate

The pre-RL calibration audit passed for completion and disabled seconds optimization:

| Head | Relative MAE vs fit item-mean | Gate |
|---|---:|---|
| Completion | -3.25% | enabled |
| Seconds proxy | +5.61% | disabled |

The calibration role also participates in checkpoint selection, so this is time-forward from fit but not an independent confirmatory reward audit. The reward model is observational and does not identify causal intervention effects.

## Policy results

The selected policy was `rl_completion_h1_b0.01`, chosen using the declared relevance floor and exact one-step expected completion proxy.

| Policy | Exact expected reward | Eligible NDCG@10 | Recall@10 | Coverage | HHI |
|---|---:|---:|---:|---:|---:|
| SFT | 0.722888 | 0.044494 | 0.036971 | 0.29535 | 0.001028 |
| Reward rerank | 0.723723 | 0.045192 | 0.036971 | 0.29510 | 0.001031 |
| Selected RL H=1, beta=0.01 | 0.723430 | 0.044024 | 0.035750 | 0.28760 | 0.001179 |
| RL H=1, beta=0.1 | 0.723235 | 0.045577 | 0.036639 | 0.29245 | 0.001067 |
| RL H=1, beta=1 | 0.723054 | 0.045302 | 0.036799 | 0.29475 | 0.001030 |
| RL H=5, beta=0.01 | 0.723324 | 0.041887 | 0.034530 | 0.29495 | 0.001337 |
| RL H=5, beta=0.1 | 0.723220 | 0.045241 | 0.035438 | 0.29535 | 0.001054 |
| RL H=5, beta=1 | 0.723039 | 0.045227 | 0.036862 | 0.29585 | 0.001025 |

Relative to SFT, the selected RL policy had:

- exact expected reward: **+0.076%**, paired user-bootstrap 95% CI approximately **[+0.070%, +0.082%]**;
- eligible NDCG: **-1.06%**, paired user-bootstrap 95% CI approximately **[-4.50%, +2.61%]**.

The declared H2 threshold was a 5% reward improvement with an eligible-NDCG non-inferiority floor. H2 is therefore **not supported**. Reranking produced a larger predicted reward gain and higher eligible NDCG without RL training. H=5 with beta=0.01 showed the clearest relevance degradation (approximately -5.86% eligible NDCG versus SFT).

## Conclusions

- H1 reward-model improvement: **not a confirmatory claim**; the calibration audit passed for completion but is not independent and the test period is reused.
- H2 RL improvement: **not supported**.
- H3 multi-step behavior: exploratory only; synthetic appended actions are not observed user transitions.
- No causal user-satisfaction, retention, or product-impact claim is supported.

The release asset is the authoritative machine-readable artifact for this run. The report should not treat these numbers as a fresh preregistered result or as evidence of causal recommendation improvement.

## Completion-only follow-up (2026-10-25)

A separate completion-only reward experiment reused the full prepared data and SFT from the later research suite. It is **not** part of the historical pilot table above. Full results and protocol are in [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).

- All **109** internal artifact checksums in `completion-20261025-203706-879.zip` were verified; archive SHA-256 is `1de1edd6d02f4f8692268234f5f1269e3056ab011c3f3c4ee922cfb745949071`.
- On the same late-calibration population (3,065 examples / 605 users), historical joint-RM completion MAE was **0.347538** versus item-mean **0.348721** (−0.339%; 95% paired-user interval [−3.379%, +2.720%]). Completion-only RM MAE was **0.349361** (+0.183%; interval [−2.794%, +3.064%]). Neither met the required −2% point-gain gate.
- Accordingly, **no RL, RL+CE or reranking policy was trained** in this follow-up. The three continued-SFT controls changed eligible NDCG by +1.58%, +2.20% and +0.89%; all paired intervals include zero. These are exploratory reused-test estimates, not a demonstrated consistent improvement.
- Completion-only checkpoint selection favored step 100 by early-calibration MSE; training loss later fell as calibration loss worsened. This is consistent with overfitting after the early checkpoint, not evidence that the selected checkpoint itself was chosen incorrectly.

This is a negative result for the hypothesis that removing the auxiliary seconds objective would make the RM pass the existing gate. It does **not** test whether RL+CE works, since reward-dependent arms were correctly stopped. The test and audit periods were previously inspected; no causal, satisfaction or retention claim follows.
