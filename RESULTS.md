# Pilot seed 42 — uploaded run results

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
