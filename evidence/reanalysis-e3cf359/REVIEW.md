# Returned reanalysis: verification and interpretation

## Scope and identity

Reviewed `analysis.7z` and `historical-verification.json` against the supplied historical ZIP, nine checkpoints and prepared data. Analysis source revision: `e3cf3596b07ddeea672c25e4d10122f39e2c770a`. Historical training revision: `68578e4ae753214b506d146e04f8207686b3c852`. This is frozen-checkpoint inference on reused data, not new training or a fresh confirmatory test.

- All 12 files bound by COMPLETE.json match their hashes.
- Embedded Python source bytes match their recorded hashes and local source after CRLF normalization.
- Historical checkpoint/data hashes agree with the previously supplied assets.
- All exported arrays corresponding to historical prepared inputs agree exactly.
- All ten policies' saved Recall/NDCG arrays reconstruct from true targets; eligible aggregates agree.
- History donors form a global cross-user permutation; NPZ arrays are pickle-free.
- Reported inference runtime: 155.75 seconds; peak allocated CUDA memory: 561,502,208 bytes (~535.5 MiB). This excludes setup/import/CPU verification and is allocated tensor memory, not total process/driver VRAM.

## Main policy result remains negative

| Policy | Eligible expected completion RM | Eligible NDCG@10 | All-target Recall@10 |
|---|---:|---:|---:|
| SFT | 0.729139 | 0.044494 | 0.036971 |
| Tuned rerank | 0.729961 | 0.045192 | 0.036971 |
| Selected historical RL | 0.729696 | 0.044024 | 0.035750 |

Selected RL versus SFT: predicted reward +0.07634%, paired user-bootstrap 95% interval [+0.07049%, +0.08226%]. Eligible NDCG -1.056%, interval [-4.498%, +2.611%]. Practical reward threshold is not reached; declared noninferiority margin is not established. These intervals condition on fixed training checkpoints and reused data.

Rerank search on selection picks weight 1 from [0, .1, 1, 10]. Weight 10 has higher proxy but violates the relevance floor; it is correctly rejected. The selected weight equals the old weight, so tuning confirms rather than changes that comparator. An exploratory paired comparison versus selected RL gives rerank +2.65% eligible NDCG, interval approximately [+0.008%, +5.71%], and +0.0363% predicted reward. The NDCG lower bound is extremely close to zero; no broad or multiplicity-adjusted superiority claim is warranted.

## New evidence: useful dependence on history within the fixed RM

| Input history | Completion MAE | Relative change versus original |
|---|---:|---:|
| Original | 0.339790 | reference |
| Cross-user shuffled | 0.363707 | +7.04%, CI [+3.43%, +10.46%] |
| All nonpadding tokens replaced by UNK | 0.378121 | +11.28%, CI [+8.40%, +14.10%] |

The correct history is useful to this fitted model on the observed-action sample, compared with these perturbations. This is stronger evidence than architecture alone. It does not establish superiority over a separately trained item-only neural model, robustness across shuffles/seeds, or valid rewards for unseen context-action pairs. All-UNK history is an out-of-distribution intervention. Completion RM versus item mean remains uncertain; seconds RM remains worse than item mean.

## New evidence: synthetic-context score increases are not specific to RL

Same matched 579-user cohort, step-five RM scores:

| Context evolution | SFT | Selected RL |
|---|---:|---:|
| Policy append | 0.759157 | 0.756589 |
| Random append | 0.761059 | 0.761714 |
| Popularity append | 0.755318 | 0.755798 |
| Actual retained observed-context replay | 0.723782 | 0.723666 |

All modes score policy proposals; random/popularity describe history updates, not which scored action produced an observed reward. Random append obtains the largest step-five proxy point estimate in both policies. The auditor remains roughly flat; average absolute disagreement is lower at step five in synthetic modes than at the start. A growing-disagreement or RL-specific reward-hacking narrative is unsupported.

The evidence is consistent with generic sensitivity/extrapolation under synthetic histories. It does not identify a unique mechanism or real-user utility loss. Observed replay is explicitly gapped: mean adjacent retained timestamp gap is 133,147.9 native bins, maximum 1,640,445. If interpreted as 5-second bins, the mean gap is ~7.7 days, maximum ~94.9 days. Consequently this is not a matched consecutive/session transition experiment. One retained rollout draw per mode is also not full sampling robustness.

## Which issues are closed?

Closed for this review: missing evaluation targets/masks, missing locally supplied checkpoints/data, historical metric reconstruction, current-source provenance, executed selection-tuned reranking, history perturbation controls, executed random/popularity/observed-context controls, measured inference resource use. Public independent reproduction still requires publishing the inputs/checkpoints in an accessible artifact store; local upload is not public release.

Still open: separately trained item-only comparator; fresh independent RM audit; alternative-reward RL and continued-SFT training; genuinely repeated training seeds; untouched evaluation period; consecutive observed transitions; user acceptance/return model and causal product conclusions (the latter are not required to complete the assignment).

## Recommendation

Integrate these new diagnostics into the main README and publish a small analysis-only release referencing historical hashes; there is no need to duplicate large historical assets in each upload. Keep the negative practical conclusion. Before new RL training, prioritize fresh item-only RM comparison and independent reward auditing. Do not expand the RL grid merely to obtain a positive proxy result.
