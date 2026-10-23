# Separate RL recommendation experiment — declared before training

Status: implementation/pilot preparation; no quality claims. Decision owner: user.
Budget: one RTX 5060 Ti 16GB, 48h after GPU access. Current BB machine has no NVIDIA device; GPU execution blocked.

## Scope and estimands

Train GPT-2 from random initialization on item IDs, then a history-conditional reward model, then KL-regularized on-policy REINFORCE. Compare policies on identical held-out contexts/candidates. This is model-based policy optimization, NOT causal offline policy evaluation. No logged propensities, no IPS/DR claims. Generated items are hypothetical context, not observed feedback.

Primary relevance estimand: user-macro NDCG@10 on next logged listen from strictly earlier timestamp history, within train-known eligible catalog. Report target coverage separately; unknown/excluded targets score zero in all-target view. Repeat consumption allowed for ALL policies. One-step top-10 and generated five-item lists are distinct estimands and reported separately.
Primary optimization estimand: expected frozen reward-model bounded completion under policy sampling. Separate top-10 average reward is a diagnostic, not equivalent to sampling return. RM validation on observed listens does not establish accuracy on unexposed items.

## Proposed steps / files

1. `PROTOCOL.md`, `sources/`, `configs/pilot.json`: predeclare data selection, split, hypotheses, source limits and budget. Save primary-source snapshots, access dates and hashes. Verify source revision.
2. `rl_recsys/data.py`: streaming Parquet audit + deterministic UID sample; global time splits; train-only catalog; discard ambiguous multi-item timestamp groups from ordering (report counts); build fixed-length contexts using strictly earlier timestamps, including unknown tokens as context. Tests for boundary equality, no future items, ties, target eligibility, deterministic sampling.
3. `rl_recsys/models.py`: GPT2Model + tied item head; predict only last-token logits (avoid B x L x V training allocation); reward head with separate history encoder; train reward vs simple baselines. Tests for causal masking, candidate indexing, padding, bounded outputs.
4. `rl_recsys/experiment.py`: SFT, reward training, frozen reference and reward, REINFORCE with detached baseline and exact full-catalog KL, horizon 1/5; uniform decode rules. Save checkpoints, hashes, config, seeds, environment, independent reward auditor. Tests for gradient isolation and deterministic CPU smoke.
5. `rl_recsys/evaluate.py`: popularity / SFT / rerank / RL on identical full eligible candidates, deterministic ties; paired user bootstrap, coverage, repetition, popularity/duration, per-step reward. Explicitly isolate confirmatory and exploratory analyses.
6. `tests/`, CLI, `README.md`, `REPORT.md`: runnable synthetic smoke (engineering only), data preparation and bounded training jobs; final report from actual artifacts, independent audits and limitations.

## Experiment design (provisional pending independent audit)

The frozen pilot uses global timestamp quantiles 70/80/90, yielding fit/calibration/selection/test roles of 70/10/10/10 in the fixed sampled-user population before catalog filtering. Calibration is used for early stopping; selection is used for RL candidate choice; test remains untouched until final evaluation. Any future 80/10/10 design is a separate protocol, not this pilot. No checkpoint selected on test. Histories can incorporate earlier logged validation/test observations in rolling teacher-forced evaluation, without weight updates. Clearly label this protocol.
Pilot: deterministic hash sample of 1000 users, train-only top-20000 items; this is a restricted-population/catalog feasibility experiment, NOT official Yambda benchmark. Full-catalog expansion only after throughput/memory measurement. Never choose users/items based on future activity. Retain excluded target counts. Context length 64; GPT2 4 layers, 128 hidden, 4 heads; start batch 64 with bounded max optimizer steps. Main configuration must be frozen before examining final test metrics.

Suggested hypotheses, not results:
- H1: history-aware RM reduces test user-macro MAE vs train-only item-mean by >=2% relative; paired user-bootstrap 95% interval must exclude no improvement. Only observed-listen population.
- H2: KL-regularized RL improves sampled proxy return >=5% relative to SFT, while relevance noninferiority lower bound for relative NDCG@10 change >-2%; both criteria required. Proxy gain alone is not a recommendation-quality win.
- H3 (exploratory): five-step rollouts exhibit larger model disagreement/context shift than one-step; no causal or confirmatory claim.
Seed 42 pilot; three seeds (42/43/44) main if time permits. 2000 user bootstrap replicates; seeds reported separately, do not treat user bootstrap as training-seed uncertainty. Small/degenerate samples => inconclusive.
Hyperparameters selected on selection only: KL beta {0.01,0.1,1}; bounded completion vs clipped seconds normalized by train-only scale; horizons {1,5}. Run fixed budget, retain failed configurations. Primary candidate selected by maximum selection reward subject to relevance floor; if none qualify report failure, do not relax floor. Report trade-off grid; no multiplicity-free significance claims for exploratory slices.

## Compute / stopping / rollback

0–4h data audit, setup, tests, throughput. 4–14h SFT+RM+baselines. 14–34h bounded RL grid. 34–44h repeated seeds/ablations if feasible. 44–48h final fixed evaluation and report. GPU clock starts when access is established; not a promise of measured runtime. Stop for leakage, RM baseline failure, nonfinite losses, memory excess, or insufficient target coverage; document next alternative (rerank or better reward/data) instead of searching test for a favorable result.
All new files in `rl_recsys/`; existing `recsys_corrected_artifact/` untouched. No migration. Rollback by removing the new directory. This checkout has no Git repository: use code-tree SHA256 manifest until a VCS revision is available.

## Verification matrix

Data: strict history time, ties, split boundary atomicity, train-only vocabulary, unknown-target coverage.
Models: no future attention, padding, no special-token recommendation, output ties.
RL: detached RM/reference, logprob gradient, identical action support and temperature, exact KL semantics, no cross-example baseline bias.
Evaluation: macro by user, all-target/eligible denominators, tie convention, paired bootstrap, degenerate intervals, identical candidate lists.
Reproduction: pinned environment, dataset revision+SHA, code manifest, config+seed, commands and finite smoke artifacts.
Independent audits: methodology thread and data/resource thread; later adversarial code audit.
