# Experiment contract: next research run

**Status: implemented prospective Experiment A contract; no completed Experiment A research-suite result is claimed.** The executed evidence is the historical joint-RM pilot plus the separate completion-only follow-up. This contract does not change historical runs. [REWARD_DESIGN.md](REWARD_DESIGN.md) documents shipped commands, artifacts and limitations. Experiment B training remains prospective and unimplemented; stage5 adds only opt-in observed-label ensemble hypothesis validation ([ENSEMBLE_AUDIT.md](../guides/ENSEMBLE_AUDIT.md)). No training is authorized by this document alone.

## 1. Primary estimand and claim boundary

Primary serving policy is **one stochastic action**, sampled from `softmax(logits / T)` at **T=1**, on the fixed fit-only catalog, with repeats allowed. Primary endpoint is user-macro, eligible-context, full-catalog expected **unpenalized completion RM**:

`V(pi) = mean_users mean_eligible_contexts sum_a pi(a|s) * frozen_RM(s,a)`.

Eligibility means the observed next target is in the catalog; report eligible counts, users and coverage of all targets. The primary reward population is therefore conditional on eligibility, not the whole service population. Report all-target results separately. Use the **same frozen evaluator RM within a matched comparison**, including for rerank and continued-SFT; never average absolute scores from different fitted RMs.

Deterministic top-10 Recall/NDCG, repeat, coverage and HHI are separate guardrails/diagnostics, not reward for the stochastic serving rule. If intended deployment is deterministic top-K, preregister a new primary list estimand (including position weights and action semantics) and align optimization/evaluation before running; the current exact expectation is not top-K utility.

This is predictive/model-based optimization on logged listens. RM accuracy, exact summation and chronological splits do not identify causal policy value, satisfaction or retention. `is_organic` is not a propensity. No labels exist for hypothetical actions or synthetic append transitions; the auditor is not gold reward. Retention requires mature future windows, verified timestamp units, session definitions and censoring; causal benefit requires a separately justified identification design, preferably randomized exposure.

## 2. Frozen roles and reward audit

Freeze dataset revision/hash, sampled users/events, catalog, temporal boundaries, prepared-data hash, model/config/source hashes and seed roles before training. Fit trains models and fit-only baselines/statistics; early calibration selects supervised checkpoints (and fits any prespecified calibrator); late calibration audits reward without fitting it. Selection chooses policies; test is evaluated only after **all** arms and choices are frozen.

Audit completion on observed eligible targets, with counts/users and paired user-bootstrap uncertainty:

- User-macro MAE/RMSE versus fit-only constant, item mean and a **fresh learned item-only RM**, trained with matched data/checkpoint budget. Report paired differences; history perturbations supplement, not replace, the item-only comparison.
- Calibration bins with predicted/observed means, signed bias and support. Preserve existing 0.1-width bins; add upper-score **top 10% and top 1%** groups on late audit, with explicit tie handling and realized group sizes. Report MAE and overprediction there. Tiny tails are inconclusive, not evidence of safety.
- Duration (`<180`, `180–300`, `>=300` seconds), fit-popularity (fit-count tertiles, ties together), repeat-in-context versus novel, and organic/recommended slices: MAE, signed bias, comparator MAE, counts/users. Freeze thresholds from fit; no tuning slices on test.
- Separate label-backed audit from unlabeled policy-proposal/append diagnostics. High proposal scores or disagreement are not observed errors; logged-target calibration need not transfer to selected actions.

**Operational gate:** retain the existing completion gate: point user-macro MAE improvement >=2% over fit item mean and better than fit constant (`reward_gate`). Report CI and learned item-only comparison even when this gate passes. Passing is permission for an exploratory optimization study, not proof of reliable counterfactual reward. Missing diagnostics block a claim of audit completeness; no new numerical tail gate is invented after seeing results. Failed completion gate skips reward-dependent arms, preserves continued-SFT and all failure artifacts. Seconds remains disabled if its own gate fails.

## 3. Experiment A — single-RM relevance-preserving optimization

Question: does adding supervised anchoring to RL improve the attainable reward–relevance tradeoff, beyond RL alone and cheap controls?

For each fixed prepared dataset and SFT checkpoint, train **one history-aware completion RM** plus the item-only diagnostic comparator, then freeze them. All policy arms start at the identical SFT weights:

| Arm | Objective / role |
|---|---|
| SFT | Unchanged reference, no additional updates |
| Continued-SFT | Next-target CE only |
| RL | H=1 REINFORCE completion minus exact forward-KL to SFT |
| RL+CE | The identical RL loss plus `lambda_ce * next_target_CE` |
| Tuned rerank | `log pi_SFT(a|s) + w * completion_RM(s,a)` over the same catalog |

Keep beta=0.01, T=1, RL learning rate and user-uniform fit sampler matched; CE uses the same fit contexts and eligible target IDs (`target - 2`). Prespecify `lambda_ce in {0.1, 1.0}` and rerank weights `{0, 0.1, 1, 10}`. Include lambda=0 via RL, not a duplicate arm. Give each trained candidate 1,000 optimizer updates and the same configured evaluation cadence; record wall time, memory and selection/inference cost because equal updates are not equal compute. Preserve selected **and final** checkpoints. No H=5 grid in the primary experiment; append rollouts remain exploratory diagnostics. Completion-minus-repeat may remain a separately labeled sensitivity, not a substitute for RL+CE or an assumed music-utility improvement.

**Common selection:** each arm uses the same selection contexts, frozen unpenalized RM and eligible NDCG floor >=98% of unchanged SFT; maximize eligible exact expected completion among eligible checkpoints/hyperparameters. Break equal scores by a frozen stable candidate order (earlier step, then lower coefficient). Retain floor failures as failures, not fallback successes. Apply this to continued-SFT and rerank too. Different search sizes remain a limitation; report every candidate and selection budget. The rerank curve is an **attainable tradeoff for this fixed candidate family and grid**, NOT a formal upper bound on RL or on achievable recommendation quality.

## 4. Experiment B — conditional uncertainty penalty, separate from A

Do not combine ensemble uncertainty and CE changes into the first causal comparison of algorithmic components. B uses the same prepared data/catalog and 3 independently initialized history RMs, calibrated by the prespecified early-calibration procedure, with late calibration held out from member/calibrator fitting. Pipeline seeds with different users/catalogs are **not** an ensemble.

Before penalty optimization, test whether member standard deviation predicts held-out absolute error of ensemble mean on late observed targets: user-bootstrap CI for their Spearman association must have lower bound >0, and top-disagreement-decile MAE must exceed bottom-decile MAE with paired user-bootstrap difference lower bound >0 (report supported users and missing groups). Report duration/popularity/repeat slices to expose confounding. These are operational prerequisites, not a guarantee of epistemic or causal validity. If unsupported, **stop B penalty arms** and report the diagnostic failure; mean-only may be reported descriptively.

Only after that gate compare matched H=1 RL using ensemble **mean** versus **mean - kappa * std**, with `kappa in {0.5, 1.0}` plus mean-only kappa=0, identical KL/budgets/seed roles. Do not add CE in B's primary contrast. Select all B arms using common **unpenalized ensemble mean** and the same relevance floor; report mean, penalty, disagreement and observed-target errors separately. Preserve single-RM A as context, not a score-comparable independent reward truth. Shared exposure bias and confident common error remain possible.

## 5. Seeds, stopping and interpretation

- Primary optimization variability: fixed prepared sample/catalog/splits, fixed SFT and RM; policy optimization seeds **42/43/44**, paired across A arms. This estimates conditional optimization variation only. Record RM/SFT seeds separately.
- Sample robustness is a **separate** outer experiment changing user/sample seed and repeating the pipeline, with within-sample comparisons. `suite.run_suite` now defaults to one fixed sample/SFT/RM with optimization seeds 42/43/44; optional `sample_seeds` changes sampling separately while preserving model initialization seeds. A new seed or old calendar split does not create untouched test data.
- Stop on nonfinite loss/gradients, source/data identity mismatch, invalid/empty eligible roles, configured deadline or resource failure; retain logs and nonzero failure status. Do not extend budgets, replace failed seeds, relax gate/floor, or choose another arm after test inspection. Finish fixed update budgets rather than peeking for favorable test stopping.
- Test bootstrap: paired users, 2,000 replicates, seed 2026, conditional on each checkpoint/seed. Report each optimization seed and across-seed dispersion; never pool users from different fitted models as independent replications. Shared users/contexts and common selection cause dependence; selection scores are optimistically selected, not unbiased final evidence.
- Historical test is already `exploratory_reuse`. Freezing before new inference prevents further leakage but does not undo past inspection. Confirmatory claims require a genuinely uninspected later period with mature labels and protocol frozen before access. No fresh test is supplied here.

## 6. Minimal acceptance and validation before a costly launch

**Engineering acceptance (all required):** (1) fixed-data matched A arms including RL+CE and rerank; (2) complete label-backed reward audit above; (3) common selection/failure semantics; (4) explicit independent sample/model/optimization seed fields; (5) all choices frozen before test, per-example targets/UIDs/populations/predictions plus selected/final weights and checksums; (6) failed gates retain continued-SFT and failure records; (7) B absent unless separately requested and disagreement gate supported. A repeat-penalty-only suite does not meet A acceptance.

**Scientific decision, not an engineering pass criterion:** retain historical practical thresholds: per fixed optimization seed, completion relative-gain 95% lower bound >=+5% and eligible NDCG relative-change lower bound >-2%, versus SFT. Claim robust practical gain only if all three prespecified seeds satisfy both; otherwise report mixed/negative/inconclusive results. Claim no advantage over simpler alternatives from SFT comparison alone: report paired RL+CE contrasts against RL, continued-SFT and tuned rerank, with CIs, compute and multiplicity/exploratory caveats. No positive outcome is required for acceptance, and even success is proxy improvement, not causal user benefit.

**Validation plan:** unit-test lambda=0 equivalence to RL, nonzero CE gradient and eligible target indexing; frozen RM/reference gradients; exact expectation on a tiny enumerable catalog; shared population and rerank softmax semantics; deterministic selection/ties and floor rejection; fit-only slices and tied tail membership; audit/checkpoint role separation; fixed-data seed identities; B gate pass/fail/insufficient-support paths; continued-SFT surviving reward failure. Run a tiny CPU end-to-end fixture proving all training/selection precedes test and bundle round-trip/hash integrity, then focused tests and full pytest. Windows launcher execution and full CUDA training remain separate hardware validation; do not claim either from CPU checks.

## 7. Shipped APIs and remaining seams

- `experiment.train(..., sft_only=True)` freezes one SFT and hash-bound `SFT_READY.json`; `reanalysis.verify_inputs(..., training_only=True)` verifies it without test artifacts.
- `reward_report` includes tails, popularity/repeat, signed errors and learned item-only comparisons. `followup.split_calibration` separates early checkpoint selection from late audit.
- `followup.train_policy(..., supervised=False, repeat_penalty=0.0, lambda_ce=0.0)` implements RL+CE and selected/final checkpoints. `run_followup(..., training_only=True)` trains one RM/comparator and reuses weights across optimization seeds; failed reward gates retain continued-SFT.
- `followup.tune_rerank` freezes the selection curve under the same RM before test. `followup_evaluate.evaluate_followup` authenticates all seeds/curves/checkpoints before test and exports eligible/all-target arrays and paired contrasts. Historical `reanalysis.run_reanalysis` remains a separate API, not the default A suite.
- `suite.run_suite(..., seeds=(42,43,44), sample_seeds=None)` freezes all comparisons before test, groups conditional effects by sample and bundles all artifacts/source/logs with ZIP64. `--sample-seeds` is optional outer sample robustness, not an ensemble or fixed-data optimization replication.
- `ensemble_audit.audit_ensemble` / `python -m rl_recsys.ensemble_audit` provides opt-in observed-label disagreement validation from explicit pickle-free NPZ exports: tied-rank Spearman and paired common-user decile error contrast, user-bootstrap CIs, >=100 common users and both lower bounds >0. Only declared identity/uncalibrated predictions are supported; no fitted calibration or calibrated claim. See [ENSEMBLE_AUDIT.md](../guides/ENSEMBLE_AUDIT.md) for provenance, slices and flags.
- No ensemble training/export pipeline, fitted calibration, ensemble-policy optimization or B orchestration is implemented; the default A suite never launches B. The prospective calibrated-member training requirement remains outstanding. Preserve source identity and completion manifests rather than patching old provenance. `run_followup` defaults to `source_mode="current"` and rejects parent source mismatch before training or historical test verification; `evaluate_followup` checks current source against the frozen follow-up provenance before test access. Deliberate cross-source historical work requires explicit `source_mode="historical_reanalysis"` (CLI follow-up: `--source-mode historical_reanalysis`), recorded in new provenance; it never rewrites old hashes. The historical `run_reanalysis` API remains intentionally cross-source.
