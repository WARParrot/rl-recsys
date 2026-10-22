# Adversarial code audit — RL recommender

**Scope:** read-only review of `rl_recsys/{data,models,evaluate,experiment}.py`, `configs/pilot.json`, and `PLAN.md`. CLI/tests were not the requested review scope; no training was run. Findings below are actionable. Line numbers refer to the reviewed files and may shift after edits.

## Verdict

**Do not treat the current run output as protocol-confirming yet.** The core REINFORCE + future-KL/direct-KL decomposition is mathematically coherent for the implemented deterministic append-item simulator, subject to the undiscounted horizon and independent-trajectory assumptions. However, the primary relevance estimand is not what the code uses for model selection/H2, the configured temporal split contradicts the plan, and final evaluation does not protect the selected-model metadata. These are material protocol/correctness issues.

## Blocking / high-severity findings

### 1. Selection and H2 use all-target NDCG, not the declared eligible-catalog primary estimand

**Files:** `rl_recsys/experiment.py:224-246, 315-347`; `rl_recsys/evaluate.py:80-87`; `PLAN.md` scope/H2.

`evaluate()` computes `ndcg` over every target, assigning zero to targets with token `1` (unknown/out-of-catalog), then returns that as `summary['ndcg']`. `summary['eligible_ndcg']` is computed separately. Training selection compares `summary['ndcg']` against the relevance floor; final H2 also pairs `raw['ndcg']`. This contradicts the plan's primary relevance estimand: user-macro NDCG on targets within the train-known eligible catalog, with all-target zero-score view reported separately.

Because user-macro eligible metrics omit users with no eligible targets while all-target metrics include them as zero, the two are not generally a constant rescaling. They can select different checkpoints and yield a different non-inferiority verdict.

**Required:** use `eligible_ndcg` (computed at user-macro level on eligible examples) for selection and H2; report all-target NDCG as a separate diagnostic. Record eligible target count and eligible-user count. Keep target coverage separately. Also ensure the selection floor uses the same estimator/denominator for baseline and candidate.

### 2. Configured split is 70/10/10/10, not the plan's 80/10/10

**Files:** `configs/pilot.json: data.quantiles`; `rl_recsys/data.py:98-101,133`; `PLAN.md` split section.

Quantiles `[0.7, 0.8, 0.9]` produce four intervals: fit <70%, calibration 70–80%, selection 80–90%, test >=90%. The plan describes train/selection/test as 80/10/10, with internal partitioning of training for RM fitting/selection. This changes training exposure and role sizes and is currently undocumented in the artifact.

**Required:** either change the config to the declared plan (and define the RM calibration sub-split), or update the plan/protocol before running and freeze the actual 70/10/10/10 design. Report exact timestamp cutoffs and role counts, not just nominal quantiles.

### 3. Selection metadata is not integrity-protected before test evaluation

**File:** `rl_recsys/experiment.py:247-252,294-323`.

`READY_FOR_TEST.json` hashes only `*.pt` checkpoints. `final_evaluation()` checks those checkpoint hashes but trusts mutable `selection.json` and `candidates.json`; it also does not verify the audit JSON hash recorded in provenance. The selected model can therefore be changed after training/selection without changing a checkpoint, including selecting a different candidate or nulling/changing the recorded selection rule. This weakens the claimed test firewall and makes the final test result non-reproducible from the READY artifact alone.

**Required:** include hashes for `selection.json`, `candidates.json`, provenance, and prepared audit in READY; verify them in final evaluation. Persist the exact chosen checkpoint name/hash and selection rule in an immutable manifest; ensure the chosen model meets the declared eligible-NDCG constraint. Reject missing/inconsistent manifest entries.

## Important correctness / estimand findings

### 4. “Independent auditor” is architecture-independent, not independently audited

**Files:** `rl_recsys/experiment.py:215-223,255-291`; `rl_recsys/models.py:37-60`; `PLAN.md`.

The auditor has a different mean-history encoder and independent parameters, but is trained on the same fit data, selected using the same calibration interval, and evaluated against the same observed-listen test labels. It is an architectural disagreement model, not independent data validation or independent measurement of user preference. Both RM and auditor share the same exposure/observational bias. This limitation is partly disclosed in final `limitations`, but naming and the earlier plan's “independent reward auditor” can overstate independence.

**Required:** call it an *independent-architecture RM diagnostic* unless a genuinely held-out RM-audit data period is reserved. Report agreement/disagreement and both models' calibration on a time-disjoint audit population. Never treat `sample_audit` agreement as independent truth.

### 5. User-bootstrap H2 omits recommendation-sampling uncertainty

**Files:** `rl_recsys/evaluate.py:13-31,49-64`; `rl_recsys/experiment.py:333-347`.

Policy reward is estimated from one sampled action per context. The paired user bootstrap resamples these already realized outcomes and is conditional on this single sampling draw; it does not quantify Monte Carlo variation of stochastic policy recommendations. The report string says conditional on seed/checkpoint, but H2's 5% threshold can still be sensitive to the one draw.

**Required:** evaluate multiple independent action samples per context or use exact full-catalog expectation for one-step reward, then aggregate at user level and bootstrap users while preserving within-user Monte Carlo replication. Report user and sampling uncertainty separately. Alternatively make deterministic expected-reward evaluation primary and sampled policy return secondary.

### 6. H1 verdict rule differs from declared hypothesis

**Files:** `rl_recsys/experiment.py:338-341`; `PLAN.md` H1.

H1 says at least 2% relative MAE reduction and CI excludes no improvement. The code requires the **upper** end of the relative CI to be `<= -2%`, which is stricter: it requires the entire CI to exceed the practical-effect threshold, not merely point estimate >=2% and CI excluding zero. This is conservative but a protocol mismatch, and labels otherwise qualifying evidence “not supported.”

**Required:** predeclare one rule and implement it exactly. If practical superiority requires the whole CI beyond -2%, state that stronger criterion in PLAN; if the existing H1 text stands, check both point estimate <= -2% and CI upper < 0.

### 7. All-target denominator semantics are underspecified in summaries

**File:** `rl_recsys/evaluate.py:80-89`.

`examples=len(eligible)` is actually the number of all examples (boolean mask length), not eligible examples. `users` counts all-target users; `eligible_ndcg` uses only users who have at least one eligible target. These fields can be misread and obscure the distinct populations/denominators.

**Required:** emit explicit names: `all_target_examples`, `eligible_target_examples`, `all_target_users`, `eligible_target_users`, `eligible_target_coverage`, and clearly label macro population for each metric. Do not call the all-target count `examples` without qualification.

### 8. Data preparation peak memory can be much larger than the raw sampled arrays

**File:** `rl_recsys/data.py:44-60,77-90,125-161`.

The sample pass accumulates arrays for every selected event, then `np.column_stack` creates a promoted dense matrix across all six columns; `np.unique(matrix, axis=0)` performs an axis-wise sort and may allocate substantial temporary memory. `read_sample` also accumulates each Parquet batch per column in Python lists before concatenation. At 1,000 users this may still be large depending on event counts, and the host/resource audit identifies limited RAM; the “bounded” claim is only partly true. The later retained contexts are capped, but all raw sample rows remain resident first.

**Required:** measure peak RSS on target data before full run. Prefer deduplicating with structured row hashes/chunked processing or a compact structured dtype; avoid `column_stack` float promotion and full axis-wise sort. At minimum add a hard row/memory guard and report sampled event count/peak RSS. Do not assume the context cap bounds preprocessing memory.

## Math review: `reinforce_loss`, future KL and direct KL

**File:** `rl_recsys/models.py:83-103`; caller `rl_recsys/experiment.py:159-185`.

For objective

\[
J=\mathbb E_{\pi}\left[\sum_t r_t-\beta\sum_t KL_t(s_t)\right]
\]

under the implemented simulator, `reinforce_loss()` uses:

- score-function reward-to-go with future KL penalties included as `-beta * kls[t+1:]`;
- a direct `+ beta * kls[t]` term for the current state's analytic KL gradient;
- leave-one-out baseline across independently sampled batch trajectories;
- detached rewards/KLs, while retaining gradients through current log-probabilities and direct KL.

That decomposition is mathematically appropriate: current `KL_t(s_t)` is action-independent conditional on state, so its score-function contribution at action `t` has expectation zero; its direct KL derivative is included. Earlier actions' effect on future states/KLs is represented by future KL in the return. Recursion/signs are consistent. No finding that this is double-counting under the stated objective.

Caveats:

- Objective is explicitly **undiscounted**; that is acceptable for fixed horizons but must remain documented.
- LOO baseline is unbiased only when other trajectories are independent of the current trajectory's sampled action. Sampling contexts with replacement does not itself invalidate this, but future tests should verify no cross-trajectory shared sampling/state coupling.
- `kls[t]` is exact categorical KL over the implemented restricted item support; this is not full Yambda catalog KL.
- Future contexts append generated item IDs without user feedback. The estimator is valid for that deterministic synthetic simulator, not for user transitions.
- The current unit test checks only finite gradients; it does not verify KL sign, future-KL credit assignment, direct KL gradient, or unbiasedness. Add analytical tiny categorical tests comparing gradients against exact enumeration for H=1 and H=2.

## Other correctness / scope observations

- Temporal history construction in `data.py:143-161` is strictly earlier in `(uid,timestamp)` order, excludes all members of tied timestamp groups, and builds later contexts from earlier singleton events, including observed validation/test events. This is consistent with the declared rolling teacher-forced context protocol, not a cold-start split. Same-timestamp ambiguity is not arbitrarily ordered.
- Train vocabulary/popularity/duration are fit-only (`data.py:101-120`); unknown targets map to 1 and receive zero rank hit. Good, provided both all-target and eligible estimands remain distinct.
- Input sequence accepts unknown token 1 as context, excludes IDs 0/1 from policy action support, and `append_action` preserves left padding/truncates oldest token. Index arithmetic is consistent.
- `RewardModel.catalog()` currently materializes a list of chunks and concatenates; at pilot catalog 20k and eval batch 128 this is likely manageable, but peak activation intermediates in `from_history` can exceed the final output size. Benchmark reranker peak VRAM. Do not extrapolate this implementation to the 934k/full catalog without a streaming top-K reranking path.
- Training samples from `fit` include both organic and recommendation-driven listens. This matches a general next-logged-listen objective, but is not policy-exposure-only reward data. Keep `organic` slices and avoid describing the RM as response to an exposed recommendation.
- The config sets `max_hours=36` but each of 12 RL candidates uses 1,000 updates, with horizon-dependent multiple full policy/RM forwards; deadline handling safely aborts without READY, but the complete grid may not fit. Pilot timing must be measured and incomplete grids must not silently be compared as if budget-matched.
- In final evaluation `results.json` is written only after all work. A crash after `per_example.npz` but before results leaves partial output and allows rerun; this is operational rather than metric correctness, but use atomic writes/manifests if tests require reliable finalization.

## Minimum changes before training/evaluation

1. Resolve 70/10/10/10 versus plan split and freeze it.
2. Use eligible NDCG for model selection and H2; explicitly report all-target metrics/coverage.
3. Protect selection/candidate/audit manifests with hashes and validate the chosen artifact before test access.
4. Clarify auditor independence; no causal or independent-truth language.
5. Quantify stochastic sampling uncertainty for H2.
6. Fix denominator field names and align H1 verdict rule with preregistration.
7. Add exact-enumeration gradient tests for REINFORCE + future/direct KL; run target-data preprocessing and VRAM peak preflights.

Until items 1–3 are resolved, a final test score can be computed, but it would not faithfully test the declared protocol.
