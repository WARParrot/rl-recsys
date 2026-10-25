# Independent code/resource audit: Yambda GPT-2 + reward model + RL

**Scope:** read-only review of `rl_recsys/data.py`, `models.py`, `evaluate.py`, `experiment.py`, `configs/pilot.json`, plus limited configuration/test context. Audit date: 2026-10-22. No implementation files modified. CLI/tests are actively being written; test observations below are a time-stamped snapshot, not a final verdict.

## Executive summary

The implementation has several strong design choices: train-only item vocabulary, explicit unknown/pad IDs, tie-group exclusion rather than invented within-bin order, bounded sample selection, hashed dataset/config provenance, and last-position-only policy logits. There are nevertheless important methodological and execution risks to address before treating pilot output as confirmatory.

**Immediate resource blocker:** 36-hour training deadline is not enforced around validation/evaluation, and final evaluation has no deadline at all. With 12 RL configurations, repeated full-catalog ranking, reward scoring, and up to 65 separate five-step rollout passes over test contexts, the declared 48-hour GPU window is not guaranteed. Preflight actual RTX 5060 Ti throughput/VRAM and bound all expensive phases before starting a full run.

**Metric/selection blocker:** PLAN declares eligible-target user-macro NDCG as primary, while candidate selection uses `selection["ndcg"]` on all targets (unknown targets score zero) and H2 uses a paired interval of that all-target metric. This can select a candidate according to a different estimand than the stated eligible-target primary metric and can dilute/mask relevance change as coverage changes. Resolve and freeze the estimand before viewing test results.

**Portability gate (installation work is still in progress):** current `pyproject.toml` directly pins `torch==2.8.0` without an explicit CUDA wheel/index strategy. This is not evidence that a fresh Linux PyPI installation will be CPU-only; PyTorch distribution varies by OS/index. It is evidence that the GPU install contract is not yet established in the reviewed files. GPU installation instructions/preflight need explicit supported CUDA wheel selection and a failing GPU smoke test on the user's target machine. The current local `.venv` is confirmed CPU-only (`2.8.0+cpu`, `torch.version.cuda=None`, `cuda.is_available=False`); no GPU training claims can be tested here.

## Findings

### High: runtime limits do not cover validation, evaluation or final test

- [`experiment.py`](../rl_recsys/experiment.py), `fit_supervised` lines 127-140: deadline checked once before each optimizer step, but `validation_loss` can process its entire validation set without a deadline check. Large validation batches/sets or slow GPU can exceed the requested window.
- `train_rl` lines 165-186 similarly checks only before each step; in-step horizon sampling and evaluation are not interruptible.
- `train` deadline is computed from `max_hours` (line 208), and candidate selection runs potentially 12 configurations (`2 rewards × 2 horizons × 3 betas`) with 1000 RL steps each. Each candidate also runs `evaluate` over selection data. Once training ends, deadline is not passed to `final_evaluation` at all.
- `final_evaluation` lines 320-337 evaluates popularity, SFT, rerank, all candidates, and rollout diagnostics. `rollout_diagnostics` is run for SFT plus each RL candidate; default grid gives 1 + 12 calls, each scanning up to 10,000 test contexts in configured batches and generating 5 actions. This is 65 rollout-step passes over test contexts, plus candidate generation/scoring. The JSON `rollout_samples: 1` is not used to bound or subsample this workload.

**Impact:** 36-hour training can consume most of the 48-hour access window and leave no time for test; a deadline exception aborts rather than checkpointing/marking an incomplete run. Final test may exceed remaining time. `READY_FOR_TEST.json` is written only after all training/selection completes.

**Recommendation:** make a phase-level compute budget with reserved final-evaluation time; check deadline within validation/evaluation loops; subsample rollout diagnostics deterministically using the configured rollout sample count; estimate cost with target GPU before full grid; persist resumable/atomic phase state or write explicit incomplete status. Do not start all grid members until a measured pilot demonstrates feasible total wall time.

### High: primary relevance estimand is inconsistent between plan and selection

- [`PLAN.md`](../docs/archive/PLAN.md), line 10: primary relevance says user-macro NDCG@10 within train-known eligible catalog; unknown targets score zero in all-target view.
- [`experiment.py`](../rl_recsys/experiment.py), lines 244-246: selection relevance floor uses `c["selection"]["ndcg"]`, which is computed by `evaluate` over every target, including unknown-token targets whose target IDs cannot occur in recommendations. This is an all-target metric.
- `final_evaluation` lines 335-337 builds paired intervals for `ndcg` and `recall`, not `eligible_ndcg`; H2 relevance non-inferiority therefore also uses all-target NDCG. `evaluate` does report `eligible_ndcg`, but that metric is not used for selection or verdict.

**Impact:** candidate admissibility and H2 use different user weights than the declared warm-catalog ranking estimand: each user's eligible ranking contribution in all-target macro metrics is multiplied by that user's target-coverage fraction. Coverage is fixed across policies for the same prepared split (it does NOT change because of RL), but differing per-user coverage can reverse comparative aggregate results. All-target metrics may be useful secondary metrics; they must not be conflated with primary eligible NDCG.

**Recommendation:** define whether H2's relevance constraint is all-target or eligible-target, report both and target coverage, and align candidate selection, paired CI and verdict with the preregistered primary estimand. Include unknown-target zero scoring as a separately named metric.

### High: GPU installation is not portable/reproducibly selected by the project metadata

- [`pyproject.toml`](../pyproject.toml), line 10 pins torch but declares no CUDA-specific index/wheel or documented platform matrix.
- `get_device` in [`experiment.py`](../rl_recsys/experiment.py), lines 40-50, does a small CUDA matmul/backward if CUDA is reported, and errors if CPU is not explicitly allowed. This catches some broken installations, but it does not report driver/runtime compatibility details, usable VRAM, target model peak memory, or run a representative model optimizer step.
- The local venv has CPU-only torch 2.8.0 and no CUDA runtime; this is expected for this host and cannot validate user's GPU setup.

**Impact:** a user can follow a generic install and receive CPU-only PyTorch; current preflight then fails, possibly only after dependency setup/data preparation. Conversely, a 32×32 test can pass although the configured training/eval workload OOMs.

**Recommendation:** keep the main lock free of CPU-specific artifact URLs; provide explicit supported install options (OS/Python/driver + official PyTorch CUDA wheel selector/index) and pin/record the resolved wheel. Preflight must print torch build, CUDA runtime, driver, GPU name/VRAM, free VRAM, precision support and run representative forward/backward/optimizer + evaluation allocation at configured shapes, then exit nonzero on inadequate memory or CPU-only torch for GPU mode.

### Medium: dataset reader materializes the selected event sample and a large duplicate-detection matrix

- [`data.py`](../rl_recsys/data.py), lines 44-60: `read_sample` performs two passes but accumulates every selected column chunk into Python lists and concatenates all selected events in RAM.
- `prepare_arrays`, lines 87-90: `np.column_stack` constructs a dense matrix across all six columns and `np.unique(axis=0)` performs global row deduplication. Mixed numeric columns can promote to float64; row-matrix plus sort/index temporaries multiply peak RAM.
- `prepare_arrays`, lines 91-120 then materializes sorted/copy arrays, masks and train catalog stats. User sampling is by UID as intended, but 1,000 Yambda users can still contribute a large, highly skewed number of events; no input-row cap exists before this allocation.

**Impact:** parquet is only ~369 MB compressed, but peak RAM can be several GB or more depending on sampled-user history lengths and NumPy temporary arrays. This may exhaust smaller user machines or cause paging. The official sequential representation would not necessarily help if fully expanded/materialized.

**Recommendation:** measure max RSS on the target machine using exact pinned versions; report selected-user event count/bytes before expensive deduplication. Use structured row keys/chunked deduplication or detect duplicates after stable sort without stacking all columns into a float matrix. Add an explicit memory budget/early rejection and document the need for RAM beyond the nominal parquet size.

### Medium: Parquet nullable/type handling is not verified at the reader boundary

- [`data.py`](../rl_recsys/data.py), lines 48-57 calls Arrow `to_numpy()` for `uid`, then converts each entire selected column with `to_numpy(zero_copy_only=False)`; lines 81-82 later require `np.isfinite` on every converted array.
- The official card declares optional playback fields for listen/non-listen schemas but does not prove that all rows in this exact listens Parquet have no nulls or how nulls convert under the pinned Arrow version. An Arrow nullable integer with nulls can materialize as object/float or raise depending on dtype/version. IDs/timestamps should remain integer typed and null validation should happen explicitly before conversion.
- HF `first-rows` endpoint for multi-event errored during the earlier data audit due to `Dictionary(UInt32, Utf8View)`; that is not proof the listens parquet fails, but it increases the need to validate the actual file with pinned `pyarrow==21.0.0`.

**Impact:** a metadata/API change or nullable value can fail unexpectedly, silently promote types, or create confusing object arrays. No sample read against the pinned package and actual dataset was part of this code review.

**Recommendation:** inspect parquet schema/row groups before conversion; explicitly reject nulls per required field (or define a deliberate listen-field null policy), assert integer ID/timestamp dtypes and ranges, and add a small real-file smoke test on the exact pinned reader. Preserve the downloaded SHA256 verification.

### Medium: same-timestamp tie policy is sound but loses more history than only tied transitions

- [`data.py`](../rl_recsys/data.py), lines 93-96 marks every row in a user/timestamp group of size >1 non-singleton; lines 148-161 omit all such rows from history and targets.
- This avoids inventing order within the 5-second timestamp bin and is defensible. However, it removes all tied-group tracks from future contexts as well, not just their within-group next-item transitions. That is a stricter estimand than “exclude ambiguous target transitions”; impact could be material if ties cluster in recommendation-generated sessions.
- Exact duplicate rows are removed first (lines 87-90); rows with same user/item/timestamp but different playback/organic values are retained and cause the entire timestamp group to be excluded, which is conservative but should be counted separately.

**Recommendation:** report unique tied groups, tied rows, and affected users/organic status; describe history exclusion in the estimand. Consider a set-valued/collapsed-bin history alternative as a distinct ablation, not as an arbitrary tie-break.

### Medium: evaluation repeatedly scores the catalog and synchronizes/copies to CPU

- [`evaluate.py`](../rl_recsys/evaluate.py), lines 62-64 fully sorts all catalog logits with `torch.argsort` to obtain top-k and creates a full-catalog multinomial distribution for every batch. `topk` can avoid sorting the unused tail.
- Lines 66-75 score reward/auditor for sampled actions and all top-k recommendations, then copy multiple tensors to CPU for each batch. On GPU each `.cpu().numpy()` synchronizes the stream. `evaluate` repeats for each selection candidate and every final policy.
- Rerank mode calls `rm.catalog(context)` (line 58), which scores each candidate in Python chunks but appends all chunk outputs before concatenating in [`models.py`](../rl_recsys/models.py), lines 65-75; memory scales with batch × catalog × two reward outputs. With pilot 20K catalog and batch 128 it is moderate but repeated and not bounded by a memory test.
- `rollout_diagnostics` repeats full policy/RM/auditor forward passes for every test context and every requested horizon. The configured `rollout_samples` is currently unused.

**Impact:** test evaluation can become a significant share of GPU wall clock and VRAM; per-batch host synchronizations limit throughput. An exact repeated policy grid can consume the promised 48h even if training fits.

**Recommendation:** use `topk` for top-K, accumulate device-side or larger CPU batches before transfer, chunk catalog reward scoring without retaining both full outputs where possible, and use deterministic configured rollout subsampling. Measure each eval mode independently on target hardware.

### Medium: data provenance is good but run-level reproducibility has gaps

Positive: data revision and expected SHA are configured; selection uses deterministic UID hash; event subsampling uses seeded NumPy RNG; prepared data and audit are hashed; model code hashes, git status, versions and device are logged.

Gaps:

- [`data.py`](../rl_recsys/data.py), lines 34-37 downloads to a `.partial` path and renames after checksum, which protects final output, but a failed checksum leaves the partial file behind. Not a correctness failure, but cleanup/resume semantics are unclear.
- Timestamp quantiles are computed from the selected sample's event rows (`np.quantile(ts, ...)`, line 98); they are deterministic for a fixed UID sample but are not the official global benchmark cutoffs. The protocol should call this a sampled-population temporal split, as PLAN does.
- `np.random.choice` selects candidate rows uniformly within each split (lines 63-66, 133-140). This is deterministic and outcome-blind, but the resulting capped sample is an event-macro sample. Later `macro` weights each represented user's retained examples equally, not each user's complete event history. Users can have very different retained counts and split coverage; this defines a sampled-context user-macro estimand, not full-user next-listen performance.
- `seed_all` sets seeds but does not enable deterministic algorithms; PyTorch/CUDA kernels and hardware may vary. Provenance explicitly avoids bitwise promises, which is appropriate, but only one seed in pilot config cannot estimate optimization variance.
- Download uses a fixed repository revision and checksum, a strong reproducibility measure; confirm the SHA is the raw file hash, not only a server-side ETag, on the target machine.

**Recommendation:** record selected-user count/rows and split per-user counts/distributions; ensure bootstrap units and metric arrays align by user/context; report seed variance separately; record driver and full install command/wheel; make retry behavior for failed downloads explicit.

### Low: categorical action semantics and sampled evaluation differ by mode

`Policy.forward` emits logits only for item rows `wte.weight[2:]`, which consistently excludes PAD/UNK action IDs. `append_action` maps action index back by +2. This is internally coherent. `evaluate` samples from the current score distribution, including popularity and rerank modes, so `sample_reward` is stochastic-policy reward, while top-10 reward is mean reward of a deterministic list. The report stores both but selection uses `sample_reward`; ensure output labels and protocol keep these distinct. Evaluation generator is fixed to the same seed for each invocation, which supports paired policy comparisons, but any changes in batch sizing/order may alter draws.

## Resource estimate / pilot recommendation

Pilot config: top-20,000 train items, GPT-2 hidden 128 × 4 layers × 4 heads, context 64, train batch 64, 250k max fit examples, 5,000 SFT steps, 3,000 steps for each of two reward models, and up to 12 RL candidates × 1,000 steps with horizons 1/5. Total nominal optimizer steps are 5,000 + 6,000 + 12,000 = **23,000** (plus calibration, selection, candidate evaluation and final test). A single RTX 5060 Ti 16GB may support the model shape, but wall time cannot be responsibly certified without a measured step and eval benchmark. Full-catalog policy output for B=64,V=20k is ~4.9 MiB fp32 per logits tensor; it is manageable, while reward/reward-auditor GPT-2 activations and optimizer states dominate training. During RL, the original SFT policy, its frozen reference deep copy, the candidate policy and the attention reward model can coexist (four transformer backbones); the mean-history auditor uses embeddings rather than a transformer. RL training retains policy gradients while inference computes frozen reference and reward; monitor peak allocation rather than estimating from parameter bytes alone.

Recommended gates on the target GPU:

1. Verify explicit CUDA torch wheel and report driver, CUDA runtime, GPU model/VRAM, package versions.
2. Run representative full-model forward/backward/AdamW step at batch 64, sequence 64, vocab 20,002; report peak allocated/reserved VRAM and median seconds over warm steps.
3. Benchmark one supervised validation pass, one RL step at horizon 5, one selection evaluation, one final evaluation mode and one rollout diagnostic pass.
4. Extrapolate 23k optimizer steps plus validation/evaluation and reserve a fixed final-test window. Reduce grid/steps before run if estimate exceeds budget; do not rely on a 36h deadline that excludes evaluation.
5. Test CPU and GPU numerical finiteness on synthetic data only; do not interpret CPU smoke as evidence of GPU runtime or statistical quality.

## Verification performed and limits

- Read the named modules/config and surrounding `PLAN.md`, `pyproject.toml`, CLI and tests for context.
- Inspected local venv: PyTorch `2.8.0+cpu`, `torch.version.cuda=None`, CUDA unavailable. GPU execution not attempted.
- Snapshot `pytest -q`: 4 passed, 1 failed in `tests/test_data.py::test_same_timestamp_rows_excluded`; failure was assertion expecting non-collapsed quantiles on a tiny tie fixture (`AssertionError: quantile collapse should make tiny tie fixture invalid`). Tests were actively being written, so notify/update before treating this as a stable product defect.
- No full Yambda file was downloaded or read in this audit; Arrow behavior on real parquet, peak RSS, GPU memory and throughput remain unverified.
- Audit scope intentionally excludes a full CLI/test review pending the user's notice that those are still in progress.

## Review disposition

Before a long run, resolve the runtime/evaluation budget, align selection/H2 metrics with the declared eligible/all-target estimand, and establish an explicit portable CUDA install + representative preflight on the user's GPU. Then add real-parquet typing/tie tests and measured memory/time gates. No test failure should be “fixed” by changing the data estimand without documenting that choice.
