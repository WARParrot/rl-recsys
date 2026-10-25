# Reward design and one-launch Experiment A

**Submission status:** Experiment A is implemented and tested, but this repository does not claim a completed research-suite result. The executed results are the historical joint-RM pilot and the separate completion-only follow-up; see [RESULTS.md](RESULTS.md) and [COMPLETION_RESULTS.md](COMPLETION_RESULTS.md).

## Implemented default (stage4)

`run_windows.bat` (or `run_research_windows.bat`) launches the implemented Experiment A workflow, not the historical full reward/horizon grid or a repeat-penalty experiment. No completed research-suite results are claimed in this repository; the executed pilot/follow-up results are linked above. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the scientific contract.

For one prepared sample/catalog/split, the suite trains **one SFT**, then **one history-aware completion RM** and one learned item-only diagnostic comparator. Early calibration selects supervised checkpoints; late calibration audits reward. The RM checkpoints are reused byte-for-byte for policy optimization seeds **42/43/44**. SFT/RM initialization uses the config seed (default 42); sampling has a separate `sample_seed` (default config seed).

Matched arms start from the same SFT:

- unchanged SFT reference;
- continued-SFT (next-target CE only);
- H=1 completion REINFORCE, beta=0.01, temperature=1;
- the same RL + next-target CE, coefficients 0.1 and 1.0;
- tuned rerank: `log pi_SFT + w * completion_RM`, weights 0, 0.1, 1, 10.

Each trained candidate gets 1,000 updates by default. Every arm selects on the same selection role: eligible NDCG >=98% of unchanged SFT, then maximum exact expected **unpenalized completion**. Earlier steps/lower coefficients break ties. Failed floors remain unselected. Rerank is tuned **before test**, under the same RM, not retuned on test or presented as a formal upper bound. All training and selection across every requested outer sample finish before the first suite test inference.

A failed completion reward gate skips RL/RL+CE/rerank but **still trains/selects/evaluates continued-SFT**. Continued-SFT remains subject to the common selection floor; survival of training does not guarantee a selected checkpoint. Gate/floor failures are retained, not replaced by favorable seeds. Repeat-adjusted reward remains a descriptive diagnostic; optional repeat-penalty training is available via the follow-up Python API, not the default suite.

## Reward and claim boundaries

Primary serving rule is one stochastic action at T=1. Primary endpoint is eligible-context, user-macro full-catalog expected completion under the frozen predictive RM. Deterministic top-K relevance, repeat, duration, coverage and HHI are separate diagnostics. All-target and eligible populations and paired user-bootstrap intervals are reported separately. No causal satisfaction, retention or online benefit is identified by this offline study.

Late audit includes constant/item-mean baselines, learned item-only comparator, calibration and tied upper-score tails, duration/popularity/repeat/organic slices and paired uncertainty. The operational completion gate remains point MAE improvement >=2% over fit item mean and better than fit constant; passing is not validation of hypothetical-action rewards. Tiny or missing slices remain inconclusive.

**No ensemble is trained.** Stage5 adds only a standalone, opt-in [ensemble hypothesis audit](ENSEMBLE_AUDIT.md) over explicitly exported observed-label NPZ predictions: user-bootstrap tied-rank association and paired common-user decile error contrast, support/degeneracy flags and fit-derived slice associations. It accepts declared identity/uncalibrated scores, fits no calibrator and makes no calibrated claim. Experiment B member training, fitted calibration, export pipeline and uncertainty-penalized policy optimization remain separate, unimplemented work. Optimization seeds and outer sample runs are not ensemble members. Seconds optimization is absent from A. Retention needs mature future labels and a separately justified design.

## Commands and seed roles

```bat
run_windows.bat
run_windows.bat research artifacts\research-my-run
```

Direct API/CLI (production training requires CUDA):

```text
python -m rl_recsys.suite --output artifacts/research.v1 --seeds 42 43 44 --steps 1000
python -m rl_recsys.suite --output artifacts/sample-robustness --sample-seeds 51 52 53
```

`--seeds` are **policy optimization seeds**, not full-pipeline repetitions. Optional `--sample-seeds` changes user sampling/context retention while keeping supervised/RM initialization fixed; each sample gets a separate prepared catalog, SFT and RM and a full matched comparison. Report these as outer sample robustness, not pure optimization variance. Effects remain grouped by sample; absolute rewards across fitted RMs are not averaged. A new sample seed does not create an untouched calendar test.

APIs: `suite.run_suite(..., seeds=(42,43,44), sample_seeds=None)`; `experiment.train(..., sft_only=True)` writes `SFT_READY.json`; `reanalysis.verify_inputs(..., training_only=True)` authenticates training inputs without requiring or opening parent test artifacts; `followup.run_followup(..., training_only=True)` freezes A; `followup_evaluate.evaluate_followup(...)` verifies all seeds and frozen rerank metadata before explicit test inference.

## Artifacts, status and operational limits

The output contains `sample_<seed>/config.json`, prepared arrays/audit, SFT/provenance/training marker, `followup/seed_<optimization-seed>/` with reward audits, curves, selected **and final** policy weights, rerank curve and hash-bound completion manifest, plus `controls-evaluation/` with per-example targets/UIDs/populations/predictions and paired comparisons. No legacy baseline test/reanalysis grid or plot generation is run by default.

`<output>.zip` (dotted names preserved) includes **all output files**, source/config/docs/tests/launchers and lossless subprocess/setup logs. ZIP64 is enabled; every member is reread and SHA256-verified; an external archive checksum is written. Raw parquet is not duplicated: download revision/hash are in the config. No automatic upload occurs. Existing outputs are not resumed or overwritten.

`suite.json` records running/complete/failed stage status, per-optimization gate status, seed roles and bundle status. Infrastructure errors abort with nonzero exit and attempt a failure archive. Bundle failure persists `bundle_failed`, the scientific status, error and traceback; it is never success. Setup failures before Python suite startup remain in `<output>-setup` and are not automatically bundled.

Windows launch installs dependencies/downloads data and requires Python 3.11, compatible NVIDIA driver, internet and substantial disk. Config time limits are caps, not runtime measurements: SFT has the configured training deadline, each optimization seed has its configured deadline (the first also trains RM/comparator), evaluation defaults to 240 minutes per sample. CPU tests do not validate Windows execution or full CUDA performance. No automatic resume, fresh uninspected test, ensemble/B training, causal identification or publication is provided.
