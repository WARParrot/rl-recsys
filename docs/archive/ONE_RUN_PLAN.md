# One-launch research suite — superseded

**Status: historical implementation plan, superseded.** The scope and seed design below are not the current launch instructions. See [EXPERIMENT_PROTOCOL.md](../research/EXPERIMENT_PROTOCOL.md) for the authoritative contract and [REWARD_DESIGN.md](../research/REWARD_DESIGN.md) for shipped APIs. Current Experiment A freezes one prepared sample/SFT/history RM and a learned item-only comparator, then pairs optimization seeds 42/43/44 across RL, RL+CE, continued-SFT and tuned rerank. Optional outer sample seeds are separate robustness runs. Experiment B remains separate and conditional; no new training results are claimed.

The remainder is retained only as historical context.

## Next experiment contract
[EXPERIMENT_PROTOCOL.md](../research/EXPERIMENT_PROTOCOL.md) is the prospective research contract. The implementation below describes the existing one-launch baseline, **not completion of that contract**. Next work must add fixed-sample matched single-RM RL+CE versus RL/continued-SFT/common-RM tuned rerank (experiment A), fuller reward audits and explicit seed roles. Ensemble mean versus mean-minus-std is separate experiment B, conditional on held-out disagreement/error validation. Do not launch the current repeat-penalty suite as if it implemented A or B.

## Scope
Preserve `run_windows.bat pilot` and `reanalyze`; make no-argument launch run a fresh, multi-seed research suite. Do not overwrite historical runs. No publication, GPU training or upload is performed by this code-editing task.

## Implementation
1. Add `rl_recsys/suite.py`: validate options, create per-seed configs/prepared inputs, train full pipelines for 42/43/44 before evaluating any seed; retain reward-gate failures, fail on other errors; run frozen reanalysis and one conditional follow-up per baseline seed; aggregate within-seed effects (never raw cross-RM reward); ZIP64 bundle all weights/data/source/logs/checksums, including failure state.
2. `experiment.py`: opt-in early-calibration checkpoint selection / late-calibration reward audit, and uniform-user sampling. Defaults preserve historical semantics. Suite enables both. The follow-up audit remains independent of its new RM checkpoint selection, not untouched relative to all earlier pipeline decisions.
3. `run_windows.bat`: default `research`, explicit `pilot` retains legacy workflow. Separate installer/launcher for research; unique output path, captured output and exact process status. No claim that rerunning old calendar boundaries creates a fresh test.
4. Add regression tests for real tiny training path with eligibility filtering, split routing, suite phase ordering, gate failures, bundle completeness and refusal to overwrite. Document commands/budget and reward rationale.

## Verification
Run focused tests then full pytest and inspect final diff/status. Windows CMD/PowerShell execution and full CUDA experiment require user hardware; report these as untested here.

## Research limits
Three full pipeline seeds include data sampling variation; conditional follow-up is one seed per independently trained baseline, not three extra independent full pipelines. Existing test remains reused; common selection is not untouched-test evidence. Primary serving estimand is stochastic T=1 full-catalog expected RM completion, not deterministic top-10 utility. Selection-tuned rerank is an attainable tradeoff, not a formal upper bound. No real retention labels/online causality or automatic public artifact publication. Completion-minus-repeat is a transparent ablation, not assumed improved music utility. Failed reward gates are scientific outcomes, not reasons to relax the threshold.
