# Research protocol amendment — pre-run gate and improvements

This amendment is an engineering and measurement upgrade, not new empirical evidence. The historical test period was inspected before these changes; its results remain descriptive.

## Required order before a fresh GPU run

1. Run `rl-recsys preflight --config configs/pilot.json --device cuda` on the target GPU and save its JSON output. The small preflight verifies representative policy/reward forward/backward execution, but does not certify full-grid runtime or peak memory.
2. Freeze a genuinely untouched later evaluation period, or explicitly label the run `exploratory_reuse`. A new seed or run directory does not create a fresh test population.
3. Prepare data with a fixed data seed/population and preserve the generated audit. The current strict protocol excludes every member of ambiguous user/timestamp groups from both history and targets; integer IDs are preserved exactly during deduplication.
4. Train SFT and both reward heads first. Validate completion and normalized-seconds heads on a time-forward audit role against constant and fit-only item-mean baselines. A weak seconds head must disable seconds optimization; reward-model architecture disagreement is a diagnostic, not independent truth.
5. Run a reduced H=1 comparison first: SFT, reward reranking, and RL. The exact one-step expected-reward utility is available for deterministic frozen-policy comparisons and should be used as the primary H=1 reward diagnostic where the full catalog fits. Sampling remains a secondary behavioral diagnostic.
6. Only after the audit gate and measured runtime pass should H=5, larger grids, or additional seeds be enabled.

## Statistical and artifact rules

- Selection and H2 relevance use eligible-target user-macro NDCG; all-target metrics are secondary diagnostics.
- Primary sampled reward intervals bootstrap per-context action-sample means by user and are conditional on realized draws. The retained sampling diagnostic is descriptive and does not feed H2.
- H1 requires point relative MAE reduction >=2%, a paired interval excluding zero, and adequate users. H2 requires >=5% reward improvement with eligible NDCG non-inferiority at -2%; reused-test outputs are labeled exploratory and cannot be treated as confirmatory support.
- Evaluation artifacts are atomic and must have a valid `EVALUATION_COMPLETE.json` hash manifest. Never edit a historical manifest to bypass provenance checks.
- New runs embed source bytes and installed-package information; final evaluation rejects source-code drift.

## Open research risks

The experiment remains model-based policy optimization against a frozen observational reward model and append-item simulator, not causal user on-policy learning. Logged listens are exposure-selected. The pilot catalog is restricted to train-known items and one training seed. Exact expected reward still does not validate causal impact. Retention/continuation rewards require a separate future-observation window, censoring treatment, and eventually randomized online validation.
