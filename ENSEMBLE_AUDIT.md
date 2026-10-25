# Stage5: opt-in ensemble hypothesis audit (not B training)

`rl_recsys.ensemble_audit` consumes **explicitly exported predictions on observed
late-calibration targets**. It never trains members/policies, invents labels,
fits calibration, accesses the historical test, or launches Experiment B. The
default A suite/Windows launchers do not call it. Existing A optimization seeds
or outer samples are not ensemble members. No new empirical result is claimed.

```text
python -m rl_recsys.ensemble_audit observed-ensemble.npz --output ensemble-audit.json
```

API: `audit_ensemble(arrays, bootstrap=2000, seed=2026)` or
`audit_npz(path, bootstrap=2000, seed=2026)` returns a JSON-safe dict. CLI writes
that dict. Invalid inputs raise an error/nonzero exit; a scientifically failed
or insufficient gate writes a report with `gate.passed=false` (not a CLI error).
No GPU, model loading or download is needed. NumPy is the only dependency.

Reports bind `input_sha256` to the exact NPZ bytes loaded (`null` for the in-memory API) and `content_sha256` to all validated export fields, including predictions, labels, early/late IDs and timestamps, member identities/roles and slice memberships/metadata. The versioned content framing sorts field names and hashes length-prefixed dtype/shape headers and C-order array bytes. Archive compression/order does not affect content identity; dtype, shape or field changes do. These hashes bind inputs, not the truth of declared training provenance.

## Explicit export contract

Write an NPZ with `numpy.savez`/`savez_compressed`, using numeric or Unicode arrays,
**never object arrays/pickle**. The producer must export real observed labels and
aligned member predictions; this tool checks consistency of declarations, not
their authenticity. Hash identifiers should refer to frozen manifests/artifacts.
There is no automatic ensemble exporter or member-training pipeline yet.

Required fields:

| Field | Shape / meaning |
|---|---|
| `predictions` | `[N,M]`, N>0, M>=3 independently initialized history RMs; finite [0,1] observed-target completion scores, identical row order |
| `labels` | `[N]` real observed completion labels, finite [0,1]; no hypothetical-action labels |
| `uid` | `[N]` stable user identities (integer or string) |
| `example_id`, `timestamp` | `[N]` unique stable event identities and finite numeric timestamps, same units as early timestamps |
| `early_example_id`, `early_timestamp` | Nonempty unique early-calibration event identities and timestamps; no overlap with late IDs; max early time strictly less than min late time |
| `data_id`, `catalog_id`, `source_id`, `split_id` | Scalar nonempty strings identifying the shared prepared data, ordered catalog, frozen model/export source, and frozen split manifest |
| `member_data_id`, `member_catalog_id`, `member_source_id`, `member_split_id` | `[M]` strings, each equal to its corresponding shared identity; mixed pipelines rejected |
| `member_id`, `member_seed` | `[M]` distinct member identities and distinct initialization seeds (integer/string) |
| `member_fit_role` | `[M]`, all `fit` |
| `member_checkpoint_role` | `[M]`, all `early_calibration`; late labels cannot select members |
| `audit_role` | Scalar `late_calibration` |
| `label_source` | Scalar `observed` |
| `calibration_status` | Scalar `identity` or `uncalibrated` only |

Identity means **no transformation, no fitted calibrator, no calibrated claim**.
Both statuses are audited as supplied. This intentionally avoids an unsound
fitted calibration pipeline. Early labels are not required or used by this tool;
early identities/timestamps document separation. Fitted calibration exports are
rejected, not treated as validated. Any future fitted method needs a declared
simple procedure fitted on early labels only and separately tested provenance.
The audit cannot establish independent initialization, label authenticity, fit
role purity or source integrity from self-declared metadata alone.

Optional slices: supply `slice__duration`, `slice__popularity`, `slice__repeat`
(or other names), each a `[N]` integer/string group identity array, plus scalar
`slice_metadata_json`. This JSON object maps each name to an object with
`derived_from: "fit"`, matching `data_id` and `catalog_id`, and a nonempty
`method` describing fixed boundaries/statistics. Use fit-only per-item duration
medians, fit popularity thresholds, and the fixed repeat-in-context definition;
do not tune boundaries on late errors. Missing/unknown groups can be explicit
string categories. Each group receives a descriptive association/CI, support and
degeneracy flags. Slices are not multiplicity-adjusted and do not alter the gate.
Missing slices do not block calculation, but block a claim of complete B
confounding diagnostics. The tool validates declarations, not derivation code.

## Estimands, ties, bootstrap and gate

- Disagreement is population SD across members (`ddof=0`). Error is absolute
  difference between ensemble mean and the observed label.
- Association is pooled-example Spearman with average ranks for exact ties,
  **not** correlation of user means. Bootstrap samples users with replacement,
  retains all their rows with multiplicity, and recomputes tied ranks on each
  paired sample. This estimates a row-weighted association with clustered
  uncertainty; it is distinct from the user-macro contrast below.
- Low/high groups are the lowest/highest `ceil(.1*N)` SD rows expanded to include
  all boundary ties. Cutoffs/groups stay fixed in bootstrap. Counts may exceed
  10%. Overlapping tails are flagged degenerate and cannot pass.
- Compute each user's low and high MAE **only for common users with both groups**.
  Report their mean paired difference (high minus low) and bootstrap those paired
  user differences. One-sided users never contribute to either reported MAE.
  Report total users, low/high/common/one-sided/neither users and row support,
  including the low/high row counts restricted to common users.
- Defaults: 2,000 user bootstrap replicates, seed 2026, percentile 95% CI.
  Degenerate Spearman draws are counted; if any draw is undefined, its interval
  is unavailable rather than silently conditioning on valid resamples. Constant
  variables, empty common support and overlapping tails cannot pass.
- Gate passes **only** when both CI lower bounds are strictly >0 and at least
  **100 common users** support the contrast. Insufficient/degenerate flags and
  explicit failure reasons accompany the estimates. This diagnostic gate is not
  automatic authorization or orchestration of B penalty training.

This is **predictive association, not causal or epistemic proof**. Disagreement
may reflect duration/popularity/repeat confounding; members may share confident
error. Logged-target validity need not transfer to policy-selected actions.

## Remaining Experiment B prerequisites

Separately implement and verify three independently initialized history RMs on
one frozen prepared dataset/catalog, early-only checkpoint selection and the
prespecified calibration decision, then explicitly export aligned real late
labels/predictions and all fit-derived slices. Identity/uncalibrated auditing
alone does **not** fulfill the protocol's prospective calibrated-ensemble
training requirement. Inspect the report and diagnostics before any B study.
Matched mean-only versus mean-minus-kappa-SD H=1 training, common unpenalized
selection, frozen provenance, and B evaluation/orchestration remain unimplemented.
No default suite training behavior changes in stage5.
