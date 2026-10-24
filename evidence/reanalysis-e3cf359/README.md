# Verified returned GPU reanalysis

Frozen historical checkpoints; analysis revision `e3cf3596b07ddeea672c25e4d10122f39e2c770a`, historical training revision `68578e4ae753214b506d146e04f8207686b3c852`. Exploratory reused test, not new training.

- [Independent review and interpretation](REVIEW.md)
- [Generated report](REPORT.md), [structured results](results.json), [selection](selection.json)
- [Historical target-based metric reconstruction](historical-verification.json)
- [Original completion manifest](COMPLETE.json)

This directory is a **compact excerpt**, not the complete analysis bundle. COMPLETE.json is retained unchanged and also references omitted evaluation-inputs.npz, per_example.npz and provenance.json. It must not be interpreted as a locally complete manifest. All 12 referenced files were available and hash-verified during review. Large arrays, supplied checkpoints and prepared data are not committed here. Public independent inference still needs access to those assets.

Summary provenance and source hashes appear in evidence-summary.json. Its archive hash binds the original supplied analysis.7z; selected-file checksums bind this checked-in excerpt. No credentials or machine-local environment package dump is included.
