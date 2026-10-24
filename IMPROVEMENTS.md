# Audit improvement implementation plan

Scope: preserve the uploaded historical experiment and add a separate, traceable exploratory reanalysis. Do not bypass its source-identity guard or overwrite its manifests. New results require execution; source changes are not empirical evidence.

1. Add `rl_recsys/reanalysis.py` (new): verify/import the supplied ZIP, all checkpoints and prepared data; export complete evaluation inputs; independently recompute eligible/all-target metrics and intervals. Reject traversal, corruption and output reuse. Bind derived artifacts to historical inputs and current source hashes.
2. Add separate inference command: all historical policies, selection-only tuned reranking, shuffled-history and history-removal sensitivity, matched synthetic append controls and explicitly gapped observed-context replay. Keep historical weights frozen. Save arrays/targets and measured runtime/memory; never call this a fresh test.
3. Add opt-in follow-up training in a separate module/run: independent time-forward reward audit, item-only RM, continued SFT, completion-minus-repeat policy; select on selection only. Restrict claims to exploratory reuse and do not relabel the existing test as untouched. Multiple seed repetitions must actually retrain, not just resample inference.
4. Extend CLI and `run_windows.bat` with an explicit reanalysis mode that imports supplied assets and runs inference without training; retain the original training path. Package complete inputs/checkpoints using a Python ZIP writer rather than silently omitting large files.
5. Add exact tiny-MDP gradient regression tests, import/verification corruption tests, metric reconstruction tests and control semantics tests. Run full suite and independently review the diff.
6. Update README with actual commands, existing-run learning curves, interpretation boundaries, required GPU work and remaining untouched-period limitation.

Verification: existing suite; CPU synthetic fixtures for import/metrics/controls; exact float64 gradient enumeration; supplied files' hashes and historical metric reconstruction. GPU inference and new research runs are deferred to the user's Windows machine.

Migration/rollback: historical manifests and outputs stay immutable; all new products go to a fresh directory. No checkpoints/data are committed to Git. Reanalysis accepts historical source differences only as an explicit new analysis with its own provenance, not as historical final evaluation.
