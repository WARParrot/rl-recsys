"""Exploratory completion-only follow-up importing an immutable stage4 suite.

No download or SFT retraining. All new training/selection precedes test inference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .artifacts import atomic_json
from .data import sha256
from .reanalysis import verify_inputs
from .suite import bundle_suite


def run_study(parent, output, *, steps=1000, seeds=(42, 43, 44), max_training_hours=4.5):
    from .followup import _validate_options
    seeds = _validate_options(seeds, steps, .1)
    if isinstance(max_training_hours, bool) or not math.isfinite(max_training_hours) or max_training_hours <= 0:
        raise ValueError("max_training_hours must be positive and finite")
    parent, output = Path(parent).resolve(), Path(output).resolve()
    sample = parent / "sample_42"
    data, run = sample / "data", sample / "run"
    if not (run / "SFT_READY.json").is_file():
        raise ValueError("Expected original suite root containing sample_42/run/SFT_READY.json")
    identity = verify_inputs(run, data, training_only=True)
    if output.exists() or Path(str(output) + ".zip").exists():
        raise FileExistsError("Use a fresh study output directory and ZIP path")
    # Never write inside the original evidence tree (or archive it recursively).
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("Study output and original suite must be separate trees")
    joint = sample / "followup" / "seed_42"
    marker = json.loads((sample / "followup" / "FOLLOWUP_COMPLETE.json").read_text(encoding="utf-8-sig"))
    old_manifest_path = sample / "followup" / "followup_manifest.json"
    if marker.get("status") != "complete" or marker["manifest_sha256"] != sha256(old_manifest_path):
        raise ValueError("Original joint follow-up manifest mismatch")
    old_prov_path = sample / "followup" / "provenance.json"
    if marker["provenance_sha256"] != sha256(old_prov_path):
        raise ValueError("Original joint provenance mismatch")
    old_prov = json.loads(old_prov_path.read_text(encoding="utf-8-sig"))
    if (old_prov["prepared_sha256"] != identity["prepared_sha256"]
            or old_prov["historical_provenance_sha256"] != identity["historical_provenance_sha256"]
            or old_prov["historical_sft_sha256"] != sha256(run / "sft.pt")):
        raise ValueError("Original joint and imported SFT/data identity mismatch")
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8-sig"))
    for name, digest in {**old_manifest["seeds"]["42"]["artifact_hashes"],
                         **old_manifest["seeds"]["42"]["checkpoint_hashes"]}.items():
        if "/" in name or "\\" in name or Path(name).name != name or sha256(joint / name) != digest:
            raise ValueError("Original joint artifact mismatch")
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    status = {"status": "running", "evaluation_status": "exploratory_reuse",
              "hypothesis": "Removing auxiliary seconds loss may improve completion modeling; not an established mechanism",
              "parent": str(parent), "parent_identity": identity,
              "reward_objective": "completion", "steps": steps, "seeds": list(seeds),
              "max_training_hours": max_training_hours, "max_evaluation_minutes": 90,
              "stages": [], "bundle_status": "pending"}

    def persist():
        atomic_json(output / "study.json", status)

    def command(label, module, *args):
        log = output / "logs" / f"{label}.log"
        log.parent.mkdir(exist_ok=True)
        argv = [sys.executable, "-u", "-m", module, *map(str, args)]
        row = {"label": label, "argv": argv, "status": "running", "log": str(log.relative_to(output))}
        status["stages"].append(row)
        persist()
        t0 = time.monotonic()
        print(f"{label} -> {log}", flush=True)
        try:
            with log.open("wb") as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=source,
                                        timeout=max_training_hours * 3600 if label == "train" else 95 * 60)
        except BaseException as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}",
                       elapsed_seconds=time.monotonic() - t0)
            persist()
            raise
        row.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - t0,
                   status="complete" if result.returncode == 0 else "failed")
        persist()
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {log}")

    persist()
    try:
        # Self-contained new archive; originals remain byte-for-byte untouched.
        shutil.copytree(data, output / "inputs" / "data")
        copied_run = output / "inputs" / "run"
        copied_run.mkdir()
        for name in ["SFT_READY.json", "provenance.json", "training.json", "sft.pt"]:
            shutil.copyfile(run / name, copied_run / name)
        baseline = output / "historical-joint"
        baseline.mkdir()
        for name in ["reward.pt", "item_only.pt", "reward_audit.json", "reward_curves.json", "config.json"]:
            shutil.copyfile(joint / name, baseline / name)
        for name in ["provenance.json", "followup_manifest.json", "FOLLOWUP_COMPLETE.json"]:
            shutil.copyfile(sample / "followup" / name, baseline / name)
        atomic_json(baseline / "IDENTITY.json", {"role": "frozen historical joint objective; not retrained; not new independent evidence",
                    "sha256": {p.name: sha256(p) for p in baseline.iterdir() if p.is_file()}})
        new_data = output / "inputs" / "data"
        copied_identity = verify_inputs(copied_run, new_data, training_only=True)
        if copied_identity != identity:
            raise ValueError("Copied input identities differ from original")
        command("train", "rl_recsys.followup", "--data-dir", new_data, "--run-dir", copied_run,
                "--output", output / "followup", "--source-mode", "historical_reanalysis",
                "--training-only-parent", "--reward-objective", "completion", "--steps", steps, "--seeds", *seeds)
        command("evaluate", "rl_recsys.cli", "evaluate-followup", "--data", new_data,
                "--run", copied_run, "--followup", output / "followup", "--output", output / "controls-evaluation",
                "--max-minutes", 90)
        new_audit = json.loads((output / "followup" / f"seed_{seeds[0]}" / "reward_audit.json").read_text())
        old_audit = json.loads((baseline / "reward_audit.json").read_text(encoding="utf-8-sig"))
        atomic_json(output / "reward-comparison.json", {
            "role": "same late calibration already inspected; descriptive comparison, not fresh confirmation",
            "historical_joint": old_audit["heads"]["completion"],
            "completion_only": new_audit["heads"]["completion"],
            "gate": new_audit["gate"],
            "conclusion": "Gate failed: retain negative result; no RL/rerank and no threshold change" if not new_audit["gate"]["heads"]["completion"]["enabled"] else "Gate admitted exploratory proxy optimization, not real user benefit",
            "limitations": ["No paired joint-versus-completion CI in this summary; each audit includes paired fit/item-only baseline contrasts",
                            "Joint and completion checkpoint selection optimize different prespecified MSE objectives",
                            "MAE gate is retained for comparability, not a guarantee of expected-utility accuracy",
                            "Secondary seconds output in completion-only models is untrained",
                            "Original 21-minute suite skipped RL: its runtime does not estimate admitted RL runtime"]})
        status["status"] = "complete; inspect reward gate and exploratory results"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        persist()
        try:
            status["bundle_status"] = "complete"
            persist()
            bundle_suite(output, source)
        except Exception:
            status.update(scientific_status=status["status"], status="bundle_failed", bundle_status="failed")
            (output / "bundle-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            persist()
            raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, help="Original research suite root, not ZIP or sample_42")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000, help="Predeclared equal per-policy updates; do not change after inspecting results")
    parser.add_argument("--max-training-hours", type=float, default=4.5,
                        help="Hard training-process cap; timeout is failure, never partial success")
    args = parser.parse_args()
    run_study(args.parent, args.output, steps=args.steps, max_training_hours=args.max_training_hours)


if __name__ == "__main__":
    main()
