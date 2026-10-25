"""Matched Experiment A suite; fixed data/SFT/RM, conditional optimization seeds.

Optional outer sample seeds repeat matched comparisons, never form an ensemble.
Every comparison finishes training and selection before any test inference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

from .config import validate_config
from .data import sha256
from .artifacts import atomic_json


def seed_effects(results):
    """Retain per-baseline effects; do not average rewards from different RMs."""
    effects = {}
    for seed, result in results.items():
        for arm, metrics in result.get("conditional_seed_effects", {}).items():
            for metric, row in metrics.items():
                for entry in row["per_seed"]:
                    effects.setdefault(arm, {}).setdefault(metric, []).append(
                        {"baseline_seed": int(seed), "relative_difference": entry["relative_difference"]})
    summary = {}
    for arm, metrics in effects.items():
        summary[arm] = {}
        for metric, entries in metrics.items():
            finite = [x["relative_difference"] for x in entries if x["relative_difference"] is not None]
            summary[arm][metric] = {"per_baseline": entries, "available_baselines": len(finite),
                "mean_relative_difference": float(np.mean(finite)) if finite else None,
                "baseline_seed_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None}
    return summary


def bundle_suite(root, source):
    """ZIP64 without duplicating weights on disk. Bind every included file."""
    root, source = Path(root), Path(source)
    destination = Path(str(root) + ".zip")
    if destination.exists():
        raise FileExistsError(destination)
    entries = [(p, p.relative_to(root).as_posix()) for p in sorted(root.rglob("*")) if p.is_file()]
    # Source snapshot is small; deliberately exclude environment, Git and other runs.
    for pattern in ("rl_recsys/*.py", "tests/*.py", "configs/*.json", "*.md", "docs/**/*.md",
                    "*.bat", "pyproject.toml"):
        entries += [(p, "source/" + p.relative_to(source).as_posix()) for p in sorted(source.glob(pattern)) if p.is_file()]
    names = [name for _, name in entries]
    if len(names) != len(set(names)) or "SHA256SUMS.json" in names:
        raise ValueError("Duplicate suite bundle paths")
    if any(p.is_symlink() for p, _ in entries):
        raise ValueError("Symlinks are not supported in suite bundles")
    checks = [{"path": name, "sha256": sha256(p)} for p, name in entries]
    temporary = destination.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for p, name in entries:
            archive.write(p, name)
        archive.writestr("SHA256SUMS.json", json.dumps(checks, indent=2))
    # Read every archived member and verify its recorded SHA256, including ZIP64 assets.
    import hashlib
    with zipfile.ZipFile(temporary) as archive:
        for entry in checks:
            digest = hashlib.sha256()
            with archive.open(entry["path"]) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Archive hash mismatch: {entry['path']}")
    temporary.replace(destination)
    atomic_json(Path(str(root) + ".zip.sha256.json"), {"path": destination.name, "sha256": sha256(destination)})
    return destination


def run_suite(config, raw, output, seeds=(42, 43, 44), steps=1000, repeat_penalty=.1, *, setup_logs=None,
              sample_seeds=None):
    """Run subprocess stages with lossless logs; a failed scientific gate is retained.

    Any failed command aborts and attempts a failure bundle. Existing output
    is never resumed/overwritten. Production training requires CUDA.
    """
    from .followup import _validate_options
    seeds = _validate_options(seeds, steps, repeat_penalty)
    if sample_seeds is not None:
        sample_seeds = _validate_options(sample_seeds, steps, repeat_penalty)
    source = Path(__file__).resolve().parents[1]
    config, raw, output = Path(config).resolve(), Path(raw).resolve(), Path(output).resolve()
    base = json.loads(config.read_text(encoding="utf-8-sig"))
    validate_config(base)
    if output.exists() or Path(str(output) + ".zip").exists():
        raise FileExistsError("Use a fresh suite output directory and ZIP path")
    if setup_logs is not None and not Path(setup_logs).is_dir():
        raise ValueError("Missing setup logs directory")
    output.mkdir(parents=True)
    if setup_logs is not None:
        shutil.copytree(setup_logs, output / "setup-logs")
    started = time.monotonic()
    samples = tuple(sample_seeds) if sample_seeds is not None else (base.get("sample_seed", base["seed"]),)
    manifest = {"status": "in progress", "optimization_seeds": list(seeds),
        "sample_seeds": list(samples), "sample_robustness": sample_seeds is not None,
        "seed_roles": {"optimization": list(seeds), "sft": base["seed"], "reward_model": base["seed"]},
        "steps": steps, "bundle_status": "pending",
        "repeat_penalty": repeat_penalty, "base_config": str(config),
        "stages": [], "baselines": {},
        "evaluation_status": "exploratory_reuse",
        "limitations": ["Historical calendar test remains reused, not fresh or confirmatory.",
            "Optimization seeds are conditional on fixed data/SFT/RM within each comparison; optional outer samples are separate robustness runs.",
            "No online user impact or retention inference; no automatic publication.",
            "Gate failures are retained, not replaced by favorable seeds; missing arms are not successes.",
            "Follow-up late audit excludes new RM checkpoint selection but has been used by its parent pipeline."]}
    atomic_json(output / "suite.json", manifest)

    def persist():
        atomic_json(output / "suite.json", manifest)

    def command(label, *args):
        index = len(manifest["stages"])
        log = output / "logs" / f"{index:02d}-{label}.log"
        log.parent.mkdir(exist_ok=True)
        argv = [sys.executable, "-u", "-m", "rl_recsys.cli", *map(str, args)]
        print(f"[{index + 1}] {label} -> {log}", flush=True)
        t0 = time.monotonic()
        record = {"label": label, "argv": argv, "status": "running", "log": str(log.relative_to(output))}
        manifest["stages"].append(record)
        persist()
        try:
            with log.open("wb") as stream:
                code = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, cwd=source).returncode
        except BaseException as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          elapsed_seconds=time.monotonic() - t0)
            persist()
            raise
        record.update(returncode=code, elapsed_seconds=time.monotonic() - t0,
                      status="complete" if code == 0 else "failed")
        persist()
        if code:
            raise RuntimeError(f"Stage {label} failed with exit {code}; see {log}")

    try:
        command("download", "download", "--config", config, "--output", raw)
        paths = {}
        # Outer samples are separate matched comparisons, not optimization seeds.
        for seed in samples:
            directory = output / f"sample_{seed}"
            directory.mkdir()
            cfg = copy.deepcopy(base)
            # Preparation uses sample_seed; supervised/RM initialization stays fixed.
            cfg["sample_seed"] = seed
            cfg["train"]["user_uniform"] = True
            cfg.setdefault("reward_audit", {})["independent_calibration"] = True
            validate_config(cfg)
            cfg_path, data, run = directory / "config.json", directory / "data", directory / "run"
            atomic_json(cfg_path, cfg)
            paths[seed] = (directory, cfg_path, data, run)
            command(f"prepare-{seed}", "prepare", "--config", cfg_path, "--input", raw, "--output", data)
            command(f"sft-{seed}", "train", "--config", cfg_path, "--data", data, "--run", run, "--sft-only")
            manifest["baselines"][str(seed)] = {"status": "sft_frozen", "sample_seed": seed,
                "sft_seed": base["seed"], "reward_model_seed": base["seed"], "optimization_seeds": list(seeds)}
            persist()
        # All controls are frozen before ANY suite test inference.
        for seed, (directory, cfg_path, data, run) in paths.items():
            command(f"followup-{seed}", "followup", "--data", data, "--run", run, "--output", directory / "followup",
                    "--seeds", *seeds, "--steps", steps, "--repeat-penalty", repeat_penalty, "--training-only-parent")
            frozen = json.loads((directory / "followup" / "followup_manifest.json").read_text())
            manifest["baselines"][str(seed)].update(status="all_arms_frozen", optimization_statuses={
                opt: row["status"] for opt, row in frozen["seeds"].items()})
            persist()
        results = {}
        for seed, (directory, cfg_path, data, run) in paths.items():
            command(f"evaluate-controls-{seed}", "evaluate-followup", "--data", data, "--run", run,
                    "--followup", directory / "followup", "--output", directory / "controls-evaluation")
            results[str(seed)] = json.loads((directory / "controls-evaluation" / "results.json").read_text())
            manifest["baselines"][str(seed)]["status"] = "evaluated"
            persist()
        manifest["conditional_optimization_effects_by_sample"] = {
            seed: row.get("conditional_seed_effects", {}) for seed, row in results.items()}
        manifest["status"] = "complete; inspect gates and limitations, not evidence of RL benefit"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        persist()
        # A failure archive is explicitly marked failed and never masks the exit status.
        try:
            manifest["bundle_status"] = "complete"
            persist()
            bundle_suite(output, source)
        except Exception:
            manifest["scientific_status"] = manifest["status"]
            manifest["status"] = "bundle_failed"
            manifest["bundle_status"] = "failed"
            (output / "bundle-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if manifest.get("error") is None:
                manifest["error"] = "Bundle creation failed; scientific outputs may be complete but are not packaged."
            persist()
            raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--raw", default="data/listens.parquet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--sample-seeds", nargs="+", type=int, help="Optional separate outer sample robustness; model seeds stay fixed")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeat-penalty", type=float, default=.1)
    parser.add_argument("--setup-logs")
    args = parser.parse_args(argv)
    run_suite(args.config, args.raw, args.output, args.seeds, args.steps, args.repeat_penalty, setup_logs=args.setup_logs, sample_seeds=args.sample_seeds)


if __name__ == "__main__":
    main()
