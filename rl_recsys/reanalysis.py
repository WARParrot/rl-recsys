"""Explicit exploratory reanalysis of an immutable historical experiment.

Import validates bytes before copying. Inference uses current code and records a
new provenance: it never edits historical manifests or bypasses final_evaluation.
"""
from __future__ import annotations

import copy
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

import numpy as np
import torch

from .artifacts import atomic_json, atomic_npz, validate_evaluation_artifacts
from .data import load_prepared, role, sha256
from .evaluate import evaluate, macro, paired_interval, ranking_metrics
from . import experiment as exp


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _safe_name(name):
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (path.is_absolute() or not path.parts or ".." in path.parts
            or any(":" in part for part in path.parts)):
        raise ValueError(f"Unsafe archive or manifest path: {name}")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"{prefix}{i}" for prefix in ["COM", "LPT"] for i in range(1, 10)}
    if any(part.endswith((" ", ".")) or part.split(".")[0].upper() in reserved for part in path.parts):
        raise ValueError(f"Unsafe Windows archive path: {name}")
    return path.as_posix()


def _manifest_name(name):
    safe = _safe_name(name)
    if "/" in safe or not safe.endswith(".pt"):
        raise ValueError(f"Invalid checkpoint name: {name}")
    return safe


def verify_inputs(run_dir, data_dir):
    """Verify complete historical inputs, including the bytes absent from old ZIPs."""
    run, data = Path(run_dir), Path(data_dir)
    ready = read_json(run / "READY_FOR_TEST.json")
    prov = read_json(run / "provenance.json")
    for name, digest in ready["checkpoint_hashes"].items():
        name = _manifest_name(name)
        if sha256(run / name) != digest:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    for name in ["selection", "candidates", "training", "provenance", "reward_audit"]:
        if sha256(run / f"{name}.json") != ready[f"{name}_sha256"]:
            raise ValueError(f"Historical metadata hash mismatch: {name}")
    if sha256(data / "prepared.npz") != prov["prepared_sha256"]:
        raise ValueError("Prepared data hash mismatch")
    audit_hash = sha256(data / "audit.json")
    if audit_hash != prov["audit_sha256"] or audit_hash != ready["prepared_audit_sha256"]:
        raise ValueError("Prepared audit hash mismatch")
    selected = read_json(run / "selection.json").get("chosen")
    selected_name = f'{selected["name"]}.pt' if selected else None
    if ready.get("selected_checkpoint") != selected_name:
        raise ValueError("Selected checkpoint does not match selection")
    if selected_name and selected_name not in ready["checkpoint_hashes"]:
        raise ValueError("Selected checkpoint not bound to READY")
    validate_evaluation_artifacts(run)
    return {"status": "verified", "checkpoint_hashes": ready["checkpoint_hashes"],
            "prepared_sha256": prov["prepared_sha256"],
            "historical_provenance_sha256": sha256(run / "provenance.json"),
            "historical_results_sha256": sha256(run / "results.json"),
            "historical_revision": prov.get("git_revision")}


def import_run(bundle, assets, output):
    """Import flat uploaded attachments or an existing run directory, without torch.load.

    `assets` contains prepared.npz and checkpoints (flat, or the usual run/data
    layout). A fresh output is required; partially failed imports are not usable.
    """
    bundle, assets, out = Path(bundle), Path(assets), Path(output)
    if out.exists():
        raise FileExistsError("Import requires a fresh output directory")
    with zipfile.ZipFile(bundle) as archive:
        names = {}
        case_names = set()
        total = 0
        for info in archive.infolist():
            safe = _safe_name(info.filename)
            if info.is_dir():
                continue
            if safe.casefold() in case_names:
                raise ValueError("Duplicate normalized ZIP path")
            case_names.add(safe.casefold())
            # Keep small historical metadata ZIPs bounded; weights are supplied separately.
            total += info.file_size
            if total > 2 * 1024**3 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Oversize archive or symlink rejected")
            names[safe] = info
        if "SHA256SUMS.json" not in names:
            raise ValueError("Missing ZIP checksum manifest")
        checks = json.loads(archive.read(names["SHA256SUMS.json"]).decode("utf-8-sig"))
        checked = set()
        for entry in checks:
            name = _safe_name(entry["path"])
            if name in checked or name not in names:
                raise ValueError("Duplicate or missing checksum entry")
            if hashlib.sha256(archive.read(names[name])).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP checksum mismatch: {name}")
            checked.add(name)
        if checked != set(names) - {"SHA256SUMS.json"}:
            raise ValueError("ZIP contains unchecked files")
        provenance_paths = [n for n in names if n.endswith("/provenance.json")]
        if len(provenance_paths) != 1:
            raise ValueError("Expected exactly one historical run")
        source_run = PurePosixPath(provenance_paths[0]).parent.as_posix()
        ready = json.loads(archive.read(names[source_run + "/READY_FOR_TEST.json"]).decode("utf-8-sig"))
        prov = json.loads(archive.read(names[source_run + "/provenance.json"]).decode("utf-8-sig"))
        sources = {}
        for name, digest in ready["checkpoint_hashes"].items():
            name = _manifest_name(name)
            choices = [assets / name, assets / source_run / name]
            source = next((p for p in choices if p.is_file()), None)
            if source is None or sha256(source) != digest:
                raise ValueError(f"Missing or mismatched supplied checkpoint: {name}")
            sources[name] = source
        prepared = next((p for p in [assets / "prepared.npz", assets / "data/prepared/prepared.npz"] if p.is_file()), None)
        if prepared is None or sha256(prepared) != prov["prepared_sha256"]:
            raise ValueError("Missing or mismatched supplied prepared.npz")
        out.mkdir(parents=True)
        # Only verified archive bytes are materialized; all paths were checked first.
        for name, info in names.items():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        for name, source in sources.items():
            shutil.copy2(source, out / source_run / name)
        shutil.copy2(prepared, out / "data/prepared/prepared.npz")
    verified = verify_inputs(out / source_run, out / "data/prepared")
    result = {**verified, "run": source_run, "data": "data/prepared",
              "bundle_sha256": sha256(bundle), "interpretation": "historical inputs; no fresh test"}
    atomic_json(out / "IMPORT_COMPLETE.json", result)
    return result


def reconstruct_metrics(run_dir, data_dir, output):
    """Recompute ranking from targets, eligible populations and saved RM arrays (CPU)."""
    identity = verify_inputs(run_dir, data_dir)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh metrics verification output")
    data = load_prepared(data_dir)
    test = role(data, "test")
    eligible = test["target"] >= 2
    policies = {}
    with np.load(Path(run_dir) / "per_example.npz", allow_pickle=False) as saved:
        policy_names = sorted(k.removesuffix("_recommendations") for k in saved.files if k.endswith("_recommendations"))
        for name in policy_names:
            uid = saved[f"{name}_uid"]
            if not np.array_equal(uid, test["uid"]):
                raise ValueError(f"Row/user mismatch: {name}")
            recs = saved[f"{name}_recommendations"]
            recall, ndcg = ranking_metrics(recs, test["target"])
            for metric, actual in [("recall", recall), ("ndcg", ndcg)]:
                if not np.allclose(actual, saved[f"{name}_{metric}"], rtol=1e-6, atol=1e-8):
                    raise ValueError(f"Saved {metric} disagrees with targets: {name}")
            exact = saved[f"{name}_exact_one_step_expected_reward"]
            _, counts = np.unique(recs, return_counts=True)
            report = {"all_target_recall": macro(recall, uid)[0], "all_target_ndcg": macro(ndcg, uid)[0],
                      "eligible_ndcg": macro(ndcg[eligible], uid[eligible])[0],
                      "eligible_expected_reward": macro(exact[eligible], uid[eligible])[0],
                      "all_target_expected_reward": macro(exact, uid)[0],
                      "coverage": len(np.unique(recs)) / len(data["vocab"]),
                      "hhi": float(((counts / counts.sum()) ** 2).sum())}
            cfg = read_json(Path(run_dir) / "provenance.json")["config"]
            if name != "sft":
                report["paired_vs_sft"] = {
                    "eligible_ndcg": paired_interval(saved["sft_ndcg"][eligible], ndcg[eligible], uid[eligible], cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"]),
                    "eligible_expected_reward": paired_interval(saved["sft_exact_one_step_expected_reward"][eligible], exact[eligible], uid[eligible], cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"])}
            policies[name] = report
    historical = read_json(Path(run_dir) / "results.json")
    for name, metrics in policies.items():
        old = historical["policies"][name]
        for key, old_key in [("all_target_recall", "recall"), ("all_target_ndcg", "ndcg"),
                             ("eligible_ndcg", "eligible_ndcg"), ("eligible_expected_reward", "eligible_exact_one_step_expected_reward"),
                             ("coverage", "catalog_coverage"), ("hhi", "recommendation_hhi")]:
            if not np.isclose(metrics[key], old[old_key], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Reconstructed metric disagrees with results.json: {name}/{key}")
    result = {"status": "verified", "inputs": identity, "examples": len(eligible),
              "eligible_examples": int(eligible.sum()), "policies": policies,
              "boundary": "ranking recomputed from targets; RM expectations remain saved outputs, not new inference"}
    atomic_json(out, result)
    return result


def flatten_arrays(values, prefix=""):
    """NPZ stays pickle-free, including nested diagnostic mode dictionaries."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flatten_arrays(value, name + "_"))
        else:
            result[name] = np.asarray(value)
    return result


def export_evaluation_inputs(data, path):
    arrays = {"vocab": data["vocab"], "popularity": data["popularity"], "durations": data["durations"]}
    for name in ["selection", "test"]:
        subset = role(data, name)
        arrays.update({f"{name}_{key}": value for key, value in subset.items()})
        arrays[f"{name}_eligible"] = subset["target"] >= 2
        arrays[f"{name}_row_id"] = np.arange(len(subset["target"]))
    atomic_npz(path, arrays)


def learning_curves(run_dir):
    logs = read_json(Path(run_dir) / "training.json")
    result = {}
    for name, records in logs.items():
        supervised = [r for r in records if "calibration_loss" in r]
        if supervised:
            best = min(supervised, key=lambda r: r["calibration_loss"])
            result[name] = {"selected_step": best["step"], "best_calibration_loss": best["calibration_loss"],
                            "best_at_first_evaluation": best is supervised[0], "curve": supervised}
    return result


def write_plots(output, result):
    """Dependency-free SVGs; numeric data remain authoritative in results.json."""
    out = Path(output)
    def chart(filename, title, series, xlabel, ylabel):
        points = [p for _, values in series for p in values]
        if not points:
            return
        xs, ys = zip(*points)
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        def xy(x, y):
            return 80 + 650 * (x-xmin) / max(xmax-xmin, 1e-12), 330 - 260 * (y-ymin) / max(ymax-ymin, 1e-12)
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="440" viewBox="0 0 1100 440">',
               '<rect width="1100" height="440" fill="white"/>',
               f'<text x="60" y="30" font-size="18">{html.escape(title)}</text>',
               '<path d="M80 60 V330 H730" fill="none" stroke="black"/>',
               f'<text x="260" y="400">{html.escape(xlabel)} [{xmin:.6g}, {xmax:.6g}]</text>',
               f'<text x="85" y="55">{html.escape(ylabel)} [{ymin:.6g}, {ymax:.6g}]</text>']
        colors = ['#1565c0', '#c62828', '#2e7d32', '#6a1b9a', '#ef6c00', '#00838f', '#4527a0', '#795548', '#37474f', '#ad1457']
        for i, (name, values) in enumerate(series):
            color = colors[i % len(colors)]
            coords = [xy(x, y) for x, y in values]
            if len(coords) > 1:
                svg.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)}" fill="none" stroke="{color}"/>')
            for x, y in coords:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            svg.append(f'<text x="750" y="{75+i*25}" fill="{color}" font-size="12">{html.escape(name)}</text>')
        svg.append('</svg>')
        (out / filename).write_text(''.join(svg), encoding='utf-8')
    chart('tradeoff.svg', 'Frozen-policy reward / relevance (exploratory)',
          [(name, [(r['eligible_exact_one_step_expected_reward'], r['eligible_ndcg'])]) for name, r in result['policies'].items()],
          'Eligible expected RM reward', 'Eligible NDCG@10')
    for name, record in result['learning_curves'].items():
        chart(f'learning-{name}.svg', f'Historical {name} calibration loss',
              [(name, [(r['step'], r['calibration_loss']) for r in record['curve']])], 'Optimizer step', 'Calibration loss')
    for name, row in result['policies'].items():
        if 'matched_rollouts' in row:
            chart(f'rollout-{name}.svg', f'Matched rollout RM trajectories: {name}',
                  [(mode, [(r['step'], r['reward']) for r in values['steps']]) for mode, values in row['matched_rollouts']['modes'].items()],
                  'Retained/synthetic step', 'RM score (not ground truth)')


def write_report(output, result):
    lines = ["# Exploratory frozen-checkpoint reanalysis", "", "This is not a fresh confirmatory test, a causal estimate, or a new training result.", "",
             "## Historical supervised checkpoint selection", "", "| Model | Selected step | Calibration loss | Best at first evaluation |", "|---|---:|---:|---|"]
    for name, row in result["learning_curves"].items():
        lines.append(f'| {name} | {row["selected_step"]} | {row["best_calibration_loss"]:.6f} | {row["best_at_first_evaluation"]} |')
    lines += ["", "## Inference", "", "| Policy | Eligible expected RM | Eligible NDCG | Recall (all) | Coverage |", "|---|---:|---:|---:|---:|"]
    for name, row in result["policies"].items():
        lines.append(f'| {name} | {row["eligible_exact_one_step_expected_reward"]:.6f} | {row["eligible_ndcg"]:.6f} | {row["recall"]:.6f} | {row["catalog_coverage"]:.4f} |')
    lines += ["", "Rerank weight is selected before test inference, on selection only. History sensitivity and append controls do not identify causal user responses.",
              "See results.json for paired intervals, control trajectories, timings and provenance; per_example.npz and evaluation-inputs.npz support independent reconstruction.",
              "Observed-context replay uses retained, potentially gapped contexts, not a consecutive user-transition simulator.", ""]
    lines += ["![Reward–relevance trade-off](tradeoff.svg)", ""]
    for name in result["learning_curves"]:
        lines += [f"![{name} calibration curve](learning-{name}.svg)", ""]
    for name, row in result["policies"].items():
        if "matched_rollouts" in row:
            lines += [f"![{name} matched rollout](rollout-{name}.svg)", ""]
    (Path(output) / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_reanalysis(data_dir, run_dir, output, *, batch_size=128, max_minutes=240,
                   rerank_weights=(0., .1, 1., 10.), diagnostics=True):
    """New inference, same historical weights/data; current source identity is explicit."""
    identity = verify_inputs(run_dir, data_dir)
    out = Path(output)
    if out.exists():
        raise FileExistsError("Reanalysis requires fresh output; never overwrite historical results")
    if batch_size <= 0 or not np.isfinite(max_minutes) or max_minutes <= 0:
        raise ValueError("Positive batch size and time budget required")
    if not rerank_weights or any(not np.isfinite(w) or w < 0 for w in rerank_weights):
        raise ValueError("Rerank weights must be finite and nonnegative")
    cfg = copy.deepcopy(read_json(Path(run_dir) / "provenance.json")["config"])
    cfg["evaluation"]["batch_size"] = batch_size
    cfg["evaluation"]["max_minutes"] = max_minutes
    exp.seed_all(cfg["seed"], cfg["train"]["threads"])
    device = exp.get_device(cfg)
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + max_minutes * 60
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data = load_prepared(data_dir)
    export_evaluation_inputs(data, out / "evaluation-inputs.npz")
    prov = {"historical_inputs": identity, "analysis": exp.provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device),
            "interpretation": "exploratory reuse; frozen historical checkpoints with current analysis code"}
    atomic_json(out / "provenance.json", prov)
    args = exp.model_args(data, cfg)
    rm = exp.restore(Path(run_dir) / "reward.pt", exp.RewardModel(*args), device)
    auditor = exp.restore(Path(run_dir) / "auditor.pt", exp.RewardModel(*args, mean_history=True), device)
    policy = exp.restore(Path(run_dir) / "sft.pt", exp.Policy(*args), device)
    selection, test = role(data, "selection"), role(data, "test")
    choices = []
    base_selection, _ = evaluate(policy, rm, auditor, selection, data, cfg, device, deadline=deadline)
    for weight in sorted(set(rerank_weights)):
        cfg["rl"]["rerank_weight"] = weight
        summary, _ = evaluate(policy, rm, auditor, selection, data, cfg, device, "rerank", deadline)
        choices.append({"weight": weight, "selection": summary})
    allowed = [c for c in choices if c["selection"]["eligible_ndcg"] >= cfg["rl"]["relevance_floor"] * base_selection["eligible_ndcg"]]
    chosen = max(allowed, key=lambda c: c["selection"]["eligible_exact_one_step_expected_reward"]) if allowed else None
    atomic_json(out / "selection.json", {"candidates": choices, "chosen": chosen,
                                         "fallback": "SFT / weight 0" if chosen is None else None,
                                         "rule": "selection-only eligible NDCG floor, then expected completion"})
    specs = [("sft", "sft.pt", "policy"), ("popularity", "sft.pt", "popularity"),
             ("rerank_historical", "sft.pt", "rerank"), ("rerank_tuned", "sft.pt", "rerank")]
    specs += [(name[:-3], name, "policy") for name in identity["checkpoint_hashes"] if name.startswith("rl_")]
    result = {"evaluation_status": "exploratory_reuse", "learning_curves": learning_curves(run_dir), "policies": {},
              "reward_model": exp.reward_report(rm, auditor, role(data, "fit", True), role(data, "test", True), cfg, device, data, deadline)}
    arrays = {}
    historical_selected = read_json(Path(run_dir) / "results.json").get("selected")
    for name, checkpoint, mode in specs:
        print(json.dumps({"phase": "reanalysis", "policy": name}), flush=True)
        exp.restore(Path(run_dir) / checkpoint, policy, device)
        cfg["rl"]["rerank_weight"] = (chosen["weight"] if chosen else 0.) if name == "rerank_tuned" else read_json(Path(run_dir) / "provenance.json")["config"]["rl"]["rerank_weight"]
        summary, raw = evaluate(policy, rm, auditor, test, data, cfg, device, mode, deadline)
        result["policies"][name] = summary
        arrays.update({f"{name}_{key}": value for key, value in raw.items()})
        if name == "sft":
            base = raw
        if name != "sft":
            mask = test["target"] >= 2
            summary["paired_vs_sft"] = {metric: paired_interval(base[metric][mask], raw[metric][mask], test["uid"][mask], cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"]) for metric in ["ndcg", "exact_one_step_expected_reward"]}
        if diagnostics and name in {"sft", historical_selected}:
            from .diagnostics import matched_rollout_controls
            controls, control_arrays = matched_rollout_controls(
                policy, rm, auditor, test, data["popularity"], batch_size=batch_size,
                k=cfg["evaluation"]["k"], temperature=cfg["rl"]["sampling_temperature"],
                device=device, seed=cfg["seed"] + 20000, deadline=deadline)
            summary["matched_rollouts"] = controls
            arrays.update(flatten_arrays(control_arrays, f"{name}_control_"))
    if diagnostics:
        from .diagnostics import reward_history_sensitivity
        sensitivity, sensitivity_arrays = reward_history_sensitivity(
            rm, role(data, "test", True), batch_size=batch_size, device=device,
            seed=cfg["seed"] + 30000, replicates=cfg["evaluation"]["bootstrap"], deadline=deadline)
        result["history_sensitivity"] = sensitivity
        arrays.update({f"history_{key}": value for key, value in sensitivity_arrays.items()})
    result["resources"] = {"elapsed_seconds": time.monotonic() - started,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    atomic_npz(out / "per_example.npz", arrays)
    atomic_json(out / "results.json", result)
    write_plots(out, result)
    write_report(out, result)
    files = [p for p in out.iterdir() if p.is_file()]
    atomic_json(out / "COMPLETE.json", {"status": "complete", "sha256": {p.name: sha256(p) for p in files}})
    return result


def bundle_reanalysis(imported, analysis, output):
    """Include data and weights, stream ZIP64, never a metadata-only handoff."""
    imported, analysis, output = Path(imported).resolve(), Path(analysis).resolve(), Path(output).resolve()
    if any(output.is_relative_to(root) or output.with_suffix(output.suffix + ".partial").is_relative_to(root)
           for root in [imported, analysis]):
        raise ValueError("Bundle output must be outside both input directories")
    complete = read_json(analysis / "COMPLETE.json")
    if complete.get("status") != "complete":
        raise ValueError("Analysis is incomplete")
    for name, digest in complete["sha256"].items():
        safe = _safe_name(name)
        if sha256(analysis / safe) != digest:
            raise ValueError(f"Analysis hash mismatch: {name}")
    imported_manifest = read_json(imported / "IMPORT_COMPLETE.json")
    verify_inputs(imported / imported_manifest["run"], imported / imported_manifest["data"])
    if output.exists():
        raise FileExistsError("Bundle already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("Partial bundle exists; inspect before retry")
    checks = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for prefix, root in [("historical", imported), ("analysis", analysis)]:
            for p in sorted(root.rglob("*")):
                if p.is_symlink():
                    raise ValueError("Symlinks are not bundle inputs")
                if p.is_file():
                    name = f"{prefix}/{p.relative_to(root).as_posix()}"
                    archive.write(p, name)
                    checks.append({"path": name, "sha256": sha256(p)})
        archive.writestr("SHA256SUMS.json", json.dumps(checks, indent=2))
    temporary.replace(output)
    return {"output": str(output), "files": len(checks), "bytes": output.stat().st_size}
