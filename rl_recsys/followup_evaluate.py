"""Explicit exploratory test inference after every follow-up seed is frozen.

Do not select seeds/arms using these test results. Seed dispersion is conditional
on the same historical SFT/data, not full-pipeline uncertainty.
"""
from pathlib import Path
import time

import numpy as np
import torch

from . import experiment as exp
from .artifacts import atomic_json, atomic_npz
from .data import load_prepared, role, sha256
from .evaluate import evaluate, paired_interval
from .followup import ItemOnlyRewardModel
from .reanalysis import read_json, verify_inputs, export_evaluation_inputs, _safe_name


def evaluate_followup(data_dir, run_dir, followup, output, batch_size=128, max_minutes=240):
    root, out = Path(followup), Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh follow-up evaluation output")
    if batch_size <= 0 or not np.isfinite(max_minutes) or max_minutes <= 0:
        raise ValueError("Positive batch/time budget required")
    identity = verify_inputs(run_dir, data_dir)
    complete = read_json(root / "FOLLOWUP_COMPLETE.json")
    if complete.get("status") != "complete" or sha256(root / "followup_manifest.json") != complete.get("manifest_sha256"):
        raise ValueError("Follow-up completion manifest mismatch")
    manifest = read_json(root / "followup_manifest.json")
    prov = read_json(root / "provenance.json")
    if not manifest["status"].startswith("follow-up complete"):
        raise ValueError("Follow-up training is not complete")
    if sha256(root / "provenance.json") != manifest["provenance_sha256"]:
        raise ValueError("Follow-up provenance mismatch")
    if identity["prepared_sha256"] != prov["prepared_sha256"] or identity["historical_provenance_sha256"] != prov["historical_provenance_sha256"]:
        raise ValueError("Follow-up historical input mismatch")
    if set(manifest["seeds"]) != {str(s) for s in prov["seeds"]}:
        raise ValueError("Follow-up seed set mismatch")
    # Validate ALL seeds before looking at any test metrics.
    for seed, row in manifest["seeds"].items():
        if not str(seed).isdigit():
            raise ValueError("Invalid seed directory")
        required_checkpoints = {"reward.pt", "item_only.pt"}
        for arm in row["policies"].values():
            selected = arm["selected_checkpoint"]
            if selected:
                if "/" in _safe_name(selected) or not selected.endswith(".pt"):
                    raise ValueError("Invalid selected checkpoint path")
                required_checkpoints.add(selected)
        if not required_checkpoints <= set(row["checkpoint_hashes"]):
            raise ValueError("Required follow-up checkpoint is not hash-bound")
        if not {"config.json", "reward_audit.json", "selection.json"} <= set(row["artifact_hashes"]):
            raise ValueError("Required follow-up metadata is not hash-bound")
        for name, digest in {**row["checkpoint_hashes"], **row["artifact_hashes"]}.items():
            safe = _safe_name(name)
            if "/" in safe or sha256(root / f"seed_{seed}" / safe) != digest:
                raise ValueError(f"Follow-up artifact mismatch: seed {seed}/{name}")
        selection = read_json(root / f"seed_{seed}" / "selection.json")
        if selection.get("policies") != row["policies"]:
            raise ValueError("Follow-up selection metadata mismatch")
        gate = read_json(root / f"seed_{seed}" / "reward_audit.json")["gate"]
        if row["policies"] and not gate["heads"]["completion"]["enabled"]:
            raise ValueError("Policies exist for a failed reward gate")
    cfg = prov["config"]
    cfg["evaluation"]["batch_size"] = batch_size
    device = exp.get_device(cfg)
    exp.seed_all(cfg["seed"], cfg["train"]["threads"])
    data = load_prepared(data_dir)
    test = role(data, "test")
    mask = test["target"] >= 2
    args = exp.model_args(data, cfg)
    out.mkdir(parents=True)
    export_evaluation_inputs(data, out / "evaluation-inputs.npz")
    atomic_json(out / "provenance.json", {"analysis": exp.provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device),
                 "training_manifest_sha256": sha256(root / "followup_manifest.json"), "historical_inputs": identity,
                 "interpretation": "exploratory test reuse; conditional control seeds, not full SFT/data retrains"})
    started = time.monotonic()
    deadline = started + max_minutes * 60
    arrays, results = {}, {"evaluation_status": "exploratory_reuse", "seeds": {}}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for seed, row in manifest["seeds"].items():
        print(f"Follow-up inference seed {seed}", flush=True)
        directory = root / f"seed_{seed}"
        seed_cfg = read_json(directory / "config.json")
        seed_cfg["evaluation"]["batch_size"] = batch_size
        exp.seed_all(int(seed), cfg["train"]["threads"])
        rm = exp.restore(directory / "reward.pt", exp.RewardModel(*args), device)
        auditor = exp.restore(directory / "item_only.pt", ItemOnlyRewardModel(*args), device)
        policy = exp.restore(Path(run_dir) / "sft.pt", exp.Policy(*args), device)
        summaries = {"reward_model": exp.reward_report(rm, auditor, role(data, "fit", True), role(data, "test", True), seed_cfg, device, data, deadline)}
        summaries["gate"] = read_json(directory / "reward_audit.json")["gate"]
        specs = [("sft", Path(run_dir) / "sft.pt")]
        specs += [(name, directory / arm["selected_checkpoint"]) for name, arm in row["policies"].items() if arm["selected_checkpoint"]]
        summaries["not_selected"] = [name for name, arm in row["policies"].items() if not arm["selected_checkpoint"]]
        summaries["policies"] = {}
        for name, checkpoint in specs:
            exp.restore(checkpoint, policy, device)
            summary, raw = evaluate(policy, rm, auditor, test, data, seed_cfg, device, deadline=deadline)
            summaries["policies"][name] = summary
            arrays.update({f"seed_{seed}_{name}_{key}": value for key, value in raw.items()})
            if name == "sft":
                base = raw
            else:
                summary["paired_vs_sft"] = {key: paired_interval(base[key][mask], raw[key][mask], test["uid"][mask], seed_cfg["evaluation"]["bootstrap"], seed_cfg["evaluation"]["bootstrap_seed"]) for key in ["ndcg", "exact_one_step_expected_reward"]}
        results["seeds"][seed] = summaries
        del rm, auditor, policy
    # Raw reward scales differ between seed-specific RMs: aggregate within-seed
    # relative effects only, retain per-seed baselines, and never pool users.
    aggregation = {}
    for name in ["continued_sft", "completion_h1", "completion_minus_repeat_h1"]:
        effects = {metric: [] for metric in ["ndcg", "exact_one_step_expected_reward"]}
        for seed, row in results["seeds"].items():
            if name in row["policies"]:
                for metric in effects:
                    effects[metric].append({"seed": int(seed), "relative_difference": row["policies"][name]["paired_vs_sft"][metric]["relative_difference"]})
        aggregation[name] = {}
        for metric, values in effects.items():
            finite = [r["relative_difference"] for r in values if r["relative_difference"] is not None]
            aggregation[name][metric] = {"per_seed": values, "available_seeds": len(finite),
                "mean_relative_difference": float(np.mean(finite)) if finite else None,
                "seed_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else None}
    results["conditional_seed_effects"] = aggregation
    results["limitations"] = ["No post-test policy/seed selection", "Different seed-specific learned RMs are not common ground-truth rewards", "Gate failures and selection-floor failures retained; missing arms are not favorable outcomes", "Conditional seed dispersion is not a confidence interval or full-pipeline seed robustness"]
    results["resources"] = {"elapsed_seconds": time.monotonic() - started,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    atomic_npz(out / "per_example.npz", arrays)
    atomic_json(out / "results.json", results)
    atomic_json(out / "COMPLETE.json", {"status": "complete", "sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    return results
