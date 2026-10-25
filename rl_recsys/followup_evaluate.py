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


def evaluate_followup(data_dir, run_dir, followup, output, batch_size=128, max_minutes=240, *,
                      source_mode="current"):
    root, out = Path(followup), Path(output)
    if out.exists():
        raise FileExistsError("Use a fresh follow-up evaluation output")
    if batch_size <= 0 or not np.isfinite(max_minutes) or max_minutes <= 0:
        raise ValueError("Positive batch/time budget required")
    if source_mode not in {"current", "historical_reanalysis"}:
        raise ValueError("source_mode must be current or historical_reanalysis")
    # Authenticate the code that trained the policies, not the historical parent
    # (which may deliberately have been imported under historical_reanalysis).
    prov = read_json(root / "provenance.json")
    if source_mode == "current":
        exp.verify_source_identity(prov)
    identity = verify_inputs(run_dir, data_dir, training_only=True)
    complete = read_json(root / "FOLLOWUP_COMPLETE.json")
    if complete.get("status") != "complete" or sha256(root / "followup_manifest.json") != complete.get("manifest_sha256"):
        raise ValueError("Follow-up completion manifest mismatch")
    manifest = read_json(root / "followup_manifest.json")
    prov = read_json(root / "provenance.json")
    if not manifest["status"].startswith("follow-up complete"):
        raise ValueError("Follow-up training is not complete")
    if (complete.get("provenance_sha256", manifest["provenance_sha256"]) != manifest["provenance_sha256"]
            or sha256(root / "provenance.json") != manifest["provenance_sha256"]):
        raise ValueError("Follow-up provenance mismatch")
    if identity["prepared_sha256"] != prov["prepared_sha256"] or identity["historical_provenance_sha256"] != prov["historical_provenance_sha256"]:
        raise ValueError("Follow-up historical input mismatch")
    if set(manifest["seeds"]) != {str(s) for s in prov["seeds"]}:
        raise ValueError("Follow-up seed set mismatch")
    if prov.get("historical_sft_sha256") and prov["historical_sft_sha256"] != sha256(Path(run_dir) / "sft.pt"):
        raise ValueError("Follow-up historical SFT mismatch")
    if prov.get("experiment") == "A":
        for checkpoint in ["reward.pt", "item_only.pt"]:
            if len({row["checkpoint_hashes"].get(checkpoint) for row in manifest["seeds"].values()}) != 1:
                raise ValueError("Experiment A requires one common frozen RM across optimization seeds")
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
            final = arm.get("final_checkpoint")
            if final:
                if "/" in _safe_name(final) or not final.endswith(".pt"):
                    raise ValueError("Invalid final checkpoint path")
                required_checkpoints.add(final)
        if not required_checkpoints <= set(row["checkpoint_hashes"]):
            raise ValueError("Required follow-up checkpoint is not hash-bound")
        if not {"config.json", "reward_audit.json", "selection.json"} <= set(row["artifact_hashes"]):
            raise ValueError("Required follow-up metadata is not hash-bound")
        for name, digest in {**row["checkpoint_hashes"], **row["artifact_hashes"]}.items():
            safe = _safe_name(name)
            if "/" in safe or sha256(root / f"seed_{seed}" / safe) != digest:
                raise ValueError(f"Follow-up artifact mismatch: seed {seed}/{name}")
        selection = read_json(root / f"seed_{seed}" / "selection.json")
        if selection.get("rerank") != row.get("rerank"):
            raise ValueError("Follow-up rerank metadata mismatch")
        if row.get("rerank"):
            if "rerank_curve.json" not in row["artifact_hashes"]:
                raise ValueError("Rerank curve is not hash-bound")
            if read_json(root / f"seed_{seed}" / "rerank_curve.json") != row["rerank"]:
                raise ValueError("Rerank curve mismatch")
            from .experiment import select_candidate
            rerank = row["rerank"]
            if rerank["selected"] != select_candidate(rerank["candidates"], rerank["selection_baseline"], prov["config"]):
                raise ValueError("Rerank selection mismatch")
        if selection.get("policies") != row["policies"]:
            raise ValueError("Follow-up selection metadata mismatch")
        gate = read_json(root / f"seed_{seed}" / "reward_audit.json")["gate"]
        if row["policies"] and not gate["heads"]["completion"]["enabled"]:
            non_sft = set(row["policies"]) - {"continued_sft"}
            if non_sft or row.get("rerank"):
                raise ValueError("Only continued-SFT may exist after a failed reward gate")
    cfg = prov["config"]
    cfg["evaluation"]["batch_size"] = batch_size
    device = exp.get_device(cfg)
    exp.seed_all(cfg["seed"], cfg["train"]["threads"])
    data = load_prepared(data_dir)
    test = role(data, "test")
    mask = test["target"] >= 2
    if not mask.any():
        raise ValueError("Empty eligible test role")
    args = exp.model_args(data, cfg)
    out.mkdir(parents=True)
    export_evaluation_inputs(data, out / "evaluation-inputs.npz")
    atomic_json(out / "provenance.json", {"analysis": exp.provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device),
                 "source_mode": source_mode,
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
        specs = [("sft", Path(run_dir) / "sft.pt", "policy")]
        specs += [(name, directory / arm["selected_checkpoint"], "policy") for name, arm in row["policies"].items() if arm["selected_checkpoint"]]
        rerank = row.get("rerank")
        if rerank and rerank["selected"] is not None:
            seed_cfg["rl"]["rerank_weight"] = rerank["selected"]["weight"]
            specs.append(("rerank_tuned", Path(run_dir) / "sft.pt", "rerank"))
        summaries["rerank_selection"] = rerank
        summaries["not_selected"] = [name for name, arm in row["policies"].items() if not arm["selected_checkpoint"]]
        if rerank and rerank["selected"] is None:
            summaries["not_selected"].append("rerank_tuned")
        summaries["policies"] = {}
        raw_policies = {}
        for name, checkpoint, mode in specs:
            exp.restore(checkpoint, policy, device)
            inference_started = time.monotonic()
            summary, raw = evaluate(policy, rm, auditor, test, data, seed_cfg, device, mode=mode, deadline=deadline)
            summary["inference_seconds"] = time.monotonic() - inference_started
            raw["target"] = test["target"]
            raw["eligible"] = mask
            # Stochastic single-action relevance is NOT deterministic top-K relevance.
            raw["sample_action_relevance"] = (raw["sample_actions"] == test["target"][:, None]).mean(1)
            summary["populations"] = {}
            for population, population_mask in [("eligible", mask), ("all_target", np.ones(len(mask), dtype=bool))]:
                summary["populations"][population] = {
                    "examples": int(population_mask.sum()),
                    "users": len(np.unique(test["uid"][population_mask])),
                    **{metric: exp.macro(raw[metric][population_mask], test["uid"][population_mask])[0]
                       for metric in ["ndcg", "recall", "exact_one_step_expected_reward", "sample_action_relevance", "top_reward"]}}
            raw_policies[name] = raw
            # Score EVERY policy under BOTH prespecified rewards, not only its
            # own objective. Repeat utility is a diagnostic hypothesis in music.
            sampled_repeat = (raw["sample_actions"][:, :, None] == test["context"][:, None, :]).any(-1).mean(1)
            penalty = prov["repeat_penalty"]
            raw["sample_repeat"] = sampled_repeat
            raw["sample_adjusted_reward"] = raw["sample_reward"] - penalty * sampled_repeat
            summary["sample_repeat"] = exp.macro(sampled_repeat, test["uid"])[0]
            summary["sample_completion_minus_repeat"] = exp.macro(raw["sample_adjusted_reward"], test["uid"])[0]
            summary["repeat_penalty"] = penalty
            summaries["policies"][name] = summary
            arrays.update({f"seed_{seed}_{name}_{key}": value for key, value in raw.items()})
            if name == "sft":
                base = raw
            else:
                summary["paired_vs_sft"] = {key: paired_interval(base[key][mask], raw[key][mask], test["uid"][mask], seed_cfg["evaluation"]["bootstrap"], seed_cfg["evaluation"]["bootstrap_seed"]) for key in ["ndcg", "exact_one_step_expected_reward"]}
        # Paired RAW eligible arrays, same users/contexts and same frozen RM.
        summaries["paired_eligible"] = {}
        for name, raw in raw_policies.items():
            if name == "sft":
                continue
            summaries["paired_eligible"][name] = {
                comparator: {metric: paired_interval(other[metric][mask], raw[metric][mask], test["uid"][mask],
                             seed_cfg["evaluation"]["bootstrap"], seed_cfg["evaluation"]["bootstrap_seed"])
                             for metric in ["ndcg", "recall", "exact_one_step_expected_reward", "sample_action_relevance", "top_reward"]}
                for comparator, other in raw_policies.items()
                if comparator in ["sft", "completion_h1", "continued_sft", "rerank_tuned"] and comparator != name}
        results["seeds"][seed] = summaries
        del rm, auditor, policy
    # Retain seed effects without pooling users. A has one common RM; legacy
    # follow-ups may instead have seed-specific RMs.
    aggregation = {}
    for name in sorted({"continued_sft", "completion_h1", "completion_minus_repeat_h1"} |
                       {name for row in results["seeds"].values() for name in row["policies"] if name != "sft"}):
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
    results["seed_roles"] = prov.get("seed_roles", {})
    results["experiment"] = prov.get("experiment", "legacy conditional controls")
    results["limitations"] = ["No post-test policy/seed selection", "A uses one frozen predictive RM, not ground-truth or causal reward; legacy seed-specific RMs are not score-comparable", "Gate failures and selection-floor failures retained; missing arms are not favorable outcomes", "Conditional seed dispersion is not a confidence interval or full-pipeline seed robustness"]
    results["resources"] = {"elapsed_seconds": time.monotonic() - started,
                            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    atomic_npz(out / "per_example.npz", arrays)
    atomic_json(out / "results.json", results)
    atomic_json(out / "COMPLETE.json", {"status": "complete", "sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}})
    return results
