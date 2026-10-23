"""Bounded training jobs; selection and final test separated on disk."""
from __future__ import annotations
import base64
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_prepared, role, sha256
from .models import Policy, RewardModel, append_action, reinforce_loss
from .evaluate import evaluate, macro, paired_interval, sampling_diagnostic, rollout_diagnostics
from .config import validate_config
from .artifacts import COMPLETION_FILE, EVALUATION_FILES, complete_evaluation


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # RTX 50-series benefits from TF32 for matmul; this changes numerics, not the protocol.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Reproducible seeds, not a bitwise cross-device determinism promise.


def verify_source_identity(prov):
    package = Path(__file__).parent
    current = {p.name: sha256(p) for p in sorted(package.glob("*.py"))}
    recorded = prov.get("code_hashes", {})
    if current != recorded:
        changed = sorted(set(current) | set(recorded))
        changed = [name for name in changed if current.get(name) != recorded.get(name)]
        raise ValueError(f"Source code differs from training provenance: {changed}")


def get_device(cfg):
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # Actual tensor kernel preflight catches incompatible wheel/architecture.
        x = torch.randn(32, 32, device="cuda", requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        return torch.device("cuda")
    if not cfg["train"]["allow_cpu"]:
        raise RuntimeError("CUDA unavailable. CPU allowed only for explicit smoke configuration")
    return torch.device("cpu")


def model_args(data, cfg):
    return len(data["vocab"]) + 2, cfg["data"]["context_length"], cfg["model"]


def batch(subset, size, rng, device):
    idx = rng.integers(len(subset["target"]), size=size)
    return {key: torch.as_tensor(value[idx], device=device) for key, value in subset.items()}


def checkpoint(path, model):
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


def restore(path, model, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def provenance(root, cfg, data_dir, device):
    code = {}
    for p in sorted(Path(__file__).parent.glob("*.py")):
        code[p.name] = sha256(p)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root).decode()
    except subprocess.CalledProcessError:
        revision, dirty = None, "not a git checkout"
    return {"config": cfg, "prepared_sha256": sha256(Path(data_dir) / "prepared.npz"),
            "audit_sha256": sha256(Path(data_dir) / "audit.json"),
            "code_hashes": code,
            "code_bytes_base64": {p.name: base64.b64encode(p.read_bytes()).decode("ascii")
                                  for p in sorted(Path(__file__).parent.glob("*.py"))},
            "installed_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
            "git_revision": revision, "git_status": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "pyarrow"]},
            "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory if device.type == "cuda" else None}


def optimize_step(loss, model, optimizer, cfg):
    if not torch.isfinite(loss):
        raise RuntimeError("Nonfinite loss; aborting rather than skipping")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def timed_out(deadline):
    return time.monotonic() >= deadline


def progress_record(path, phase, step, total, started, **fields):
    elapsed = max(time.monotonic() - started, 1e-6)
    record = {"kind": "progress", "phase": phase, "step": step, "total_steps": total,
              "elapsed_seconds": elapsed, "steps_per_second": step / elapsed,
              "eta_seconds": max(total - step, 0) * elapsed / max(step, 1), **fields}
    if torch.cuda.is_available():
        record.update(gpu_allocated_mb=round(torch.cuda.memory_allocated() / 2**20, 1),
                      gpu_reserved_mb=round(torch.cuda.memory_reserved() / 2**20, 1))
    line = json.dumps(record)
    print(line, flush=True)
    if path is not None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def check_deadline(deadline, label):
    if deadline is not None and timed_out(deadline):
        raise TimeoutError(f"Deadline reached during {label}")


@torch.no_grad()
def validation_loss(model, subset, cfg, device, reward=False, deadline=None):
    model.eval()
    losses = []
    for start in range(0, len(subset["target"]), cfg["evaluation"]["batch_size"]):
        check_deadline(deadline, "validation")
        end = start + cfg["evaluation"]["batch_size"]
        ctx = torch.as_tensor(subset["context"][start:end], device=device)
        target = torch.as_tensor(subset["target"][start:end], device=device)
        if reward:
            y = torch.as_tensor(np.column_stack([subset["reward"][start:end], subset["seconds"][start:end]]), device=device)
            loss = ((model(ctx, target) - y) ** 2).mean(-1)
        else:
            loss = F.cross_entropy(model(ctx), target - 2, reduction="none")
        losses.extend(loss.cpu().tolist())
    return macro(np.array(losses), subset["uid"])[0]


def fit_supervised(model, fit, valid, steps, path, cfg, device, deadline, reward=False, progress_path=None, phase="supervised"):
    rng = np.random.default_rng(cfg["seed"] + (101 if reward else 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    best = float("inf")
    log = []
    start = time.monotonic()
    progress_every = cfg["train"].get("progress_every", cfg["train"]["eval_every"])
    for step in range(steps):
        if timed_out(deadline):
            raise TimeoutError("Job deadline reached; final evaluation intentionally not run")
        model.train()
        b = batch(fit, cfg["train"]["batch_size"], rng, device)
        if reward:
            y = torch.stack([b["reward"], b["seconds"]], -1)
            loss = F.mse_loss(model(b["context"], b["target"]), y)
        else:
            loss = F.cross_entropy(model(b["context"]), b["target"] - 2)
        norm = optimize_step(loss, model, optimizer, cfg)
        if (step + 1) % progress_every == 0 or step + 1 == steps:
            progress_record(progress_path, phase, step + 1, steps, start, loss=float(loss), grad_norm=norm)
        if (step + 1) % cfg["train"]["eval_every"] == 0 or step + 1 == steps:
            val = validation_loss(model, valid, cfg, device, reward, deadline)
            record = {"step": step + 1, "loss": float(loss), "calibration_loss": val,
                      "grad_norm": norm, "elapsed_seconds": time.monotonic() - start}
            print(json.dumps(record), flush=True)
            log.append(record)
            if val < best:
                best = val
                checkpoint(path, model)
    if not path.exists():
        raise RuntimeError("No checkpoint produced")
    restore(path, model, device)
    return log


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def train_rl(policy, reference, rm, fit, cfg, device, beta, horizon, reward_index, deadline, progress_path=None, phase="rl"):
    """Current-policy samples in deterministic append-item simulator, NOT users."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["rl_lr"])
    rng = np.random.default_rng(cfg["seed"] + 999)
    temperature = cfg["rl"]["sampling_temperature"]
    log = []
    start = time.monotonic()
    progress_every = cfg["train"].get("progress_every", cfg["train"]["eval_every"])
    for step in range(cfg["train"]["rl_steps"]):
        if timed_out(deadline):
            raise TimeoutError("RL deadline reached")
        # Dropout off for matching sampling/logprob/KL policy distributions.
        policy.eval()
        context = batch(fit, cfg["train"]["batch_size"], rng, device)["context"]
        logprobs, rewards, kls = [], [], []
        for _ in range(horizon):
            logits = policy(context) / temperature
            logp = logits.log_softmax(-1)
            p = logp.exp()
            with torch.no_grad():
                ref_logp = (reference(context) / temperature).log_softmax(-1)
                action = torch.multinomial(p.detach(), 1).squeeze(1)
                reward = rm(context, action + 2)[:, reward_index]
            logprobs.append(logp.gather(1, action[:, None]).squeeze(1))
            rewards.append(reward)
            kls.append((p * (logp - ref_logp)).sum(-1))
            context = append_action(context, action)
        loss = reinforce_loss(logprobs, rewards, kls, beta)
        optimize_step(loss, policy, optimizer, cfg)
        if (step + 1) % progress_every == 0 or step + 1 == cfg["train"]["rl_steps"]:
            progress_record(progress_path, phase, step + 1, cfg["train"]["rl_steps"], start,
                            loss=float(loss), proxy_reward=float(torch.stack(rewards).mean()),
                            exact_kl=float(torch.stack(kls).mean()), horizon=horizon)
        if (step + 1) % cfg["train"]["eval_every"] == 0 or step + 1 == cfg["train"]["rl_steps"]:
            record = {"step": step + 1, "loss": float(loss),
                      "proxy_reward": float(torch.stack(rewards).mean()),
                      "exact_kl": float(torch.stack(kls).mean())}
            log.append(record)
            print(json.dumps(record), flush=True)
    return log


def evaluation_decisions(h1_decision, h2_decision):
    """Keep numeric rule decisions separate from the reused-test interpretation."""
    return {"evaluation_status": "exploratory_reuse",
            "descriptive_decisions": {"H1": h1_decision, "H2": h2_decision},
            "verdicts": {"H1": f"exploratory reuse; {h1_decision}",
                         "H2": f"exploratory reuse; {h2_decision}",
                         "H3": "exploratory reuse; no confirmatory verdict"}}


def select_candidate(candidates, base_summary, cfg):
    """Select only from the declared eligible completion/horizon-1 population.

    All-target metrics may be reported descriptively, but cannot affect this
    confirmatory-style selection rule.
    """
    eligible = [c for c in candidates
                if c["reward"] == "completion" and c["horizon"] == 1
                and c["selection"]["eligible_ndcg"] >= cfg["rl"]["relevance_floor"] * base_summary["eligible_ndcg"]]
    return max(eligible, key=lambda c: c["selection"]["eligible_sample_reward"]) if eligible else None


def train(config, data_dir, run_dir):
    cfg = validate_config(config)
    seed_all(cfg["seed"], cfg["train"]["threads"])
    device = get_device(cfg)
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "provenance.json").exists():
        raise FileExistsError("Use a fresh run directory; no silent overwrite/resume")
    data = load_prepared(data_dir)
    audit = json.loads((Path(data_dir) / "audit.json").read_text())
    if audit.get("config") != cfg:
        raise ValueError("Prepared-data config mismatch. Reprepare for this configuration/seed")
    dump(run / "provenance.json", provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device))
    deadline = time.monotonic() + cfg["train"]["max_hours"] * 3600
    fit, calibration, selection = [role(data, r, eligible=(r != "selection")) for r in ["fit", "calibration", "selection"]]
    if any(not len(x["target"]) for x in [fit, calibration, selection]):
        raise ValueError("Empty eligible split")
    args = model_args(data, cfg)
    policy = Policy(*args).to(device)
    logs = {}
    progress_path = run / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    logs["sft"] = fit_supervised(policy, fit, calibration, cfg["train"]["sft_steps"], run / "sft.pt", cfg, device, deadline,
                                  progress_path=progress_path, phase="sft")
    rm = RewardModel(*args).to(device)
    logs["reward"] = fit_supervised(rm, fit, calibration, cfg["train"]["reward_steps"], run / "reward.pt", cfg, device, deadline, True,
                                    progress_path=progress_path, phase="reward")
    torch.manual_seed(cfg["seed"] + 12345)
    auditor = RewardModel(*args, mean_history=True).to(device)
    logs["auditor"] = fit_supervised(auditor, fit, calibration, cfg["train"]["reward_steps"], run / "auditor.pt", cfg, device, deadline, True,
                                      progress_path=progress_path, phase="auditor")
    reference = freeze(copy.deepcopy(policy))
    freeze(rm)
    freeze(auditor)
    base_summary, _ = evaluate(policy, rm, auditor, selection, data, cfg, device, deadline=deadline)
    candidates = []
    for reward_index, reward_name in enumerate(["completion", "seconds"]):
        if reward_name not in cfg["rl"]["rewards"]:
            continue
        for horizon in cfg["rl"]["horizons"]:
            for beta in cfg["rl"]["betas"]:
                name = f"rl_{reward_name}_h{horizon}_b{beta:g}"
                candidate = restore(run / "sft.pt", Policy(*args), device)
                # Equal initial stochastic conditions; each candidate has fixed training budget.
                torch.manual_seed(cfg["seed"] + 999)
                logs[name] = train_rl(candidate, reference, rm, fit, cfg, device, beta, horizon, reward_index, deadline,
                                       progress_path=progress_path, phase=name)
                checkpoint(run / f"{name}.pt", candidate)
                summary, _ = evaluate(candidate, rm, auditor, selection, data, cfg, device, deadline=deadline)
                candidates.append({"name": name, "reward": reward_name, "horizon": horizon,
                                   "beta": beta, "selection": summary})
                dump(run / "training.json", logs)
                dump(run / "candidates.json", candidates)
                del candidate
    # H2 is completion-only, horizon 1. Other grid members are exploratory.
    chosen = select_candidate(candidates, base_summary, cfg)
    dump(run / "selection.json", {"baseline": base_summary, "chosen": chosen,
                                  "rule": "horizon1 completion; eligible-NDCG relevance floor then max eligible sampled completion proxy",
                                  "all_candidates_failed_floor": chosen is None})
    dump(run / "training.json", logs)
    chosen = json.loads((run / "selection.json").read_text()).get("chosen")
    manifest = {"status": "training complete; test unexamined",
                "checkpoint_hashes": {p.name: sha256(p) for p in sorted(run.glob("*.pt"))},
                "selection_sha256": sha256(run / "selection.json"),
                "candidates_sha256": sha256(run / "candidates.json"),
                "training_sha256": sha256(run / "training.json"),
                "provenance_sha256": sha256(run / "provenance.json"),
                "prepared_audit_sha256": sha256(Path(data_dir) / "audit.json"),
                "selected_checkpoint": (f'{chosen["name"]}.pt' if chosen else None)}
    if chosen and manifest["selected_checkpoint"] not in manifest["checkpoint_hashes"]:
        raise RuntimeError("Selected checkpoint is missing from the training manifest")
    dump(run / "READY_FOR_TEST.json", manifest)


@torch.no_grad()
def reward_report(rm, auditor, fit, test, cfg, device, data=None):
    predictions = []
    audit_predictions = []
    for start in range(0, len(test["target"]), cfg["evaluation"]["batch_size"]):
        end = start + cfg["evaluation"]["batch_size"]
        ctx = torch.as_tensor(test["context"][start:end], device=device)
        items = torch.as_tensor(test["target"][start:end], device=device)
        predictions.append(rm(ctx, items).cpu().numpy())
        audit_predictions.append(auditor(ctx, items).cpu().numpy())
    pred = np.concatenate(predictions)
    apred = np.concatenate(audit_predictions)
    size = int(max(fit["target"].max(), test["target"].max())) + 1
    counts = np.bincount(fit["target"], minlength=size)
    heads = {}
    for head, (name, label_key) in enumerate([("completion", "reward"), ("seconds", "seconds")]):
        mean = float(fit[label_key].mean())
        sums = np.bincount(fit["target"], weights=fit[label_key], minlength=size)
        item_mean = np.divide(sums, counts, out=np.full(size, mean), where=counts > 0)
        y = test[label_key]
        baseline_error = np.abs(item_mean[test["target"]] - y)
        error = np.abs(pred[:, head] - y)
        report = {"examples": len(y), "users": len(np.unique(test["uid"])),
                  "units": "bounded fraction" if head == 0 else "played-seconds proxy / seconds_cap",
                  "mae": macro(error, test["uid"])[0],
                  "rmse": macro((pred[:, head] - y)**2, test["uid"])[0]**.5,
                  "auditor_mae": macro(np.abs(apred[:, head] - y), test["uid"])[0],
                  "constant_mae": macro(np.abs(mean-y), test["uid"])[0],
                  "item_mean_mae": macro(baseline_error, test["uid"])[0],
                  "paired_mae": paired_interval(baseline_error, error, test["uid"], cfg["evaluation"]["bootstrap"],
                                               cfg["evaluation"]["bootstrap_seed"]),
                  "calibration": [], "slices": {}}
        for lo in np.arange(0, 1, .1):
            mask = (pred[:, head] >= lo) & (pred[:, head] < lo + .1 if lo < .9 else pred[:, head] <= 1)
            if mask.any():
                report["calibration"].append({"lower": float(lo), "n": int(mask.sum()),
                                              "predicted": float(pred[mask, head].mean()), "observed": float(y[mask].mean())})
        for label, mask in [("organic", test["organic"] == 1), ("recommended", test["organic"] == 0)]:
            if mask.any():
                report["slices"][label] = {"n": int(mask.sum()), "mae": macro(error[mask], test["uid"][mask])[0]}
        if data is not None:
            duration = data["durations"][test["target"] - 2]
            for label, mask in [("duration_lt_180s", duration < 180),
                                ("duration_180_to_300s", (duration >= 180) & (duration < 300)),
                                ("duration_ge_300s", duration >= 300)]:
                if mask.any():
                    report["slices"][label] = {"n": int(mask.sum()),
                                              "mae": macro(error[mask], test["uid"][mask])[0],
                                              "item_mean_mae": macro(baseline_error[mask], test["uid"][mask])[0]}
        if head == 1:
            report["mae_seconds_proxy"] = report["mae"] * cfg["rl"]["seconds_cap"]
        heads[name] = report
    # Preserve the original completion fields for consumers of historical results.
    return {**heads["completion"], "heads": heads}


def final_evaluation(data_dir, run_dir):
    run = Path(run_dir)
    if any((run / name).exists() for name in (*EVALUATION_FILES, COMPLETION_FILE)):
        raise FileExistsError("Evaluation artifacts already exist; do not repeatedly inspect test")
    ready = json.loads((run / "READY_FOR_TEST.json").read_text())
    for name, digest in ready["checkpoint_hashes"].items():
        if sha256(run / name) != digest:
            raise ValueError("Checkpoint changed after selection")
    for name in ["selection.json", "candidates.json", "training.json", "provenance.json"]:
        key = name.replace(".json", "_sha256")
        if sha256(run / name) != ready[key]:
            raise ValueError(f"{name} changed after selection")
    if sha256(Path(data_dir) / "audit.json") != ready["prepared_audit_sha256"]:
        raise ValueError("Prepared audit changed after selection")
    selected_manifest = json.loads((run / "selection.json").read_text()).get("chosen")
    selected_checkpoint = f'{selected_manifest["name"]}.pt' if selected_manifest else None
    if selected_checkpoint != ready["selected_checkpoint"]:
        raise ValueError("Selected checkpoint does not match the immutable test manifest")
    prov = json.loads((run / "provenance.json").read_text())
    cfg = validate_config(prov["config"])
    verify_source_identity(prov)
    if sha256(Path(data_dir) / "prepared.npz") != prov["prepared_sha256"]:
        raise ValueError("Prepared data changed")
    seed_all(cfg["seed"], cfg["train"]["threads"])
    device = get_device(cfg)
    evaluation_provenance = provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device)
    data = load_prepared(data_dir)
    args = model_args(data, cfg)
    rm = restore(run / "reward.pt", RewardModel(*args), device)
    auditor = restore(run / "auditor.pt", RewardModel(*args, mean_history=True), device)
    policy = restore(run / "sft.pt", Policy(*args), device)
    fit = role(data, "fit", True)
    test = role(data, "test")
    evaluation_deadline = time.monotonic() + cfg["evaluation"]["max_minutes"] * 60
    results = {"interpretation": "restricted-catalog observed-next-listen and model-based diagnostics; not user impact; final evaluation is time-bounded by the configured job budget",
               "reward_model": reward_report(rm, auditor, fit, role(data, "test", True), cfg, device, data), "policies": {}}
    arrays = {}
    base = None
    selected = json.loads((run / "selection.json").read_text())["chosen"]
    specs = [("popularity", "popularity", "sft"), ("sft", "policy", "sft"), ("rerank", "rerank", "sft")]
    candidates = json.loads((run / "candidates.json").read_text())
    specs += [(c["name"], "policy", c["name"]) for c in candidates]
    for name, mode, source in specs:
        restore(run / f"{source}.pt", policy, device)
        summary, raw = evaluate(policy, rm, auditor, test, data, cfg, device, mode, deadline=evaluation_deadline)
        results["policies"][name] = summary
        for key, value in raw.items():
            arrays[f"{name}_{key}"] = value
        if name == "sft":
            base = raw
        if name == "sft" or name.startswith("rl_"):
            results["policies"][name]["hypothetical_rollout"] = rollout_diagnostics(policy, rm, auditor, test, cfg, device,
                                                                                       deadline=evaluation_deadline)
        if name.startswith("rl_") or name == "rerank":
            eligible_mask = test["target"] >= 2
            results["policies"][name]["paired_vs_sft"] = {
                **{metric: paired_interval(base[metric][eligible_mask], raw[metric][eligible_mask],
                                           test["uid"][eligible_mask], cfg["evaluation"]["bootstrap"],
                                           cfg["evaluation"]["bootstrap_seed"])
                   for metric in ["sample_reward", "sample_audit"]},
                "eligible_ndcg": paired_interval(base["ndcg"][eligible_mask], raw["ndcg"][eligible_mask],
                                                   test["uid"][eligible_mask], cfg["evaluation"]["bootstrap"],
                                                   cfg["evaluation"]["bootstrap_seed"]),
                "all_target_ndcg": paired_interval(base["ndcg"], raw["ndcg"], test["uid"],
                                                     cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"]),
                "all_target_recall": paired_interval(base["recall"], raw["recall"], test["uid"],
                                                       cfg["evaluation"]["bootstrap"], cfg["evaluation"]["bootstrap_seed"]),
                "sampling_diagnostic": {metric: sampling_diagnostic(base[metric + "_draws"][eligible_mask],
                                                                       raw[metric + "_draws"][eligible_mask],
                                                                       test["uid"][eligible_mask])
                                        for metric in ["sample_reward", "sample_audit"]}}
    h1 = results["reward_model"]["paired_mae"]
    h1_supported = (h1["relative_difference"] is not None and h1["relative_difference"] <= -.02
                    and h1["relative_ci95"] is not None and h1["relative_ci95"][1] < 0
                    and h1["adequate_users"])
    h1_decision = "supported" if h1_supported else "not supported" if h1["adequate_users"] else "inconclusive"
    h2_decision = "not supported"
    if selected:
        comparison = results["policies"][selected["name"]]["paired_vs_sft"]
        gain, relevance = comparison["sample_reward"], comparison["eligible_ndcg"]
        if gain["adequate_users"] and relevance["relative_ci95"] is not None:
            ok = gain["relative_ci95"] is not None and gain["relative_ci95"][0] >= .05 and relevance["relative_ci95"][0] > -.02
            h2_decision = "supported (proxy-only)" if ok else "not supported"
        else:
            h2_decision = "inconclusive"
    # This test interval is reused for the amended protocol: numbers remain
    # descriptive, but cannot be emitted as unqualified confirmatory verdicts.
    results.update(evaluation_decisions(h1_decision, h2_decision))
    results["selected"] = selected["name"] if selected else None
    results["limitations"] = ["No exposure probabilities, user simulator validation or causal identification",
        "Independent auditor shares observational bias and is only a disagreement diagnostic",
        "User bootstrap is conditional on trained seed; multi-seed robustness remains separate",
        "Uniform event subsampling yields a sampled-context user-macro estimand; inspect audit for coverage",
        "The pilot uses a declared 70/10/10/10 fit/calibration/selection/test temporal split; this supersedes any 80/10/10 prose in earlier planning.",
        "All grid test results except preselected H2 are exploratory; no post-test selection permitted"]
    complete_evaluation(run, results, arrays, evaluation_provenance)
    return results
