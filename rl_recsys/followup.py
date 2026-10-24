"""Opt-in conditional training controls on a fixed historical prepared sample.

Run explicitly with ``python -m rl_recsys.followup --help``. No test role is
accessed here. These seeds do NOT retrain the end-to-end data/SFT baseline.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import experiment as exp
from .data import load_prepared, role, sha256
from .models import Policy, RewardModel, reinforce_loss


LIMITATIONS = [
    "Conditional training-control seeds on one fixed imported data sample and historical SFT; not full end-to-end baseline retrains.",
    "Historical selection/test may already have been inspected; no independent fresh test or confirmatory user-impact claim.",
    "Late calibration is independent of NEW reward-model checkpoint selection only; it is an observational gate, not causal validation.",
    "Selection is reused for policy checkpoint/model selection; test is not evaluated by this command.",
    "Item-only RM is a learned observational comparator, not an independent causal auditor.",
]


class ItemOnlyRewardModel(nn.Module):
    """Learned item embedding + MLP; ignores context, predicts two bounded heads."""

    def __init__(self, vocab_size, length, model_cfg):
        super().__init__()
        hidden = model_cfg["hidden"]
        self.item = nn.Embedding(vocab_size, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def history(self, context):
        # Preserve a nonempty history dimension for the evaluator's expansions.
        return self.item.weight.new_zeros((len(context), 1))

    def from_history(self, hidden, items):
        return self.head(self.item(items)).sigmoid()

    def forward(self, context, items):
        return self.from_history(self.history(context), items)

    @torch.no_grad()
    def catalog(self, context, chunk_size=1024):
        ids = torch.arange(2, self.item.num_embeddings, device=context.device)
        return self.from_history(None, ids)[None].expand(len(context), -1, -1)


def split_calibration(calibration):
    """Split at the median UNIQUE timestamp (median and later go to audit).

    No event-count median: uneven timestamp frequencies must not move ties
    across the boundary. Fail closed if two chronological periods cannot exist.
    """
    timestamps = np.asarray(calibration["timestamp"])
    if timestamps.ndim != 1 or not timestamps.size or not np.isfinite(timestamps).all():
        raise ValueError("Calibration timestamps must be nonempty and finite")
    unique = np.unique(timestamps)
    if len(unique) < 2:
        raise ValueError("Calibration needs at least two unique timestamps")
    boundary = unique[len(unique) // 2]
    early_mask = timestamps < boundary
    late_mask = ~early_mask
    early = {key: value[early_mask] for key, value in calibration.items()}
    late = {key: value[late_mask] for key, value in calibration.items()}
    assert not np.any(early_mask & late_mask)
    assert early["timestamp"].max() < late["timestamp"].min()
    return early, late


class UserEventSampler:
    """Uniform users, then uniform retained fit events within each sampled user."""

    def __init__(self, fit):
        self.fit = fit
        users, inverse = np.unique(fit["uid"], return_inverse=True)
        if not len(users):
            raise ValueError("Empty fit users")
        order = np.argsort(inverse, kind="stable")
        self.groups = np.split(order, np.cumsum(np.bincount(inverse))[:-1])

    def batch(self, size, rng, device):
        groups = rng.integers(len(self.groups), size=size)
        indices = np.array([self.groups[g][rng.integers(len(self.groups[g]))] for g in groups])
        return {key: torch.as_tensor(value[indices], device=device) for key, value in self.fit.items()}


def adjusted_completion(completion, context, items, repeat_penalty):
    repeat = context.eq(items[:, None]).any(-1).to(completion.dtype)
    return completion - repeat_penalty * repeat, repeat


def train_policy(policy, reference, rm, auditor, fit, selection, data, cfg,
                 device, output, name, steps, deadline, *, supervised=False,
                 repeat_penalty=0.0):
    """Local H1 REINFORCE (or continued SFT); select on selection, never audit.

    All arms execute exactly ``steps`` optimizer updates, even if the selected
    checkpoint is earlier. Final and selected weights are persisted separately.
    Selection always maximizes unpenalized completion under the same relevance
    floor; repeat and adjusted reward are separate descriptive metrics.
    """
    output = Path(output)
    exp.check_deadline(deadline, name)
    exp.freeze(reference)
    exp.freeze(rm)
    exp.freeze(auditor)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg["train"]["rl_lr"])
    rng = np.random.default_rng(cfg["seed"] + 999)
    sampler = UserEventSampler(fit)
    logs, candidates = [], []
    baseline, _ = exp.evaluate(reference, rm, auditor, selection, data, cfg, device, deadline=deadline)
    best = None
    started = time.monotonic()
    for step in range(1, steps + 1):
        exp.check_deadline(deadline, name)
        # Match user weighting across continued-SFT and both RL controls.
        b = sampler.batch(cfg["train"]["batch_size"], rng, device)
        if supervised:
            policy.train()
            loss = F.cross_entropy(policy(b["context"]), b["target"] - 2)
            fields = {}
        else:
            # Sampling, score function, and exact KL use the same dropout-free distribution.
            policy.eval()
            logp = (policy(b["context"]) / cfg["rl"]["sampling_temperature"]).log_softmax(-1)
            p = logp.exp()
            with torch.no_grad():
                ref_logp = (reference(b["context"]) / cfg["rl"]["sampling_temperature"]).log_softmax(-1)
                action = torch.multinomial(p.detach(), 1).squeeze(1)
                completion = rm(b["context"], action + 2)[:, 0]
                reward, repeat = adjusted_completion(completion, b["context"], action + 2, repeat_penalty)
            kl = (p * (logp - ref_logp)).sum(-1)
            loss = reinforce_loss([logp.gather(1, action[:, None]).squeeze(1)], [reward], [kl], .01)
            fields = {"completion": float(completion.mean()), "sample_repeat": float(repeat.mean()),
                      "adjusted_reward": float(reward.mean()), "exact_kl": float(kl.detach().mean())}
        norm = exp.optimize_step(loss, policy, optimizer, cfg)
        if step % cfg["train"]["eval_every"] == 0 or step == steps:
            exp.check_deadline(deadline, "policy selection")
            summary, raw = exp.evaluate(policy, rm, auditor, selection, data, cfg, device, deadline=deadline)
            if "sample_actions" in raw:
                sampled_repeat = (raw["sample_actions"][:, :, None] == selection["context"][:, None, :]).any(-1).mean(1)
                summary["sample_repeat"] = exp.macro(sampled_repeat, selection["uid"])[0]
                summary["sample_completion_minus_repeat"] = exp.macro(
                    raw["sample_reward"] - repeat_penalty * sampled_repeat, selection["uid"])[0]
            candidate = {"name": name, "reward": "completion", "horizon": 1,
                         "step": step, "selection": summary}
            candidates.append(candidate)
            chosen = exp.select_candidate(candidates, baseline, cfg)
            if chosen is candidate:
                exp.checkpoint(output / f"{name}.pt", policy)
                best = candidate
            logs.append({"step": step, "loss": float(loss.detach()), "grad_norm": norm,
                         "elapsed_seconds": time.monotonic() - started,
                         "selection": summary, **fields})
            exp.dump(output / f"{name}_curve.json", logs)
    exp.checkpoint(output / f"{name}_final.pt", policy)
    return {"name": name, "optimizer_steps": steps, "selected": best,
            "selected_checkpoint": f"{name}.pt" if best else None,
            "final_checkpoint": f"{name}_final.pt", "repeat_penalty": repeat_penalty,
            "lr": optimizer.param_groups[0]["lr"], "beta": None if supervised else .01,
            "sampling": "uniform users then uniform fit events", 
            "selection_baseline": baseline,
            "selection_rule": "historical SFT relevance floor, then max eligible exact one-step unpenalized completion; repeat reported separately"}


def _validate_options(seeds, steps, repeat_penalty):
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds) or
            any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)):
        raise ValueError("seeds must be distinct nonnegative 32-bit integers")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(repeat_penalty, bool) or not np.isfinite(repeat_penalty) or repeat_penalty < 0:
        raise ValueError("repeat_penalty must be finite and nonnegative")
    return seeds


def run_followup(data_dir, run_dir, output, seeds=(42, 43, 44), steps=1000, repeat_penalty=.1):
    """Train fresh follow-up artifacts only; requires CUDA and a fresh output path.

    Parent ``reanalysis.verify_inputs(run, data)`` authenticates imported inputs.
    Historical reward/auditor weights are NEVER restored. The config comes from
    historical provenance; only explicit follow-up controls below are overridden.
    Returns the follow-up manifest, including per-seed gate failures (no RL).
    """
    seeds = _validate_options(seeds, steps, repeat_penalty)
    data_dir, run, output = Path(data_dir), Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError("Follow-up requires a fresh output directory")
    # Lazy import permits unit tests and API inspection before the parent lands.
    from .reanalysis import verify_inputs
    imported_manifest = verify_inputs(run, data_dir)
    historical = json.loads((run / "provenance.json").read_text(encoding="utf-8-sig"))
    cfg = copy.deepcopy(historical["config"])
    cfg["train"].update(eval_every=50, rl_steps=steps, allow_cpu=False)
    cfg["rl"].update(rewards=["completion"], horizons=[1], betas=[.01])
    device = exp.get_device(cfg)
    if device.type != "cuda":
        raise RuntimeError("Follow-up training requires CUDA")
    data = load_prepared(data_dir)
    fit = role(data, "fit", True)
    early, late = split_calibration(role(data, "calibration", True))
    selection = role(data, "selection")
    if not len(fit["target"]) or not np.any(selection["target"] >= 2):
        raise ValueError("Empty fit or eligible selection")
    output.mkdir(parents=True, exist_ok=False)
    provenance = exp.provenance(Path(__file__).resolve().parents[1], cfg, data_dir, device)
    provenance.update(historical_manifest=imported_manifest,
                      historical_provenance_sha256=sha256(run / "provenance.json"),
                      historical_sft_sha256=sha256(run / "sft.pt"),
                      seeds=list(seeds), steps=steps, repeat_penalty=repeat_penalty,
                      limitations=LIMITATIONS,
                      calibration_split={"rule": "median unique timestamp; ties together; median in late audit",
                                         "selection_examples": len(early["target"]),
                                         "audit_examples": len(late["target"]),
                                         "selection_max_timestamp": int(early["timestamp"].max()),
                                         "audit_min_timestamp": int(late["timestamp"].min())})
    exp.dump(output / "provenance.json", provenance)
    manifest = {"status": "in progress; test not evaluated", "seeds": {}, "limitations": LIMITATIONS}
    exp.dump(output / "followup_manifest.json", manifest)
    args = exp.model_args(data, cfg)
    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        exp.seed_all(seed, cfg["train"]["threads"])
        directory = output / f"seed_{seed}"
        directory.mkdir()
        exp.dump(directory / "config.json", seed_cfg)
        deadline = time.monotonic() + cfg["train"]["max_hours"] * 3600
        logs = {}
        rm = RewardModel(*args).to(device)
        auditor = ItemOnlyRewardModel(*args).to(device)
        for name, model in [("reward", rm), ("item_only", auditor)]:
            logs[name] = exp.fit_supervised(model, fit, early, cfg["train"]["reward_steps"],
                                            directory / f"{name}.pt", seed_cfg, device, deadline,
                                            reward=True, phase=name)
        exp.dump(directory / "reward_curves.json", logs)
        # Late labels are consumed here only, never passed to an optimizer or selector.
        report = exp.reward_report(rm, auditor, fit, late, seed_cfg, device, data, deadline=deadline)
        report.update(role="late_calibration_reward_audit", independent_of_new_rm_selection=True,
                      comparator="fresh learned item-only RM (auditor_mae); gate compares attention RM to fit-only constant/item-mean",
                      gate=exp.reward_gate(report, seed_cfg),
                      checkpoint_hashes={f"{n}.pt": sha256(directory / f"{n}.pt") for n in ["reward", "item_only"]})
        exp.dump(directory / "reward_audit.json", report)
        result = {"status": "reward gate failed; no policy training", "policies": {}}
        if report["gate"]["heads"]["completion"]["enabled"]:
            exp.freeze(rm)
            exp.freeze(auditor)
            reference = exp.freeze(exp.restore(run / "sft.pt", Policy(*args), device))
            for name, supervised, penalty in [("continued_sft", True, 0.),
                                               ("completion_h1", False, 0.),
                                               ("completion_minus_repeat_h1", False, repeat_penalty)]:
                exp.seed_all((seed + 999) % 2**32, cfg["train"]["threads"])
                policy = exp.restore(run / "sft.pt", Policy(*args), device)
                result["policies"][name] = train_policy(
                    policy, reference, rm, auditor, fit, selection, data, seed_cfg,
                    device, directory, name, steps, deadline,
                    supervised=supervised, repeat_penalty=penalty)
                del policy
            result["status"] = "training complete; test not evaluated"
        exp.dump(directory / "selection.json", result)
        result["checkpoint_hashes"] = {p.name: sha256(p) for p in sorted(directory.glob("*.pt"))}
        result["artifact_hashes"] = {p.name: sha256(p) for p in sorted(directory.glob("*.json"))}
        manifest["seeds"][str(seed)] = result
        exp.dump(output / "followup_manifest.json", manifest)
    manifest["status"] = "follow-up complete; inspect per-seed gates; test not evaluated"
    manifest["provenance_sha256"] = sha256(output / "provenance.json")
    exp.dump(output / "followup_manifest.json", manifest)
    exp.dump(output / "FOLLOWUP_COMPLETE.json", {"status": "complete",
             "manifest_sha256": sha256(output / "followup_manifest.json"),
             "provenance_sha256": sha256(output / "provenance.json")})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--repeat-penalty", type=float, default=.1)
    args = parser.parse_args()
    run_followup(args.data_dir, args.run_dir, args.output, args.seeds, args.steps, args.repeat_penalty)


if __name__ == "__main__":
    main()
