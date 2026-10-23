"""Full restricted-catalog evaluation; user-macro metrics, paired uncertainty."""
import time
import numpy as np
import torch
from .models import append_action


def macro(values, users):
    unique, inverse = np.unique(users, return_inverse=True)
    means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return float(means.mean()), means, unique


def paired_interval(before, after, users, replicates=2000, seed=2026):
    _, a, u = macro(np.asarray(before), users)
    _, b, _ = macro(np.asarray(after), users)
    rng = np.random.default_rng(seed)
    deltas = []
    relative = []
    for _ in range(replicates):
        idx = rng.integers(len(u), size=len(u))
        av, bv = a[idx].mean(), b[idx].mean()
        deltas.append(bv - av)
        relative.append((bv - av) / av if av else np.nan)
    finite = np.asarray(relative)[np.isfinite(relative)]
    return {"users": len(u), "before": float(a.mean()), "after": float(b.mean()),
            "difference": float((b - a).mean()),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "relative_difference": float((b.mean() - a.mean()) / a.mean()) if a.mean() else None,
            "relative_ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "uncertainty": "paired user bootstrap conditional on this seed/checkpoint; no seed uncertainty",
            "adequate_users": bool(len(u) >= 100)}


def ranking_metrics(recs, targets):
    hits = recs == np.asarray(targets)[:, None]
    recall = hits.any(1).astype(float)
    ndcg = (hits / np.log2(np.arange(recs.shape[1]) + 2)).sum(1)
    return recall, ndcg


@torch.no_grad()
def evaluate(policy, rm, auditor, subset, data, cfg, device, mode="policy", deadline=None):
    for model in [policy, rm, auditor]:
        model.eval()
    batch_size = cfg["evaluation"]["batch_size"]
    k = min(cfg["evaluation"]["k"], len(data["vocab"]))
    rows = {key: [] for key in ["recommendations", "sample_reward", "sample_audit", "sample_seconds",
                               "top_reward", "top_audit", "repeat", "duration", "popularity"]}
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 10000)
    for start in range(0, len(subset["target"]), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Deadline reached during evaluation")
        context = torch.as_tensor(subset["context"][start:start + batch_size], device=device)
        scores = policy(context).float()
        rh = rm.history(context)
        ah = auditor.history(context)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Deadline reached during evaluation")
        if mode == "popularity":
            scores = torch.as_tensor(data["popularity"], device=device).float().log1p().expand(len(context), -1)
        elif mode == "rerank":
            scores = scores.log_softmax(-1) + cfg["rl"]["rerank_weight"] * rm.catalog(context)[:, :, 0]
        elif mode != "policy":
            raise ValueError(mode)
        # Stable score-descending ties use ascending token/item ID.
        rec = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]
        dist = (scores / cfg["rl"]["sampling_temperature"]).softmax(-1)
        action = torch.multinomial(dist, 1, generator=generator)
        rec_np = rec.cpu().numpy()
        sample_r = rm.from_history(rh, action.squeeze(1) + 2)
        sample_a = auditor.from_history(ah, action.squeeze(1) + 2)
        top_r = rm.from_history(rh[:, None, :].expand(-1, k, -1), rec + 2)
        top_a = auditor.from_history(ah[:, None, :].expand(-1, k, -1), rec + 2)
        rows["recommendations"].append(rec_np + 2)
        rows["sample_reward"].append(sample_r[:, 0].cpu().numpy())
        rows["sample_seconds"].append(sample_r[:, 1].cpu().numpy())
        rows["sample_audit"].append(sample_a[:, 0].cpu().numpy())
        rows["top_reward"].append(top_r[:, :, 0].mean(1).cpu().numpy())
        rows["top_audit"].append(top_a[:, :, 0].mean(1).cpu().numpy())
        rows["repeat"].append(((rec + 2)[:, :, None] == context[:, None, :]).any(-1).float().mean(1).cpu().numpy())
        rows["duration"].append(data["durations"][rec_np].mean(1))
        rows["popularity"].append(data["popularity"][rec_np].mean(1))
    out = {key: np.concatenate(value) for key, value in rows.items()}
    out["recall"], out["ndcg"] = ranking_metrics(out["recommendations"], subset["target"])
    out["uid"] = subset["uid"]
    summary = {key: macro(out[key], out["uid"])[0] for key in out if key not in ["uid", "recommendations"]}
    eligible = subset["target"] >= 2
    eligible_users = np.unique(subset["uid"][eligible])
    summary.update(all_target_examples=len(subset["target"]),
                   eligible_target_examples=int(eligible.sum()),
                   all_target_users=len(np.unique(subset["uid"])),
                   eligible_target_users=len(eligible_users),
                   target_coverage=float(eligible.mean()),
                   eligible_ndcg=macro(out["ndcg"][eligible], subset["uid"][eligible])[0] if eligible.any() else None,
                   catalog_coverage=len(np.unique(out["recommendations"])) / len(data["vocab"]))
    _, counts = np.unique(out["recommendations"], return_counts=True)
    summary["recommendation_hhi"] = float(((counts / counts.sum()) ** 2).sum())
    return summary, out


@torch.no_grad()
def rollout_diagnostics(policy, rm, auditor, subset, cfg, device, horizon=5, deadline=None):
    """Synthetic context only: not measured user transitions/session value."""
    result = []
    bs = cfg["evaluation"]["batch_size"]
    samples = max(1, int(cfg["evaluation"].get("rollout_samples", 1)))
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 20000)
    buffers = [[] for _ in range(horizon)]
    for sample in range(samples):
        for start in range(0, len(subset["target"]), bs):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Deadline reached during rollout diagnostics")
            context = torch.as_tensor(subset["context"][start:start + bs], device=device)
            for t in range(horizon):
                action = torch.multinomial((policy(context) / cfg["rl"]["sampling_temperature"]).softmax(-1), 1,
                                           generator=generator).squeeze(1)
                r = rm(context, action + 2)[:, 0]
                a = auditor(context, action + 2)[:, 0]
                repeat = (context == (action + 2)[:, None]).any(1).float()
                buffers[t].append(torch.stack([r, a, (r-a).abs(), repeat], 1).cpu().numpy())
                context = append_action(context, action)
    for t, values in enumerate(buffers):
        values = np.concatenate(values)
        result.append({"step": t + 1, **{key: macro(values[:, i], subset["uid"])[0]
                      for i, key in enumerate(["reward", "audit_reward", "disagreement", "repeat"])}})
    return result
