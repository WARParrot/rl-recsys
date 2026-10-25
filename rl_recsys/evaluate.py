"""Full restricted-catalog evaluation; user-macro metrics, paired uncertainty."""
import time
import numpy as np
import torch
from .models import append_action


def _validated_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if (not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite numeric values")
    return values


def macro(values, users):
    values = _validated_vector(values, "values")
    users = np.asarray(users)
    if users.ndim != 1 or users.size == 0 or len(users) != len(values):
        raise ValueError("users must be a non-empty one-dimensional array matching values")
    if any(uid is None or (isinstance(uid, (int, float, complex, np.number))
                           and not np.isfinite(uid)) for uid in users):
        raise ValueError("users must contain finite IDs")
    unique, inverse = np.unique(users, return_inverse=True)
    means = np.bincount(inverse, weights=values) / np.bincount(inverse)
    return float(means.mean()), means, unique


def paired_interval(before, after, users, replicates=2000, seed=2026, *, root_mse=False):
    before = _validated_vector(before, "before")
    after = _validated_vector(after, "after")
    users = np.asarray(users)
    if before.shape != after.shape or users.ndim != 1 or len(users) != len(before):
        raise ValueError("before, after, and users must be matching one-dimensional arrays")
    if isinstance(replicates, (bool, np.bool_)) or not isinstance(replicates, (int, np.integer)) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    _, a, u = macro(before, users)
    _, b, _ = macro(np.asarray(after), users)
    if root_mse and ((before < 0).any() or (after < 0).any()):
        raise ValueError("root_mse requires nonnegative squared errors")
    transform = np.sqrt if root_mse else lambda x: x
    av_point, bv_point = transform(a.mean()), transform(b.mean())
    rng = np.random.default_rng(seed)
    deltas = []
    relative = []
    for _ in range(replicates):
        idx = rng.integers(len(u), size=len(u))
        av, bv = transform(a[idx].mean()), transform(b[idx].mean())
        deltas.append(bv - av)
        relative.append((bv - av) / av if av else np.nan)
    finite = np.asarray(relative)[np.isfinite(relative)]
    return {"users": len(u), "before": float(av_point), "after": float(bv_point),
            "difference": float(bv_point - av_point) if root_mse else float((b - a).mean()),
            "ci95": np.quantile(deltas, [.025, .975]).tolist(),
            "relative_difference": float((bv_point - av_point) / av_point) if av_point else None,
            "relative_ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "uncertainty": "paired user bootstrap conditional on this seed/checkpoint; no seed uncertainty",
            "adequate_users": bool(len(u) >= 100)}


@torch.no_grad()
def exact_one_step_expected_reward(policy, reward_model, contexts, users, batch_size=256, temperature=1.0, device=None,
                                   deadline=None, logits=None):
    """Return (user-macro reward, per-context rewards) without action sampling.

    Integrates softmax(policy logits / temperature) against reward head zero over
    all real items (catalog column j maps to token j + 2). Models are put in eval
    mode and evaluated without gradients; weights and contexts are not updated.
    This is a conditional model-based one-step utility, not measured user impact.
    """
    contexts = torch.as_tensor(contexts)
    if contexts.ndim != 2 or contexts.shape[0] == 0:
        raise ValueError("contexts must be a non-empty [contexts, history] array")
    users = np.asarray(users)
    # Reuse macro validation without computing any model outputs on invalid input.
    macro(np.zeros(len(contexts)), users)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    device = torch.device(device) if device is not None else next(policy.parameters(), contexts).device
    policy.eval()
    reward_model.eval()
    values = []
    for start in range(0, len(contexts), batch_size):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Deadline reached during exact one-step evaluation")
        context = contexts[start:start + batch_size].to(device=device, dtype=torch.long)
        scores = policy(context) if logits is None else logits[start:start + batch_size]
        probabilities = (scores.float() / temperature).softmax(-1)
        history = reward_model.history(context)
        expected = torch.zeros(len(context), device=device)
        # Bound the B x catalog x hidden intermediate on large catalogs.
        for offset in range(0, probabilities.shape[-1], batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Deadline reached during exact one-step evaluation")
            catalog = torch.arange(offset, min(offset + batch_size, probabilities.shape[-1]), device=device) + 2
            expanded_history = history[:, None, :].expand(-1, len(catalog), -1).reshape(-1, history.shape[-1])
            items = catalog.expand(len(context), -1).reshape(-1)
            rewards = reward_model.from_history(expanded_history, items).reshape(len(context), len(catalog), -1)[:, :, 0]
            expected += (probabilities[:, offset:offset + len(catalog)] * rewards).sum(-1)
        values.append(expected.cpu())
    values = torch.cat(values)
    return macro(values.numpy(), users)[0], values.numpy()


# Concise alias for callers that do not need to emphasize exact integration.
expected_one_step_reward = exact_one_step_expected_reward


def sampling_diagnostic(before, after, users):
    """Descriptive variation across retained action draws; not a significance interval."""
    before, after, users = np.asarray(before), np.asarray(after), np.asarray(users)
    if before.shape != after.shape or before.ndim != 2 or len(users) != len(before):
        raise ValueError("Expected matching [contexts, action_samples] arrays and users")
    unique, inverse = np.unique(users, return_inverse=True)
    before_user = np.zeros((len(unique), before.shape[1]), dtype=np.float64)
    after_user = np.zeros_like(before_user)
    counts = np.bincount(inverse)
    np.add.at(before_user, inverse, before)
    np.add.at(after_user, inverse, after)
    before_user /= counts[:, None]
    after_user /= counts[:, None]
    differences = after_user - before_user
    return {"users": len(unique), "action_samples": int(before.shape[1]),
            "difference": float(differences.mean()),
            "sampling_sd_of_user_difference": float(differences.mean(0).std(ddof=1)) if before.shape[1] > 1 else 0.0,
            "uncertainty": "descriptive variation across retained action draws; not a confidence interval or training-seed uncertainty"}


def action_support_diagnostic(recommendations, reward_scores, support_targets, users=None):
    """Quantify empirical action support and score shift (not propensity/OPE)."""
    rec = np.asarray(recommendations)
    scores = np.asarray(reward_scores)
    support_targets = np.asarray(support_targets)
    if (rec.ndim != 2 or not rec.size or scores.shape != rec.shape
            or support_targets.ndim != 1 or not np.isfinite(scores).all()):
        raise ValueError("recommendations, reward_scores, and support_targets have incompatible shapes or scores")
    support = np.unique(support_targets[support_targets >= 2])
    if rec.dtype.kind not in "iu" or np.any(rec < 2):
        raise ValueError("recommendations, reward_scores, and support_targets have incompatible shapes")
    mask = np.isin(rec, support)
    flat_scores = scores.reshape(-1)
    flat_mask = mask.reshape(-1)
    result = {"support_unique_actions": int(len(support)),
              "recommendation_actions": int(mask.size),
              "unsupported_action_rate": float((~flat_mask).mean()),
              "supported_action_rate": float(flat_mask.mean()),
              "supported_unique_action_rate": float(np.isin(np.unique(rec), support).mean()),
              "reward_score_mean": float(flat_scores.mean())}
    for label, group in (("supported", flat_mask), ("unsupported", ~flat_mask)):
        result[f"{label}_actions"] = int(group.sum())
        result[f"{label}_reward_score_mean"] = float(flat_scores[group].mean()) if group.any() else None
    if users is not None:
        result["user_macro_unsupported_rate"] = macro((~mask).mean(1), users)[0]
    return result


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
    action_samples = max(1, int(cfg["evaluation"].get("action_samples", 1)))
    rows = {key: [] for key in ["recommendations", "exact_one_step_expected_reward", "sample_reward", "sample_audit", "sample_seconds",
                               "sample_reward_draws", "sample_audit_draws", "top_reward", "top_reward_actions", "sample_actions", "top_audit",
                               "repeat", "duration", "popularity"]}
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
        _, exact = exact_one_step_expected_reward(
            policy, rm, context, subset["uid"][start:start + batch_size],
            batch_size=batch_size, temperature=cfg["rl"]["sampling_temperature"],
            device=device, deadline=deadline, logits=scores)
        rows["exact_one_step_expected_reward"].append(np.asarray(exact).reshape(-1))
        # Stable score-descending ties use ascending token/item ID.
        rec = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]
        dist = (scores / cfg["rl"]["sampling_temperature"]).softmax(-1)
        actions = torch.multinomial(dist, action_samples, replacement=True, generator=generator)
        rec_np = rec.cpu().numpy()
        expanded_history = rh[:, None, :].expand(-1, action_samples, -1).reshape(-1, rh.shape[-1])
        expanded_audit = ah[:, None, :].expand(-1, action_samples, -1).reshape(-1, ah.shape[-1])
        action_ids = actions.reshape(-1) + 2
        sample_r = rm.from_history(expanded_history, action_ids).reshape(len(context), action_samples, 2)
        sample_a = auditor.from_history(expanded_audit, action_ids).reshape(len(context), action_samples, 2)
        top_r = rm.from_history(rh[:, None, :].expand(-1, k, -1), rec + 2)
        top_a = auditor.from_history(ah[:, None, :].expand(-1, k, -1), rec + 2)
        rows["recommendations"].append(rec_np + 2)
        rows["sample_actions"].append(actions.cpu().numpy() + 2)
        rows["sample_reward"].append(sample_r[:, :, 0].mean(1).cpu().numpy())
        rows["sample_seconds"].append(sample_r[:, :, 1].mean(1).cpu().numpy())
        rows["sample_audit"].append(sample_a[:, :, 0].mean(1).cpu().numpy())
        rows["sample_reward_draws"].append(sample_r[:, :, 0].cpu().numpy())
        rows["sample_audit_draws"].append(sample_a[:, :, 0].cpu().numpy())
        rows["top_reward"].append(top_r[:, :, 0].mean(1).cpu().numpy())
        rows["top_reward_actions"].append(top_r[:, :, 0].cpu().numpy())
        rows["top_audit"].append(top_a[:, :, 0].mean(1).cpu().numpy())
        rows["repeat"].append(((rec + 2)[:, :, None] == context[:, None, :]).any(-1).float().mean(1).cpu().numpy())
        rows["duration"].append(data["durations"][rec_np].mean(1))
        rows["popularity"].append(data["popularity"][rec_np].mean(1))
    out = {key: np.concatenate(value) for key, value in rows.items()}
    out["recall"], out["ndcg"] = ranking_metrics(out["recommendations"], subset["target"])
    out["uid"] = subset["uid"]
    summary = {key: macro(out[key], out["uid"])[0] for key in out
               if key not in ["uid", "recommendations", "sample_reward_draws", "sample_audit_draws", "top_reward_actions", "sample_actions"]}
    summary["sample_reward_sampling_sd"] = float(np.std([
        macro(out["sample_reward_draws"][:, i], out["uid"])[0] for i in range(action_samples)], ddof=1
    )) if action_samples > 1 else 0.0
    eligible = subset["target"] >= 2
    eligible_users = np.unique(subset["uid"][eligible])
    summary.update(all_target_examples=len(subset["target"]),
                   eligible_target_examples=int(eligible.sum()),
                   all_target_users=len(np.unique(subset["uid"])),
                   eligible_target_users=len(eligible_users),
                   target_coverage=float(eligible.mean()),
                   eligible_ndcg=macro(out["ndcg"][eligible], subset["uid"][eligible])[0] if eligible.any() else None,
                   eligible_exact_one_step_expected_reward=macro(out["exact_one_step_expected_reward"][eligible], subset["uid"][eligible])[0] if eligible.any() else None,
                   eligible_sample_reward=macro(out["sample_reward"][eligible], subset["uid"][eligible])[0] if eligible.any() else None,
                   eligible_sample_audit=macro(out["sample_audit"][eligible], subset["uid"][eligible])[0] if eligible.any() else None,
                   catalog_coverage=len(np.unique(out["recommendations"])) / len(data["vocab"]))
    _, counts = np.unique(out["recommendations"], return_counts=True)
    summary["recommendation_hhi"] = float(((counts / counts.sum()) ** 2).sum())
    if "fit_target" in data:
        summary["action_support"] = action_support_diagnostic(
            out["recommendations"], out["top_reward_actions"],
            data["fit_target"], out["uid"])
        summary["sample_action_support"] = action_support_diagnostic(
            out["sample_actions"], out["sample_reward_draws"], data["fit_target"], out["uid"])
        summary["action_support"]["reference"] = "retained eligible fit targets; marginal item support only, not context-action or causal support"
    return summary, out


@torch.no_grad()
def rollout_diagnostics(policy, rm, auditor, subset, cfg, device, horizon=5, deadline=None):
    """Synthetic context only: not measured user transitions/session value."""
    result = []
    bs = cfg["evaluation"]["batch_size"]
    samples = max(1, int(cfg["evaluation"].get("rollout_samples", 1)))
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 20000)
    sample_metrics = [[[] for _ in range(horizon)] for _ in range(samples)]
    for sample in range(samples):
        buffers = [[] for _ in range(horizon)]
        for start in range(0, len(subset["target"]), bs):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Deadline reached during rollout diagnostics")
            context = torch.as_tensor(subset["context"][start:start + bs], device=device)
            initial_scores = policy(context) / cfg["rl"]["sampling_temperature"]
            k = min(cfg["evaluation"]["k"], initial_scores.shape[-1])
            initial_top = torch.argsort(initial_scores, descending=True, stable=True)[:, :k]
            for t in range(horizon):
                scores = initial_scores if t == 0 else policy(context) / cfg["rl"]["sampling_temperature"]
                probabilities = scores.softmax(-1)
                current_top = torch.argsort(scores, descending=True, stable=True)[:, :k]
                overlap = (current_top[:, :, None] == initial_top[:, None, :]).any(-1).float().mean(-1)
                entropy = -(probabilities * scores.log_softmax(-1)).sum(-1)
                action = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
                r = rm(context, action + 2)[:, 0]
                a = auditor(context, action + 2)[:, 0]
                repeat = (context == (action + 2)[:, None]).any(1).float()
                buffers[t].append(torch.stack([r, a, (r-a).abs(), repeat, overlap, entropy], 1).cpu().numpy())
                context = append_action(context, action)
        for t, values in enumerate(buffers):
            values = np.concatenate(values)
            sample_metrics[sample][t] = {key: macro(values[:, i], subset["uid"])[0]
                                         for i, key in enumerate(["reward", "audit_reward", "disagreement", "repeat",
                                                                  "initial_topk_overlap", "policy_entropy"])}
    for t in range(horizon):
        per_sample = [sample_metrics[s][t] for s in range(samples)]
        result.append({"step": t + 1,
                       **{key: float(np.mean([row[key] for row in per_sample])) for key in per_sample[0]},
                       "sampling_sd": {key: float(np.std([row[key] for row in per_sample], ddof=1)) if samples > 1 else 0.0
                                       for key in per_sample[0]},
                       "per_sample": per_sample})
    return result
