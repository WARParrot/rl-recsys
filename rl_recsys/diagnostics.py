"""Exploratory frozen-model diagnostics, not off-policy or causal estimates.

Public functions return ``(summary, arrays)``. Arrays retain row identities for
independent reconstruction. Deadlines are absolute ``time.monotonic()`` values.
Token 0 is PAD, 1 is UNK, and policy columns map to tokens starting at 2.
"""
import time

import numpy as np
import torch

from .evaluate import macro, paired_interval
from .models import append_action


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Deadline reached during diagnostics")


def _inputs(subset, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    context = np.asarray(subset["context"])
    n = len(context)
    if (context.ndim != 2 or not n or not context.shape[1]
            or context.dtype.kind not in "iu" or (context < 0).any()
            or (context[:, -1] == 0).any()
            or ((context[:, :-1] != 0) & (context[:, 1:] == 0)).any()):
        raise ValueError("context must contain nonempty left-padded integer histories")
    for key in ("uid", "target"):
        if np.asarray(subset[key]).shape != (n,):
            raise ValueError(f"{key} must match context rows")
    if np.asarray(subset["target"]).dtype.kind not in "iu" or (np.asarray(subset["target"]) < 1).any():
        raise ValueError("targets must be positive integer tokens")
    macro(np.zeros(n), subset["uid"])
    return context


def _device(model, device):
    return torch.device(device) if device is not None else next(model.parameters(), torch.empty(0)).device


def _cross_user_permutation(users, seed):
    """A global bijection of rows, with no same-user donor (never batch-local)."""
    rng = np.random.default_rng(seed)
    unique = np.unique(users)
    groups = [rng.permutation(np.flatnonzero(users == u)) for u in rng.permutation(unique)]
    largest = max(map(len, groups))
    if largest * 2 > len(users):
        raise ValueError("Cross-user history permutation requires no user to own more than half the eligible rows")
    order = np.concatenate(groups)
    donor = np.empty(len(users), dtype=np.int64)
    donor[order] = np.roll(order, largest)
    return donor


@torch.no_grad()
def reward_history_sensitivity(reward_model, subset, *, batch_size=256, device=None,
                               seed=2026, replicates=2000, deadline=None):
    """Observed-target reward/seconds MAE under original, UNK and shuffled histories.

    Only target>=2 rows enter; UNK replaces every non-PAD token. Shuffling is
    one seeded global whole-history permutation across different users, shared
    by all batches. Impossible cross-user bijections raise ValueError. Labels
    remain those of the original observed actions; no synthetic action is labeled.
    ``paired_interval`` reports perturbed-minus-original absolute-error deltas.
    """
    context = _inputs(subset, batch_size)
    keep = np.flatnonzero(np.asarray(subset["target"]) >= 2)
    if not len(keep):
        raise ValueError("No eligible observed targets")
    users = np.asarray(subset["uid"])[keep]
    context = context[keep]
    targets = np.asarray(subset["target"])[keep]
    labels = np.column_stack([np.asarray(subset[k])[keep] for k in ("reward", "seconds")])
    if labels.shape != (len(keep), 2) or not np.isfinite(labels).all():
        raise ValueError("reward and seconds must be finite observed labels")
    donor = _cross_user_permutation(users, seed)
    variants = {"original": context, "all_unk": np.where(context == 0, 0, 1),
                "shuffled": context[donor]}
    device = _device(reward_model, device)
    reward_model.eval()
    arrays = {"row_index": keep, "uid": users, "target": targets, "labels": labels,
              "shuffled_donor_row_index": keep[donor]}
    summary = {"examples": len(keep), "users": len(np.unique(users)), "seed": seed,
               "interpretation": "history sensitivity on eligible observed actions, not causal effects",
               "variants": {}}
    for name, histories in variants.items():
        chunks = []
        for start in range(0, len(keep), batch_size):
            _deadline(deadline)
            ctx = torch.as_tensor(histories[start:start + batch_size], dtype=torch.long, device=device)
            item = torch.as_tensor(targets[start:start + batch_size], dtype=torch.long, device=device)
            chunks.append(reward_model(ctx, item).cpu().numpy())
        predictions = np.concatenate(chunks)
        if predictions.shape != labels.shape or not np.isfinite(predictions).all():
            raise ValueError("Reward model must return finite [batch, 2] predictions")
        arrays[name + "_prediction"] = predictions
        error = np.abs(predictions - labels)
        arrays[name + "_absolute_error"] = error
        stats = {}
        for j, head in enumerate(("reward", "seconds")):
            stats[head + "_mae"] = macro(error[:, j], users)[0]
            if name != "original":
                stats[head + "_paired_mae"] = paired_interval(
                    arrays["original_absolute_error"][:, j], error[:, j], users,
                    replicates=replicates, seed=seed)
                change = predictions[:, j] - arrays["original_prediction"][:, j]
                stats[head + "_prediction_change"] = macro(change, users)[0]
                stats[head + "_absolute_prediction_change"] = macro(np.abs(change), users)[0]
        summary["variants"][name] = stats
    _deadline(deadline)
    return summary, arrays


@torch.no_grad()
def matched_rollout_controls(policy, reward_model, auditor, subset, popularity, *,
                             horizon=5, batch_size=256, k=10, temperature=1.0,
                             device=None, seed=2026, deadline=None):
    """Compare policy proposals under policy/random/popularity/observed histories.

    All modes score sampled *policy* actions; only the context transition differs.
    Random appends uniform catalog actions, popularity appends count-proportional
    actions, and policy appends its own proposal. Observed mode is explicitly
    ``observed_context_replay``: step t uses the t-th actual retained context and
    never appends subsampled targets. Users need >=max(5,horizon) retained rows.
    Rows are ordered by timestamp (ties rejected), with elapsed gaps exported in
    native timestamp units. This is NOT consecutive-event/session replay.

    Observed labels/MAE appear only in replay's separate observed-target arrays,
    restricted to eligible targets at their actual recorded contexts. They never
    label policy proposals or other synthetic actions. Metrics are user-macro.
    Seeds and cohort are shared across modes; no models or inputs are trained.
    """
    context = _inputs(subset, batch_size)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    pop = np.asarray(popularity, dtype=float)
    if pop.ndim != 1 or not len(pop) or not np.isfinite(pop).all() or (pop < 0).any() or pop.sum() <= 0:
        raise ValueError("popularity must be nonnegative finite counts with positive total")
    if (context >= len(pop) + 2).any() or (np.asarray(subset["target"]) >= len(pop) + 2).any():
        raise ValueError("Context or target token outside catalog")
    timestamps = np.asarray(subset["timestamp"])
    if timestamps.shape != (len(context),) or timestamps.dtype.kind not in "iu":
        raise ValueError("timestamp must be a matching integer vector")
    users = np.asarray(subset["uid"])
    selected = []
    for uid in np.unique(users):
        rows = np.flatnonzero(users == uid)
        rows = rows[np.argsort(timestamps[rows], kind="stable")]
        if len(rows) >= max(5, horizon):
            if (np.diff(timestamps[rows].astype(np.int64)) <= 0).any():
                raise ValueError("Replay requires strictly increasing timestamps per user")
            selected.append(rows[:horizon])
    if not selected:
        raise ValueError("No matched users with at least max(5, horizon) retained rows")
    rows = np.stack(selected)
    cohort_users = users[rows[:, 0]]
    times = timestamps[rows]
    gaps = np.diff(times.astype(np.int64), axis=1)
    device = _device(policy, device)
    for model in (policy, reward_model, auditor):
        model.eval()
    arrays = {"uid": cohort_users, "row_index": rows, "timestamp": times,
              "elapsed_from_start": times - times[:, :1], "elapsed_gaps": gaps, "modes": {}}
    summary = {"users": len(rows), "horizon": horizon, "seed": seed,
               "cohort_minimum_rows": max(5, horizon),
               "interpretation": "matched synthetic append controls and gapped observed_context_replay; not measured user transitions",
               "gap_units": "native prepared timestamp units",
               "elapsed_gaps": {"min": int(gaps.min()), "max": int(gaps.max()),
                                "mean": float(gaps.mean())} if gaps.size else None, "modes": {}}
    names = ("reward", "audit_reward", "disagreement", "repeat", "initial_topk_overlap", "policy_entropy")
    for mode in ("policy", "random", "popularity", "observed_context_replay"):
        # CPU draws keep streams independent of model device and batch partition.
        proposal_rng = torch.Generator().manual_seed(seed)
        append_rng = torch.Generator().manual_seed(seed + 1)
        current = context[rows[:, 0]].copy()
        metrics = np.empty((len(rows), horizon, len(names)))
        actions = np.empty((len(rows), horizon), dtype=np.int64)
        appended = np.full_like(actions, -1)
        replay_predictions = np.full((len(rows), horizon, 2), np.nan)
        initial_top = None
        for step in range(horizon):
            if mode == "observed_context_replay":
                current = context[rows[:, step]]
            top_parts, proposal_parts, entropy_parts = [], [], []
            for start in range(0, len(rows), batch_size):
                _deadline(deadline)
                ctx = torch.as_tensor(current[start:start + batch_size], dtype=torch.long, device=device)
                scores = policy(ctx).float().cpu() / temperature
                if scores.shape != (len(ctx), len(pop)) or not torch.isfinite(scores).all():
                    raise ValueError("Policy must return finite [batch, catalog] logits")
                probs = scores.softmax(-1)
                top_parts.append(torch.argsort(scores, descending=True, stable=True)[:, :min(k, len(pop))])
                proposal_parts.append(torch.multinomial(probs, 1, generator=proposal_rng).squeeze(1))
                entropy_parts.append(-(probs * scores.log_softmax(-1)).sum(-1))
                # Do not retain cohort x catalog logits/probabilities/sort buffers.
                del scores, probs
            top = torch.cat(top_parts)
            proposal = torch.cat(proposal_parts)
            entropy = torch.cat(entropy_parts)
            if initial_top is None:
                initial_top = top
            actions[:, step] = proposal.numpy() + 2
            overlap = (top[:, :, None] == initial_top[:, None, :]).any(-1).float().mean(-1)
            for start in range(0, len(rows), batch_size):
                _deadline(deadline)
                stop = min(start + batch_size, len(rows))
                ctx = torch.as_tensor(current[start:stop], dtype=torch.long, device=device)
                token = torch.as_tensor(actions[start:stop, step], device=device)
                r = reward_model(ctx, token)[:, 0]
                a = auditor(ctx, token)[:, 0]
                repeat = (ctx == token[:, None]).any(-1).float()
                metrics[start:stop, step] = np.column_stack([
                    r.cpu().numpy(), a.cpu().numpy(), (r-a).abs().cpu().numpy(),
                    repeat.cpu().numpy(), overlap[start:stop].numpy(), entropy[start:stop].numpy()])
                if mode == "observed_context_replay":
                    target = np.asarray(subset["target"])[rows[start:stop, step]]
                    eligible = target >= 2
                    if eligible.any():
                        pred = reward_model(ctx[torch.as_tensor(eligible, device=device)],
                                            torch.as_tensor(target[eligible], dtype=torch.long, device=device))
                        replay_predictions[start:stop, step][eligible] = pred.cpu().numpy()
            if mode != "observed_context_replay" and step + 1 < horizon:
                if mode == "policy":
                    append = proposal
                else:
                    weights = torch.ones(len(pop), dtype=torch.float64) if mode == "random" else torch.tensor(pop)
                    append = torch.multinomial(weights, len(rows), replacement=True, generator=append_rng)
                appended[:, step] = append.numpy() + 2
                current = append_action(torch.as_tensor(current), append).numpy()
        out = {"actions": actions, "appended_tokens": appended,
               **{name: metrics[:, :, j] for j, name in enumerate(names)}}
        steps = [{"step": t + 1, **{name: macro(metrics[:, t, j], cohort_users)[0]
                                          for j, name in enumerate(names)}} for t in range(horizon)]
        mode_summary = {"steps": steps}
        if mode == "observed_context_replay":
            targets = np.asarray(subset["target"])[rows]
            labels = np.stack([np.asarray(subset[key])[rows] for key in ("reward", "seconds")], -1)
            if not np.isfinite(labels).all():
                raise ValueError("Observed labels must be finite")
            eligible = targets >= 2
            out.update(observed_target=targets, observed_labels=labels,
                       observed_target_prediction=replay_predictions, observed_eligible=eligible)
            flat_users = np.broadcast_to(cohort_users[:, None], targets.shape)[eligible]
            mode_summary["observed_target_mae"] = {
                head: macro(np.abs(replay_predictions[:, :, j][eligible] - labels[:, :, j][eligible]), flat_users)[0]
                if eligible.any() else None for j, head in enumerate(("reward", "seconds"))}
            mode_summary["observed_eligible_examples"] = int(eligible.sum())
        arrays["modes"][mode] = out
        summary["modes"][mode] = mode_summary
    _deadline(deadline)
    return summary, arrays
