"""Label-backed diagnostics only: never score or label unexposed actions."""
import math

import numpy as np

from .evaluate import macro, paired_interval


def audit_groups(fit, subset, data=None):
    """Freeze popularity cut points on fit items; include entire boundary ties.

    Prepared popularity counts include all fit events, not just sampled contexts.
    Without prepared metadata, use the supplied fit targets (and say so).
    """
    target = subset["target"]
    eligible = target >= 2
    repeat = eligible & np.any(subset["context"] == target[:, None], axis=1)
    masks = {"organic": subset["organic"] == 1,
             "recommended": subset["organic"] == 0,
             "repeat": repeat, "novel": eligible & ~repeat}
    if data is not None and "popularity" in data:
        counts = np.asarray(data["popularity"])
        tokens = np.arange(len(counts)) + 2
        source = "prepared fit-only event counts"
    else:
        tokens, counts = np.unique(fit["target"][fit["target"] >= 2], return_counts=True)
        source = "supplied fit target counts"
    # Inverse empirical CDF cut points (no interpolation), over items, not events.
    ordered = np.sort(counts)
    cuts = [int(ordered[math.ceil(len(ordered) * q) - 1]) for q in (1/3, 2/3)] if len(ordered) else []
    metadata = {"popularity": {"source": source, "cut_points": cuts,
                               "fit_items": len(tokens),
                               "tie_policy": "low <= first cut; middle > first and <= second; high > second; ties never split"},
                "duration": {"source": "prepared fit-only per-item median seconds",
                             "cut_points": [180, 300], "available": data is not None and "durations" in data}}
    if cuts:
        lookup = dict(zip(tokens.tolist(), counts.tolist()))
        supported = np.array([t in lookup for t in target])
        popularity = np.array([lookup.get(t, 0) for t in target])
        masks.update(popularity_low=supported & (popularity <= cuts[0]),
                     popularity_middle=supported & (popularity > cuts[0]) & (popularity <= cuts[1]),
                     popularity_high=supported & (popularity > cuts[1]))
    if metadata["duration"]["available"]:
        durations = np.asarray(data["durations"])
        supported = eligible & (target - 2 < len(durations))
        duration = np.full(len(target), np.nan)
        duration[supported] = durations[target[supported] - 2]
        masks.update(duration_lt_180s=supported & (duration < 180),
                     duration_180_to_300s=supported & (duration >= 180) & (duration < 300),
                     duration_ge_300s=supported & (duration >= 300))
    return masks, metadata


def label_summary(mask, prediction, auditor, observed, baseline_error, users, evaluation):
    """User-macro errors; paired differences are after minus before."""
    n = int(mask.sum())
    result = {"n": n, "users": int(len(np.unique(users[mask]))),
              "mae": None, "bias": None, "item_mean_mae": None,
              "auditor_mae": None, "paired_mae": None, "auditor_paired_mae": None}
    if not n:
        return result
    uid = users[mask]
    residual = prediction[mask] - observed[mask]
    error = np.abs(residual)
    aerror = np.abs(auditor[mask] - observed[mask])
    result.update(mae=macro(error, uid)[0], bias=macro(residual, uid)[0],
                  item_mean_mae=macro(baseline_error[mask], uid)[0], auditor_mae=macro(aerror, uid)[0],
                  paired_mae=paired_interval(baseline_error[mask], error, uid,
                                            evaluation["bootstrap"], evaluation["bootstrap_seed"]),
                  auditor_paired_mae=paired_interval(aerror, error, uid,
                                                    evaluation["bootstrap"], evaluation["bootstrap_seed"]))
    return result


def score_tails(prediction, auditor, observed, baseline_error, users, evaluation):
    """Top ceil(fraction*n) observed rows, expanded to include all cutoff ties."""
    result = {}
    for name, fraction in (("top10", .1), ("top1", .01)):
        requested = max(1, math.ceil(len(prediction) * fraction))
        cutoff = float(np.sort(prediction)[-requested])
        mask = prediction >= cutoff
        row = label_summary(mask, prediction, auditor, observed, baseline_error, users, evaluation)
        # Descriptive support warning, not an additional numerical reward gate.
        row.update(fraction=fraction, requested_n=requested, cutoff=cutoff,
                   tie_policy="score >= cutoff; include all cutoff ties",
                   row_indices=np.flatnonzero(mask).tolist(), realized_fraction=float(mask.mean()),
                   cutoff_tie_n=int(np.sum(prediction == cutoff)),
                   predicted=float(prediction[mask].mean()), observed=float(observed[mask].mean()),
                   overprediction=macro(np.maximum(prediction[mask] - observed[mask], 0), users[mask])[0],
                   insufficient_examples=row["n"] < 100, insufficient_users=row["users"] < 100,
                   support_rule="descriptive only: fewer than 100 examples or 100 users is inconclusive; not a gate",
                   insufficiency_flag=row["n"] < 100 or row["users"] < 100)
        result[name] = row
    return result
