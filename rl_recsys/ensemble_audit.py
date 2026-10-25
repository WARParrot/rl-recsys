"""Opt-in observed-label ensemble hypothesis audit; no training or calibration fit.

See docs/guides/ENSEMBLE_AUDIT.md for the pickle-free NPZ contract and estimands.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


def _text(value, name):
    a = np.asarray(value)
    if a.shape != () or a.dtype.kind not in "US" or not str(a.item()).strip():
        raise ValueError(f"{name} must be a nonempty scalar string")
    return str(a.item())


def _ids(value, name, size=None):
    a = np.asarray(value)
    if a.ndim != 1 or (size is not None and len(a) != size) or a.dtype.kind not in "USiu":
        raise ValueError(f"{name} must be a one-dimensional string/integer identity array")
    if a.dtype.kind in "US" and any(not str(x).strip() for x in a):
        raise ValueError(f"{name} contains empty identities")
    return a.astype(str)


def _finite(value, name, shape, bounded=False):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all():
        raise ValueError(f"{name} must have shape {shape} and finite numeric values")
    if bounded and ((a < 0).any() or (a > 1).any()):
        raise ValueError(f"{name} must be in [0,1]")
    return a.astype(float)


def validate_export(arrays):
    """Validate declared provenance, not authenticity of labels/model training."""
    required = {
        "predictions", "labels", "uid", "example_id", "timestamp",
        "early_example_id", "early_timestamp", "data_id", "catalog_id", "source_id",
        "split_id", "member_id", "member_seed", "member_data_id", "member_catalog_id",
        "member_source_id", "member_split_id", "member_fit_role", "member_checkpoint_role",
        "label_source", "calibration_status", "audit_role",
    }
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    # Reject object arrays even for unrecognized fields; never load with pickle.
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise ValueError("Object arrays are not permitted")
    p = np.asarray(arrays["predictions"])
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 3:
        raise ValueError("predictions must be nonempty [examples,members] with >=3 members")
    n, m = p.shape
    result = {"predictions": _finite(p, "predictions", (n, m), True),
              "labels": _finite(arrays["labels"], "labels", (n,), True),
              "uid": _ids(arrays["uid"], "uid", n)}
    late = _ids(arrays["example_id"], "example_id", n)
    early = _ids(arrays["early_example_id"], "early_example_id")
    if not len(early) or len(np.unique(late)) != n or len(np.unique(early)) != len(early):
        raise ValueError("Early/late example identities must be nonempty and unique")
    if np.intersect1d(early, late).size:
        raise ValueError("Early calibration and late audit examples overlap")
    lt = _finite(arrays["timestamp"], "timestamp", (n,))
    et = _finite(arrays["early_timestamp"], "early_timestamp", (len(early),))
    if et.max() >= lt.min():
        raise ValueError("Early calibration must strictly precede late audit (including ties)")
    provenance = {}
    for key in ("data_id", "catalog_id", "source_id", "split_id"):
        provenance[key] = _text(arrays[key], key)
        if not np.all(_ids(arrays[f"member_{key}"], f"member_{key}", m) == provenance[key]):
            raise ValueError(f"Member {key} mismatch")
    for key in ("member_id", "member_seed"):
        ids = _ids(arrays[key], key, m)
        if len(np.unique(ids)) != m:
            raise ValueError(f"{key} must identify independently initialized distinct members")
        provenance[key] = ids.tolist()
    for key, expected in (("member_fit_role", "fit"),
                          ("member_checkpoint_role", "early_calibration")):
        if not np.all(_ids(arrays[key], key, m) == expected):
            raise ValueError(f"{key} must be {expected}")
    if _text(arrays["audit_role"], "audit_role") != "late_calibration":
        raise ValueError("Only late_calibration observed-target audits are supported")
    if _text(arrays["label_source"], "label_source") != "observed":
        raise ValueError("Explicit observed labels required; no synthetic labels")
    status = _text(arrays["calibration_status"], "calibration_status")
    if status not in ("identity", "uncalibrated"):
        raise ValueError("Only declared identity/uncalibrated predictions supported; no calibrated claim")
    slices = {}
    keys = [key for key in arrays if key.startswith("slice__")]
    if keys:
        metadata = json.loads(_text(arrays.get("slice_metadata_json"), "slice_metadata_json"))
        if not isinstance(metadata, dict):
            raise ValueError("slice_metadata_json must be an object")
        for key in keys:
            name = key[len("slice__"):]
            meta = metadata.get(name, {})
            if (not isinstance(meta, dict) or meta.get("derived_from") != "fit"
                    or meta.get("data_id") != provenance["data_id"]
                    or meta.get("catalog_id") != provenance["catalog_id"]
                    or not isinstance(meta.get("method"), str) or not meta["method"].strip()):
                raise ValueError(f"Slice {name} needs matching fit-derived metadata and method")
            slices[name] = (_ids(arrays[key], key, n), meta)
    result.update(provenance=provenance, calibration_status=status, slices=slices,
                  early_examples=len(early))
    return result


def content_digest(arrays):
    """Hash validated export fields, preserving dtype, shape and C-order bytes.

    Sorted names and length-prefixed headers/payloads make framing unambiguous.
    Includes all supplied fields, including early/late IDs, times and slice metadata.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        header = json.dumps([name, value.dtype.str, list(np.asarray(arrays[name]).shape)],
                            separators=(",", ":"), ensure_ascii=True).encode("ascii")
        for part in (header, value.tobytes(order="C")):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def tied_ranks(values):
    """One-based average ranks, with exact ties kept together."""
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(x, y):
    if len(x) < 2:
        return None
    a, b = tied_ranks(x), tied_ranks(y)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else None


def _interval(draws, replicates):
    valid = [x for x in draws if x is not None and np.isfinite(x)]
    # Do not silently drop degenerate resamples to obtain a passing interval.
    return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) == replicates else None,
            "valid_replicates": len(valid), "degenerate_replicates": replicates - len(valid)}


def _association(sd, error, uid, replicates, seed):
    users, inverse = np.unique(uid, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(users))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.integers(len(users), size=len(users))])
        draws.append(spearman(sd[indices], error[indices]))
    value = spearman(sd, error)
    return {"n": len(sd), "users": len(users), "spearman": value,
            "degenerate": value is None, "insufficient_users": len(users) < 100,
            **_interval(draws, replicates)}


def audit_ensemble(arrays, *, bootstrap=2000, seed=2026):
    """Audit explicit observed labels; return JSON-safe diagnostics, never a policy."""
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 2:
        raise ValueError("bootstrap must be an integer >=2")
    data = validate_export(arrays)
    p, y, uid = data["predictions"], data["labels"], data["uid"]
    sd = p.std(axis=1, ddof=0)
    error = np.abs(p.mean(axis=1) - y)
    association = _association(sd, error, uid, bootstrap, seed)
    k = max(1, math.ceil(len(sd) * .1))
    low_cut, high_cut = np.sort(sd)[k - 1], np.sort(sd)[-k]
    low, high = sd <= low_cut, sd >= high_cut
    lu, hu = np.unique(uid[low]), np.unique(uid[high])
    common = np.intersect1d(lu, hu)
    differences, low_errors, high_errors = [], [], []
    for user in common:
        lo, hi = error[low & (uid == user)].mean(), error[high & (uid == user)].mean()
        low_errors.append(lo)
        high_errors.append(hi)
        differences.append(hi - lo)
    differences = np.asarray(differences)
    rng = np.random.default_rng(seed)
    draws = [float(differences[rng.integers(len(common), size=len(common))].mean())
             for _ in range(bootstrap)] if len(common) else []
    overlap = bool(np.any(low & high))
    contrast = {"low_cutoff": float(low_cut), "high_cutoff": float(high_cut),
                "requested_decile_n": k, "low_n": int(low.sum()), "high_n": int(high.sum()),
                "low_users": len(lu), "high_users": len(hu), "common_users": len(common),
                "low_only_users": len(np.setdiff1d(lu, hu)), "high_only_users": len(np.setdiff1d(hu, lu)),
                "neither_users": len(np.setdiff1d(np.unique(uid), np.union1d(lu, hu))),
                "common_low_n": int((low & np.isin(uid, common)).sum()),
                "common_high_n": int((high & np.isin(uid, common)).sum()),
                "low_mae": float(np.mean(low_errors)) if len(common) else None,
                "high_mae": float(np.mean(high_errors)) if len(common) else None,
                "high_minus_low_mae": float(differences.mean()) if len(common) else None,
                "degenerate": overlap or not len(common), "overlapping_groups": overlap,
                "insufficient_users": len(common) < 100, **_interval(draws, bootstrap)}
    reasons = []
    for name, stat in (("association", association), ("decile_contrast", contrast)):
        if stat["degenerate"]:
            reasons.append(f"{name}:degenerate")
        if stat["insufficient_users"]:
            reasons.append(f"{name}:insufficient_users")
        if stat["ci95"] is None or stat["ci95"][0] <= 0:
            reasons.append(f"{name}:lower_bound_not_positive_or_unavailable")
    slices = {}
    for name, (values, metadata) in data["slices"].items():
        slices[name] = {"metadata": metadata, "groups": {
            str(value): _association(sd[values == value], error[values == value],
                                     uid[values == value], bootstrap, seed)
            for value in np.unique(values)}}
    return {"schema_version": 2, "provenance": data["provenance"],
            "input_sha256": None, "content_sha256": content_digest(arrays),
            "content_digest_format": "sorted fields; length-prefixed JSON [name,dtype.str,shape] and C-order bytes; SHA256 v1",
            "n": len(y), "members": p.shape[1], "users": len(np.unique(uid)),
            "early_examples": data["early_examples"],
            "calibration": {"status": data["calibration_status"], "method": "identity (no transformation)",
                            "fitted": False, "calibrated_claim": False},
            "bootstrap": {"replicates": bootstrap, "seed": seed, "unit": "user", "ci": "percentile 95%"},
            "estimands": {"association": "pooled-row tied-rank Spearman; paired user-cluster resampling; ranks recomputed",
                          "contrast": "user-macro high-minus-low MAE on common users only; paired user resampling",
                          "disagreement": "population member SD (ddof=0)",
                          "deciles": "ceil(0.1*n), include all boundary ties; fixed groups in bootstrap; overlapping groups fail"},
            "association": association, "decile_contrast": contrast, "slices": slices,
            "gate": {"passed": not reasons, "reasons": reasons, "minimum_common_users": 100},
            "limitations": ["Predictive association, not causal or epistemic proof; shared confident error remains possible.",
                            "Declared provenance is checked for consistency, not independently authenticated.",
                            "No calibration is fitted; no ensemble training, policy optimization or B launch is implemented.",
                            "Slice associations are descriptive/exploratory, not additional gates."]}


def audit_npz(path, *, bootstrap=2000, seed=2026):
    # Load and hash the same bytes, rather than reopening a possibly changed file.
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        report = audit_ensemble({key: archive[key] for key in archive.files}, bootstrap=bootstrap, seed=seed)
    report["input_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    report = audit_npz(args.input, bootstrap=args.bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
