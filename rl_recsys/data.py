"""Point-in-time preparation; no arbitrary order inside timestamp ties."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq


COLUMNS = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"]
_INTEGER_COLUMNS = ("uid", "item_id", "timestamp")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(config, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = config["data"]["sha256"]
    if output.exists():
        if sha256(output) != expected:
            raise ValueError("Existing dataset hash mismatch; refusing overwrite")
        return
    url = f'https://huggingface.co/datasets/yandex/yambda/resolve/{config["data"]["revision"]}/flat/50m/listens.parquet'
    temp = output.with_suffix(".partial")
    urllib.request.urlretrieve(url, temp)
    if sha256(temp) != expected:
        raise ValueError("Downloaded dataset hash mismatch")
    temp.rename(output)


def uid_key(uid, seed):
    return hashlib.sha256(f"{seed}:{int(uid)}".encode()).digest()


def read_sample(path, users, seed):
    """Two passes; selection depends on UID only, not future event counts."""
    pf = pq.ParquetFile(path)
    ids = set()
    for batch in pf.iter_batches(columns=["uid"], batch_size=262144):
        ids.update(np.unique(batch.column(0).to_numpy()).tolist())
    selected = set(sorted(ids, key=lambda x: uid_key(x, seed))[:users])
    parts = {c: [] for c in COLUMNS}
    for batch in pf.iter_batches(columns=COLUMNS, batch_size=262144):
        a = batch.column(0).to_numpy()
        keep = np.isin(a, list(selected))
        if keep.any():
            for i, c in enumerate(COLUMNS):
                parts[c].append(batch.column(i).to_numpy(zero_copy_only=False)[keep])
    if not parts["uid"]:
        raise ValueError("Empty sample")
    return {c: np.concatenate(parts[c]) for c in COLUMNS}, sorted(selected)


def _bounded_indices(y, limit, seed):
    # Uniform deterministic subsampling within role; chosen before observing outcomes.
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(y), min(len(y), limit), replace=False))


def prepare_arrays(raw, cfg):
    """Roles: fit < t0, RM-calibration [t0,t1), selection [t1,t2), test >=t2.

    Singleton user-timestamp groups define unambiguous targets/history tokens.
    Ambiguous groups are excluded entirely, without manufacturing an ordering.
    Exact duplicate rows are removed first; counts are reported.
    """
    dc = cfg["data"]
    from .config import validate_temporal_config
    validate_temporal_config(dc)
    arrays = {k: np.asarray(raw[k]) for k in COLUMNS}
    n = len(arrays["uid"])
    if not n or any(len(a) != n for a in arrays.values()):
        raise ValueError("Empty or inconsistent columns")
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Missing/nonfinite input: audit before proceeding")
    # Reject float IDs even when integral: precision may already have been
    # lost before preparation. Outputs use nonnegative signed int64 IDs/times.
    for name in _INTEGER_COLUMNS:
        values = arrays[name]
        if values.dtype.kind not in "iu":
            raise ValueError(f"Expected integer dtype for {name}")
        if int(values.min()) < 0 or int(values.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"Out-of-range integer values in {name}")
        arrays[name] = values.astype(np.int64, copy=False)
    if np.any(arrays["played_ratio_pct"] < 0) or np.any(arrays["track_length_seconds"] <= 0):
        raise ValueError("Invalid playback values; do not silently clean")
    if not np.isin(arrays["is_organic"], [0, 1]).all():
        raise ValueError("Invalid organic flag")
    row_dtype = np.dtype([(name, arrays[name].dtype) for name in COLUMNS])
    rows = np.empty(n, dtype=row_dtype)
    for name in COLUMNS:
        rows[name] = arrays[name]
    _, unique_idx = np.unique(rows, return_index=True)
    unique_idx.sort()
    arrays = {c: a[unique_idx] for c, a in arrays.items()}
    order = np.lexsort((arrays["timestamp"], arrays["uid"]))
    arrays = {c: a[order] for c, a in arrays.items()}
    uid, ts = arrays["uid"], arrays["timestamp"]
    previous = np.r_[False, (uid[1:] == uid[:-1]) & (ts[1:] == ts[:-1])]
    following = np.r_[(uid[1:] == uid[:-1]) & (ts[1:] == ts[:-1]), False]
    singleton = ~(previous | following)
    ambiguous = ~singleton
    ambiguous_group_starts = (~previous) & following
    ambiguous_group_count = int(ambiguous_group_starts.sum())
    ambiguous_user_count = int(np.unique(uid[ambiguous]).size)
    # Calendar cutoffs are frozen in the protocol and must not be estimated from
    # the complete (future-inclusive) sample. Quantiles are retained only for
    # direct callers/legacy configs and are explicitly marked in the audit.
    if dc.get("cutoffs") is not None:
        cutoffs = np.asarray(dc["cutoffs"], dtype=np.int64)
        cutoff_source = "frozen_calendar_config"
    else:
        if not dc.get("allow_derived_cutoffs", False):
            raise ValueError("data.cutoffs is required; legacy quantiles need allow_derived_cutoffs=true")
        cutoffs = np.quantile(ts, dc["quantiles"], method="nearest").astype(np.int64)
        cutoff_source = "legacy_sample_quantiles_future_inclusive"
    if len(cutoffs) != 3 or not np.all(np.diff(cutoffs) > 0):
        raise ValueError("Cutoffs must contain three strictly increasing timestamps")
    train_mask = (ts < cutoffs[0]) & singleton
    items, counts = np.unique(arrays["item_id"][train_mask], return_counts=True)
    if not len(items):
        raise ValueError("No training items")
    ranked = np.lexsort((items, -counts))[:dc["catalog_size"]]
    vocab = np.sort(items[ranked])
    # IDs 0=PAD, 1=unknown; eligible IDs start at 2. No future token vocabulary.
    mapping = {int(item): i + 2 for i, item in enumerate(vocab)}
    tokens = np.array([mapping.get(int(x), 1) for x in arrays["item_id"]], dtype=np.int32)
    pop = np.zeros(len(vocab), dtype=np.float64)
    duration = np.zeros(len(vocab), dtype=np.float64)
    known_train = train_mask & (tokens >= 2)
    pop[:] = np.bincount(tokens[known_train] - 2, minlength=len(vocab))
    train_order = np.argsort(tokens[known_train], kind="stable")
    sorted_tokens = tokens[known_train][train_order]
    sorted_duration = arrays["track_length_seconds"][known_train][train_order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_tokens)) + 1]
    ends = np.r_[starts[1:], len(sorted_tokens)]
    for start, end in zip(starts, ends):
        duration[sorted_tokens[start] - 2] = np.median(sorted_duration[start:end])
    roles = {r: {k: [] for k in ["context", "target", "uid", "timestamp", "reward", "seconds", "organic"]}
             for r in ["fit", "calibration", "selection", "test"]}
    roles_names = list(roles)
    maxlen = dc["context_length"]
    # Pick rows before constructing contexts, keeping memory bounded by configured caps.
    prior_history = np.zeros(len(uid), dtype=bool)
    seen = False
    for i in range(len(uid)):
        if i == 0 or uid[i] != uid[i - 1]:
            seen = False
        prior_history[i] = seen
        seen = seen or bool(singleton[i])
    row_roles = np.searchsorted(cutoffs, ts, side="right")
    retained = np.zeros(len(uid), dtype=bool)
    pre_stats = {}
    for offset, name in enumerate(roles_names):
        candidates = np.flatnonzero(singleton & prior_history & (row_roles == offset))
        limit = dc["max_train"] if name == "fit" else dc["max_test"] if name == "test" else dc["max_valid"]
        chosen = candidates[_bounded_indices(candidates, limit, cfg.get("sample_seed", cfg["seed"]) + offset)]
        retained[chosen] = True
        pre_stats[name] = {"total": len(candidates), "eligible": int((tokens[candidates] >= 2).sum()),
                           "users": len(np.unique(uid[candidates])), "retained": len(chosen)}
    history = []
    last_uid = None
    for i in range(len(uid)):
        if uid[i] != last_uid:
            history = []
            last_uid = uid[i]
        if not singleton[i]:
            continue
        if history and retained[i]:
            role = roles[roles_names[row_roles[i]]]
            ctx = [0] * max(0, maxlen - len(history)) + history[-maxlen:]
            values = {"context": ctx, "target": tokens[i], "uid": uid[i], "timestamp": ts[i],
                      "reward": min(arrays["played_ratio_pct"][i] / 100., 1.),
                      "seconds": min(arrays["played_ratio_pct"][i] / 100. * arrays["track_length_seconds"][i],
                                     cfg["rl"]["seconds_cap"]) / cfg["rl"]["seconds_cap"],
                      "organic": arrays["is_organic"][i]}
            for k, v in values.items():
                role[k].append(v)
        history.append(int(tokens[i]))
    result = {}
    role_stats = {}
    for offset, (name, role) in enumerate(roles.items()):
        role = {k: np.asarray(v, dtype=np.float32 if k in ["reward", "seconds"] else np.int64)
                for k, v in role.items()}
        if not len(role["target"]):
            raise ValueError(f"No examples for {name}")
        role_stats[name] = pre_stats[name]
        for k, v in role.items():
            result[f"{name}_{k}"] = v
    result.update(vocab=vocab, popularity=pop, durations=duration, cutoffs=cutoffs)
    audit = {"input_rows": n, "exact_duplicates_removed": n - len(uid),
             "ambiguous_rows_excluded": int(ambiguous.sum()),
             "ambiguous_timestamp_groups": ambiguous_group_count,
             "ambiguous_users": ambiguous_user_count,
             "pct_above_100": int((arrays["played_ratio_pct"] > 100).sum()),
             "roles": role_stats, "cutoffs": cutoffs.tolist(), "cutoff_source": cutoff_source,
             "audit_metadata": {"cutoff_source": cutoff_source,
                                "future_information_used_for_cutoff_formation": cutoff_source != "frozen_calendar_config",
                                "timestamp_unit": "dataset timestamp integer; boundaries are inclusive on lower, exclusive on upper role intervals"},
             "catalog": len(vocab),
             "history_policy": "strictly earlier singleton timestamp bins; repeats allowed; rolling observed history",
             "duration_policy": "train-only per-item median duration; seconds reward is proxy, not wall-clock"}
    return result, audit


def prepare(path, config, output):
    from .config import validate_config
    config = validate_config(config)
    if sha256(path) != config["data"]["sha256"]:
        raise ValueError("Dataset SHA256 differs from protocol")
    raw, ids = read_sample(path, config["data"]["users"], config.get("sample_seed", config["seed"]))
    arrays, audit = prepare_arrays(raw, config)
    audit.update(dataset_sha256=sha256(path), selected_uids=ids, config=config)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "prepared.npz", **arrays)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def load_prepared(path):
    with np.load(Path(path) / "prepared.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def role(data, name, eligible=False):
    result = {k[len(name) + 1:]: v for k, v in data.items() if k.startswith(name + "_")}
    if eligible:
        mask = result["target"] >= 2
        result = {k: v[mask] for k, v in result.items()}
    return result
