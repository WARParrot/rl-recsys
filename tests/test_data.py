import numpy as np
import pytest
from rl_recsys.data import prepare_arrays


def cfg():
    return {"seed": 42, "data": {"users": 10, "catalog_size": 20, "context_length": 3,
            "quantiles": [.6, .75, .9], "allow_derived_cutoffs": True,
            "max_train": 100, "max_valid": 100, "max_test": 100},
            "rl": {"seconds_cap": 600}}


def raw():
    # Every target has a strictly earlier singleton context; one same-timestamp pair is excluded.
    uid = np.repeat([1, 2, 3, 4], 5)
    ts = np.tile([1, 2, 3, 4, 5], 4)
    item = np.arange(20)
    return {"uid": uid, "item_id": item, "timestamp": ts, "is_organic": np.zeros(20),
            "played_ratio_pct": np.full(20, 80.), "track_length_seconds": np.full(20, 100.)}


def test_preparation_is_deterministic_and_strict():
    a, audit = prepare_arrays(raw(), cfg())
    b, audit2 = prepare_arrays(raw(), cfg())
    for key in a:
        if isinstance(a[key], np.ndarray): np.testing.assert_array_equal(a[key], b[key])
    assert audit == audit2
    assert audit["ambiguous_rows_excluded"] == 0
    assert audit["ambiguous_timestamp_groups"] == 0
    assert audit["ambiguous_users"] == 0
    assert np.all(a["fit_context"][:, -1] != 0)
    assert audit["cutoff_source"] == "legacy_sample_quantiles_future_inclusive"
    assert audit["audit_metadata"]["future_information_used_for_cutoff_formation"] is True


def test_frozen_cutoffs_are_used_without_future_quantile_formation():
    x = raw()
    c = cfg()
    c["data"]["cutoffs"] = [3, 4, 5]
    c["data"]["allow_derived_cutoffs"] = False
    prepared, audit = prepare_arrays(x, c)
    assert audit["cutoffs"] == [3, 4, 5]
    # Alter the future volume/range without changing any earlier observations.
    future = {name: values[-4:].copy() for name, values in x.items()}
    future["timestamp"][:] = [100, 101, 102, 103]
    extended = {name: np.concatenate([values, future[name]]) for name, values in x.items()}
    changed, _ = prepare_arrays(extended, c)
    for key in prepared:
        if key.startswith(("fit_", "calibration_", "selection_")) or key in ("vocab", "popularity", "durations", "cutoffs"):
            np.testing.assert_array_equal(prepared[key], changed[key])
    assert np.all(prepared["calibration_timestamp"] == 3)
    assert np.all(prepared["selection_timestamp"] == 4)
    assert np.all(prepared["test_timestamp"] == 5)
    assert audit["cutoff_source"] == "frozen_calendar_config"
    assert audit["audit_metadata"]["future_information_used_for_cutoff_formation"] is False


def test_same_timestamp_rows_excluded_and_audited():
    x = raw()
    x["timestamp"][1] = x["timestamp"][0]
    a, audit = prepare_arrays(x, cfg())
    assert audit["ambiguous_rows_excluded"] == 2
    assert audit["ambiguous_timestamp_groups"] == 1
    assert audit["ambiguous_users"] == 1
    assert 0 not in a["vocab"] and 1 not in a["vocab"]


def test_exact_dedup_preserves_large_integer_ids():
    x = raw()
    x["item_id"][:2] = [2**53, 2**53 + 1]
    x["timestamp"][1] = x["timestamp"][0]
    duplicate = {name: values[:1].copy() for name, values in x.items()}
    x = {name: np.concatenate([values, duplicate[name]]) for name, values in x.items()}

    _, audit = prepare_arrays(x, cfg())
    assert audit["exact_duplicates_removed"] == 1
    assert audit["ambiguous_rows_excluded"] == 2
    assert audit["ambiguous_timestamp_groups"] == 1

    # Large user IDs with otherwise identical rows must also remain distinct.
    x = raw()
    x["uid"] = np.repeat([2**53, 2**53 + 1, 3, 4], 5)
    x["item_id"][5:10] = x["item_id"][:5]
    prepared, audit = prepare_arrays(x, cfg())
    assert audit["exact_duplicates_removed"] == 0
    assert {2**53, 2**53 + 1}.issubset(set(prepared["fit_uid"].tolist()))


@pytest.mark.parametrize("column", ["uid", "item_id", "timestamp"])
@pytest.mark.parametrize("invalid", [1.0, 1.5, -1, 2**63])
def test_invalid_integer_column_is_rejected(column, invalid):
    x = raw()
    dtype = float if isinstance(invalid, float) else np.uint64 if invalid == 2**63 else np.int64
    x[column] = x[column].astype(dtype)
    x[column][0] = invalid
    with pytest.raises(ValueError, match="integer"):
        prepare_arrays(x, cfg())


def test_multiple_tie_groups_count_users_once_after_dedup():
    x = raw()
    x["timestamp"][[1, 3, 6]] = x["timestamp"][[0, 2, 5]]
    x = {name: np.concatenate([values, values[:1]]) for name, values in x.items()}
    _, audit = prepare_arrays(x, cfg())
    assert audit["exact_duplicates_removed"] == 1
    assert audit["ambiguous_rows_excluded"] == 6
    assert audit["ambiguous_timestamp_groups"] == 3
    assert audit["ambiguous_users"] == 2
