import numpy as np
from rl_recsys.data import prepare_arrays


def cfg():
    return {"seed": 42, "data": {"users": 10, "catalog_size": 20, "context_length": 3,
            "quantiles": [.6, .75, .9], "max_train": 100, "max_valid": 100, "max_test": 100},
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
    assert np.all(a["fit_context"][:, -1] != 0)


def test_same_timestamp_rows_excluded():
    x = raw()
    x["timestamp"][1] = x["timestamp"][0]
    a, audit = prepare_arrays(x, cfg())
    assert audit["ambiguous_rows_excluded"] == 2
    assert len(a["fit_target"]) >= 0
