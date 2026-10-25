import json

import numpy as np
import pytest

from rl_recsys.ensemble_audit import audit_ensemble, audit_npz, main, spearman, tied_ranks


def exported(users=120):
    n = users * 10
    spread = np.tile(np.linspace(.01, .2, 10), users)
    return dict(
        predictions=np.column_stack((.5 - spread, np.full(n, .5), .5 + spread)),
        labels=.5 - spread, uid=np.repeat(np.arange(users), 10),
        example_id=np.array([f"late-{i}" for i in range(n)]), timestamp=np.arange(n) + 100,
        early_example_id=np.array(["early-1", "early-2"]), early_timestamp=np.array([1, 2]),
        data_id=np.array("data-sha"), catalog_id=np.array("catalog-sha"),
        source_id=np.array("code-sha"), split_id=np.array("split-sha"),
        member_data_id=np.array(["data-sha"] * 3), member_catalog_id=np.array(["catalog-sha"] * 3),
        member_source_id=np.array(["code-sha"] * 3), member_split_id=np.array(["split-sha"] * 3),
        member_id=np.array(["rm1", "rm2", "rm3"]), member_seed=np.array([42, 43, 44]),
        member_fit_role=np.array(["fit"] * 3), member_checkpoint_role=np.array(["early_calibration"] * 3),
        audit_role=np.array("late_calibration"), label_source=np.array("observed"),
        calibration_status=np.array("identity"),
    )


def audit(a, **kwargs):
    return audit_ensemble(a, bootstrap=30, **kwargs)


def test_tied_rank_spearman():
    np.testing.assert_equal(tied_ranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
    assert spearman([1, 1, 2], [3, 3, 1]) == pytest.approx(-1)
    assert spearman([1, 1], [1, 2]) is None


def test_pass_deterministic_and_json_safe():
    a = exported()
    first = audit(a, seed=91)
    assert first == audit(a, seed=91)
    json.dumps(first, allow_nan=False)
    assert first["gate"]["passed"]
    assert first["association"]["ci95"][0] > 0
    contrast = first["decile_contrast"]
    assert contrast["common_users"] == 120
    assert contrast["low_n"] == contrast["high_n"] == 120
    assert contrast["high_minus_low_mae"] == pytest.approx(.19)
    assert not first["calibration"]["calibrated_claim"]


def test_degenerate_constant_sd_and_error():
    a = exported()
    a["predictions"][:] = .5
    a["labels"][:] = .5
    r = audit(a)
    assert not r["gate"]["passed"]
    assert r["association"]["degenerate"]
    assert r["association"]["ci95"] is None
    assert r["decile_contrast"]["overlapping_groups"]


def test_insufficient_even_with_positive_bounds():
    r = audit(exported(99))
    assert r["decile_contrast"]["ci95"][0] > 0
    assert r["decile_contrast"]["common_users"] == 99
    assert not r["gate"]["passed"]
    assert audit(exported(100))["gate"]["passed"]


def test_common_users_only_and_missing_groups():
    a = exported()
    # First user's low error must not contribute after losing their high group.
    a["uid"][9] = 999
    a["labels"][0] = 0
    r = audit(a)
    c = r["decile_contrast"]
    assert c["common_users"] == 119
    assert c["low_only_users"] == c["high_only_users"] == 1
    assert c["high_minus_low_mae"] == pytest.approx(.19)
    a["uid"] = np.arange(len(a["uid"]))
    c = audit(a)["decile_contrast"]
    assert c["common_users"] == 0
    assert c["ci95"] is None and c["degenerate"]


def test_negative_association_blocks_gate():
    a = exported()
    a["labels"] = .5 - np.tile(np.linspace(.2, .01, 10), 120)
    r = audit(a)
    assert r["association"]["ci95"][1] < 0
    assert r["decile_contrast"]["ci95"][1] < 0
    assert not r["gate"]["passed"]


@pytest.mark.parametrize("field,value", [
    ("member_data_id", ["wrong"] * 3), ("member_catalog_id", ["wrong"] * 3),
    ("member_source_id", ["wrong"] * 3), ("member_split_id", ["wrong"] * 3),
    ("member_seed", [42, 42, 44]), ("member_id", ["same"] * 3),
    ("member_fit_role", ["late_calibration"] * 3),
    ("member_checkpoint_role", ["late_calibration"] * 3),
    ("calibration_status", "isotonic"), ("label_source", "synthetic"),
    ("audit_role", "test"), ("early_timestamp", [1, 100]),
    ("early_example_id", ["late-1", "early-2"]),
])
def test_metadata_mismatch(field, value):
    a = exported()
    a[field] = np.asarray(value)
    with pytest.raises(ValueError):
        audit(a)


@pytest.mark.parametrize("field", ["predictions", "labels"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_bad_scores(field, bad):
    a = exported()
    a[field].flat[0] = bad
    with pytest.raises(ValueError):
        audit(a)


def test_missing_member_shape_object_and_labels():
    for modify in (
        lambda a: a.pop("labels"),
        lambda a: a.update(predictions=a["predictions"][:, :2]),
        lambda a: a.update(uid=np.zeros(2)),
        lambda a: a.update(extra=np.array([{}], dtype=object)),
    ):
        a = exported()
        modify(a)
        with pytest.raises(ValueError):
            audit(a)


def test_slice_metadata_and_associations():
    a = exported()
    a["slice__duration"] = np.tile(["short"] * 5 + ["long"] * 5, 120)
    with pytest.raises(ValueError):
        audit(a)
    a["slice_metadata_json"] = np.array(json.dumps({"duration": {
        "derived_from": "fit", "data_id": "data-sha", "catalog_id": "catalog-sha",
        "method": "fit per-item median; fixed 180s boundary"}}))
    r = audit(a)
    assert r["slices"]["duration"]["groups"]["short"]["n"] == 600
    assert r["slices"]["duration"]["groups"]["long"]["spearman"] > .99
    a["slice_metadata_json"] = np.array(str(a["slice_metadata_json"]).replace('"fit"', '"late"'))
    with pytest.raises(ValueError):
        audit(a)


@pytest.mark.parametrize('field', ['predictions', 'labels', 'uid', 'example_id',
    'timestamp', 'early_timestamp', 'early_example_id', 'member_seed', 'member_id',
    'dtype', 'shape', 'slice_metadata'])
def test_content_digest_binds_validated_inputs(field):
    a = exported(2)
    before = audit(a)['content_sha256']
    if field == 'dtype':
        a['predictions'] = a['predictions'].astype('float32')
    elif field == 'shape':
        for key in ['predictions', 'labels', 'uid', 'example_id', 'timestamp']:
            a[key] = a[key][:-1]
    elif field == 'slice_metadata':
        a['slice__duration'] = np.full(20, 'short')
        a['slice_metadata_json'] = np.array(json.dumps({'duration': {
            'derived_from': 'fit', 'data_id': 'data-sha', 'catalog_id': 'catalog-sha',
            'method': 'fixed threshold'}}))
    elif field == 'member_seed':
        a[field][0] += 10
    elif a[field].dtype.kind in 'US':
        a[field][0] = 'new-id'
    else:
        a[field].flat[0] += .001 if a[field].dtype.kind == 'f' else 1
    assert audit(a)['content_sha256'] != before
    assert audit(dict(reversed(list(a.items()))))['content_sha256'] == audit(a)['content_sha256']


def test_npz_cli_and_pickle_rejection(tmp_path):
    source, output = tmp_path / "observed.npz", tmp_path / "report.json"
    np.savez(source, **exported())
    main([str(source), "--output", str(output), "--bootstrap", "20"])
    report = json.loads(output.read_text())
    assert report['gate']['passed']
    import hashlib
    assert report['input_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report['content_sha256'] == audit(exported())['content_sha256']
    # Same arrays in a different ZIP representation: content stable, file hash changes.
    np.savez_compressed(source, **exported())
    compressed = audit_npz(source, bootstrap=20)
    assert compressed['content_sha256'] == report['content_sha256']
    assert compressed['input_sha256'] != report['input_sha256']
    np.savez(source, malicious=np.array([{}], dtype=object))
    with pytest.raises(ValueError):
        audit_npz(source, bootstrap=20)


def test_cluster_bootstrap_matches_explicit_user_resampling():
    a = exported(12)
    rng = np.random.default_rng(123)
    a["labels"] = rng.uniform(.1, .9, len(a["labels"]))
    r = audit(a, seed=72)
    sd = a["predictions"].std(axis=1)
    error = np.abs(a["predictions"].mean(axis=1) - a["labels"])
    rng = np.random.default_rng(72)
    correlations, differences = [], []
    # Identity validation normalizes integer IDs to strings; use its stable order.
    users = np.unique(a["uid"].astype(str)).astype(int)
    for _ in range(30):
        sampled = users[rng.integers(12, size=12)]
        indices = np.concatenate([np.arange(user * 10, (user + 1) * 10) for user in sampled])
        correlations.append(spearman(sd[indices], error[indices]))
        differences.append(np.mean([error[user * 10 + 9] - error[user * 10] for user in sampled]))
    np.testing.assert_allclose(r["association"]["ci95"], np.quantile(correlations, [.025, .975]))
    np.testing.assert_allclose(r["decile_contrast"]["ci95"], np.quantile(differences, [.025, .975]))
    assert r == audit(a, seed=72)
    assert r["association"]["ci95"] != audit(a, seed=73)["association"]["ci95"]


def test_boundary_ties_expand_deciles():
    a = exported()
    a["predictions"][1::10] = a["predictions"][0]
    c = audit(a)["decile_contrast"]
    assert c["requested_decile_n"] == 120
    assert c["low_n"] == c["common_low_n"] == 240
    assert c["high_n"] == 120
    assert not c["overlapping_groups"]


def test_degenerate_resamples_not_dropped():
    a = exported(1)
    a["uid"][1:] = 2
    # One user has a single row; resampling only that user gives undefined ranks.
    r = audit(a)
    assert r["association"]["degenerate_replicates"] > 0
    assert r["association"]["ci95"] is None
    assert not r["gate"]["passed"]
