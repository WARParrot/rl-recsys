"""Tiny actual CPU inference with hash-bound follow-up fixture; no training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl_recsys import followup_evaluate as fe
from rl_recsys.data import sha256
from rl_recsys.followup import ItemOnlyRewardModel
from rl_recsys.models import Policy, RewardModel


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen_followup(tmp_path, monkeypatch):
    cfg = json.loads(Path("configs/pilot.json").read_text())
    cfg["model"] = {"hidden": 8, "layers": 1, "heads": 2, "dropout": 0.}
    cfg["data"]["context_length"] = 3
    cfg["train"].update(allow_cpu=True, threads=1)
    cfg["evaluation"].update(k=2, bootstrap=10, action_samples=2)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    prepared, run, root = [tmp_path / name for name in ["data", "historical", "followup"]]
    for directory in [prepared, run, root]:
        directory.mkdir()
    data = {"vocab": np.array([10, 20, 30]), "popularity": np.array([5., 3., 1.]),
            "durations": np.array([100., 200., 300.])}
    for name in ["fit", "selection", "test"]:
        targets = np.array([2, 3, 1, 4, 2, 3]) if name != "fit" else np.array([2, 3, 4, 4, 2, 3])
        fields = {"context": np.array([[0, 2, 3], [0, 3, 4], [0, 2, 4]] * 2),
                  "target": targets, "uid": np.repeat([101, 202], 3),
                  "timestamp": np.arange(6), "reward": np.linspace(.1, .9, 6).astype("float32"),
                  "seconds": np.linspace(.1, .5, 6).astype("float32"), "organic": np.zeros(6, dtype=int)}
        data.update({f"{name}_{key}": value for key, value in fields.items()})
    np.savez(prepared / "prepared.npz", **data)
    write_json(prepared / "audit.json", {})
    write_json(run / "provenance.json", {"config": cfg})
    args = (5, 3, cfg["model"])
    torch.save(Policy(*args).state_dict(), run / "sft.pt")
    identity = {"prepared_sha256": sha256(prepared / "prepared.npz"),
                "historical_provenance_sha256": sha256(run / "provenance.json"),
                "checkpoint_hashes": {"sft.pt": sha256(run / "sft.pt")}}
    # Historical authentication has its own suite; still bind these exact fixture bytes.
    def verify(run_dir, data_dir, **kwargs):
        assert kwargs == {"training_only": True}
        assert run_dir == run and data_dir == prepared
        assert sha256(run / "sft.pt") == identity["checkpoint_hashes"]["sft.pt"]
        assert sha256(prepared / "prepared.npz") == identity["prepared_sha256"]
        return identity
    monkeypatch.setattr(fe, "verify_inputs", verify)
    monkeypatch.setattr(fe.exp, "get_device", lambda config: torch.device("cpu"))
    # Avoid installed-package/git inspection; real inference and checkpoint loading remain intact.
    monkeypatch.setattr(fe.exp, "provenance", lambda *a: {"fixture": "CPU inference", "config": cfg})
    prov = {"config": cfg, "seeds": [42, 43], "repeat_penalty": .1,
            "code_hashes": {p.name: sha256(p) for p in Path(fe.exp.__file__).parent.glob('*.py')},
            "prepared_sha256": identity["prepared_sha256"],
            "historical_provenance_sha256": identity["historical_provenance_sha256"]}
    write_json(root / "provenance.json", prov)
    manifest = {"status": "follow-up complete; test not evaluated", "seeds": {},
                "provenance_sha256": sha256(root / "provenance.json")}
    selected_names = ["continued_sft", "completion_h1"]
    for seed in [42, 43]:
        directory = root / f"seed_{seed}"
        directory.mkdir()
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed
        write_json(directory / "config.json", seed_cfg)
        torch.save(RewardModel(*args).state_dict(), directory / "reward.pt")
        torch.save(ItemOnlyRewardModel(*args).state_dict(), directory / "item_only.pt")
        gate = {"status": "passed" if seed == 42 else "failed",
                "heads": {"completion": {"enabled": seed == 42}},
                "enabled_rewards": ["completion"] if seed == 42 else []}
        write_json(directory / "reward_audit.json", {"gate": gate})
        policies = {}
        if seed == 42:
            for name in selected_names:
                torch.save(Policy(*args).state_dict(), directory / f"{name}.pt")
                policies[name] = {"selected_checkpoint": f"{name}.pt"}
            policies["completion_minus_repeat_h1"] = {"selected_checkpoint": None}
        write_json(directory / "selection.json", {"policies": policies})
        manifest["seeds"][str(seed)] = {"policies": policies,
            "status": "training complete" if seed == 42 else "reward gate failed; no policy training",
            "checkpoint_hashes": {p.name: sha256(p) for p in directory.glob("*.pt")},
            "artifact_hashes": {p.name: sha256(p) for p in directory.glob("*.json")}}
    write_json(root / "followup_manifest.json", manifest)
    write_json(root / "FOLLOWUP_COMPLETE.json", {"status": "complete", "manifest_sha256": sha256(root / "followup_manifest.json"), "provenance_sha256": sha256(root / "provenance.json")})
    return prepared, run, root, tmp_path / "evaluated", data, manifest


def test_source_mismatch_before_any_test_access(frozen_followup, monkeypatch):
    prepared, run, root, out, _, manifest = frozen_followup
    prov = json.loads((root / 'provenance.json').read_text())
    prov['code_hashes']['followup.py'] = 'changed-source'
    write_json(root / 'provenance.json', prov)
    manifest['provenance_sha256'] = sha256(root / 'provenance.json')
    write_json(root / 'followup_manifest.json', manifest)
    write_json(root / 'FOLLOWUP_COMPLETE.json', {
        'status': 'complete', 'manifest_sha256': sha256(root / 'followup_manifest.json'),
        'provenance_sha256': sha256(root / 'provenance.json')})
    monkeypatch.setattr(fe, 'load_prepared', lambda *a: pytest.fail('test data access'))
    monkeypatch.setattr(fe, 'verify_inputs', lambda *a, **kw: pytest.fail('parent access'))
    with pytest.raises(ValueError, match='Source code differs'):
        fe.evaluate_followup(prepared, run, root, out)
    assert not out.exists()


def test_real_cpu_inference_selected_arms_masks_arrays_and_failed_seed(frozen_followup):
    prepared, run, root, out, data, manifest = frozen_followup
    result = fe.evaluate_followup(prepared, run, root, out, batch_size=2, max_minutes=2)
    assert result["evaluation_status"] == "exploratory_reuse"
    successful, failed = [result["seeds"][seed] for seed in ["42", "43"]]
    assert set(successful["policies"]) == {"sft", "continued_sft", "completion_h1"}
    assert successful["not_selected"] == ["completion_minus_repeat_h1"]
    assert failed["gate"]["status"] == "failed"
    assert set(failed["policies"]) == {"sft"}
    for seed in ["42", "43"]:
        baseline = result["seeds"][seed]["policies"]["sft"]
        assert baseline["all_target_examples"] == 6
        assert baseline["eligible_target_examples"] == 5
        assert result["seeds"][seed]["reward_model"]["heads"]["completion"]["examples"] == 5
    for arm in ["continued_sft", "completion_h1"]:
        paired = successful["policies"][arm]["paired_vs_sft"]
        assert set(paired) == {"ndcg", "exact_one_step_expected_reward"}
        assert paired["ndcg"]["users"] == 2
        aggregate = result["conditional_seed_effects"][arm]["ndcg"]
        assert aggregate["available_seeds"] == 1
        assert aggregate["per_seed"][0]["seed"] == 42
        assert aggregate["seed_sd"] is None
    assert result["conditional_seed_effects"]["completion_minus_repeat_h1"]["ndcg"]["available_seeds"] == 0
    with np.load(out / "evaluation-inputs.npz", allow_pickle=False) as inputs:
        for name in ["selection", "test"]:
            np.testing.assert_array_equal(inputs[f"{name}_target"], data[f"{name}_target"])
            np.testing.assert_array_equal(inputs[f"{name}_eligible"], data[f"{name}_target"] >= 2)
            np.testing.assert_array_equal(inputs[f"{name}_row_id"], np.arange(6))
    with np.load(out / "per_example.npz", allow_pickle=False) as arrays:
        for seed, names in [(42, ["sft", "continued_sft", "completion_h1"]), (43, ["sft"])]:
            for name in names:
                prefix = f"seed_{seed}_{name}_"
                for key in ["uid", "ndcg", "exact_one_step_expected_reward", "recommendations", "sample_actions"]:
                    assert arrays[prefix + key].shape[0] == 6
                np.testing.assert_array_equal(arrays[prefix + "uid"], data["test_uid"])
        assert all(arrays[key].dtype.kind != "O" for key in arrays.files)
        # Eligible paired baseline can be reconstructed without dropping unknown targets in stored arrays.
        mask = data["test_target"] >= 2
        from rl_recsys.evaluate import macro
        expected = macro(arrays["seed_42_sft_ndcg"][mask], data["test_uid"][mask])[0]
        assert successful["policies"]["completion_h1"]["paired_vs_sft"]["ndcg"]["before"] == pytest.approx(expected)
    complete = json.loads((out / "COMPLETE.json").read_text())
    assert complete["status"] == "complete"
    for name, digest in complete["sha256"].items():
        assert sha256(out / name) == digest
    with pytest.raises(FileExistsError):
        fe.evaluate_followup(prepared, run, root, out)


@pytest.mark.parametrize("marker", ["missing", "wrong_hash", "incomplete"])
def test_completion_marker_required_before_inference(frozen_followup, monkeypatch, marker):
    prepared, run, root, out, _, _ = frozen_followup
    path = root / "FOLLOWUP_COMPLETE.json"
    if marker == "missing":
        path.unlink()
    else:
        value = json.loads(path.read_text())
        value["manifest_sha256" if marker == "wrong_hash" else "status"] = "invalid"
        write_json(path, value)
    monkeypatch.setattr(fe.exp, "get_device", lambda *a: pytest.fail("Unsealed run reached inference"))
    with pytest.raises((ValueError, FileNotFoundError)):
        fe.evaluate_followup(prepared, run, root, out)
    assert not out.exists()


@pytest.mark.parametrize("mutation", ["missing_checkpoint", "missing_metadata", "unsafe_selected", "seed_mismatch", "selection_mismatch", "failed_gate_policies"])
def test_manifest_firewall_rejects_inconsistent_or_unbound_inputs(frozen_followup, monkeypatch, mutation):
    prepared, run, root, out, _, manifest = frozen_followup
    row = manifest["seeds"]["42"]
    if mutation == "missing_checkpoint":
        row["checkpoint_hashes"].pop("continued_sft.pt")
    elif mutation == "missing_metadata":
        row["artifact_hashes"].pop("config.json")
    elif mutation == "unsafe_selected":
        row["policies"]["continued_sft"]["selected_checkpoint"] = "../outside.pt"
    elif mutation == "seed_mismatch":
        manifest["seeds"].pop("43")
    elif mutation == "selection_mismatch":
        row["policies"]["continued_sft"]["selected_checkpoint"] = None
    else:
        audit_path = root / "seed_42" / "reward_audit.json"
        audit = json.loads(audit_path.read_text())
        audit["gate"]["heads"]["completion"]["enabled"] = False
        write_json(audit_path, audit)
        row["artifact_hashes"]["reward_audit.json"] = sha256(audit_path)
    write_json(root / "followup_manifest.json", manifest)
    write_json(root / "FOLLOWUP_COMPLETE.json", {"status": "complete", "manifest_sha256": sha256(root / "followup_manifest.json"), "provenance_sha256": sha256(root / "provenance.json")})
    monkeypatch.setattr(fe.exp, "get_device", lambda *a: pytest.fail("Invalid manifest reached inference"))
    with pytest.raises(ValueError):
        fe.evaluate_followup(prepared, run, root, out)
    assert not out.exists()


@pytest.mark.parametrize("relative", ["seed_43/reward.pt", "seed_43/config.json", "seed_42/reward_audit.json", "provenance.json"])
def test_corrupt_hash_rejected_for_all_seeds_before_any_evaluation(frozen_followup, monkeypatch, relative):
    prepared, run, root, out, _, _ = frozen_followup
    with (root / relative).open("ab") as stream:
        stream.write(b"corrupt")
    def forbidden(*args, **kwargs):
        pytest.fail("Corrupt artifact reached device/data/model evaluation")
    monkeypatch.setattr(fe.exp, "get_device", forbidden)
    monkeypatch.setattr(fe, "load_prepared", forbidden)
    monkeypatch.setattr(fe, "evaluate", forbidden)
    # Corrupt provenance itself may fail JSON parsing before digest comparison.
    with pytest.raises((ValueError, json.JSONDecodeError)):
        fe.evaluate_followup(prepared, run, root, out)
    assert not out.exists()
