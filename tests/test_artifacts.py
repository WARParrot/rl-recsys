import numpy as np
import pytest

from rl_recsys.artifacts import complete_evaluation, validate_evaluation_artifacts
from rl_recsys.experiment import select_candidate


def test_evaluation_manifest_hashes_atomic_artifacts(tmp_path):
    complete_evaluation(tmp_path, {"evaluation_status": "exploratory_reuse"},
                        {"metric": np.array([1.0, 2.0])}, {"seed": 1})
    manifest = validate_evaluation_artifacts(tmp_path)
    assert manifest["status"] == "complete"
    assert set(manifest["sha256"]) == {"results.json", "per_example.npz", "evaluation-provenance.json"}
    with np.load(tmp_path / "per_example.npz") as values:
        np.testing.assert_array_equal(values["metric"], [1.0, 2.0])
    with pytest.raises(FileExistsError):
        complete_evaluation(tmp_path, {}, {}, {})


def test_validation_rejects_corruption_and_missing_completion(tmp_path):
    complete_evaluation(tmp_path, {}, {"x": np.array([3])}, {})
    (tmp_path / "results.json").write_text('{"corrupt": true}')
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_evaluation_artifacts(tmp_path)
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="Incomplete"):
        validate_evaluation_artifacts(missing)


@pytest.mark.parametrize("failed_name", ["per_example.npz", "results.json", "EVALUATION_COMPLETE.json"])
def test_failed_publish_has_no_completion_marker(tmp_path, monkeypatch, failed_name):
    from rl_recsys import artifacts
    original = artifacts.os.replace

    def fail(source, destination):
        if destination.name == failed_name:
            raise OSError("injected publish failure")
        original(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        complete_evaluation(tmp_path, {}, {"x": np.array([1])}, {})
    assert not (tmp_path / "EVALUATION_COMPLETE.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        validate_evaluation_artifacts(tmp_path)


def test_candidate_selection_uses_eligible_not_all_target_ranking():
    cfg = {"rl": {"relevance_floor": 0.98}}
    base = {"eligible_ndcg": 1.0, "all_target_ndcg": 1.0}
    candidates = [
        {"name": "eligible-winner", "reward": "completion", "horizon": 1,
         "selection": {"eligible_ndcg": 1.0, "eligible_sample_reward": .4,
                        "all_target_ndcg": .1}},
        {"name": "all-target-winner", "reward": "completion", "horizon": 1,
         "selection": {"eligible_ndcg": .99, "eligible_sample_reward": .2,
                        "all_target_ndcg": 9.0}},
    ]
    assert select_candidate(candidates, base, cfg)["name"] == "eligible-winner"


def test_candidate_selection_excludes_ineligible_grid_members():
    cfg = {"rl": {"relevance_floor": .98}}
    base = {"eligible_ndcg": 1.0}
    candidates = [{"name": "seconds", "reward": "seconds", "horizon": 1,
                   "selection": {"eligible_ndcg": 1.0, "eligible_sample_reward": 99}}]
    assert select_candidate(candidates, base, cfg) is None
