"""Pre-RL audit is a durable firewall, not a post-training report."""
import json
import numpy as np
import pytest
import torch

from rl_recsys import experiment as exp


def setup_training(monkeypatch, tmp_path, enabled):
    cfg = {"seed": 42, "train": {"threads": 1, "max_hours": 1,
           "sft_steps": 1, "reward_steps": 1},
           "rl": {"rewards": ["completion", "seconds"], "horizons": [1],
                  "betas": [.1], "relevance_floor": .98}}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "audit.json").write_text(json.dumps({"config": cfg}))
    monkeypatch.setattr(exp, "validate_config", lambda config: config)
    monkeypatch.setattr(exp, "seed_all", lambda *args: None)
    monkeypatch.setattr(exp, "get_device", lambda cfg: torch.device("cpu"))
    monkeypatch.setattr(exp, "load_prepared", lambda path: {})
    subset = {"target": np.array([2])}
    roles = []
    def role(data, name, eligible=False):
        roles.append(name)
        assert name != "test"
        return subset
    monkeypatch.setattr(exp, "role", role)
    monkeypatch.setattr(exp, "model_args", lambda *args: ())
    class Model(torch.nn.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(1, 1)
    monkeypatch.setattr(exp, "Policy", Model)
    monkeypatch.setattr(exp, "RewardModel", Model)
    monkeypatch.setattr(exp, "provenance", lambda *args: {"config": cfg})
    def fit(model, fit, valid, steps, path, *args, **kwargs):
        exp.checkpoint(path, model)
        return [{"loss": 1.}]
    monkeypatch.setattr(exp, "fit_supervised", fit)
    def report(*args, **kwargs):
        return {"heads": {name: {"mae": .2 if name in enabled else .6,
                                 "constant_mae": .5,
                                 "paired_mae": {"relative_difference": -.2 if name in enabled else .2}}
                          for name in ["completion", "seconds"]}}
    monkeypatch.setattr(exp, "reward_report", report)
    calls = []
    run = tmp_path / "run"
    def train_rl(*args, **kwargs):
        assert (run / "reward_audit.json").exists()
        calls.append(args[8])  # reward_index
        return []
    monkeypatch.setattr(exp, "train_rl", train_rl)
    monkeypatch.setattr(exp, "evaluate", lambda *args, **kwargs: (
        {"eligible_ndcg": 1., "eligible_exact_one_step_expected_reward": .5}, {}))
    return cfg, data_dir, run, roles, calls


def test_failed_gate_persists_audit_and_stops_before_rl(monkeypatch, tmp_path):
    cfg, data, run, roles, calls = setup_training(monkeypatch, tmp_path, [])
    with pytest.raises(RuntimeError, match="Pre-RL reward audit gate failed"):
        exp.train(cfg, data, run)
    audit = json.loads((run / "reward_audit.json").read_text())
    assert audit["role"] == "calibration"
    assert audit["gate"]["status"] == "failed"
    assert set(audit["checkpoint_hashes"]) == {"sft.pt", "reward.pt", "auditor.pt"}
    assert (run / "training.json").exists()
    assert not (run / "READY_FOR_TEST.json").exists()
    assert not calls and "test" not in roles


def test_weak_head_is_skipped_and_audit_is_bound_to_ready(monkeypatch, tmp_path):
    cfg, data, run, roles, calls = setup_training(monkeypatch, tmp_path, ["completion"])
    exp.train(cfg, data, run)
    assert calls == [0]
    candidates = json.loads((run / "candidates.json").read_text())
    assert [c["reward"] for c in candidates] == ["completion"]
    ready = json.loads((run / "READY_FOR_TEST.json").read_text())
    assert ready["reward_audit_sha256"] == exp.sha256(run / "reward_audit.json")
    # Hash validation must reject audit edits before test data/model loading.
    (run / "reward_audit.json").write_text("{}")
    with pytest.raises(ValueError, match="reward_audit.json changed"):
        exp.final_evaluation(data, run)


def test_selection_uses_exact_not_sampled_reward():
    def candidate(name, exact, sampled):
        return {"name": name, "reward": "completion", "horizon": 1,
                "selection": {"eligible_ndcg": 1.,
                              "eligible_exact_one_step_expected_reward": exact,
                              "eligible_sample_reward": sampled}}
    chosen = exp.select_candidate([candidate("exact", .8, .1), candidate("sampled", .5, .9)],
                                  {"eligible_ndcg": 1.}, {"rl": {"relevance_floor": .98}})
    assert chosen["name"] == "exact"
