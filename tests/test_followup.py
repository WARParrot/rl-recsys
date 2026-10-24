"""Small CPU unit/mocked orchestration tests; no real training jobs or test-role evaluation."""
import copy
import json
import sys
import time
import types

import numpy as np
import pytest
import torch

from rl_recsys import followup as fu


def subset(timestamps=(1, 1, 2, 3, 4, 4)):
    n = len(timestamps)
    return {"context": np.tile([0, 2], (n, 1)), "target": np.full(n, 2),
            "uid": np.arange(n) % 2, "timestamp": np.array(timestamps),
            "reward": np.full(n, .5, dtype=np.float32),
            "seconds": np.full(n, .2, dtype=np.float32), "organic": np.ones(n)}


def test_split_unique_median_not_event_median_and_ties_stay_together():
    data = subset([1] * 20 + [2, 3, 4, 5])
    early, late = fu.split_calibration(data)
    assert set(early["timestamp"]) == {1, 2}
    assert set(late["timestamp"]) == {3, 4, 5}
    assert early["timestamp"].max() < late["timestamp"].min()
    assert len(early["target"]) + len(late["target"]) == len(data["target"])
    for timestamps in [[], [1, 1], [1, float("nan")]]:
        with pytest.raises(ValueError):
            fu.split_calibration(subset(timestamps))


def test_item_only_ignores_context_and_supports_evaluator_shapes():
    model = fu.ItemOnlyRewardModel(7, 3, {"hidden": 4})
    contexts = torch.tensor([[0, 2, 3], [4, 5, 6]])
    items = torch.tensor([2, 2])
    out = model(contexts, items)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out[0], out[1])
    assert ((out > 0) & (out < 1)).all()
    hidden = model.history(contexts)
    assert model.from_history(hidden[:, None].expand(-1, 3, -1), torch.ones(2, 3, dtype=torch.long)).shape == (2, 3, 2)
    assert model.catalog(contexts).shape == (2, 5, 2)
    out.sum().backward()
    assert model.item.weight.grad is not None


def test_uniform_users_then_events_not_uniform_events():
    fit = subset(range(101))
    fit["uid"] = np.array([0] * 100 + [1])
    sampler = fu.UserEventSampler(fit)
    b = sampler.batch(12000, np.random.default_rng(42), "cpu")
    assert .47 < float(b["uid"].float().mean()) < .53
    # All 100 events of the large user remain sampleable, not just its first.
    assert len(torch.unique(b["timestamp"][b["uid"] == 0])) == 100


def test_repeat_adjustment_uses_item_tokens_not_action_indices():
    completion = torch.tensor([.8, .8, .8])
    context = torch.tensor([[0, 2], [2, 4], [0, 1]])
    reward, repeat = fu.adjusted_completion(completion, context, torch.tensor([2, 3, 2]), .1)
    torch.testing.assert_close(repeat, torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(reward, torch.tensor([.7, .8, .8]))
    torch.testing.assert_close(fu.adjusted_completion(completion, context, torch.tensor([2, 3, 2]), 0)[0], completion)


class TinyPolicy(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([.1, -.1]))

    def forward(self, context):
        return self.logits.expand(len(context), -1)


class TinyRM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, context, items):
        value = items.float() / 4 + self.bias
        return torch.stack([value, value], -1)


def small_config():
    return {"seed": 42, "model": {"hidden": 4},
            "train": {"threads": 1, "lr": .001, "rl_lr": .001, "batch_size": 8,
                      "grad_clip": 1., "eval_every": 50, "reward_steps": 2, "max_hours": 1},
            "rl": {"sampling_temperature": 1., "relevance_floor": .98}}


@pytest.mark.parametrize("supervised", [False, True])
def test_local_loop_exact_steps_selection_and_frozen_models(monkeypatch, tmp_path, supervised):
    cfg = small_config()
    cfg["train"]["eval_every"] = 1
    policy = TinyPolicy()
    reference = fu.exp.freeze(copy.deepcopy(policy))
    rm = fu.exp.freeze(TinyRM())
    initial_rm = copy.deepcopy(rm.state_dict())
    scores = iter([(.9, .4), (.9, .6), (.9, .5), (.1, .99)])
    def evaluate(*args, **kwargs):
        ndcg, reward = next(scores)
        return {"eligible_ndcg": ndcg, "eligible_exact_one_step_expected_reward": reward,
                "repeat": .2}, {}
    monkeypatch.setattr(fu.exp, "evaluate", evaluate)
    optimizer_steps = []
    original = fu.exp.optimize_step
    def optimize(*args):
        optimizer_steps.append(1)
        return original(*args)
    monkeypatch.setattr(fu.exp, "optimize_step", optimize)
    result = fu.train_policy(policy, reference, rm, rm, subset(), subset(), {}, cfg,
                             torch.device("cpu"), tmp_path, "arm", 3, time.monotonic() + 30,
                             supervised=supervised, repeat_penalty=.1)
    assert len(optimizer_steps) == 3
    assert result["selected"]["step"] == 1  # later high proxy fails relevance floor
    assert result["selected_checkpoint"] == "arm.pt"
    assert (tmp_path / "arm_final.pt").exists()
    assert len(json.loads((tmp_path / "arm_curve.json").read_text())) == 3
    for key, value in rm.state_dict().items():
        torch.testing.assert_close(value, initial_rm[key])
    assert rm.bias.grad is None and reference.logits.grad is None
    if not supervised:
        assert result["beta"] == .01
        assert result["sampling"] == "uniform users then uniform fit events"


def test_deadline_stops_local_loop_before_optimizer(monkeypatch, tmp_path):
    monkeypatch.setattr(fu.exp, "evaluate", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(fu.exp, "optimize_step", lambda *a: pytest.fail("deadline bypassed"))
    with pytest.raises(TimeoutError):
        fu.train_policy(TinyPolicy(), TinyPolicy(), TinyRM(), TinyRM(), subset(), subset(), {},
                        small_config(), "cpu", tmp_path, "arm", 1, time.monotonic() - 1)


@pytest.mark.parametrize("enabled", [False, True])
def test_orchestration_gate_provenance_no_historical_rm_or_test(monkeypatch, tmp_path, enabled):
    run, data_dir, output = [tmp_path / name for name in ["historical", "data", "fresh"]]
    run.mkdir()
    data_dir.mkdir()
    cfg = small_config()
    (run / "provenance.json").write_text(json.dumps({"config": cfg}))
    (run / "sft.pt").write_bytes(b"historical sft")
    verify_calls = []
    def verify(r, d):
        verify_calls.append((r, d))
        return {"verified": True}
    monkeypatch.setitem(sys.modules, "rl_recsys.reanalysis", types.SimpleNamespace(verify_inputs=verify))
    monkeypatch.setattr(fu.exp, "get_device", lambda c: torch.device("cuda"))
    monkeypatch.setattr(fu.exp, "seed_all", lambda *a: None)
    monkeypatch.setattr(fu.exp, "provenance", lambda root, c, *a: {"config": c, "code_hashes": {"followup.py": "mock"}})
    monkeypatch.setattr(fu, "load_prepared", lambda *a: {})
    roles = []
    def role(data, name, eligible=False):
        roles.append(name)
        assert name != "test"
        return subset()
    monkeypatch.setattr(fu, "role", role)
    monkeypatch.setattr(fu.exp, "model_args", lambda *a: ())
    class MockModel(TinyPolicy):
        def to(self, *args):
            return self
    monkeypatch.setattr(fu, "Policy", MockModel)
    monkeypatch.setattr(fu, "RewardModel", MockModel)
    monkeypatch.setattr(fu, "ItemOnlyRewardModel", MockModel)
    fits = []
    def fit(model, fit, valid, steps, path, c, *a, **kw):
        assert c["train"]["eval_every"] == 50
        assert valid["timestamp"].max() == 2
        fits.append(path.name)
        fu.exp.checkpoint(path, model)
        return [{"step": steps, "loss": .1}]
    monkeypatch.setattr(fu.exp, "fit_supervised", fit)
    def report(rm, auditor, fit, audit, *a, **kw):
        assert audit["timestamp"].min() == 3
        return {"heads": {"completion": {"mae": .2 if enabled else .8,
                         "constant_mae": .5,
                         "paired_mae": {"relative_difference": -.2 if enabled else .2}}}}
    monkeypatch.setattr(fu.exp, "reward_report", report)
    restored = []
    def restore(path, model, *a):
        restored.append(path.name)
        assert path.name == "sft.pt"
        return model
    monkeypatch.setattr(fu.exp, "restore", restore)
    arms = []
    def train(*args, **kw):
        directory, name, steps = args[9:12]
        assert (directory / "reward_audit.json").exists()
        assert not args[2].training and not args[2].logits.requires_grad
        assert steps == 3
        arms.append((name, kw["supervised"], kw["repeat_penalty"]))
        fu.exp.checkpoint(directory / f"{name}.pt", args[0])
        return {"optimizer_steps": steps, "selected_checkpoint": f"{name}.pt"}
    monkeypatch.setattr(fu, "train_policy", train)
    manifest = fu.run_followup(data_dir, run, output, seeds=(42, 43), steps=3, repeat_penalty=.25)
    assert verify_calls == [(run, data_dir)]
    assert roles == ["fit", "calibration", "selection"]
    assert fits == ["reward.pt", "item_only.pt"] * 2
    assert len(arms) == (6 if enabled else 0)
    if enabled:
        assert arms[:3] == [("continued_sft", True, 0.), ("completion_h1", False, 0.),
                            ("completion_minus_repeat_h1", False, .25)]
    else:
        assert not restored
    prov = json.loads((output / "provenance.json").read_text())
    assert prov["historical_sft_sha256"] == fu.sha256(run / "sft.pt")
    assert prov["historical_manifest"] == {"verified": True}
    assert prov["seeds"] == [42, 43]
    assert "Conditional training-control" in prov["limitations"][0]
    assert prov["config"]["train"]["allow_cpu"] is False
    for seed in ["42", "43"]:
        assert "reward.pt" in manifest["seeds"][seed]["checkpoint_hashes"]
        assert "reward_audit.json" in manifest["seeds"][seed]["artifact_hashes"]
    with pytest.raises(FileExistsError):
        fu.run_followup(data_dir, run, output)


@pytest.mark.parametrize("seeds,steps,penalty", [((), 1, .1), ((42, 42), 1, .1),
                          ((True,), 1, .1), ((42,), 0, .1), ((42,), 1, -.1), ((42,), 1, float("nan"))])
def test_invalid_options_fail_before_import_or_output(tmp_path, seeds, steps, penalty):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        fu.run_followup(tmp_path, tmp_path, output, seeds, steps, penalty)
    assert not output.exists()


def test_cuda_required_before_output_creation(monkeypatch, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "provenance.json").write_text(json.dumps({"config": small_config()}))
    monkeypatch.setitem(sys.modules, "rl_recsys.reanalysis", types.SimpleNamespace(verify_inputs=lambda *a: {}))
    monkeypatch.setattr(fu.exp, "get_device", lambda cfg: torch.device("cpu"))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        fu.run_followup(tmp_path, run, tmp_path / "output")
    assert not (tmp_path / "output").exists()
