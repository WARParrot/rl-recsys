"""Small CPU-only checks of label provenance and matched-control semantics."""
import time

import numpy as np
import pytest
import torch
from torch import nn

from rl_recsys.diagnostics import matched_rollout_controls, reward_history_sensitivity
from rl_recsys.evaluate import macro


class FakeReward(nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        self.calls = []

    def forward(self, context, items):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.calls.append((context.clone(), items.clone()))
        value = context.float().sum(-1) / 50 + items.float() / 100 + self.offset
        return torch.stack((value, value / 2), -1)


class FakePolicy(nn.Module):
    def forward(self, context):
        assert not torch.is_grad_enabled()
        assert not self.training
        return torch.stack((context[:, -1].float(), -context[:, -1].float(),
                            torch.zeros(len(context))), -1)


def subset():
    # Deliberately gapped contexts: not append(target) continuations.
    return {"context": np.tile([[0, 2, 3], [2, 4, 4], [0, 0, 3], [3, 2, 4], [4, 4, 3]], (2, 1)),
            "uid": np.repeat([10, 20], 5), "target": np.array([2, 1, 4, 3, 2] * 2),
            "timestamp": np.tile([10, 20, 70, 100, 1000], 2),
            "reward": np.linspace(0, 1, 10), "seconds": np.linspace(0, .5, 10)}


def test_sensitivity_global_shuffle_labels_and_macro():
    data = subset()
    before = data["context"].copy()
    rm = FakeReward()
    summary, out = reward_history_sensitivity(rm, data, batch_size=3, replicates=30)
    eligible = data["target"] >= 2
    assert summary["examples"] == 8
    np.testing.assert_array_equal(out["row_index"], np.flatnonzero(eligible))
    np.testing.assert_array_equal(data["context"], before)
    donors = out["shuffled_donor_row_index"]
    assert np.all(data["uid"][donors] != out["uid"])
    assert set(donors) == set(out["row_index"])
    original_calls = rm.calls[:3]
    unk_calls = rm.calls[3:6]
    shuffle_calls = rm.calls[6:]
    np.testing.assert_array_equal(torch.cat([c for c, _ in unk_calls]), np.where(before[eligible] == 0, 0, 1))
    np.testing.assert_array_equal(torch.cat([c for c, _ in shuffle_calls]), before[donors])
    for calls in (original_calls, unk_calls, shuffle_calls):
        np.testing.assert_array_equal(torch.cat([t for _, t in calls]), data["target"][eligible])
    for variant in ("original", "all_unk", "shuffled"):
        err = np.abs(out[variant + "_prediction"] - out["labels"])
        assert summary["variants"][variant]["reward_mae"] == pytest.approx(macro(err[:, 0], out["uid"])[0])
    paired = summary["variants"]["all_unk"]["reward_paired_mae"]
    assert paired["difference"] == pytest.approx(summary["variants"]["all_unk"]["reward_mae"] - summary["variants"]["original"]["reward_mae"])
    _, other = reward_history_sensitivity(FakeReward(), data, batch_size=1, replicates=30)
    np.testing.assert_array_equal(donors, other["shuffled_donor_row_index"])
    for variant in ("original", "all_unk", "shuffled"):
        np.testing.assert_allclose(out[variant + "_prediction"], other[variant + "_prediction"])


def test_sensitivity_rejects_impossible_shuffle_and_deadline():
    data = subset()
    data["uid"][:] = 1
    with pytest.raises(ValueError, match="Cross-user"):
        reward_history_sensitivity(FakeReward(), data)
    with pytest.raises(TimeoutError):
        reward_history_sensitivity(FakeReward(), subset(), deadline=time.monotonic() - 1)


def test_rollout_actual_contexts_matched_starts_gaps_and_labels():
    data = subset()
    # One user with too few retained rows must be absent in every mode.
    data = {key: np.concatenate((value, value[:2]), axis=0) for key, value in data.items()}
    data["uid"][-2:] = 30
    rm, auditor = FakeReward(), FakeReward(.1)
    summary, out = matched_rollout_controls(FakePolicy(), rm, auditor, data, [1, 2, 3], batch_size=1, k=2)
    assert summary["users"] == 2
    np.testing.assert_array_equal(out["uid"], [10, 20])
    np.testing.assert_array_equal(out["elapsed_gaps"], [[10, 50, 30, 900]] * 2)
    assert summary["elapsed_gaps"]["max"] == 900
    modes = out["modes"]
    assert set(modes) == {"policy", "random", "popularity", "observed_context_replay"}
    for name, mode in modes.items():
        np.testing.assert_array_equal(mode["actions"][:, 0], modes["policy"]["actions"][:, 0])
        np.testing.assert_allclose(mode["reward"][:, 0], modes["policy"]["reward"][:, 0])
        np.testing.assert_allclose(mode["disagreement"], .1, atol=1e-7)
        np.testing.assert_array_equal(mode["initial_topk_overlap"][:, 0], 1)
        for t in range(5):
            assert summary["modes"][name]["steps"][t]["reward"] == pytest.approx(mode["reward"][:, t].mean())
        if name != "observed_context_replay":
            assert "observed_labels" not in mode
            assert (mode["appended_tokens"][:, :-1] >= 2).all()
        else:
            assert (mode["appended_tokens"] == -1).all()
    replay = modes["observed_context_replay"]
    np.testing.assert_array_equal(modes["policy"]["appended_tokens"][:, :-1], modes["policy"]["actions"][:, :-1])
    actual_contexts = data["context"][out["row_index"]]
    expected = actual_contexts.sum(-1) / 50 + replay["observed_target"] / 100
    eligible = replay["observed_eligible"]
    np.testing.assert_allclose(replay["observed_target_prediction"][:, :, 0][eligible], expected[eligible], rtol=1e-6)
    assert np.isnan(replay["observed_target_prediction"][~eligible]).all()
    # Proposal reward also uses actual replay context, not a synthetic append.
    np.testing.assert_allclose(replay["reward"], actual_contexts.sum(-1) / 50 + replay["actions"] / 100, rtol=1e-6)
    assert summary["modes"]["observed_context_replay"]["observed_eligible_examples"] == 8


def test_rollout_batch_invariance_and_popularity_append():
    kwargs = dict(policy=FakePolicy(), reward_model=FakeReward(), auditor=FakeReward(),
                  subset=subset(), popularity=[0, 1, 0], seed=31)
    _, a = matched_rollout_controls(**kwargs, batch_size=1)
    _, b = matched_rollout_controls(**kwargs, batch_size=7)
    for name in a["modes"]:
        for key in ("actions", "appended_tokens", "reward", "policy_entropy"):
            np.testing.assert_allclose(a["modes"][name][key], b["modes"][name][key])
    assert (a["modes"]["popularity"]["appended_tokens"][:, :-1] == 3).all()


@pytest.mark.parametrize("problem", ["ties", "short", "deadline", "popularity"])
def test_rollout_rejects_invalid_inputs(problem):
    data = subset()
    kwargs = {}
    if problem == "ties":
        data["timestamp"][1] = data["timestamp"][0]
    elif problem == "short":
        data = {key: value[:4] for key, value in data.items()}
    elif problem == "deadline":
        kwargs["deadline"] = time.monotonic() - 1
    pop = [0, 0, 0] if problem == "popularity" else [1, 2, 3]
    with pytest.raises(TimeoutError if problem == "deadline" else ValueError):
        matched_rollout_controls(FakePolicy(), FakeReward(), FakeReward(), data, pop, **kwargs)
