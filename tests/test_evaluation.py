import numpy as np
import pytest
import torch

from rl_recsys.evaluate import (evaluate, macro, paired_interval, sampling_diagnostic,
                                rollout_diagnostics, exact_one_step_expected_reward,
                                action_support_diagnostic)
from rl_recsys.experiment import reward_report, verify_source_identity, reward_gate


class Policy(torch.nn.Module):
    def forward(self, context):
        return torch.zeros((len(context), 3))


class Reward(torch.nn.Module):
    def history(self, context):
        return context[:, -1:].float()

    def from_history(self, hidden, items):
        return torch.stack([items.float() / 5, torch.zeros_like(items).float() + .25], -1)

    def forward(self, context, items):
        return self.from_history(self.history(context), items)


def fixture():
    subset = dict(context=np.array([[0, 2], [2, 3], [0, 4]]),
                  target=np.array([2, 1, 4]), uid=np.array([10, 10, 20]),
                  reward=np.array([.4, .6, .8]), seconds=np.array([.25, .25, .25]),
                  organic=np.array([0, 1, 1]))
    data = dict(vocab=np.arange(3), popularity=np.ones(3), durations=np.ones(3) * 100)
    cfg = dict(seed=42, rl=dict(sampling_temperature=1., seconds_cap=600.),
               evaluation=dict(batch_size=2, k=1, action_samples=8, rollout_samples=3,
                               bootstrap=50, bootstrap_seed=2026))
    return subset, data, cfg


@pytest.mark.parametrize('values,users', [
    ([], []), ([1.], []), ([1., 2.], [1]), ([[1.]], [1]),
    ([1.], [[1]]), ([np.nan], [1]), ([np.inf], [1]),
    ([-np.inf], [1]), ([1.], [np.nan]), ([1.], [np.inf]),
])
def test_macro_rejects_invalid_inputs(values, users):
    with pytest.raises(ValueError):
        macro(values, users)


@pytest.mark.parametrize('before,after,users', [
    ([], [], []), ([1.], [1., 2.], [1]), ([1.], [1.], []),
    ([[1.]], [1.], [1]), ([1.], [[1.]], [1]),
    ([np.nan], [1.], [1]), ([1.], [np.inf], [1]),
    ([1.], [-np.inf], [1]), ([1.], [1.], [np.nan]),
])
def test_paired_interval_rejects_invalid_inputs(before, after, users):
    with pytest.raises(ValueError):
        paired_interval(before, after, users)


@pytest.mark.parametrize('replicates', [0, -1, 1.5, 2.0, True, np.bool_(False), np.nan, np.inf, '2', None])
def test_paired_interval_rejects_invalid_replicates(replicates):
    with pytest.raises(ValueError, match='replicates'):
        paired_interval([1.], [2.], [1], replicates=replicates)


def test_macro_and_paired_interval_preserve_user_weighting():
    score, means, users = macro([1., 3., 8.], ['a', 'a', 'b'])
    assert score == 5.
    np.testing.assert_array_equal(means, [2., 8.])
    np.testing.assert_array_equal(users, ['a', 'b'])
    result = paired_interval([1., 3., 8.], [2., 4., 9.], [1, 1, 2], replicates=np.int64(20))
    assert result['before'] == 5.
    assert result['difference'] == 1.
    assert result['ci95'] == [1., 1.]
    zero = paired_interval([0.], [1.], [1], replicates=1)
    assert zero['relative_difference'] is None
    assert zero['relative_ci95'] is None


def test_exact_one_step_expected_reward_categorical():
    class CategoricalPolicy(torch.nn.Module):
        def forward(self, context):
            assert not self.training and not torch.is_grad_enabled()
            return torch.tensor([[.2, .3, .5], [.5, .25, .25], [.1, .2, .7]])[context[:, 0]].log()

    class CatalogReward(Reward):
        def from_history(self, hidden, items):
            assert not self.training and not torch.is_grad_enabled()
            reward = (items.float() - 2) * .5 + hidden[:, 0] * .1
            return torch.stack([reward, torch.ones_like(reward) * 99], -1)

    contexts = np.array([[0, 2], [1, 3], [2, 4]])
    # Sum p(a|x) r(x,a): .65+.2, .375+.3, .8+.4.
    expected = [.85, .675, 1.2]
    for batch_size in (1, 2, 10):
        score, raw = exact_one_step_expected_reward(
            CategoricalPolicy(), CatalogReward(), contexts, [1, 1, 2], batch_size=batch_size)
        np.testing.assert_allclose(raw, expected, rtol=1e-6)
        assert score == pytest.approx(((.85 + .675) / 2 + 1.2) / 2)
    score, raw = exact_one_step_expected_reward(
        CategoricalPolicy(), CatalogReward(), contexts, [1, 1, 2], temperature=2.)
    p = np.sqrt([[.2, .3, .5], [.5, .25, .25], [.1, .2, .7]])
    p /= p.sum(1, keepdims=True)
    np.testing.assert_allclose(raw, p @ [0., .5, 1.] + [.2, .3, .4], rtol=1e-6)


def test_replicated_reward_and_eligible_population():
    subset, data, cfg = fixture()
    data['fit_target'] = np.array([2, 2, 4])
    summary, raw = evaluate(Policy(), Reward(), Reward(), subset, data, cfg, torch.device('cpu'))
    assert raw['sample_reward_draws'].shape == (3, 8)
    assert summary['action_support']['unsupported_action_rate'] == 0
    assert summary['action_support']['unsupported_reward_score_mean'] is None
    assert summary['sample_action_support']['unsupported_action_rate'] == pytest.approx(
        np.mean(raw['sample_actions'] == 3))
    assert raw['top_reward_actions'].shape == raw['recommendations'].shape
    np.testing.assert_allclose(raw['sample_reward'], raw['sample_reward_draws'].mean(1))
    mask = subset['target'] >= 2
    assert summary['eligible_sample_reward'] == macro(raw['sample_reward'][mask], subset['uid'][mask])[0]
    np.testing.assert_allclose(raw['exact_one_step_expected_reward'], [.6, .6, .6])
    assert summary['eligible_exact_one_step_expected_reward'] == pytest.approx(.6)
    assert summary['eligible_target_examples'] == 2
    assert summary['ndcg'] == .25
    assert summary['eligible_ndcg'] == .5


def test_rollout_replicates_keep_user_macro_and_all_steps():
    subset, _, cfg = fixture()
    result = rollout_diagnostics(Policy(), Reward(), Reward(), subset, cfg, torch.device('cpu'))
    assert result[0]['initial_topk_overlap'] == 1
    assert all(row['policy_entropy'] > 0 for row in result)
    assert len(result) == 5
    assert all(len(row['per_sample']) == 3 for row in result)
    for row in result:
        assert row['reward'] == pytest.approx(np.mean([s['reward'] for s in row['per_sample']]))
        assert 0 <= row['repeat'] <= 1


def test_both_reward_heads_have_independent_validation():
    subset, _, cfg = fixture()
    subset['target'] = np.array([2, 3, 4])
    subset['reward'] = np.array([.4, .6, .8])
    subset['seconds'] = np.array([.1, .2, .3])
    fit = dict(subset)
    fit['reward'] = np.array([.4, .6, .8])
    fit['seconds'] = np.array([.1, .2, .3])
    result = reward_report(Reward(), Reward(), fit, subset, cfg, torch.device('cpu'),
                           data={'durations': np.array([100., 200., 350.])})
    assert result['heads']['completion']['mae'] > 0
    assert result['heads']['seconds']['mae'] > 0
    assert result['heads']['seconds']['mae_seconds_proxy'] == pytest.approx(result['heads']['seconds']['mae'] * 600)
    assert result['heads']['completion']['mae'] != result['heads']['seconds']['mae']
    assert len(result['heads']['seconds']['calibration']) > 0
    assert set(result['heads']['seconds']['slices']) == {
        'organic', 'recommended', 'repeat', 'novel',
        'popularity_low', 'popularity_middle', 'popularity_high',
        'duration_lt_180s', 'duration_180_to_300s', 'duration_ge_300s'}


def test_reward_gate_disables_heads_that_do_not_beat_item_mean():
    cfg = {"rl": {"rewards": ["completion", "seconds"]},
           "reward_audit": {"min_relative_improvement": 0.02}}
    audit = {"heads": {
        "completion": {"paired_mae": {"relative_difference": -0.03}, "mae": .8, "constant_mae": .9},
        "seconds": {"paired_mae": {"relative_difference": 0.01}, "mae": 1., "constant_mae": .9},
    }}
    gate = reward_gate(audit, cfg)
    assert gate["enabled_rewards"] == ["completion"]
    assert gate["heads"]["seconds"]["enabled"] is False
    assert reward_gate({"heads": {"completion": {"paired_mae": {"relative_difference": None}, "mae": 1, "constant_mae": 1}}}, cfg)["status"] == "failed"


def test_sampling_diagnostic_preserves_draw_pairing():
    a = np.repeat(np.array([.2, .4, .8])[:, None], 4, axis=1)
    b = a + .1
    assert sampling_diagnostic(a, b, np.array([1, 1, 2]))['difference'] == pytest.approx(.1)


def test_action_support_reports_ood_score_shift():
    result = action_support_diagnostic(
        np.array([[2, 3], [4, 2]]), np.array([[.2, .8], [.9, .1]]),
        np.array([2, 4]), np.array([10, 10]))
    assert result["unsupported_action_rate"] == pytest.approx(.25)
    assert result["supported_reward_score_mean"] == pytest.approx(.4)
    assert result["unsupported_reward_score_mean"] == pytest.approx(.8)
    assert result["user_macro_unsupported_rate"] == pytest.approx(.25)


def test_sampling_diagnostic_preserves_draw_pairing():
    a = np.repeat(np.array([.2, .4, .8])[:, None], 4, axis=1)
    b = a + .1
    r = sampling_diagnostic(a, b, np.array([1, 1, 2]))
    assert r['difference'] == pytest.approx(.1)
    assert r['sampling_sd_of_user_difference'] == pytest.approx(0)
    with pytest.raises(ValueError):
        sampling_diagnostic(a, b[:, :2], np.array([1, 1, 2]))


def test_source_identity_rejects_missing_or_changed_source():
    from pathlib import Path
    from rl_recsys import experiment
    from rl_recsys.data import sha256
    hashes = {p.name: sha256(p) for p in Path(experiment.__file__).parent.glob('*.py')}
    verify_source_identity({'code_hashes': hashes})
    hashes['models.py'] = 'changed'
    with pytest.raises(ValueError, match='models.py'):
        verify_source_identity({'code_hashes': hashes})
