import copy

import numpy as np
import pytest
import torch

from rl_recsys.experiment import reward_report, reward_gate
from rl_recsys.reward_audit import audit_groups, label_summary, score_tails


EVAL = dict(batch_size=3, bootstrap=20, bootstrap_seed=2026)


def test_tails_include_boundary_ties_and_explicit_rows():
    scores = np.r_[np.full(97, .2), [.8, .9, .9]]
    tails = score_tails(scores, scores - .1, np.zeros(100), np.ones(100), np.arange(100), EVAL)
    assert tails['top1']['row_indices'] == [98, 99]
    assert tails['top1']['requested_n'] == 1
    assert tails['top1']['cutoff_tie_n'] == 2
    assert tails['top1']['realized_fraction'] == .02
    assert tails['top1']['mae'] == .9
    assert tails['top1']['bias'] == .9
    assert tails['top1']['overprediction'] == .9
    assert tails['top1']['insufficiency_flag']
    assert tails['top10']['n'] == 100  # cutoff falls in a large tie
    assert not tails['top10']['insufficiency_flag']
    tiny = score_tails(np.array([.5]), np.array([.4]), np.array([.7]),
                       np.array([.1]), np.array([1]), EVAL)['top1']
    assert tiny['insufficient_examples'] and tiny['insufficient_users']
    assert tiny['overprediction'] == 0
    assert tiny['bias'] == pytest.approx(-.2)


def test_fit_only_popularity_ties_duration_boundaries_and_unknowns():
    fit = {'target': np.array([2, 2, 3, 4, 5, 6, 7])}
    subset = dict(target=np.arange(1, 9), context=np.array([[0, t] for t in range(1, 9)]),
                  organic=np.zeros(8))
    data = dict(popularity=np.array([1, 1, 1, 2, 2, 9]),
                durations=np.array([179, 180, 299, 300, 301, 100]))
    masks, meta = audit_groups(fit, subset, data)
    assert meta['popularity']['cut_points'] == [1, 2]
    assert np.flatnonzero(masks['popularity_low']).tolist() == [1, 2, 3]
    assert np.flatnonzero(masks['popularity_middle']).tolist() == [4, 5]
    assert np.flatnonzero(masks['popularity_high']).tolist() == [6]
    assert np.flatnonzero(masks['duration_180_to_300s']).tolist() == [2, 3]
    assert np.flatnonzero(masks['duration_ge_300s']).tolist() == [4, 5]
    assert not masks['repeat'][0]  # unknown token is not an observed item repeat
    changed = copy.deepcopy(subset)
    changed['target'] = np.array([7] * 8)
    assert audit_groups(fit, changed, data)[1] == meta
    fallback, metadata = audit_groups(fit, subset)
    assert metadata['popularity']['source'] == 'supplied fit target counts'
    assert not metadata['duration']['available']
    assert 'duration_lt_180s' not in fallback
    assert metadata['popularity']['cut_points'] == [1, 1]
    assert not fallback['popularity_middle'].any()


def test_slice_user_macro_bias_comparators_and_empty_support():
    row = label_summary(np.ones(3, dtype=bool), np.array([.2, .4, .9]),
                        np.array([.1, .1, .1]), np.zeros(3), np.ones(3),
                        np.array([1, 1, 2]), EVAL)
    assert row['n'] == 3 and row['users'] == 2
    assert row['mae'] == pytest.approx(.6)
    assert row['bias'] == pytest.approx(.6)
    assert row['auditor_mae'] == pytest.approx(.1)
    assert row['paired_mae']['difference'] == pytest.approx(-.4)
    assert row['auditor_paired_mae']['difference'] == pytest.approx(.5)
    empty = label_summary(np.zeros(3, dtype=bool), None, None, None, None,
                          np.array([1, 1, 2]), EVAL)
    assert empty['n'] == empty['users'] == 0
    assert empty['mae'] is None and empty['paired_mae'] is None


class ObservedReward(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        self.seen = []

    def forward(self, context, target):
        assert not self.training and not torch.is_grad_enabled()
        self.seen.extend(target.tolist())
        return torch.stack([target.float() / 10 + self.offset,
                            target.float() / 20 + self.offset], -1)


def test_rmse_comparators_unequal_user_counts_and_root_per_replicate():
    from rl_recsys.evaluate import macro
    subset = dict(target=np.array([2, 3, 4]), uid=np.array([1, 1, 2]),
                  context=np.array([[0, 2], [0, 2], [3, 4]]),
                  reward=np.array([0., .1, .8]), seconds=np.array([0., .1, .4]),
                  organic=np.array([1, 0, 1]))
    fit = copy.deepcopy(subset)
    fit['reward'] = np.array([.1, .3, .5])
    cfg = dict(evaluation=EVAL, rl=dict(seconds_cap=600))
    result = reward_report(ObservedReward(), ObservedReward(.05), fit, subset, cfg, 'cpu')
    y, uid = subset['reward'], subset['uid']
    pred = np.array([.2, .3, .4])
    assert result['rmse'] == pytest.approx(np.sqrt(macro((pred-y)**2, uid)[0]))
    assert result['rmse'] != pytest.approx(np.sqrt(np.mean((pred-y)**2)))
    for name, values in [('constant', np.full(3, fit['reward'].mean())),
                         ('item_mean', fit['reward']), ('auditor', pred + .05)]:
        a = macro((values-y)**2, uid)[1]
        b = macro((pred-y)**2, uid)[1]
        rng = np.random.default_rng(EVAL['bootstrap_seed'])
        draws = []
        for _ in range(EVAL['bootstrap']):
            idx = rng.integers(2, size=2)
            draws.append(np.sqrt(b[idx].mean()) - np.sqrt(a[idx].mean()))
        stat = result[f'history_vs_{name}_paired_rmse']
        assert result[f'{name}_rmse'] == pytest.approx(np.sqrt(a.mean()))
        assert stat['difference'] == pytest.approx(np.sqrt(b.mean()) - np.sqrt(a.mean()))
        np.testing.assert_allclose(stat['ci95'], np.quantile(draws, [.025, .975]), atol=2e-8)
        assert result[f'history_vs_{name}_paired_mae']['users'] == 2


def test_report_additions_preserve_historical_metrics_and_gate():
    subset = dict(target=np.array([2, 3, 4]), uid=np.array([1, 1, 2]),
                  context=np.array([[0, 2], [0, 2], [3, 4]]),
                  reward=np.array([.1, .2, .3]), seconds=np.array([.05, .1, .15]),
                  organic=np.array([1, 0, 1]))
    cfg = dict(evaluation=EVAL, rl=dict(seconds_cap=600, rewards=['completion']),
               reward_audit=dict(min_relative_improvement=.02))
    rm, auditor = ObservedReward(), ObservedReward(.05)
    result = reward_report(rm, auditor, subset, subset, cfg, torch.device('cpu'))
    assert rm.seen == auditor.seen == [2, 3, 4]  # no hypothetical action labels
    assert result['mae'] == result['heads']['completion']['mae'] == pytest.approx(.1)
    assert result['item_mean_mae'] == 0
    assert result['auditor_paired_mae']['before'] == 0  # old comparator unchanged
    assert result['history_vs_auditor_paired_mae']['difference'] == pytest.approx(-.05)
    bins = result['calibration']
    assert sum(b['n'] for b in bins) == 3
    for b in bins:
        assert b['bias'] == pytest.approx(b['predicted'] - b['observed'])
        assert b['users'] >= 1
    assert result['slices']['repeat']['n'] == 2
    assert result['slices']['novel']['n'] == 1
    assert result['heads']['seconds']['mae_seconds_proxy'] == pytest.approx(30)
    gate = reward_gate(result, cfg)
    reduced = {'heads': {name: {key: value for key, value in head.items()
                               if key in ('mae', 'constant_mae', 'paired_mae')}
                         for name, head in result['heads'].items()}}
    assert reward_gate(reduced, cfg) == gate
