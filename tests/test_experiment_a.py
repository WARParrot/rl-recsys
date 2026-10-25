"""Experiment A CPU loss/selection contracts; no downloads or GPU jobs."""
import copy
import json
import time

import numpy as np
import pytest
import torch

from rl_recsys import followup as fu
from test_followup import TinyPolicy, TinyRM, small_config, subset


@pytest.mark.parametrize('training_only', [True, False])
def test_source_mismatch_stops_before_training_or_test_verification(monkeypatch, tmp_path, training_only):
    run = tmp_path / 'parent'
    run.mkdir()
    (run / 'provenance.json').write_text(json.dumps({'config': small_config(), 'code_hashes': {}}))
    from rl_recsys import reanalysis
    monkeypatch.setattr(reanalysis, 'verify_inputs', lambda *a, **kw: pytest.fail('verification/test access'))
    monkeypatch.setattr(fu, 'load_prepared', lambda *a: pytest.fail('data access'))
    monkeypatch.setattr(fu.exp, 'get_device', lambda *a: pytest.fail('training initialization'))
    with pytest.raises(ValueError, match='Source code differs'):
        fu.run_followup(tmp_path, run, tmp_path / 'out', training_only=training_only)
    assert not (tmp_path / 'out').exists()


def test_explicit_historical_source_mode_allows_parent_import_only(monkeypatch, tmp_path):
    run = tmp_path / 'parent'
    run.mkdir()
    (run / 'provenance.json').write_text(json.dumps({'config': small_config(), 'code_hashes': {}}))
    from rl_recsys import reanalysis
    calls = []
    monkeypatch.setattr(reanalysis, 'verify_inputs', lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(fu.exp, 'get_device', lambda *a: torch.device('cpu'))
    with pytest.raises(RuntimeError, match='requires CUDA'):
        fu.run_followup(tmp_path, run, tmp_path / 'out', training_only=True,
                        source_mode='historical_reanalysis')
    assert calls == [{'training_only': True}]


def test_mixed_loss_zero_equivalence_logged_targets_and_nonzero_ce_gradient(monkeypatch, tmp_path):
    cfg = small_config()
    cfg['train']['eval_every'] = 1
    fit = subset()
    fit['target'] = np.array([1, 3, 3, 3, 3, 3])  # exclude unknown; token 3 => class 1
    monkeypatch.setattr(fu.exp, 'evaluate', lambda *a, **kw: (
        {'eligible_ndcg': .5, 'eligible_exact_one_step_expected_reward': .5}, {}))
    gradients = []
    original = fu.exp.optimize_step
    def optimize(loss, policy, optimizer, config):
        gradients.append(torch.autograd.grad(loss, policy.logits, retain_graph=True)[0].clone())
        return original(loss, policy, optimizer, config)
    monkeypatch.setattr(fu.exp, 'optimize_step', optimize)
    logs = []
    for index, coefficient in enumerate([None, 0., 1.]):
        torch.manual_seed(100)
        policy = TinyPolicy()
        reference, rm = copy.deepcopy(policy), TinyRM()
        directory = tmp_path / str(index)
        directory.mkdir()
        kwargs = {} if coefficient is None else {'lambda_ce': coefficient}
        fu.train_policy(policy, reference, rm, rm, fit, subset(), {}, cfg, 'cpu', directory,
                        'mixed', 1, time.monotonic() + 30, **kwargs)
        logs.append(json.loads((directory / 'mixed_curve.json').read_text())[0])
        assert reference.logits.grad is None and rm.bias.grad is None
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    expected_ce = torch.tensor([.1, -.1]).softmax(0) - torch.tensor([0., 1.])
    torch.testing.assert_close(gradients[2] - gradients[1], expected_ce)
    assert logs[0]['loss'] == logs[1]['loss']
    assert logs[2]['loss'] == pytest.approx(logs[2]['rl_loss'] + logs[2]['ce_loss'])
    assert logs[2]['ce_loss'] == pytest.approx(float(torch.nn.functional.cross_entropy(
        torch.tensor([[.1, -.1]]), torch.tensor([1]))))


def test_rerank_selection_curve_common_floor_ties_and_no_fallback(monkeypatch):
    cfg = small_config()
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((args[5]['rl'].get('rerank_weight'), kwargs.get('mode')))
        w = args[5]['rl'].get('rerank_weight', 0.)
        return {'eligible_ndcg': .5 if w != 10 else .1,
                'eligible_exact_one_step_expected_reward': .6 if w else .4}, {}
    monkeypatch.setattr(fu.exp, 'evaluate', evaluate)
    result = fu.tune_rerank(None, None, None, {}, {}, cfg, 'cpu', time.monotonic() + 10)
    assert [c['weight'] for c in result['candidates']] == [0., .1, 1., 10.]
    assert result['selected']['weight'] == .1  # tie lower coefficient; high score fails floor
    assert calls[1:] == [(w, 'rerank') for w in [0., .1, 1., 10.]]
    candidates = copy.deepcopy(result['candidates'])
    for candidate in candidates:
        candidate['selection']['eligible_ndcg'] = 0.
    assert fu.exp.select_candidate(candidates, result['selection_baseline'], cfg) is None
