"""Offline regression coverage for completion-only objective and imported study."""
import copy
import json
from pathlib import Path
import time
import types

import numpy as np
import pytest
import torch

from rl_recsys import experiment as exp
from rl_recsys import completion_study as study
from rl_recsys.models import RewardModel
from rl_recsys.data import sha256
from test_followup import subset


def config():
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/pilot.json').read_text())
    cfg['train'].update(reward_objective='completion', allow_cpu=True, threads=1,
                        batch_size=4, eval_every=1, progress_every=1)
    cfg['model'].update(hidden=8, heads=1, layers=1, dropout=0.)
    cfg['evaluation'].update(batch_size=4, bootstrap=8)
    return cfg


def test_loss_excludes_seconds_and_has_zero_secondary_gradient():
    cfg = config()
    prediction = torch.tensor([[.2, .7], [.4, .9]], requires_grad=True)
    completion = torch.tensor([.8, .1])
    seconds = torch.tensor([float('nan'), float('nan')])
    loss = exp.reward_squared_error(prediction, completion, seconds, cfg).mean()
    torch.testing.assert_close(loss, (prediction[:, 0] - completion).square().mean())
    loss.backward()
    assert torch.equal(prediction.grad[:, 1], torch.zeros(2))
    assert prediction.grad[:, 0].abs().sum() > 0
    cfg['train']['reward_objective'] = 'joint'
    seconds = torch.tensor([.3, .5])
    torch.testing.assert_close(exp.reward_squared_error(prediction, completion, seconds, cfg).mean(),
                               torch.nn.functional.mse_loss(prediction, torch.stack([completion, seconds], -1)))
    cfg['train']['reward_objective'] = 'typo'
    with pytest.raises(ValueError):
        exp.reward_squared_error(prediction, completion, seconds, cfg)


def test_real_cpu_completion_train_select_restore_and_architecture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = config()
    model = RewardModel(5, 2, cfg['model'])
    fit = subset()
    fit['seconds'][:] = float('nan')  # ignored by the completion objective
    path = tmp_path / 'reward.pt'
    logs = exp.fit_supervised(model, fit, fit, 3, path, cfg, 'cpu', time.monotonic() + 30, reward=True)
    assert len(logs) == 3
    assert all(np.isfinite(row['calibration_loss']) for row in logs)
    assert exp.validation_loss(model, fit, cfg, 'cpu', reward=True) == pytest.approx(
        min(row['calibration_loss'] for row in logs))
    # Same original architecture/state dict; existing evaluator can restore it.
    restored = exp.restore(path, RewardModel(5, 2, cfg['model']), 'cpu')
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key])
    assert restored(torch.tensor([[0, 2]]), torch.tensor([2])).shape == (1, 2)
    with torch.no_grad():
        prediction = restored(torch.as_tensor(fit['context']), torch.as_tensor(fit['target']))[:, 0].numpy()
    expected = exp.macro((prediction - fit['reward']) ** 2, fit['uid'])[0]
    assert exp.validation_loss(restored, fit, cfg, 'cpu', reward=True) == pytest.approx(expected)


def test_untrained_seconds_never_admitted_even_if_point_mae_passes():
    cfg = config()
    audit = {'heads': {head: {'mae': .1, 'constant_mae': .5,
                             'paired_mae': {'relative_difference': -.5}}
                       for head in ['completion', 'seconds']}}
    gate = exp.reward_gate(audit, cfg)
    assert gate['enabled_rewards'] == ['completion']
    assert not gate['heads']['seconds']['enabled']


def parent_fixture(tmp_path):
    parent = tmp_path / 'original research'
    sample = parent / 'sample_42'
    run, data, joint = sample / 'run', sample / 'data', sample / 'followup' / 'seed_42'
    for directory in [run, data, joint]:
        directory.mkdir(parents=True)
    for name in ['SFT_READY.json', 'provenance.json', 'training.json', 'sft.pt']:
        (run / name).write_text('{}')
    for name in ['prepared.npz', 'audit.json']:
        (data / name).write_bytes(b'original')
    audit = {'heads': {'completion': {'mae': .35}}}
    for name in ['reward.pt', 'item_only.pt', 'reward_audit.json', 'reward_curves.json', 'config.json']:
        (joint / name).write_text(json.dumps(audit) if name == 'reward_audit.json' else '{}')
    manifest = {'seeds': {'42': {'checkpoint_hashes': {p.name: sha256(p) for p in joint.glob('*.pt')},
                               'artifact_hashes': {p.name: sha256(p) for p in joint.glob('*.json')}}}}
    (joint.parent / 'provenance.json').write_text(json.dumps({
        'prepared_sha256': 'prepared', 'historical_provenance_sha256': 'provenance',
        'historical_sft_sha256': sha256(run / 'sft.pt')}))
    (joint.parent / 'followup_manifest.json').write_text(json.dumps(manifest))
    (joint.parent / 'FOLLOWUP_COMPLETE.json').write_text(json.dumps({
        'status': 'complete', 'provenance_sha256': sha256(joint.parent / 'provenance.json'),
        'manifest_sha256': sha256(joint.parent / 'followup_manifest.json')}))
    return parent


def test_study_order_import_identities_bundle_and_no_original_writes(monkeypatch, tmp_path):
    parent = parent_fixture(tmp_path)
    before = {str(p): sha256(p) for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(study, 'verify_inputs', lambda *a, **kw: {'prepared_sha256': 'prepared', 'historical_provenance_sha256': 'provenance'})
    out = tmp_path / 'new output.v1'
    calls = []
    def run(argv, **kw):
        calls.append(argv)
        if 'rl_recsys.followup' in argv:
            assert '--source-mode' in argv and 'historical_reanalysis' in argv
            assert '--reward-objective' in argv and 'completion' in argv
            directory = out / 'followup' / 'seed_42'
            directory.mkdir(parents=True)
            (directory / 'reward_audit.json').write_text(json.dumps({'heads': {'completion': {'mae': .34}},
                'gate': {'heads': {'completion': {'enabled': False}}}}))
        else:
            assert (out / 'followup' / 'seed_42' / 'reward_audit.json').exists()
            assert argv[4] == 'evaluate-followup'
            assert '--source-mode' not in argv  # new follow-up must match current source
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(study.subprocess, 'run', run)
    monkeypatch.setattr(study, 'bundle_suite', lambda root, source: calls.append('bundle'))
    result = study.run_study(parent, out)
    assert result['status'].startswith('complete')
    assert calls[-1] == 'bundle' and len(calls) == 3
    assert 'Gate failed' in json.loads((out / 'reward-comparison.json').read_text())['conclusion']
    assert before == {str(p): sha256(p) for p in parent.rglob('*') if p.is_file()}
    assert (out / 'inputs' / 'run' / 'sft.pt').exists()
    with pytest.raises(FileExistsError):
        study.run_study(parent, out)
    with pytest.raises(ValueError, match='separate trees'):
        study.run_study(parent, parent / 'nested')


def test_study_failure_is_bundled_not_success(monkeypatch, tmp_path):
    parent = parent_fixture(tmp_path)
    monkeypatch.setattr(study, 'verify_inputs', lambda *a, **kw: {
        'prepared_sha256': 'prepared', 'historical_provenance_sha256': 'provenance'})
    monkeypatch.setattr(study.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=3))
    bundled = []
    monkeypatch.setattr(study, 'bundle_suite', lambda *a: bundled.append(True))
    out = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='train failed'):
        study.run_study(parent, out)
    assert bundled == [True]
    assert json.loads((out / 'study.json').read_text())['status'] == 'failed'


def test_timeout_is_failure_with_recorded_cap(monkeypatch, tmp_path):
    parent = parent_fixture(tmp_path)
    monkeypatch.setattr(study, 'verify_inputs', lambda *a, **kw: {
        'prepared_sha256': 'prepared', 'historical_provenance_sha256': 'provenance'})
    def timeout(argv, **kw):
        assert kw['timeout'] == 4.5 * 3600
        raise study.subprocess.TimeoutExpired(argv, kw['timeout'])
    monkeypatch.setattr(study.subprocess, 'run', timeout)
    monkeypatch.setattr(study, 'bundle_suite', lambda *a: None)
    out = tmp_path / 'timeout'
    with pytest.raises(study.subprocess.TimeoutExpired):
        study.run_study(parent, out)
    status = json.loads((out / 'study.json').read_text())
    assert status['status'] == 'failed'
    assert status['stages'][0]['status'] == 'failed'
    assert 'TimeoutExpired' in status['stages'][0]['error']


def test_original_joint_provenance_tamper_rejected_before_output(monkeypatch, tmp_path):
    parent = parent_fixture(tmp_path)
    monkeypatch.setattr(study, 'verify_inputs', lambda *a, **kw: {
        'prepared_sha256': 'prepared', 'historical_provenance_sha256': 'provenance'})
    (parent / 'sample_42/followup/provenance.json').write_text('{}')
    out = tmp_path / 'out'
    with pytest.raises(ValueError, match='provenance mismatch'):
        study.run_study(parent, out)
    assert not out.exists()
