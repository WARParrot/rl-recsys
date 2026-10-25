import copy
import hashlib
import json
from pathlib import Path
import types
import zipfile

import numpy as np
import pytest
import torch

from rl_recsys import suite, experiment as exp
from rl_recsys.data import role
from rl_recsys.followup import split_calibration
from rl_recsys.models import Policy, RewardModel


def test_bundle_complete_zip64_source_and_hashes(tmp_path):
    root, source = tmp_path / 'run', tmp_path / 'src'
    (root / 'seed_42').mkdir(parents=True)
    source.mkdir()
    (source / 'rl_recsys').mkdir()
    (source / 'rl_recsys' / 'x.py').write_text('source')
    (root / 'seed_42' / 'weights.pt').write_bytes(b'weights')
    (root / 'suite.json').write_text('{"status":"failed"}')
    destination = suite.bundle_suite(root, source)
    with zipfile.ZipFile(destination) as z:
        checks = json.loads(z.read('SHA256SUMS.json'))
        assert {e['path'] for e in checks} == {'seed_42/weights.pt', 'suite.json', 'source/rl_recsys/x.py'}
        for e in checks:
            assert hashlib.sha256(z.read(e['path'])).hexdigest() == e['sha256']
    with pytest.raises(FileExistsError):
        suite.bundle_suite(root, source)


@pytest.mark.parametrize('samples', [None, (51, 52)])
@pytest.mark.parametrize('gate_passed', [True, False])
def test_suite_freezes_all_before_test_and_keeps_gate_failures(monkeypatch, tmp_path, samples, gate_passed):
    config = Path(__file__).resolve().parents[1] / 'configs/pilot.json'
    calls = []
    def run(argv, **kwargs):
        command = argv[4:]
        name = command[0]
        calls.append(command)
        if name == 'followup':
            p = Path(command[command.index('--output') + 1]); p.mkdir()
            (p / 'followup_manifest.json').write_text(json.dumps({'seeds': {
                str(seed): {'status': 'training complete' if gate_passed else 'reward gate failed; continued-SFT control only'}
                for seed in (42, 43, 44)}}))
        if name == 'evaluate-followup':
            p = Path(command[command.index('--output') + 1]); p.mkdir()
            (p / 'results.json').write_text('{"conditional_seed_effects":{}}')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(suite.subprocess, 'run', run)
    monkeypatch.setattr(suite, 'bundle_suite', lambda *a: None)
    out = tmp_path / 'suite.v1'
    result = suite.run_suite(config, tmp_path / 'raw', out, steps=1, sample_seeds=samples)
    names = [x[0] for x in calls]
    count = len(samples) if samples else 1
    assert max(i for i, name in enumerate(names) if name in {'followup', 'train', 'prepare'}) < names.index('evaluate-followup')
    assert names.count('train') == count and names.count('evaluate-followup') == count
    assert 'evaluate' not in names and 'reanalyze' not in names
    for call in calls:
        if call[0] == 'train':
            assert '--sft-only' in call
        if call[0] == 'followup':
            assert '--training-only-parent' in call
            assert call[call.index('--seeds') + 1:call.index('--steps')] == ['42', '43', '44']
    for sample in samples or (42,):
        cfg = json.loads((out / f'sample_{sample}/config.json').read_text())
        assert cfg['reward_audit']['independent_calibration']
        assert cfg['seed'] == 42 and cfg['sample_seed'] == sample
        statuses = result['baselines'][str(sample)]['optimization_statuses']
        assert len(statuses) == 3
        assert all(('continued-SFT' in status) == (not gate_passed) for status in statuses.values())
    assert all(stage['status'] == 'complete' for stage in result['stages'])
    assert result['bundle_status'] == 'complete'
    with pytest.raises(FileExistsError):
        suite.run_suite(config, tmp_path / 'raw', out)


def test_suite_error_not_success_and_failure_bundle(monkeypatch, tmp_path):
    config = Path(__file__).resolve().parents[1] / 'configs/pilot.json'
    monkeypatch.setattr(suite.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=7))
    bundles = []
    monkeypatch.setattr(suite, 'bundle_suite', lambda *a: bundles.append(a))
    with pytest.raises(RuntimeError, match='exit 7'):
        suite.run_suite(config, tmp_path/'raw', tmp_path/'suite', seeds=(42,))
    assert json.loads((tmp_path/'suite/suite.json').read_text())['status'] == 'failed'
    assert len(bundles) == 1


def test_real_tiny_supervised_rm_rl_with_eligible_targets(tmp_path):
    # Real optimization catches the incorrectly alleged UNK production-path bug.
    torch.set_num_threads(1)
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/pilot.json').read_text())
    cfg['train'].update(batch_size=4, eval_every=1, progress_every=1, user_uniform=True, rl_steps=1)
    cfg['model'].update(hidden=8, layers=1, heads=1, dropout=0.)
    raw = {'fit_context': np.array([[0,2],[0,3],[2,3],[3,2]]),
           'fit_target': np.array([1,2,3,2]), 'fit_uid': np.array([1,1,2,2]),
           'fit_timestamp': np.array([1,2,3,4]), 'fit_reward': np.array([.3,.5,.8,.7],dtype=np.float32),
           'fit_seconds': np.array([.1,.2,.3,.4],dtype=np.float32)}
    fit = role(raw,'fit',True)
    assert len(fit['target']) == 3 and (fit['target'] >= 2).all()
    early, late = split_calibration(fit)
    assert early['timestamp'].max() < late['timestamp'].min()
    import time
    deadline = time.monotonic()+30
    p = Policy(4,2,cfg['model'])
    exp.fit_supervised(p,fit,early,1,tmp_path/'sft.pt',cfg,'cpu',deadline)
    rm = RewardModel(4,2,cfg['model'])
    exp.fit_supervised(rm,fit,early,1,tmp_path/'rm.pt',cfg,'cpu',deadline,reward=True)
    exp.train_rl(p,exp.freeze(copy.deepcopy(p)),exp.freeze(rm),fit,cfg,'cpu',.01,1,0,deadline)
    assert (tmp_path/'sft.pt').exists() and all(torch.isfinite(x).all() for x in p.parameters())
