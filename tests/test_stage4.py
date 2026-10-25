"""Stage4 orchestration and training-only integrity regressions (CPU, offline)."""
import json
from pathlib import Path

import pytest

from rl_recsys import suite
from rl_recsys.data import sha256
from rl_recsys.reanalysis import verify_inputs
from test_reanalysis import fixture_run
from test_followup_evaluate import frozen_followup
from rl_recsys import followup_evaluate as fe


@pytest.mark.parametrize('mode', ['ready', 'sft', 'gate_failed'])
@pytest.mark.parametrize('tamper', [None, 'sft.pt', 'provenance.json', 'prepared.npz'])
def test_training_only_without_test_artifacts_and_tamper(tmp_path, mode, tamper):
    run, data, _, _ = fixture_run(tmp_path)
    for name in ['results.json', 'per_example.npz', 'evaluation-provenance.json', 'EVALUATION_COMPLETE.json']:
        (run / name).unlink()
    if mode != 'ready':
        (run / 'READY_FOR_TEST.json').unlink()
    if mode == 'sft':
        marker = {'checkpoint_hashes': {'sft.pt': sha256(run / 'sft.pt')},
                  'prepared_sha256': sha256(data / 'prepared.npz'),
                  'prepared_audit_sha256': sha256(data / 'audit.json'),
                  'provenance_sha256': sha256(run / 'provenance.json'),
                  'training_sha256': sha256(run / 'training.json')}
        (run / 'SFT_READY.json').write_text(json.dumps(marker))
    elif mode == 'gate_failed':
        for name in ['reward.pt', 'auditor.pt']:
            (run / name).write_bytes(b'weights')
        (run / 'reward_audit.json').write_text(json.dumps({
            'gate': {'status': 'failed'},
            'provenance_sha256': sha256(run / 'provenance.json'),
            'checkpoint_hashes': {name: sha256(run / name) for name in ['sft.pt', 'reward.pt', 'auditor.pt']}}))
    if tamper:
        target = data / tamper if tamper == 'prepared.npz' else run / tamper
        target.write_bytes(target.read_bytes() + b' ')
        with pytest.raises(ValueError, match='hash mismatch'):
            verify_inputs(run, data, training_only=True)
    else:
        result = verify_inputs(run, data, training_only=True)
        assert result['historical_results_sha256'] is None
        assert result['status'] == 'training inputs verified'
        assert not (run / 'results.json').exists()


def test_real_cpu_sft_only_parent_never_constructs_reward_or_reads_test(monkeypatch, tmp_path):
    import numpy as np
    from rl_recsys import experiment as exp
    cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/pilot.json').read_text())
    cfg['train'].update(allow_cpu=True, sft_steps=1, threads=1, batch_size=2, eval_every=1)
    cfg['model'].update(hidden=8, heads=1, layers=1, dropout=0.)
    cfg['reward_audit'] = {'independent_calibration': True}
    data_dir, run = tmp_path / 'data', tmp_path / 'run'
    data_dir.mkdir()
    arrays = {'vocab': np.array([10, 20]), 'popularity': np.array([2., 1.]), 'durations': np.array([100., 200.])}
    for name in ['fit', 'calibration', 'selection']:
        fields = {'context': np.array([[0, 2], [0, 3], [2, 3], [3, 2]]),
                  'target': np.array([2, 3, 2, 3]), 'uid': np.array([1, 1, 2, 2]),
                  'timestamp': np.arange(4), 'reward': np.array([.3, .4, .5, .6], dtype='float32'),
                  'seconds': np.array([.1, .2, .3, .4], dtype='float32')}
        arrays.update({f'{name}_{key}': value for key, value in fields.items()})
    np.savez(data_dir / 'prepared.npz', **arrays)  # Deliberately NO test role.
    (data_dir / 'audit.json').write_text(json.dumps({'config': cfg}))
    monkeypatch.setattr(exp, 'provenance', lambda root, config, data, device: {
        'config': config, 'prepared_sha256': sha256(data / 'prepared.npz'),
        'audit_sha256': sha256(data / 'audit.json')})
    monkeypatch.setattr(exp, 'RewardModel', lambda *a, **kw: pytest.fail('SFT parent constructed RM'))
    exp.train(cfg, data_dir, run, sft_only=True)
    assert (run / 'sft.pt').exists() and (run / 'SFT_READY.json').exists()
    assert not (run / 'READY_FOR_TEST.json').exists()
    assert verify_inputs(run, data_dir, training_only=True)['checkpoint_hashes'].keys() == {'sft.pt'}
    with pytest.raises(FileExistsError):
        exp.train(cfg, data_dir, run, sft_only=True)


def test_bundle_failure_persists_error_and_scientific_failure(monkeypatch, tmp_path):
    import types
    monkeypatch.setattr(suite.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=7))
    def fail(*a):
        raise OSError('disk full')
    monkeypatch.setattr(suite, 'bundle_suite', fail)
    out = tmp_path / 'run.v2'
    config = Path(__file__).resolve().parents[1] / 'configs/pilot.json'
    with pytest.raises(OSError, match='disk full'):
        suite.run_suite(config, tmp_path / 'raw', out)
    row = json.loads((out / 'suite.json').read_text())
    assert row['status'] == 'bundle_failed' and row['bundle_status'] == 'failed'
    assert row['scientific_status'] == 'failed'
    assert 'exit 7' in row['error']
    assert row['stages'][0]['status'] == 'failed'
    assert (out / 'bundle-error.txt').exists()


@pytest.mark.parametrize('enabled', [True, False])
def test_common_rm_multiple_optimization_seeds_and_failed_gate(frozen_followup, enabled):
    import shutil
    prepared, run, root, out, _, manifest = frozen_followup
    prov = json.loads((root / 'provenance.json').read_text())
    prov.update(experiment='A', seeds=[42, 43, 44])
    (root / 'provenance.json').write_text(json.dumps(prov))
    manifest['provenance_sha256'] = sha256(root / 'provenance.json')
    source = root / 'seed_42'
    for seed in [42, 43, 44]:
        directory = root / f'seed_{seed}'
        if seed != 42:
            shutil.rmtree(directory, ignore_errors=True)
            shutil.copytree(source, directory)
        audit = {'gate': {'status': 'passed' if enabled else 'failed',
                         'heads': {'completion': {'enabled': enabled}}}}
        (directory / 'reward_audit.json').write_text(json.dumps(audit))
        policies = json.loads((directory / 'selection.json').read_text())['policies']
        if not enabled:
            policies = {'continued_sft': policies['continued_sft']}
        (directory / 'selection.json').write_text(json.dumps({'policies': policies}))
        manifest['seeds'][str(seed)] = {'policies': policies,
            'checkpoint_hashes': {p.name: sha256(p) for p in directory.glob('*.pt')},
            'artifact_hashes': {p.name: sha256(p) for p in directory.glob('*.json')}}
    (root / 'followup_manifest.json').write_text(json.dumps(manifest))
    (root / 'FOLLOWUP_COMPLETE.json').write_text(json.dumps({'status': 'complete',
        'manifest_sha256': sha256(root / 'followup_manifest.json'),
        'provenance_sha256': sha256(root / 'provenance.json')}))
    result = fe.evaluate_followup(prepared, run, root, out, batch_size=2, max_minutes=2)
    assert result['conditional_seed_effects']['continued_sft']['ndcg']['available_seeds'] == 3
    assert result['conditional_seed_effects']['continued_sft']['ndcg']['seed_sd'] is not None
    for row in result['seeds'].values():
        assert ('completion_h1' in row['policies']) == enabled
        assert 'continued_sft' in row['policies']


def test_dotted_bundle_name_is_preserved(tmp_path):
    root = tmp_path / 'research.v1'
    source = tmp_path / 'src'
    root.mkdir(); source.mkdir()
    (root / 'log.txt').write_text('complete log')
    result = suite.bundle_suite(root, source)
    assert result == tmp_path / 'research.v1.zip'
    assert (tmp_path / 'research.v1.zip.sha256.json').exists()
    assert not (tmp_path / 'research.zip').exists()
