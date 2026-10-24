import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from rl_recsys.artifacts import complete_evaluation
from rl_recsys.data import sha256
from rl_recsys.reanalysis import (import_run, verify_inputs, reconstruct_metrics,
                                  export_evaluation_inputs, bundle_reanalysis)


def fixture_run(tmp_path):
    root = tmp_path / 'source'
    run, data = root / 'runs/pilot-seed42', root / 'data/prepared'
    run.mkdir(parents=True)
    data.mkdir(parents=True)
    arrays = {'vocab': np.array([10, 20]), 'popularity': np.array([2., 1.]), 'durations': np.array([100., 200.])}
    for name in ['selection', 'test']:
        arrays.update({f'{name}_context': np.array([[0, 2], [2, 3], [3, 2]]),
                       f'{name}_target': np.array([2, 1, 3]), f'{name}_uid': np.array([1, 1, 2]),
                       f'{name}_timestamp': np.array([1, 2, 3]), f'{name}_reward': np.array([1., .5, 0.])})
    np.savez(data / 'prepared.npz', **arrays)
    (data / 'audit.json').write_text('{}')
    (run / 'sft.pt').write_bytes(b'fixture checkpoint; never torch.load')
    cfg = {'evaluation': {'bootstrap': 20, 'bootstrap_seed': 2026}}
    prov = {'config': cfg, 'prepared_sha256': sha256(data / 'prepared.npz'), 'audit_sha256': sha256(data / 'audit.json')}
    for name, value in [('provenance', prov), ('selection', {'chosen': None}), ('candidates', []), ('training', {}), ('reward_audit', {})]:
        (run / f'{name}.json').write_text(json.dumps(value))
    ready = {'checkpoint_hashes': {'sft.pt': sha256(run / 'sft.pt')}, 'selected_checkpoint': None,
             'prepared_audit_sha256': sha256(data / 'audit.json')}
    ready.update({f'{n}_sha256': sha256(run / f'{n}.json') for n in ['provenance', 'selection', 'candidates', 'training', 'reward_audit']})
    (run / 'READY_FOR_TEST.json').write_text(json.dumps(ready))
    raw = {'sft_recommendations': np.array([[2], [3], [3]]), 'sft_uid': arrays['test_uid'],
           'sft_recall': np.array([1., 0., 1.]), 'sft_ndcg': np.array([1., 0., 1.]),
           'sft_exact_one_step_expected_reward': np.array([.2, .4, .6])}
    results = {'policies': {'sft': {'recall': .75, 'ndcg': .75, 'eligible_ndcg': 1.,
                                   'eligible_exact_one_step_expected_reward': .4,
                                   'catalog_coverage': 1., 'recommendation_hhi': 5/9}}}
    complete_evaluation(run, results, raw, {})
    checks = []
    bundle = tmp_path / 'input.zip'
    with zipfile.ZipFile(bundle, 'w') as z:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix not in {'.pt', '.npz'} or p.name == 'per_example.npz':
                name = p.relative_to(root).as_posix()
                z.write(p, name)
                checks.append({'path': name, 'sha256': sha256(p)})
        z.writestr('SHA256SUMS.json', json.dumps(checks))
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'sft.pt').write_bytes((run / 'sft.pt').read_bytes())
    (assets / 'prepared.npz').write_bytes((data / 'prepared.npz').read_bytes())
    return run, data, bundle, assets


def test_import_and_reconstruct_from_actual_targets(tmp_path):
    run, data, bundle, assets = fixture_run(tmp_path)
    result = import_run(bundle, assets, tmp_path / 'imported')
    assert result['status'] == 'verified'
    imported = tmp_path / 'imported'
    reconstructed = reconstruct_metrics(imported / result['run'], imported / result['data'], tmp_path / 'verified.json')
    assert reconstructed['eligible_examples'] == 2
    assert reconstructed['policies']['sft']['eligible_ndcg'] == 1
    with pytest.raises(FileExistsError):
        import_run(bundle, assets, imported)
    with pytest.raises(FileExistsError):
        reconstruct_metrics(run, data, tmp_path / 'verified.json')


def test_reject_corrupted_checkpoints_and_metadata(tmp_path):
    run, data, bundle, assets = fixture_run(tmp_path)
    (assets / 'sft.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checkpoint'):
        import_run(bundle, assets, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    (run / 'selection.json').write_text('{}')
    with pytest.raises(ValueError, match='metadata'):
        verify_inputs(run, data)


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:\\absolute', 'a/../../escape', 'CON', 'logs/NUL.txt', 'name.'])
def test_reject_archive_traversal(tmp_path, name):
    p = tmp_path / 'bad.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr(name, 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        import_run(p, tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_complete_inputs_include_targets_eligibility_and_row_ids(tmp_path):
    _, data, _, _ = fixture_run(tmp_path)
    with np.load(data / 'prepared.npz') as source:
        export_evaluation_inputs(dict(source), tmp_path / 'inputs.npz')
    with np.load(tmp_path / 'inputs.npz', allow_pickle=False) as result:
        assert result['test_eligible'].tolist() == [True, False, True]
        assert result['test_target'].tolist() == [2, 1, 3]
        assert result['test_row_id'].tolist() == [0, 1, 2]


def test_manifest_checkpoint_path_traversal_rejected(tmp_path):
    run, data, _, _ = fixture_run(tmp_path)
    ready = json.loads((run / 'READY_FOR_TEST.json').read_text())
    ready['checkpoint_hashes'] = {'../sft.pt': 'x'}
    (run / 'READY_FOR_TEST.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError, match='Unsafe'):
        verify_inputs(run, data)


def test_bundle_contains_weights_and_prepared_data(tmp_path):
    _, _, bundle, assets = fixture_run(tmp_path)
    imported = tmp_path / 'imported'
    import_run(bundle, assets, imported)
    analysis = tmp_path / 'analysis'
    analysis.mkdir()
    (analysis / 'results.json').write_text('{}')
    (analysis / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'sha256': {'results.json': sha256(analysis / 'results.json')}}))
    result = bundle_reanalysis(imported, analysis, tmp_path / 'complete.zip')
    assert result['files'] > 0
    with zipfile.ZipFile(tmp_path / 'complete.zip') as z:
        assert 'historical/runs/pilot-seed42/sft.pt' in z.namelist()
        assert 'historical/data/prepared/prepared.npz' in z.namelist()
        for entry in json.loads(z.read('SHA256SUMS.json')):
            assert hashlib.sha256(z.read(entry['path'])).hexdigest() == entry['sha256']


def test_bundle_rejects_self_inclusion_before_creating_partial(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        bundle_reanalysis(tmp_path / 'historical', tmp_path / 'analysis', tmp_path / 'analysis' / 'result.zip')
    assert not (tmp_path / 'analysis').exists()


def test_case_insensitive_zip_collisions_rejected(tmp_path):
    p = tmp_path / 'bad.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('Run/a.json', '{}')
        z.writestr('run/A.JSON', '{}')
    with pytest.raises(ValueError, match='Duplicate'):
        import_run(p, tmp_path, tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_reconstruction_detects_bad_saved_relevance_even_with_valid_manifest(tmp_path):
    run, data, _, _ = fixture_run(tmp_path)
    with np.load(run / 'per_example.npz') as source:
        arrays = dict(source)
    arrays['sft_recall'] = np.zeros(3)
    np.savez(run / 'per_example.npz', **arrays)
    manifest = json.loads((run / 'EVALUATION_COMPLETE.json').read_text())
    manifest['sha256']['per_example.npz'] = sha256(run / 'per_example.npz')
    (run / 'EVALUATION_COMPLETE.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='recall disagrees'):
        reconstruct_metrics(run, data, tmp_path / 'out.json')
