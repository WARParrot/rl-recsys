"""Hash-bound A rerank round-trip, sharing legacy CPU bundle fixture."""
import copy
import json

import numpy as np
import pytest

from test_followup_evaluate import frozen_followup, write_json
from rl_recsys import followup_evaluate as fe
from rl_recsys.data import sha256


def seal(root, manifest):
    write_json(root / 'followup_manifest.json', manifest)
    write_json(root / 'FOLLOWUP_COMPLETE.json', {'status': 'complete',
        'manifest_sha256': sha256(root / 'followup_manifest.json'),
        'provenance_sha256': sha256(root / 'provenance.json')})


def test_rerank_frozen_before_test_same_rm_populations_and_pairwise(frozen_followup, monkeypatch):
    prepared, run, root, out, data, manifest = frozen_followup
    directory = root / 'seed_42'
    candidate = {'name': 'rerank_tuned', 'reward': 'completion', 'horizon': 1, 'step': 0,
                 'weight': .1, 'selection': {'eligible_ndcg': .5,
                 'eligible_exact_one_step_expected_reward': .6}}
    rerank = {'selected': candidate, 'candidates': [candidate],
              'selection_baseline': {'eligible_ndcg': .5}}
    row = manifest['seeds']['42']
    row['rerank'] = rerank
    selection = json.loads((directory / 'selection.json').read_text())
    selection['rerank'] = rerank
    write_json(directory / 'selection.json', selection)
    write_json(directory / 'rerank_curve.json', rerank)
    for name in ['selection.json', 'rerank_curve.json']:
        row['artifact_hashes'][name] = sha256(directory / name)
    seal(root, manifest)
    original = fe.evaluate
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((id(args[1]), kwargs['mode'], args[5]['rl'].get('rerank_weight')))
        return original(*args, **kwargs)
    monkeypatch.setattr(fe, 'evaluate', evaluate)
    result = fe.evaluate_followup(prepared, run, root, out, batch_size=2, max_minutes=2)
    assert len({call[0] for call in calls[:4]}) == 1
    assert calls[3][1:] == ('rerank', .1)
    seed = result['seeds']['42']
    assert seed['rerank_selection'] == rerank
    pair = seed['paired_eligible']['completion_h1']['rerank_tuned']
    assert pair['exact_one_step_expected_reward']['users'] == 2
    assert pair['sample_action_relevance']['users'] == 2
    with np.load(out / 'per_example.npz') as arrays:
        prefix = 'seed_42_rerank_tuned_'
        np.testing.assert_array_equal(arrays[prefix + 'eligible'], data['test_target'] >= 2)
        expected = (arrays[prefix + 'sample_actions'] == data['test_target'][:, None]).mean(1)
        np.testing.assert_array_equal(arrays[prefix + 'sample_action_relevance'], expected)
    populations = seed['policies']['rerank_tuned']['populations']
    assert populations['eligible']['examples'] == 5
    assert populations['all_target']['examples'] == 6
    assert 'top_reward' in populations['eligible']


def test_final_checkpoint_must_be_hash_bound_before_test(frozen_followup, monkeypatch):
    prepared, run, root, out, _, manifest = frozen_followup
    manifest['seeds']['42']['policies']['completion_h1']['final_checkpoint'] = 'missing_final.pt'
    seal(root, manifest)
    monkeypatch.setattr(fe, 'load_prepared', lambda *a: pytest.fail('test read before validation'))
    with pytest.raises(ValueError, match='checkpoint is not hash-bound'):
        fe.evaluate_followup(prepared, run, root, out)
