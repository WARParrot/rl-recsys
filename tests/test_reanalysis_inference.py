"""Tiny actual-model CPU inference exercises the Windows reanalysis orchestration."""
import copy
import json
import numpy as np
import torch

from rl_recsys import reanalysis as ra
from rl_recsys.models import Policy, RewardModel


def test_real_tiny_inference_controls_and_pickle_free_export(tmp_path, monkeypatch):
    cfg = json.loads(open('configs/pilot.json').read())
    cfg['model'] = {'hidden': 8, 'layers': 1, 'heads': 2, 'dropout': 0.}
    cfg['data']['context_length'] = 3
    cfg['train']['allow_cpu'] = True
    cfg['train']['threads'] = 1
    cfg['evaluation'].update(k=2, bootstrap=10, action_samples=2)
    data = {'vocab': np.array([10, 20, 30]), 'popularity': np.array([5., 3., 1.]),
            'durations': np.array([100., 200., 300.])}
    for role in ['fit', 'selection', 'test']:
        data.update({f'{role}_context': np.array([[0, 2, 3]] * 10),
                     f'{role}_target': np.array([2, 3, 4, 2, 3] * 2),
                     f'{role}_uid': np.repeat([1, 2], 5),
                     f'{role}_timestamp': np.tile(np.arange(1, 6), 2),
                     f'{role}_reward': np.linspace(.1, .9, 10).astype('float32'),
                     f'{role}_seconds': np.linspace(.1, .5, 10).astype('float32'),
                     f'{role}_organic': np.zeros(10, dtype=int)})
    run, prepared = tmp_path / 'run', tmp_path / 'data'
    run.mkdir(); prepared.mkdir()
    np.savez(prepared / 'prepared.npz', **data)
    (prepared / 'audit.json').write_text('{}')
    (run / 'provenance.json').write_text(json.dumps({'config': cfg}))
    (run / 'results.json').write_text(json.dumps({'selected': 'rl_completion_h1_b0.01'}))
    (run / 'training.json').write_text(json.dumps({'reward': [{'step': 1, 'calibration_loss': .2}]}))
    args = (5, 3, cfg['model'])
    models = {'sft.pt': Policy(*args), 'reward.pt': RewardModel(*args),
              'auditor.pt': RewardModel(*args, mean_history=True),
              'rl_completion_h1_b0.01.pt': Policy(*args)}
    for name, model in models.items():
        torch.save(model.state_dict(), run / name)
    monkeypatch.setattr(ra, 'verify_inputs', lambda *args: {'checkpoint_hashes': dict.fromkeys(models, 'fixture')})
    result = ra.run_reanalysis(prepared, run, tmp_path / 'analysis', batch_size=3,
                               max_minutes=2, rerank_weights=[0., 1.])
    assert result['evaluation_status'] == 'exploratory_reuse'
    assert 'history_sensitivity' in result
    assert 'matched_rollouts' in result['policies']['sft']
    out = tmp_path / 'analysis'
    assert (out / 'COMPLETE.json').exists()
    with np.load(out / 'per_example.npz', allow_pickle=False) as arrays:
        for key in arrays.files:
            assert arrays[key].dtype.kind != 'O'
        assert 'sft_control_modes_observed_context_replay_observed_target' in arrays
    with np.load(out / 'evaluation-inputs.npz', allow_pickle=False) as arrays:
        assert arrays['test_target'].shape == (10,)
