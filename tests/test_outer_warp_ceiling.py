"""Ceiling adapter checks: prepare/verify, tamper rejection, tiny Warp-train run."""
import copy
from pathlib import Path
import pytest
from tools import outer_warp_ceiling as r

INNER = Path('results/bldc/outer-inner-freeze-v1')
TINY_LEARNER = dict(learning_starts=8, batch_size=8, buffer_size=200, critic_widths=[8, 8])


def test_prepare_verify_and_tamper(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=100, seeds=[20], widths=['128x2'], validation_every=50,
              n_envs=2, learner_extra=TINY_LEARNER)
    p = r.verify(root)
    assert p['protocol'] == 'outer-warp-ceiling-v1'
    assert p['n_envs'] == 2
    cfg = r.read(root / 'config.json')
    assert cfg['total_timesteps'] == 200 and cfg['validation_every'] == 100
    assert cfg['learner']['learning_starts'] == 8
    assert len(cfg['training_cases']) == 40
    assert len(p['records']) == 1
    (root / 'config.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_tiny_warp_train_select(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=100, seeds=[20], widths=['128x2'], validation_every=50,
              n_envs=2, learner_extra=TINY_LEARNER)
    r.run_arm(root, '128x2', 20)
    m = r.read(root / '128x2-seed20/manifest.json')
    assert m['status'] == 'completed' and m['gradient_updates'] > 0
    assert m['outer_actions'] == 200
    r.save(root / 'run_summary.json', [dict(label='128x2', seed=20, arm='128x2-seed20', returncode=0)])
    result = r.select(root)
    assert len(result['rows']) == 1 and result['final_qualification'] is False
    with pytest.raises(FileExistsError):
        r.select(root)
