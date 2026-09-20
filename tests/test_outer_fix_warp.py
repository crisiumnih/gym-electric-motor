"""Fix-warp checks: prepare/verify, contract dispatch, tiny Warp-train run."""
from pathlib import Path
import pytest
from tools import outer_fix_warp as r

INNER = Path('results/bldc/outer-inner-freeze-v1')


def test_prepare_verify_and_dispatch(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER)
    p = r.verify(root)
    assert p['protocol'] == 'outer-fix-warp-v1'
    assert [s for _, s in [(x['label'], x['seed']) for x in p['records']]] == [24, 25]
    cfg = r.read(root / 'config.json')
    assert cfg['contract']['version'] == 'bldc-outer-speed-v2'
    assert cfg['contract']['memory_time_s'] == 0.05
    assert cfg['reward']['memory_divisor'] == 0.05 and cfg['reward']['memory_cost'] == 0.005
    assert cfg['n_envs'] == 8 and cfg['total_timesteps'] == 2000000
    assert len(cfg['training_cases']) == 40
    (root / 'config.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_tiny_warp_train_select(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=16, seeds=[24], validation_every=8,
              learner_extra=dict(learning_starts=4, batch_size=8, buffer_size=200, critic_widths=[8, 8]))
    r.run(root, '128x2', 24)
    m = r.read(root / '128x2-seed24/manifest.json')
    assert m['status'] == 'completed' and m['gradient_updates'] > 0
    r.save(root / 'run_summary.json', [dict(label='128x2', seed=24, arm='128x2-seed24', returncode=0)])
    result = r.select(root)
    assert len(result['rows']) == 1 and result['final_qualification'] is False
