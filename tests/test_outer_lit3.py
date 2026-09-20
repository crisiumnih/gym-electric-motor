"""Lit3 checks: per-screen prepare/verify/dispatch + tiny TD3 warp-train."""
from pathlib import Path
import pytest
from tools import outer_lit3 as r

INNER = Path('results/bldc/outer-inner-freeze-v1')


def test_all_screens_prepare_verify_and_dispatch(tmp_path):
    for screen, sc in r.SCREENS.items():
        root = tmp_path / screen
        r.prepare(root, screen, INNER)
        p = r.verify(root, screen)
        assert p['protocol'] == sc['protocol']
        cfg = r.read(root / 'config.json')
        assert cfg['contract']['version'] == 'bldc-outer-speed-v2'
        assert cfg['learner']['algorithm'] == 'td3'
        assert cfg['reward']['memory_divisor'] == 0.05
        assert cfg['n_envs'] == 8 and cfg['total_timesteps'] == 2000000
        (root / 'config.json').write_text('{}')
        with pytest.raises(ValueError):
            r.verify(root, screen)


def test_tiny_td3_warp_train_select(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'td3'
    r.prepare(root, 'td3', INNER, budget=16, seeds=[26], validation_every=8,
              learner_extra=dict(learning_starts=4, batch_size=8, buffer_size=200, critic_widths=[8, 8]))
    r.run(root, 'td3', '128x2', 26)
    m = r.read(root / '128x2-seed26/manifest.json')
    assert m['status'] == 'completed' and m['gradient_updates'] > 0
    r.save(root / 'run_summary.json', [dict(label='128x2', seed=26, arm='128x2-seed26', returncode=0)])
    result = r.select(root, 'td3')
    assert len(result['rows']) == 1 and result['final_qualification'] is False
