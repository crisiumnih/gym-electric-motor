"""IASA runner checks: prepare/verify/dispatch + tiny two-arm smoke (Warp)."""
from pathlib import Path
import pytest
from tools import outer_iasa_v1 as r

INNER = Path('results/bldc/outer-inner-freeze-v1')


def test_prepare_verify_and_dispatch(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER)
    p = r.verify(root)
    assert p['protocol'] == 'outer-iasa-v1'
    assert {(x['arm'], x['seed']) for x in p['records']} == {
        ('direct-td3', 32), ('direct-td3', 33), ('direct-td3', 34), ('direct-td3', 35),
        ('iasa-td3', 32), ('iasa-td3', 33), ('iasa-td3', 34), ('iasa-td3', 35)}
    cfgs = r.read(root / 'configs.json')
    assert cfgs['iasa-td3']['contract_version'] == 'bldc-outer-speed-v3'
    assert cfgs['direct-td3']['contract_version'] == 'bldc-outer-speed-v2'
    assert cfgs['iasa-td3']['learner']['gradient_steps'] == 8
    (root / 'configs.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_tiny_two_arm_smoke(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    monkeypatch.setattr(r.original, 'training_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=16, seeds=[32], validation_every=8,
              learner_extra=dict(learning_starts=4, batch_size=8, buffer_size=200, critic_widths=[8, 8]))
    from stable_baselines3 import DDPG
    import copy
    for arm in ['direct-td3', 'iasa-td3']:
        child_cfg = r.read(root / 'configs.json')[arm]
        inner = DDPG.load(root / 'inner_model.zip', device='cpu')
        r.evaluation.baselines(child_cfg['plant'], child_cfg['inner_study'], inner,
                               child_cfg['validation_cases'], root / child_cfg['baseline_dir'])
    for arm in ['direct-td3', 'iasa-td3']:
        r.run_arm(root, arm, 32)
    for job in ['direct-td3-seed32', 'iasa-td3-seed32']:
        m = r.read(root / job / 'manifest.json')
        assert m['status'] == 'completed' and m['gradient_updates'] > 0, job
