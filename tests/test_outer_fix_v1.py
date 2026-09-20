"""Fix-v1 checks: contract values, prepare/verify, 10x-gain equivalence (CPU)."""
import copy
from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import DDPG
from tools import outer_fix_v1 as r
from tools.outer_rl.environment import OuterEnv

INNER = Path('results/bldc/outer-inner-freeze-v1')


def _env(cls, case):
    m = r.read(INNER / 'manifest.json')
    cfg = r.read(INNER / 'inner_config.json')
    plant = copy.deepcopy(cfg['plant'])
    plant['controller']['current_reference_limit_a'] = cfg['study']['reference_limit_a']
    inner = DDPG.load(INNER / 'inner_model.zip', device='cpu')
    inner.policy.set_training_mode(False)
    env = cls(plant, cfg['study'], inner, [case])
    return env


def _case():
    return dict(name='fixture', duration_s=0.05, reference=[[0., 0.], [.01, 3.]],
                disturbance=[[0., 0.]], stress=False)


def test_gain_gives_10x_signal_at_identical_memory_cost():
    v1, v2 = _env(OuterEnv, _case()), _env(r.FixV1Env, _case())
    o1, _ = v1.reset(seed=0)
    o2, _ = v2.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-0.2, 0.2, (1,)).astype(np.float32)
        o1, _, _, _, i1 = v1.step(a)
        o2, _, _, _, i2 = v2.step(a)
        assert abs(o2[6] - 10 * o1[6]) < 1e-4 or abs(o2[6]) >= 1.0 - 1e-6
        assert i2['reward_terms']['memory'] == pytest.approx(i1['reward_terms']['memory'], abs=1e-6)
    assert abs(o2[6]) > abs(o1[6])


def test_prepare_verify_and_tamper(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER)
    p = r.verify(root)
    assert p['protocol'] == 'outer-memory-gain-v1'
    assert [s for _, s in [(x['label'], x['seed']) for x in p['records']]] == [24, 25]
    cfg = r.read(root / 'fix' / 'config.json')
    assert cfg['contract']['version'] == 'bldc-outer-speed-v2'
    assert cfg['contract']['memory_time_s'] == 0.05
    assert cfg['learner']['device'] == 'cuda' and cfg['learner']['gamma'] == 0.995
    assert cfg['reward']['failure'] == -5200.0
    (root / 'fix' / 'config.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_tiny_fix_train_select(tmp_path, monkeypatch):
    """Needs GPU; run after the ceiling queue frees it."""
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=40, seeds=[24], validation_every=20)
    real_cuda = r.cuda_learner

    def small_cuda(env, widths, seed, cfg):
        c = copy.deepcopy(cfg)
        c['learner'].update(learning_starts=2, batch_size=8, buffer_size=100, critic_widths=[8, 8])
        model = real_cuda(env, [8, 8], seed, c)
        assert model.device.type == 'cuda'
        assert model.replay_buffer.size() == 0
        return model

    monkeypatch.setattr(r, 'cuda_learner', small_cuda)
    r.original.torch.set_num_threads(1)
    child = root / 'fix'
    c = r.read(child / 'config.json')
    inner = DDPG.load(child / 'inner_model.zip', device='cpu')
    with r.environment():
        r.evaluation.baselines(c['plant'], c['inner_study'], inner, c['validation_cases'], child / 'baselines')
    r.run(root, '128x2', 24)
    m = r.read(child / '128x2-seed24/manifest.json')
    assert m['status'] == 'completed' and m['gradient_updates'] > 0 and m['physics_steps'] == 400
    r.save(root / 'run_summary.json', [dict(label='128x2', seed=24, arm='128x2-seed24', returncode=0)])
    result = r.select(root)
    assert len(result['rows']) == 1 and result['final_qualification'] is False
