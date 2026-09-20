"""v2 CUDA adapter checks: device dispatch, v1 value-match, tiny real pipeline."""
import copy
from pathlib import Path
import pytest
from stable_baselines3 import DDPG
from tools import outer_horizon_v2 as r

INNER = Path('results/bldc/outer-inner-freeze-v1')
V1REF = Path('results/bldc/outer-horizon-v1')


def test_v2_config_device_and_gamma_dispatch(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER, V1REF)
    p = r.verify(root)
    assert p['protocol'] == 'outer-discount-horizon-v2'
    assert len(p['records']) == 4
    assert {s for _, s in [(x['group'], x['seed']) for x in p['records']]} == {18, 19}
    configs = [r.read(root / s / 'config.json') for s in ['g0995', 'g0999']]
    assert configs[0]['learner'].pop('gamma') == .995
    assert configs[1]['learner'].pop('gamma') == .999
    assert configs[0]['learner']['device'] == 'cuda' == configs[1]['learner']['device']
    assert configs[0]['reward']['failure'] == -5200
    assert configs[0] == configs[1]
    (root / 'g0999/config.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_cuda_learner_places_outer_on_cuda():
    import gymnasium as gym
    env = gym.make('Pendulum-v1')
    cfg = dict(learner=dict(device='cuda', learning_rate=1e-4, gamma=.995, tau=.005,
        buffer_size=100, learning_starts=10, batch_size=8, train_freq=1, gradient_steps=1,
        noise_sigma=.05, critic_widths=[8, 8]))
    model = r.cuda_learner(env, [8, 8], 0, cfg)
    assert model.device.type == 'cuda'
    model.learn(20)
    assert model.num_timesteps == 20


def test_real_tiny_cuda_selection(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'training_cases', tiny)
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, V1REF, budget=40, seeds=[18], validation_every=20, strict=False)
    real_cuda_learner = r.cuda_learner
    def small_cuda(env, widths, seed, cfg):
        c = copy.deepcopy(cfg)
        c['learner'].update(learning_starts=2, batch_size=8, buffer_size=100, critic_widths=[8, 8])
        model = real_cuda_learner(env, [8, 8], seed, c)
        assert model.device.type == 'cuda'
        assert model.replay_buffer.size() == 0
        return model
    monkeypatch.setattr(r, 'cuda_learner', small_cuda)
    r.original.torch.set_num_threads(1)
    queue = []
    for group in ['g0995', 'g0999']:
        child = root / group
        c = r.read(child / 'config.json')
        inner = DDPG.load(child / 'inner_model.zip', device='cpu')
        with r.parent.environment(group):
            r.evaluation.baselines(c['plant'], c['inner_study'], inner, c['validation_cases'], child / 'baselines')
        r.run(root, group, 18)
        m = r.read(child / '128x2-seed18/manifest.json')
        assert m['status'] == 'completed' and m['gradient_updates'] > 0 and m['physics_steps'] == 400
        queue.append(dict(group=group, seed=18, arm='128x2-seed18', returncode=0))
    r.save(root / 'run_summary.json', queue)
    result = r.select(root)
    assert len(result['rows']) == 2 and result['final_qualification'] is False
