"""TD3 screen checks: prepare/verify/dispatch, tiny TD3 train on Warp."""
from pathlib import Path
import pytest
from tools import outer_td3_v1 as r

INNER = Path('results/bldc/outer-inner-freeze-v1')


def test_prepare_verify_and_dispatch(tmp_path):
    root = tmp_path / 'study'
    r.prepare(root, INNER)
    p = r.verify(root)
    assert p['protocol'] == 'outer-td3-v1'
    assert [s for _, s in [(x['label'], x['seed']) for x in p['records']]] == [26, 27]
    cfg = r.read(root / 'config.json')
    assert cfg['contract']['version'] == 'bldc-outer-speed-v1'
    assert cfg['contract']['memory_time_s'] == 0.5
    assert cfg['learner']['td3']['policy_delay'] == 2
    assert cfg['learner']['gamma'] == 0.995 and cfg['learner']['device'] == 'cuda'
    assert cfg['reward']['shape'] == 'l1' and cfg['reward']['failure'] == -5200.0
    assert cfg['n_envs'] == 8 and cfg['total_timesteps'] == 2000000
    (root / 'config.json').write_text('{}')
    with pytest.raises(ValueError):
        r.verify(root)


def test_tiny_td3_train_select(tmp_path, monkeypatch):
    tiny = lambda: [dict(name='fixture', duration_s=.004, reference=[[0., 0.], [.001, 1.]], disturbance=[[0., 0.]], stress=False)]
    monkeypatch.setattr(r.original, 'validation_cases', tiny)
    root = tmp_path / 'study'
    r.prepare(root, INNER, budget=16, seeds=[26], validation_every=8,
              learner_extra=dict(learning_starts=4, batch_size=8, buffer_size=200, critic_widths=[8, 8]))
    import copy
    real_td3 = r.td3_learner

    def small_td3(env, widths, seed, cfg):
        c = copy.deepcopy(cfg)
        c['learner'].update(learning_starts=2, batch_size=8, buffer_size=100, critic_widths=[8, 8])
        model = real_td3(env, [8, 8], seed, c)
        from stable_baselines3 import TD3 as _T
        assert isinstance(model, _T) and model.device.type == 'cuda'
        assert model.replay_buffer.size() == 0
        return model

    monkeypatch.setattr(r.ceiling, 'learner', small_td3)
    r.original.torch.set_num_threads(1)
    r.run(root, '128x2', 26)
    m = r.read(root / '128x2-seed26/manifest.json')
    assert m['status'] == 'completed' and m['gradient_updates'] > 0
    r.save(root / 'run_summary.json', [dict(label='128x2', seed=26, arm='128x2-seed26', returncode=0)])
    result = r.select(root)
    assert len(result['rows']) == 1 and result['final_qualification'] is False
