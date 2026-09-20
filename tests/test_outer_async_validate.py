"""Async validation mechanics: adoption predicate, polling, promotion (CPU)."""
import json
import time
from pathlib import Path
import pytest
from tools import outer_async_validate as r


class FakeProc:
    def __init__(self, rc=None):
        self._rc = rc
        self.returncode = rc

    def poll(self):
        return self._rc


def _pend(tmp_path, heartbeat_age_s=0.0, result=None):
    pend = tmp_path / '_pending' / '100'
    pend.mkdir(parents=True)
    hb = pend / 'heartbeat.json'
    hb.write_text(json.dumps(dict(state='started')) + '\n')
    if heartbeat_age_s:
        import os
        t = time.time() - heartbeat_age_s
        os.utime(hb, (t, t))
    if result is not None:
        (pend / 'result.json').write_text(json.dumps(result))
    return pend


def test_adoptable_staleness_rules():
    assert r.adoptable(dict(step=100), 200, 0) is True
    assert r.adoptable(dict(step=100), 50, 0) is False  # from the future
    assert r.adoptable(dict(step=100), 200, 100) is False  # already adopted
    assert r.adoptable(None, 200, 0) is False


def test_poll_ready_running_dead(tmp_path):
    pend = _pend(tmp_path, result=dict(step=100, score=[0, 0, 1.0]))
    assert r.poll(pend, FakeProc())['step'] == 100
    pend2 = _pend(tmp_path / 'b')
    assert r.poll(pend2, FakeProc()) is None
    with pytest.raises(RuntimeError):
        r.poll(pend2, FakeProc(rc=1))


def test_poll_stale_heartbeat(tmp_path):
    pend = _pend(tmp_path, heartbeat_age_s=r.HEARTBEAT_TIMEOUT_S + 10)
    with pytest.raises(TimeoutError):
        r.poll(pend, FakeProc())


def test_promote_moves_validation(tmp_path):
    pend = tmp_path / '_pending' / '100'
    (pend / 'validation').mkdir(parents=True)
    (pend / 'validation' / 'metrics.json').write_text('[]')
    dest = tmp_path / 'validation' / '100'
    r.promote(pend, dest)
    assert (dest / 'metrics.json').exists() and not (pend / 'validation').exists()
    with pytest.raises(FileExistsError):
        r.promote(pend, dest)
