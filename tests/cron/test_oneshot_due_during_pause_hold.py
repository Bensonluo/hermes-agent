"""A one-shot that comes due while ``hermes pause`` (ESTOP) holds the cron tick fires once after resume.

``hermes pause`` skips dispatch from inside the tick (the ESTOP early return in
``cron/scheduler_tick.py``) rather than through the ticker loop's drain gate, so the #147894
dispatch-hold stamp was never written for it: the first due scan after a pause longer than the
120s one-shot grace retired a one-shot that came due during the pause — or one run by hand just
before it — as ``missed`` without ever firing it (#153683). A paused tick now stamps the same
hold, so the grace counts from the end of the pause; a drain hold's MAX_EXTENSION_SECONDS
staleness cap applies unchanged, and the tick still skips the due scan entirely while paused.
"""

import time
from datetime import datetime, timedelta, UTC

import pytest

from agent import estop
from cron import dispatch_hold, scheduler
from cron.jobs import get_due_jobs, load_jobs, save_jobs


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    (tmp_path / "cron").mkdir()
    estop._logged_components.clear()
    estop.engage(reason="test pause")
    return tmp_path


def _oneshot(jid, run_at):
    return {
        "id": jid, "name": jid, "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at.isoformat()},
        "next_run_at": run_at.isoformat(), "last_run_at": None,
        "enabled": True, "state": "scheduled", "repeat": {"times": 1, "completed": 0},
        "deliver": "local", "run_claim": None, "fire_claim": None,
    }


def _scan_at(monkeypatch, when):
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: when)
    return sorted(d["id"] for d in get_due_jobs())


def test_paused_tick_stamps_the_dispatch_hold(cron_store):
    assert scheduler.tick(verbose=False) == 0
    held = dispatch_hold.read(cron_store / "cron")
    assert held is not None, "a paused tick must stamp the dispatch hold like a drain-held one"
    assert held["last"] <= time.time()


def test_oneshots_held_by_a_pause_fire_once_after_resume(cron_store, monkeypatch):
    t0 = time.time()
    assert scheduler.tick(verbose=False) == 0  # the ticker keeps ticking (and holding) while paused
    start = datetime.fromtimestamp(t0, UTC)
    save_jobs([
        # `hermes cron run` in the minute before the pause: an explicit request to run.
        _oneshot("run-by-hand", start - timedelta(seconds=90)),
        # comes due while the pause is engaged.
        _oneshot("due-mid-pause", start + timedelta(seconds=30)),
    ])

    # First due scan after `hermes resume`: both are past the 120s grace counted from their own
    # due time, but within it counted from the end of the hold.
    assert _scan_at(monkeypatch, start + timedelta(seconds=170)) == [
        "due-mid-pause", "run-by-hand"]
    assert sorted(j["id"] for j in load_jobs()) == ["due-mid-pause", "run-by-hand"]
