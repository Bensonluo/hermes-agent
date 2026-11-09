"""Crash accounting must spend the dispatcher's ``failure_limit`` (#140426).

``dispatch_once`` already handed the config-resolved ``kanban.failure_limit`` to
``release_stale_claims``, ``recompute_ready`` and both spawn-failure paths, but
``detect_crashed_workers`` dropped it: an ordinary worker crash was booked against
the hardcoded ``DEFAULT_FAILURE_LIMIT`` instead, so cards gave up after 2 crashes
no matter what the operator configured (field report: ``kanban.failure_limit: 4``
still logging ``gave_up {'failures': 3, 'effective_limit': 2}``).

The board fixture (isolated HERMES_HOME, zero crash grace, real dispatch ticks)
is shared with the managed-worker-loss suite that introduced it.
"""

from __future__ import annotations

import json

from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_managed_worker_loss import board as board  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import

CONFIGURED = kbd.DEFAULT_FAILURE_LIMIT + 2  # strictly above the hardcoded default


def _crash(board, tid: str, *, prior_failures: int = 0, tick_limit: int = CONFIGURED) -> dict:
    """One seeded managed-worker loss plus one real dispatcher tick."""
    board.running(tid, claim_expires_in=900, heartbeat_age=30,
                  worker_lost="managed_worker_failed")
    if prior_failures:
        board.conn.execute(
            "UPDATE tasks SET consecutive_failures = ? WHERE id = ?", (prior_failures, tid))
        board.conn.commit()
    board.tick(failure_limit=tick_limit)
    return board.state(tid)


def test_crash_accounting_spends_the_dispatcher_failure_limit(board):
    """DEFAULT_FAILURE_LIMIT crashes stay ready under a higher configured limit."""
    row = _crash(board, "t_cfg", prior_failures=kbd.DEFAULT_FAILURE_LIMIT - 1)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == kbd.DEFAULT_FAILURE_LIMIT
    kinds = [r[0] for r in board.conn.execute(
        "SELECT kind FROM task_events WHERE task_id = 't_cfg' ORDER BY id")]
    assert "gave_up" not in kinds


def test_crash_accounting_trips_at_the_configured_limit(board):
    """The breaker trips on the configured budget, and the gave_up event says so."""
    row = _crash(board, "t_limit", prior_failures=CONFIGURED - 1)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == CONFIGURED
    payload = json.loads(next(board.conn.execute(
        "SELECT payload FROM task_events WHERE task_id = 't_limit' AND kind = 'gave_up'"))[0])
    assert payload["effective_limit"] == CONFIGURED
    assert payload["limit_source"] == "dispatcher"


def test_systemic_crashes_still_trip_below_the_configured_limit(board):
    """>= 3 identical fingerprints in one tick stay operator-sticky at one crash."""
    for i in range(3):
        board.running(f"t_sys{i}", claim_expires_in=900, heartbeat_age=30,
                      worker_lost="managed_worker_failed")
    board.tick(failure_limit=kbd.DEFAULT_FAILURE_LIMIT + 9)
    for i in range(3):
        row = board.state(f"t_sys{i}")
        assert row["status"] == "blocked"
        assert row["consecutive_failures"] == 1
