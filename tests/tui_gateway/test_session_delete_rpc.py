"""``session.delete``: argument checks, live-session eviction (#155831), the profile's db and sessions dir,
and the onboarding run a deleted chat ends first (moved out of test_tui_gateway_server.py)."""

from pathlib import Path

from tui_gateway import server


class _NoOnboardingRun:
    """A fake state DB whose sessions carry no onboarding run (session.delete reads it first)."""

    def get_session_model_config_value(self, *_args):
        return None

    def resolve_resume_session_id(self, *_args):
        return None


def test_session_delete_requires_session_id(monkeypatch):
    """Empty / missing session_id is a 4006 client error (no DB call)."""
    called: list[tuple] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, *a, **kw):
            called.append((a, kw))
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request({"id": "1", "method": "session.delete", "params": {}})
    assert "error" in resp
    assert resp["error"]["code"] == 4006
    assert called == []


def test_session_delete_returns_db_unavailable_when_no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_db_error", "locked")

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "abc"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert resp["error"]["data"]["code"] == "storage_locked"


def test_session_delete_evicts_live_session_then_deletes(monkeypatch):
    """A live session bound to the target key is interrupted and torn down, then the row is deleted (#155831)."""
    events: list[str] = []
    torn: list[dict] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            events.append("delete")
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_interrupt_session_turn",
                        lambda sid, session, **_kw: events.append("interrupt"))
    monkeypatch.setattr(server, "_teardown_popped_session",
                        lambda session, *, end_reason="tui_close": (torn.append(session) or events.append(f"teardown:{end_reason}") or True))
    monkeypatch.setitem(server._sessions, "live", {"session_key": "key-live", "running": True})
    try:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.delete",
                "params": {"session_id": "key-live"},
            }
        )
    finally:
        server._sessions.pop("live", None)

    assert resp.get("result") == {"deleted": "key-live"}, resp
    assert events == ["interrupt", "teardown:tui_delete", "delete"], events
    assert torn and torn[0].get("_sid") == "live"
    assert "live" not in server._sessions


def test_session_delete_evicts_idle_live_session_without_interrupt(monkeypatch):
    """An idle live record (no running turn) is still evicted, but never interrupted."""
    events: list[str] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            events.append("delete")
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_interrupt_session_turn",
                        lambda sid, session, **_kw: events.append("interrupt"))
    monkeypatch.setattr(server, "_teardown_popped_session",
                        lambda session, *, end_reason="tui_close": (events.append(f"teardown:{end_reason}") or True))
    monkeypatch.setitem(server._sessions, "idle", {"session_key": "key-idle"})
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.delete", "params": {"session_id": "key-idle"}}
        )
    finally:
        server._sessions.pop("idle", None)

    assert resp.get("result") == {"deleted": "key-idle"}, resp
    assert events == ["teardown:tui_delete", "delete"], events


def test_session_delete_evicts_every_live_record_for_the_key(monkeypatch):
    """Duplicate running live records for one stored key (post-resume races) are all claimed, each one
    interrupted, before the single row delete."""
    events: list[str] = []
    torn: list[str] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            events.append("delete")
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_interrupt_session_turn",
                        lambda sid, session, **_kw: events.append(f"interrupt:{sid}"))
    monkeypatch.setattr(server, "_teardown_popped_session",
                        lambda session, *, end_reason="tui_close": (torn.append(session.get("_sid")) or True))
    monkeypatch.setitem(server._sessions, "a", {"session_key": "dup", "running": True})
    monkeypatch.setitem(server._sessions, "b", {"session_key": "dup", "running": True})
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.delete", "params": {"session_id": "dup"}}
        )
    finally:
        server._sessions.pop("a", None)
        server._sessions.pop("b", None)

    assert resp.get("result") == {"deleted": "dup"}, resp
    assert sorted(torn) == ["a", "b"]
    assert sorted(events) == ["delete", "interrupt:a", "interrupt:b"]


def test_session_delete_skips_a_foreign_profile_live_record(monkeypatch):
    """A coincidental key match in another profile's store is left alone: its row is not the one deleted."""
    events: list[str] = []

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            events.append("delete")
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_interrupt_session_turn", lambda *a, **_kw: events.append("interrupt"))
    monkeypatch.setattr(server, "_teardown_popped_session",
                        lambda session, *, end_reason="tui_close": events.append("teardown") or True)
    monkeypatch.setitem(server._sessions, "foreign", {"session_key": "x-key", "profile_home": "/tmp/other-home"})
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.delete", "params": {"session_id": "x-key"}}
        )
        assert resp.get("result") == {"deleted": "x-key"}, resp
        assert events == ["delete"], events
        assert "foreign" in server._sessions
    finally:
        server._sessions.pop("foreign", None)


def test_session_delete_refuses_when_eviction_teardown_fails(monkeypatch):
    """Teardown failure keeps the 4023 refusal and the row — and re-queues the record, so a retry re-enters
    eviction instead of deleting the row bare under a runtime that could not be stopped."""
    calls: list[str] = []
    outcomes: list[object] = ["raise", True]  # first teardown fails, the retry succeeds

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            calls.append("delete")
            return True

    def _teardown(session, *, end_reason="tui_close"):
        if outcomes and outcomes[0] == "raise":
            outcomes.pop(0)
            raise RuntimeError("finalize blew up")
        calls.append("teardown")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_interrupt_session_turn", lambda *a, **_kw: None)
    monkeypatch.setattr(server, "_teardown_popped_session", _teardown)
    monkeypatch.setitem(server._sessions, "live", {"session_key": "key-live"})
    try:
        first = server.handle_request(
            {"id": "1", "method": "session.delete", "params": {"session_id": "key-live"}}
        )
        assert first["error"]["code"] == 4023, first
        assert "finalize blew up" in first["error"]["message"], first
        assert calls == [], "delete_session must not run while the runtime could not be stopped"
        assert "live" in server._sessions, "the failed eviction must re-queue the record"

        second = server.handle_request(
            {"id": "2", "method": "session.delete", "params": {"session_id": "key-live"}}
        )
        assert second.get("result") == {"deleted": "key-live"}, second
        assert calls == ["teardown", "delete"], calls
    finally:
        server._sessions.pop("live", None)


def test_session_delete_fails_closed_when_active_snapshot_raises(monkeypatch):
    """Concurrent ``_sessions`` mutation from another RPC thread can raise
    ``RuntimeError: dictionary changed size during iteration``.  When the
    handler can't enumerate active sessions safely it must refuse the
    delete (fail closed) rather than fall through and allow it."""

    class _DB(_NoOnboardingRun):
        def delete_session(self, *a, **kw):
            raise AssertionError("delete must not run when active snapshot fails")

    class _ExplodingDict:
        def values(self):
            raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_sessions", _ExplodingDict())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036


def test_session_delete_returns_4007_when_missing(monkeypatch):
    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            return False

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "ghost"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 4007


def test_session_delete_propagates_db_exception(monkeypatch):
    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            raise RuntimeError("disk full")

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "x"}}
    )

    assert "error" in resp
    assert resp["error"]["code"] == 5036
    assert "disk full" in resp["error"]["message"]


def test_session_delete_success_returns_deleted_id(monkeypatch):
    """Happy path — DB delete succeeds, response carries the deleted id
    and the on-disk sessions dir is forwarded so transcript files get
    cleaned up alongside the row."""
    captured: dict = {}

    class _DB(_NoOnboardingRun):
        def delete_session(self, sid, sessions_dir=None, **_kw):
            captured["sid"] = sid
            captured["sessions_dir"] = sessions_dir
            return True

    monkeypatch.setattr(server, "_get_db", lambda: _DB())

    resp = server.handle_request(
        {"id": "1", "method": "session.delete", "params": {"session_id": "old-1"}}
    )

    assert "result" in resp, resp
    assert resp["result"] == {"deleted": "old-1"}
    assert captured["sid"] == "old-1"
    # sessions_dir must be forwarded so transcript files get cleaned up
    # too — not just the SQLite row.  The autouse _isolate_hermes_home
    # fixture pins HERMES_HOME to a temp dir; the handler should append
    # /sessions to it.
    assert captured["sessions_dir"] is not None
    assert str(captured["sessions_dir"]).endswith("sessions")




# --------------------------------------------------------------------------
# session.* profile scoping (app-global remote mode) — #62503
# --------------------------------------------------------------------------


def test_session_delete_honors_params_profile_sessions_dir(monkeypatch, tmp_path):
    """Issue #62503: delete must target the profile state.db + sessions dir."""
    profile_home = tmp_path / "profiles" / "mlperf"
    (profile_home / "sessions").mkdir(parents=True)
    captured: dict = {}

    class ProfileDB(_NoOnboardingRun):
        def __init__(self, db_path=None):
            captured["db_path"] = db_path

        def delete_session(self, sid, sessions_dir=None, **_kw):
            captured["sid"] = sid
            captured["sessions_dir"] = sessions_dir
            return True

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(server, "_profile_home", lambda p: profile_home if p == "mlperf" else None)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr("hermes_state_registry.acquire", ProfileDB)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "session.delete",
            "params": {"session_id": "old-ml", "profile": "mlperf"},
        }
    )
    assert "result" in resp, resp
    assert resp["result"] == {"deleted": "old-ml"}
    assert str(captured["db_path"]).endswith("state.db")
    assert Path(captured["sessions_dir"]) == profile_home / "sessions"
    assert captured.get("closed") is True
