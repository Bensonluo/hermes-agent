"""``hermes -z`` exit-code contract: the outcome is judged from the turn result, not from
whether any text was printed (#111770 — an incomplete or failed run that left an explanation on
stdout used to exit 0, so scripts treated a half-done job as success)."""

from hermes_cli import oneshot


def _run(monkeypatch, tmp_path, response, result):
    usage = tmp_path / "usage.json"
    oneshot._write_usage_file(str(usage), dict(result))
    return oneshot._oneshot_exit_code(response, result), usage


def test_incomplete_run_with_text_exits_nonzero_and_reports_why(monkeypatch, tmp_path):
    import json

    partial = {"final_response": "Got as far as step 2.", "completed": False, "partial": True,
               "turn_exit_reason": "iteration_limit"}
    code, usage = _run(monkeypatch, tmp_path, "Got as far as step 2.", partial)
    assert code == 2
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["completed"] is False and report["partial"] is True
    assert report["turn_exit_reason"] == "iteration_limit"

    interrupted = {"final_response": "Stopping.", "completed": False, "interrupted": True}
    code, usage = _run(monkeypatch, tmp_path, "Stopping.", interrupted)
    assert code == 130
    assert json.loads(usage.read_text(encoding="utf-8"))["interrupted"] is True

    failed = {"final_response": "Provider returned 401.", "completed": False, "failed": True}
    code, _ = _run(monkeypatch, tmp_path, "Provider returned 401.", failed)
    assert code == 2


def test_completed_run_still_exits_zero(monkeypatch, tmp_path):
    code, _ = _run(monkeypatch, tmp_path, "Paris.", {"final_response": "Paris.", "completed": True, "failed": False})
    assert code == 0


def test_pre_llm_call_veto_exits_nonzero_for_unattended_callers(monkeypatch, tmp_path):
    """A ``pre_llm_call`` veto settles a completed-looking turn (the block message is the
    reply), but the model never ran: an unattended caller must not see exit 0 (#154234)."""
    import json

    from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON

    vetoed = {"final_response": "BLOCKED: policy", "completed": True, "failed": False,
              "interrupted": False, "turn_exit_reason": PRE_LLM_BLOCK_EXIT_REASON}
    code, usage = _run(monkeypatch, tmp_path, "BLOCKED: policy", vetoed)
    assert code == 2
    report = json.loads(usage.read_text(encoding="utf-8"))
    assert report["turn_exit_reason"] == PRE_LLM_BLOCK_EXIT_REASON

    # An empty block message is still "the task was not done", not the exit-1 no-text branch.
    code, _ = _run(monkeypatch, tmp_path, "", dict(vetoed, final_response=""))
    assert code == 2
