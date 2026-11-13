"""The terminal tool surfaces a masked pipeline failure: when the shell's
PIPESTATUS shows the first pipeline stage failed while ``$?`` (the tail's
status) is 0, the result carries a hint instead of a bare success (issue
#155298 — an agent reading only ``exit_code`` ships broken code)."""

import json

from tools.environments.local import LocalEnvironment
from tools.terminal_tool_result import finalize_foreground_result

# The issue's regression command: a stage that fails printing nothing at all.
_MASKED_CMD = 'python3 -c "import sys; sys.exit(7)" 2>&1 | tail -25'


def _finalize(command: str, result: dict, env=None) -> dict:
    return json.loads(finalize_foreground_result(
        command=command, result=result, env=env, env_type="local",
        effective_task_id="t", task_id="t", session_id="s", session_key="***",
        workdir=None, command_cwd=None, approval_note=None))


def test_finalize_flags_masked_pipeline_failure():
    flagged = _finalize(
        _MASKED_CMD, {"output": "", "returncode": 0, "pipeline_status": [7, 0]})
    # Advisory only — exit_code keeps the shell's own semantics.
    assert flagged["exit_code"] == 0
    assert "PIPESTATUS" in flagged.get("hint", "")
    assert "7" in flagged["hint"]


def test_finalize_silent_on_clean_pipeline():
    clean = _finalize(
        "echo hi | tail -1", {"output": "hi\n", "returncode": 0, "pipeline_status": [0, 0]})
    assert "hint" not in clean


def test_live_backend_end_to_end():
    """Real bash through the production wrapper: PIPESTATUS is captured on the
    backend's live path and the finalizer turns it into a model-facing hint."""
    env = LocalEnvironment(cwd=".", timeout=30)
    try:
        result = env.execute(_MASKED_CMD)
        assert result["returncode"] == 0
        assert result.get("pipeline_status") == [7, 0]
        out = _finalize(_MASKED_CMD, result, env=env)
        assert "PIPESTATUS" in out.get("hint", "")
    finally:
        env.cleanup()
