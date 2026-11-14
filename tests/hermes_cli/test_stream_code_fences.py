"""Streamed code fences must stay unmodified in ``strip`` display mode.

``display.final_response_markdown: strip`` removes markdown markers as lines
stream in. Without fence tracking, a streamed ``#`` comment lost its hash and
``__init__`` became ``init`` inside code blocks (#155746). Fence delimiter
lines toggle ``_stream_fence_*`` state and are dropped; lines inside an open
fence are emitted unmodified (no marker stripping, no table buffering), and
the fence state resets at turn boundaries.
"""
import re

import pytest


def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


@pytest.fixture
def cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as climod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "strip"
    cli.show_timestamps = False
    cli._reset_stream_state()

    emitted = []
    monkeypatch.setattr(climod, "_cprint", lambda s: emitted.append(s))
    monkeypatch.setattr(climod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def test_streamed_code_fence_contents_are_verbatim(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta(
        "```python\n"
        "# leading comment\n"
        "class Thing:\n"
        "    def __init__(self, **kwargs):\n"
        "```\n"
    )
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "# leading comment" in lines
    assert "    def __init__(self, **kwargs):" in lines
    assert not any("```" in ln for ln in lines)
    # The delimiter line (with its info string) is a marker, not content.
    assert "python" not in lines


def test_streamed_prose_markers_still_stripped_around_fences(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta(
        "## Heading\n\n```text\n**not emphasis**\n```\n\n**real emphasis**\n"
    )
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "Heading" in lines
    assert "**not emphasis**" in lines
    assert "real emphasis" in lines


def test_streamed_fence_state_resets_between_turns(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("```python\nx = 1\n```\n")
    cli._stream_delta(None)  # turn boundary: flush + reset streaming state
    cli._stream_delta("# Prose heading after the boundary\n")
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "Prose heading after the boundary" in lines
    assert "# Prose heading after the boundary" not in lines


def test_streamed_unclosed_fence_flushes_trailing_partial_verbatim(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("```python\ndef __setup__(self):")
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "def __setup__(self):" in lines


def test_streamed_table_rows_inside_fence_are_not_buffered(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta(
        "```\n| alpha | beta |\n| ------ | ---- |\n| 1 | 2 |\n```\n"
    )
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "| alpha | beta |" in lines
    assert "| ------ | ---- |" in lines
    assert "| 1 | 2 |" in lines


def test_stream_truncated_on_closing_fence_swallows_the_delimiter(cli_stub):
    cli, emitted = cli_stub
    # No trailing newline: the closing delimiter is still in the partial buffer
    # at end-of-stream and must be swallowed, not printed.
    cli._stream_delta("```python\ncode = 1\n```")
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "code = 1" in lines
    assert not any("```" in ln for ln in lines)


def test_streamed_spinner_preview_skips_fenced_code(cli_stub):
    cli, _ = cli_stub
    cli._invalidate = lambda **kwargs: None
    cli._stream_delta("```python\n")
    cli._stream_delta("x" * 90)  # long partial code line, no newline
    # The preview tail is 60 chars; with the fence open none of it may leak
    # into the spinner text.
    assert getattr(cli, "_spinner_text", "") == ""


def test_streamed_tilde_fences_are_tracked_too(cli_stub):
    cli, emitted = cli_stub
    cli._stream_delta("~~~python\n# kept comment\n~~~\n")
    cli._flush_stream()

    lines = [_strip_ansi(e) for e in emitted]
    assert "# kept comment" in lines
    assert not any("~~~" in ln for ln in lines)
