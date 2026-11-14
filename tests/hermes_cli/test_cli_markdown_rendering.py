from io import StringIO

import pytest
from rich.console import Console
from rich.markdown import Markdown

from cli import _render_final_assistant_content


def _render_to_text(renderable) -> str:
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(renderable)
    return buf.getvalue()


def test_final_assistant_content_uses_markdown_renderable():
    renderable = _render_final_assistant_content("# Title\n\n- one\n- two")

    assert isinstance(renderable, Markdown)
    output = _render_to_text(renderable)
    assert "Title" in output
    assert "one" in output
    assert "two" in output




def test_final_assistant_content_keeps_non_path_markdown_escapes():
    renderable = _render_final_assistant_content(r"1\. Not an ordered list")

    output = _render_to_text(renderable)
    assert "1. Not an ordered list" in output
    assert r"1\." not in output






def test_strip_mode_preserves_lists():
    renderable = _render_final_assistant_content(
        "**Formatting**\n- Ran prettier\n- Files changed\n- Verified clean",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "- Ran prettier" in output
    assert "- Files changed" in output
    assert "- Verified clean" in output
    assert "**" not in output




def test_strip_mode_preserves_blockquotes():
    renderable = _render_final_assistant_content(
        "> This is quoted text\n> Another quoted line",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "> This is quoted" in output
    assert "> Another quoted" in output






def test_strip_mode_preserves_cron_asterisks_in_plain_text():
    renderable = _render_final_assistant_content("* * * * *", mode="strip")

    output = _render_to_text(renderable)
    assert "* * * * *" in output

    # Still treat the canonical 3-asterisk Markdown horizontal rule as decoration.
    renderable = _render_final_assistant_content("* * *", mode="strip")
    output = _render_to_text(renderable)
    assert "* * *" not in output




def test_strip_mode_preserves_intraword_underscores_in_snake_case_identifiers():
    renderable = _render_final_assistant_content(
        "Let me look at test_case_with_underscores and SOME_CONST "
        "then /tmp/snake_case_dir/file_with_name.py",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "test_case_with_underscores" in output
    assert "SOME_CONST" in output
    assert "snake_case_dir" in output
    assert "file_with_name" in output


def test_strip_mode_still_strips_boundary_underscore_emphasis():
    renderable = _render_final_assistant_content(
        "say _hi_ and __bold__ now",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "say hi and bold now" in output


def test_strip_mode_keeps_code_fence_contents_verbatim():
    renderable = _render_final_assistant_content(
        "Here is the snippet:\n\n"
        "```python\n"
        "# leading comment\n"
        "class Thing:\n"
        "    def __init__(self, **kwargs):\n"
        "        pass\n"
        "```\n",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "# leading comment" in output
    assert "def __init__(self, **kwargs):" in output
    assert "```" not in output
    # The fence delimiter line (with its info string) is a marker, not content.
    assert "\npython\n" not in output


def test_strip_mode_still_strips_markers_outside_code_fences():
    renderable = _render_final_assistant_content(
        "# Heading\n\n**bold** and _em_ text\n\n"
        "```text\n# not a heading\n**bold** _em_\n```",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "Heading" in output
    assert "# Heading" not in output
    assert "bold and em text" in output
    assert "# not a heading" in output
    assert "**bold** _em_" in output


def test_strip_mode_handles_tilde_fences_and_unclosed_fences():
    renderable = _render_final_assistant_content(
        "~~~python\nx = 1\n~~~\n\nafter the block",
        mode="strip",
    )

    output = _render_to_text(renderable)
    assert "x = 1" in output
    assert "~~~" not in output
    assert "after the block" in output

    # A fence left open (truncated stream) still protects what follows.
    renderable = _render_final_assistant_content(
        "```python\n# comment inside an unclosed fence",
        mode="strip",
    )
    output = _render_to_text(renderable)
    assert "# comment inside an unclosed fence" in output


def test_strip_mode_tilde_fence_keeps_inner_backtick_runs_verbatim():
    renderable = _render_final_assistant_content(
        "~~~\n```\ninner\n```\n~~~",
        mode="strip",
    )

    output = _render_to_text(renderable)
    # A backtick run inside a tilde fence is content, not a closing delimiter.
    assert "```" in output
    assert "inner" in output
    assert "~~~" not in output


@pytest.mark.parametrize(
    "text, expect_in, expect_not_in",
    [
        # A backtick fence whose info string contains a backtick does not open:
        # the line is prose, so its "#" line is stripped as a heading.
        ("``` a`b\n# heading text\n", ["heading text"], ["# heading text"]),
        # A closing fence must be at least as long as the opening run.
        ("````\ncode\n```\n````", ["```", "code"], []),
        # Trailing non-whitespace disqualifies a closing fence.
        ("```\ncode\n``` tail\n", ["``` tail"], []),
        # Three leading spaces still open a fence; four make it prose.
        ("   ```\n# kept\n   ```", ["# kept"], ["```"]),
        ("    ```\n# heading text\n", ["heading text"], ["# heading text"]),
        # A longer closing run is a valid close: following prose is stripped again.
        ("```\ncode\n````\n**bold** after", ["bold after"], ["````"]),
    ],
)
def test_strip_mode_fence_delimiter_rules(text, expect_in, expect_not_in):
    output = _render_to_text(_render_final_assistant_content(text, mode="strip"))
    for fragment in expect_in:
        assert fragment in output
    for fragment in expect_not_in:
        assert fragment not in output
