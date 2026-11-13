"""Trailing empty assistant messages on /v1/chat/completions (#154758).

OpenAI-compatible clients that pre-allocate the reply slot — Open WebUI sends
the conversation with a trailing ``assistant`` message carrying no visible
payload — used to trip the current-input selection in
``_handle_chat_completions``: it reads ``conversation_messages[-1]`` as the
turn input, saw the empty placeholder instead of the user prompt below it,
and rejected an otherwise valid request with HTTP 400
``No user message found in messages``.

The handler strips trailing assistant messages that carry no visible payload
before selecting the current input.  A trailing assistant message WITH payload
(assistant prefill) keeps today's semantics, and requests whose last real
message is still empty keep failing with the same 400.
"""

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)

_AGENT_RESULT = (
    {"final_response": "ok", "messages": [], "api_calls": 1},
    {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
)


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    return app


async def _post_chat(cli, messages):
    return await cli.post(
        "/v1/chat/completions",
        json={"model": "hermes-agent", "messages": messages, "stream": False},
    )


class TestTrailingEmptyAssistantStripped:
    @pytest.mark.asyncio
    async def test_reporter_repro_falls_through_to_user_turn(self, adapter):
        """The minimal repro from #154758 used to 400."""
        app = _create_app(adapter)
        run = AsyncMock(return_value=_AGENT_RESULT)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=run):
                resp = await _post_chat(cli, [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": ""},
                ])
                assert resp.status == 200
        kwargs = run.call_args.kwargs
        assert kwargs["user_message"] == "Hello"
        assert kwargs["conversation_history"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("trailing", [
        [""],
        ["", "   "],
        [[{"type": "text", "text": "   "}]],
    ])
    async def test_placeholder_shape_variants_all_fall_through(self, adapter, trailing):
        app = _create_app(adapter)
        run = AsyncMock(return_value=_AGENT_RESULT)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=run):
                messages = [{"role": "user", "content": "Hello"}] + [
                    {"role": "assistant", "content": content} for content in trailing]
                resp = await _post_chat(cli, messages)
                assert resp.status == 200
        assert run.call_args.kwargs["user_message"] == "Hello"
        assert run.call_args.kwargs["conversation_history"] == []

    @pytest.mark.asyncio
    async def test_prior_turns_survive_as_history(self, adapter):
        app = _create_app(adapter)
        run = AsyncMock(return_value=_AGENT_RESULT)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=run):
                resp = await _post_chat(cli, [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": ""},
                ])
                assert resp.status == 200
        kwargs = run.call_args.kwargs
        assert kwargs["user_message"] == "second question"
        assert kwargs["conversation_history"] == [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]


class TestPrefillAndRejectionSemanticsUnchanged:
    @pytest.mark.asyncio
    async def test_trailing_assistant_with_payload_still_drives_the_turn(self, adapter):
        """Assistant prefill keeps today's semantics: payload means current input."""
        app = _create_app(adapter)
        run = AsyncMock(return_value=_AGENT_RESULT)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=run):
                resp = await _post_chat(cli, [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Continue the story"},
                ])
                assert resp.status == 200
        assert run.call_args.kwargs["user_message"] == "Continue the story"
        assert run.call_args.kwargs["conversation_history"] == [
            {"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_assistant_only_request_still_rejected(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await _post_chat(cli, [{"role": "assistant", "content": ""}])
            assert resp.status == 400
            data = await resp.json()
            assert "No user message" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_trailing_user_message_still_rejected(self, adapter):
        app = _create_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await _post_chat(cli, [
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "   "},
            ])
            assert resp.status == 400
            data = await resp.json()
            assert "No user message" in data["error"]["message"]
