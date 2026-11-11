"""The classic client renders a slash.exec refusal that names its check (#153927)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_cli.gateway_client import GatewayRPCError

_DETAIL = ("this session's source is telegram; slash commands run on sessions the gateway owns "
           "locally — run the command in the chat that owns the session")


@pytest.mark.asyncio
async def test_permission_denied_with_detail_names_the_check(capsys):
    from hermes_cli.gateway_chat_commands import run_command
    rpc = AsyncMock(side_effect=GatewayRPCError("permission_denied",
                                                data={"reason": "permission_denied", "detail": _DETAIL}))
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id='s', submit=AsyncMock())
    await run_command(view, '/review', 'the PR as a whole')
    out = capsys.readouterr().out
    assert '/review refused:' in out and 'telegram' in out and 'owns the session' in out


@pytest.mark.asyncio
async def test_bare_permission_denied_still_raises():
    """A refusal without a named check (e.g. a capability refusal) surfaces unchanged."""
    from hermes_cli.gateway_chat_commands import run_command
    rpc = AsyncMock(side_effect=GatewayRPCError("permission_denied"))
    view = SimpleNamespace(client=SimpleNamespace(rpc=rpc), session_id='s', submit=AsyncMock())
    with pytest.raises(GatewayRPCError):
        await run_command(view, '/review', 'the PR as a whole')
