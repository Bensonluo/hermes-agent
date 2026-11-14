"""Configured server headers on MCP OAuth requests (#155708).

An identity-aware proxy in front of an MCP server (Cloudflare Access service
tokens, API-gateway keys) authenticates by request header. Those headers ride
every MCP request because the transports hand them to the SDK client, but the
SDK builds OAuth discovery, dynamic client registration, token exchange and
refresh as bare ``httpx.Request`` objects sent with ``client.send()``, which
never merges the client's default headers — so ``hermes mcp login`` behind such
a proxy is 302'd before the browser even opens.

The provider now stamps the configured headers onto OAuth requests that stay on
the server's own origin (scheme/host/port). A third-party authorization server
never sees them — the same boundary ``strict_redirect_headers`` enforces on the
transport path — and a header the SDK already set is never overwritten.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK required for OAuth support",
)

from tools.mcp_oauth_provider import (
    origin_scoped_headers,
    stamp_configured_origin_headers,
)

SERVER_URL = "https://mcp.example.com/mcp"
CF_HEADERS = {"CF-Access-Client-Id": "repro-id.access", "CF-Access-Client-Secret": "repro-secret"}


# ---------------------------------------------------------------------------
# Origin scoping of the configured headers
# ---------------------------------------------------------------------------


def test_same_origin_target_gets_the_configured_headers():
    scoped = origin_scoped_headers("https://mcp.example.com/.well-known/oauth-authorization-server",
                                   SERVER_URL, CF_HEADERS)
    assert scoped == CF_HEADERS


def test_explicit_default_port_matches_the_origin():
    assert origin_scoped_headers("https://mcp.example.com:443/register", SERVER_URL, CF_HEADERS) == CF_HEADERS
    # http vs https is a different origin even on the same host.
    assert origin_scoped_headers("http://mcp.example.com/.well-known/x", SERVER_URL, CF_HEADERS) == {}


def test_other_host_port_or_scheme_is_out_of_scope():
    for url in (
        "https://other.example.com/.well-known/oauth-authorization-server",
        "https://mcp.example.com:8443/token",
        "https://mcp.example.com.evil.net/token",
    ):
        assert origin_scoped_headers(url, SERVER_URL, CF_HEADERS) == {}


def test_sdk_owned_and_hop_by_hop_headers_are_never_forwarded():
    scoped = origin_scoped_headers("https://mcp.example.com/token", SERVER_URL, {
        **CF_HEADERS,
        "Authorization": "Bearer already-set-by-sdk",
        "Content-Type": "application/x-www-form-urlencoded",
        "Content-Length": "3",
        "Host": "mcp.example.com",
        "Connection": "keep-alive",
        "Transfer-Encoding": "chunked",
        "Cookie": "session=1",
        "mcp-session-id": "abc",
        "mcp-protocol-version": "2026-11-05",
        "Keep-Alive": "timeout=5",
        "Proxy-Authorization": "Basic another-credential",
        "Proxy-Authenticate": "Basic",
        "TE": "trailers",
        "Trailer": "X-Checksum",
        "Upgrade": "h2c",
        "Expect": "100-continue",
    })
    assert scoped == CF_HEADERS


def test_header_case_does_not_matter_for_the_denylist():
    scoped = origin_scoped_headers("https://mcp.example.com/token", SERVER_URL,
                                   {"AUTHORIZATION": "Bearer x", "Mcp-Session-Id": "abc", **CF_HEADERS})
    assert scoped == CF_HEADERS


def test_unset_configuration_scopes_to_nothing():
    assert origin_scoped_headers("https://mcp.example.com/token", SERVER_URL, None) == {}
    assert origin_scoped_headers("https://mcp.example.com/token", SERVER_URL, {}) == {}


def test_malformed_urls_scope_to_nothing_and_never_raise():
    """OAuth request URLs can come from server-advertised metadata; a bad port or IPv6 literal
    must fail the scoping, not raise out of the stamping helper (#155708 review finding)."""
    for url in ("https://as.example.com:abc/token", "http://[::1", "not a url"):
        assert origin_scoped_headers(url, SERVER_URL, CF_HEADERS) == {}
    # A malformed server URL never matches anything either — including another malformed URL.
    assert origin_scoped_headers("https://mcp.example.com/token", "https://mcp.example.com:abc", CF_HEADERS) == {}
    assert origin_scoped_headers("https://mcp.example.com:abc", "https://mcp.example.com:abc", CF_HEADERS) == {}


def test_ipv6_literals_and_userinfo_follow_the_same_origin_rules():
    """Positive regression guard: brackets strip consistently on both sides of the comparison,
    the port still separates origins, and userinfo is not part of an origin."""
    assert origin_scoped_headers("http://[::1]:8080/x", "http://[::1]:8080/mcp", CF_HEADERS) == CF_HEADERS
    assert origin_scoped_headers("http://[::1]:8081/x", "http://[::1]:8080/mcp", CF_HEADERS) == {}
    assert origin_scoped_headers("https://user:pass@mcp.example.com/token", SERVER_URL, CF_HEADERS) == CF_HEADERS


def test_sdk_set_header_is_not_overwritten():
    from tools.mcp_tool import sdk_httpx

    request = sdk_httpx().Request(
        "POST", "https://mcp.example.com/token", headers={"CF-Access-Client-Id": "sdk-set"})
    stamp_configured_origin_headers(request, SERVER_URL, CF_HEADERS)
    assert request.headers["CF-Access-Client-Id"] == "sdk-set"
    assert request.headers["CF-Access-Client-Secret"] == "repro-secret"


# ---------------------------------------------------------------------------
# The SDK auth-flow pump: discovery requests seen end to end
# ---------------------------------------------------------------------------


async def _noop_redirect(url: str) -> None:
    pass


async def _noop_callback():
    return ("code", None)


async def _make_flow(tmp_path, monkeypatch, *, origin_headers=None):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    assert _HERMES_PROVIDER_CLS is not None
    httpx = sdk_httpx()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    storage = HermesTokenStorage("srv")
    await storage.set_tokens(
        OAuthToken(access_token="old_access", token_type="Bearer", expires_in=3600, refresh_token="old_refresh")
    )
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="test-client",
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
    )
    provider = _HERMES_PROVIDER_CLS(
        server_name="srv",
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
            client_name="Hermes Agent",
        ),
        storage=storage,
        redirect_handler=_noop_redirect,
        callback_handler=_noop_callback,
        origin_headers=origin_headers,
    )
    req = httpx.Request("POST", "https://example.com/mcp")
    return httpx, req, provider.async_auth_flow(req)


@pytest.mark.asyncio
async def test_same_origin_discovery_request_carries_configured_headers(tmp_path, monkeypatch):
    httpx, req, flow = await _make_flow(tmp_path, monkeypatch, origin_headers=CF_HEADERS)

    outbound = await flow.__anext__()
    assert outbound is req  # the caller's MCP request first, untouched
    fake_401 = httpx.Response(
        401, request=outbound,
        headers={"www-authenticate": 'Bearer resource_metadata="https://example.com/.well-known/oauth-protected-resource"'},
    )
    discovery = await flow.asend(fake_401)

    assert discovery is not req, "SDK must have yielded its own discovery request"
    assert ".well-known" in str(discovery.url)
    assert discovery.headers.get("CF-Access-Client-Id") == "repro-id.access"
    assert discovery.headers.get("CF-Access-Client-Secret") == "repro-secret"

    await flow.aclose()


@pytest.mark.asyncio
async def test_callers_own_mcp_request_is_left_untouched(tmp_path, monkeypatch):
    """The caller's MCP request already carries the transport's headers — the OAuth layer must
    not decorate it a second time."""
    _httpx, req, flow = await _make_flow(tmp_path, monkeypatch, origin_headers=CF_HEADERS)

    outbound = await flow.__anext__()
    assert outbound is req
    assert "CF-Access-Client-Id" not in outbound.headers

    await flow.aclose()


@pytest.mark.asyncio
async def test_third_party_authorization_server_discovery_gets_no_configured_headers(tmp_path, monkeypatch):
    """A protected-resource challenge on another origin must not leak the server's proxy
    credentials to that origin."""
    httpx, _req, flow = await _make_flow(tmp_path, monkeypatch, origin_headers=CF_HEADERS)

    outbound = await flow.__anext__()
    fake_401 = httpx.Response(
        401, request=outbound,
        headers={"www-authenticate": 'Bearer resource_metadata="https://other.example.net/.well-known/oauth-protected-resource"'},
    )
    discovery = await flow.asend(fake_401)

    assert ".well-known" in str(discovery.url)
    assert "example.com" not in str(discovery.url).split("/")[2]
    assert "CF-Access-Client-Id" not in discovery.headers
    assert "CF-Access-Client-Secret" not in discovery.headers

    await flow.aclose()


# ---------------------------------------------------------------------------
# MCPOAuthManager wiring
# ---------------------------------------------------------------------------


def _interactive_manager(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.mcp_oauth._is_interactive", lambda: True)
    reset_manager_for_tests()
    return get_manager()


def test_manager_passes_server_headers_to_the_provider(tmp_path, monkeypatch):
    manager = _interactive_manager(tmp_path, monkeypatch)
    provider = manager.get_or_build_provider("srv", SERVER_URL, {}, server_headers=CF_HEADERS)
    assert provider is not None
    assert provider._hermes_origin_headers == CF_HEADERS


def test_manager_rebuilds_the_provider_when_configured_headers_change(tmp_path, monkeypatch):
    manager = _interactive_manager(tmp_path, monkeypatch)
    first = manager.get_or_build_provider("srv", SERVER_URL, {}, server_headers=CF_HEADERS)
    changed = {**CF_HEADERS, "X-Extra": "1"}
    second = manager.get_or_build_provider("srv", SERVER_URL, {}, server_headers=changed)
    assert second is not first
    assert second._hermes_origin_headers["X-Extra"] == "1"
    # Unchanged headers reuse the cached provider, as before.
    assert manager.get_or_build_provider("srv", SERVER_URL, {}, server_headers=changed) is second


def test_transport_passes_only_the_configured_headers(tmp_path, monkeypatch):
    """The transport hands the CONFIGURED headers to the manager — not its connect-time dict,
    which additionally carries the identity overlay and the mcp-protocol-version seed. CLI paths
    pass the configured set too, so entries built from either side compare equal and the
    provider cache is not thrashed between `hermes mcp login` and a runtime connect (#155708)."""
    from types import SimpleNamespace

    from tools import mcp_tool_transport
    from tools.mcp_oauth_manager import reset_manager_for_tests

    captured = {}

    class FakeManager:
        def get_or_build_provider(self, name, server_url, oauth_config, server_headers=None):
            captured["server_headers"] = server_headers
            return object()

    monkeypatch.setattr("tools.mcp_oauth_manager.get_manager", lambda: FakeManager())
    reset_manager_for_tests()
    server = SimpleNamespace(name="srv", _auth_type="oauth")

    config = {"url": SERVER_URL, "auth": "oauth", "headers": dict(CF_HEADERS)}
    mcp_tool_transport.MCPServerTransportMixin._build_oauth_auth(server, SERVER_URL, config)
    assert captured["server_headers"] == CF_HEADERS

    # A server without configured headers passes None, and a non-oauth server never asks.
    mcp_tool_transport.MCPServerTransportMixin._build_oauth_auth(
        server, SERVER_URL, {"url": SERVER_URL, "auth": "oauth"})
    assert captured["server_headers"] is None
    server._auth_type = "none"
    assert mcp_tool_transport.MCPServerTransportMixin._build_oauth_auth(server, SERVER_URL, config) is None


# ---------------------------------------------------------------------------
# Pre-flight metadata discovery (its own bare client) — on the wire
# ---------------------------------------------------------------------------


def _recording_server(reply_status=404):
    seen: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
            body = b"{}"
            self.send_response(reply_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_preflight_metadata_discovery_carries_configured_headers_on_the_wire(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    server, seen = _recording_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_manager_for_tests()
        provider = _HERMES_PROVIDER_CLS(
            server_name="srv",
            server_url=f"{base}/mcp",
            client_metadata=OAuthClientMetadata(
                redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")], client_name="Hermes Agent"),
            storage=HermesTokenStorage("srv"),
            redirect_handler=_noop_redirect,
            callback_handler=_noop_callback,
            origin_headers=CF_HEADERS,
        )

        asyncio.run(provider._prefetch_oauth_metadata())

        assert seen, "no discovery request reached the server"
        assert all(".well-known" in path for path, _ in seen)
        assert all(headers.get("cf-access-client-id") == "repro-id.access" for _, headers in seen)
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Device flow end to end against a header-authenticating proxy stand-in
# ---------------------------------------------------------------------------


def _access_proxy_server():
    """A stand-in for Cloudflare Access service auth: any request without the configured
    service-token header is 302'd to a login origin; with it, a full RFC 8628 server answers."""
    from tools.mcp_oauth_device import DEVICE_GRANT

    seen: list[tuple[str, str, bool]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, payload=None, headers=None):
            data = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            if payload is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)

        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            has_service_token = "cf-access-client-id" in {k.lower() for k in self.headers}
            seen.append((self.command, self.path.split("?")[0], has_service_token))
            if not has_service_token:
                return self._reply(
                    302, headers={"Location": "https://example-team.cloudflareaccess.com/cdn-cgi/access/login"})
            base = f"http://127.0.0.1:{self.server.server_port}"
            path = self.path.split("?")[0]
            if path == "/mcp":
                return self._reply(401, {"error": "unauthorized"},
                                   {"WWW-Authenticate": f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource"'})
            if path.startswith("/.well-known/oauth-protected-resource"):
                return self._reply(200, {"resource": f"{base}/mcp", "authorization_servers": [base]})
            if path == "/.well-known/oauth-authorization-server":
                return self._reply(200, {
                    "issuer": base,
                    "authorization_endpoint": f"{base}/authorize",
                    "token_endpoint": f"{base}/token",
                    "registration_endpoint": f"{base}/register",
                    "device_authorization_endpoint": f"{base}/device",
                    "response_types_supported": ["code"],
                    "code_challenge_methods_supported": ["S256"],
                    "grant_types_supported": [DEVICE_GRANT, "refresh_token"],
                })
            if path == "/register":
                return self._reply(201, {"client_id": "repro-client",
                                         "redirect_uris": ["http://localhost:8420/callback"]})
            if path == "/device":
                return self._reply(200, {"device_code": "fixture-device-code", "user_code": "TEST-CODE",
                                         "verification_uri": f"{base}/verify", "interval": 0.01, "expires_in": 30})
            if path == "/token":
                return self._reply(200, {"access_token": "at", "refresh_token": "rt",
                                         "token_type": "Bearer", "expires_in": 3600})
            return self._reply(404, {})

        do_GET = do_POST = do_HEAD = do_DELETE = _handle

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_device_login_behind_a_header_authenticating_proxy_succeeds(tmp_path, monkeypatch, capsys):
    from tools.mcp_oauth_device import login_device

    server, seen = _access_proxy_server()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        asyncio.run(login_device("srv", f"{base}/mcp", {"timeout": 20},
                                 server_headers={"CF-Access-Client-Id": "repro-id.access",
                                                 "CF-Access-Client-Secret": "repro-secret"}))

        assert seen, "no request reached the server"
        header_less = [(method, path) for method, path, ok in seen if not ok]
        assert not header_less, f"OAuth request(s) left without the configured headers: {header_less}"
        paths = {path for _, path, _ in seen}
        assert any(p.startswith("/.well-known/oauth-protected-resource") for p in paths)
        assert "/register" in paths and "/device" in paths and "/token" in paths
    finally:
        server.shutdown()
        server.server_close()
