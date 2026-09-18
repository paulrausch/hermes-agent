"""Transport-level tests: X-Origin-* is stamped per-request, never on the
long-lived shared httpx client.

The regression this guards: setting the headers on the shared AsyncClient would
attach one conversation's identity to another conversation's call.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway import origin_attestation as oa  # noqa: E402
from gateway import session_context as sc  # noqa: E402
from tools import mcp_origin_headers as moh  # noqa: E402

SECRET = "test-secret-do-not-use"
SUBMIT_ARGS = {"op": "agents.job_submit"}
TOOLS_CALL_BODY = b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}'
INITIALIZE_BODY = b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}'


class FakeServer:
    """Stands in for MCPServerTask: only _pending_origin_headers matters."""

    def __init__(self):
        self._pending_origin_headers = None


@contextmanager
def session(**kw):
    tokens = sc.set_session_vars(**kw)
    try:
        yield
    finally:
        sc.clear_session_vars(tokens)


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv(oa.ORIGIN_SECRET_ENV, SECRET)
    return SECRET


def _request(body=TOOLS_CALL_BODY, headers=None):
    return httpx.Request(
        "POST", "http://127.0.0.1:8765/mcp",
        content=body, headers=headers or {},
    )


async def _run(server, request):
    hook = moh.make_origin_request_hook(server)
    await hook(request)
    return request


@pytest.mark.asyncio
async def test_hook_stamps_pending_headers_onto_tools_call(secret):
    server = FakeServer()
    with session(platform="telegram", chat_id="111", session_id="sess-a"):
        server._pending_origin_headers = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    req = await _run(server, _request())
    assert req.headers["x-origin-chat-id"] == "111"
    assert req.headers["x-origin-session-id"] == "sess-a"
    assert req.headers["x-origin-attestation"]


@pytest.mark.asyncio
async def test_hook_is_a_noop_when_nothing_pending():
    server = FakeServer()
    req = await _run(server, _request())
    assert not [k for k in req.headers if k.lower().startswith("x-origin-")]


@pytest.mark.asyncio
async def test_hook_skips_non_tools_call_requests(secret):
    """A handshake/initialize POST must not spend a single-use attestation."""
    server = FakeServer()
    with session(platform="telegram", chat_id="111", session_id="sess-a"):
        server._pending_origin_headers = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    req = await _run(server, _request(body=INITIALIZE_BODY))
    assert not [k for k in req.headers if k.lower().startswith("x-origin-")]


@pytest.mark.asyncio
async def test_shared_client_headers_are_never_mutated(secret):
    """The AsyncClient is long-lived and shared: its headers must stay clean."""
    captured = []

    async def handler(request):
        captured.append(dict(request.headers))
        return httpx.Response(200, json={"ok": True})

    server = FakeServer()
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        headers={"Authorization": "Bearer t"},
        event_hooks={"request": [moh.make_origin_request_hook(server)]},
    ) as client:
        # conversation A
        with session(platform="telegram", chat_id="111", user_id="alice",
                     session_id="sess-a"):
            server._pending_origin_headers = moh.mint_origin_headers_for_call(
                "control-plane", "jsbc_call", SUBMIT_ARGS)
        await client.post("http://x/mcp", content=TOOLS_CALL_BODY)
        server._pending_origin_headers = None

        # the shared client itself never acquired origin state
        assert not [k for k in client.headers
                    if k.lower().startswith("x-origin-")]

        # conversation B
        with session(platform="slack", chat_id="222", user_id="bob",
                     session_id="sess-b"):
            server._pending_origin_headers = moh.mint_origin_headers_for_call(
                "control-plane", "jsbc_call", SUBMIT_ARGS)
        await client.post("http://x/mcp", content=TOOLS_CALL_BODY)
        server._pending_origin_headers = None

        # a call with nothing pending carries no origin at all
        await client.post("http://x/mcp", content=TOOLS_CALL_BODY)

    assert len(captured) == 3
    a, b, c = captured
    assert a["x-origin-chat-id"] == "111"
    assert a["x-origin-user-id"] == "alice"
    assert b["x-origin-chat-id"] == "222"
    assert b["x-origin-user-id"] == "bob"
    assert a["x-origin-attestation"] != b["x-origin-attestation"]
    assert a["x-origin-nonce"] != b["x-origin-nonce"]
    # no bleed-through onto the third, header-free call
    assert not [k for k in c if k.lower().startswith("x-origin-")]
    # the server's own auth header still rides along
    assert a["authorization"] == "Bearer t"


@pytest.mark.asyncio
async def test_stale_origin_headers_are_scrubbed_before_injection(secret):
    """A retried request object must not keep the previous conversation's."""
    server = FakeServer()
    with session(platform="telegram", chat_id="111", session_id="sess-a"):
        stale = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    with session(platform="slack", chat_id="222", session_id="sess-b"):
        server._pending_origin_headers = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)

    req = await _run(server, _request(headers=stale))
    assert req.headers["x-origin-chat-id"] == "222"
    assert req.headers["x-origin-session-id"] == "sess-b"
    assert req.headers.get_list("x-origin-chat-id") == ["222"]
    assert "111" not in [v for k, v in req.headers.items()
                         if k.lower().startswith("x-origin-")]
