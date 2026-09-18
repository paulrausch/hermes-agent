"""Scope + mint ``X-Origin-*`` headers for outbound control-plane tool calls.

Kept out of ``tools/mcp_tool.py`` so the transport file only gains a couple of
call sites. See ``gateway/origin_attestation.py`` for the contract itself.

Scoping is deliberately narrow: only the control-plane MCP server, and only the
tool call that submits a job. Every other MCP server on the box sees exactly
the traffic it saw before.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# Only these MCP servers ever receive origin headers. Overridable for tests /
# renamed deployments via a comma-separated env var.
DEFAULT_ORIGIN_SERVERS = frozenset({"control-plane"})
ORIGIN_SERVERS_ENV = "HERMES_ORIGIN_ATTESTATION_SERVERS"

# The dispatching tool, and the op that actually creates a job. The plane only
# reads the headers on job_submit; minting on anything else would spend a
# single-use attestation for nothing.
ORIGIN_TOOL_NAMES = frozenset({"jsbc_call"})
ORIGIN_OPS = frozenset({"agents.job_submit"})


def _origin_servers() -> frozenset:
    raw = (os.environ.get(ORIGIN_SERVERS_ENV) or "").strip()
    if not raw:
        return DEFAULT_ORIGIN_SERVERS
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _extract_op(args: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(args, Mapping):
        return ""
    op = args.get("op")
    return op.strip() if isinstance(op, str) else ""


def should_mint_origin(
    server_name: str,
    tool_name: str,
    args: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Whether this specific outbound call should carry an origin attestation.

    ``args`` is consulted for the *op name only* — routing identity itself is
    never taken from tool arguments.
    """
    if server_name not in _origin_servers():
        return False
    if tool_name not in ORIGIN_TOOL_NAMES:
        return False
    return _extract_op(args) in ORIGIN_OPS


def mint_origin_headers_for_call(
    server_name: str,
    tool_name: str,
    args: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Return freshly minted headers for this call, or ``{}``.

    Never raises. A fresh nonce is minted per call; the result is single-use
    and must not be cached.
    """
    try:
        if not should_mint_origin(server_name, tool_name, args):
            return {}
        from gateway.origin_attestation import build_origin_headers

        headers = build_origin_headers()
        if headers:
            logger.debug(
                "origin attestation: minted %d X-Origin-* headers for %s/%s",
                len(headers), server_name, tool_name,
            )
        else:
            logger.warning(
                "origin attestation: scoped to mint for %s/%s but built 0 headers"
                " — dispatch will be refused by an attested plane",
                server_name, tool_name,
            )
        return headers
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "origin attestation: mint failed for %s/%s (%s)"
            " — dispatch will be refused by an attested plane",
            server_name, tool_name, exc, exc_info=True,
        )
        return {}


# Names we may write onto an outbound request. Used to scrub any stale value
# before injecting, so a request can never carry two conversations' headers.
_ORIGIN_HEADER_PREFIX = "x-origin-"


def make_origin_request_hook(server):
    """Build an httpx ``request`` event hook that stamps per-call headers.

    Why a hook and not ``client.headers``: the ``httpx.AsyncClient`` behind an
    MCP HTTP transport is long-lived and shared by every call to that server.
    Mutating ``client.headers`` would leak one conversation's identity onto
    another conversation's call. The MCP SDK also builds and sends its own
    ``Request`` objects, so a per-request ``headers=`` kwarg is not reachable
    from the tool handler either — the request hook is the one place that can
    see the individual ``tools/call`` POST.

    The headers are handed over via ``server._pending_origin_headers``, which
    the tool handler sets while holding the server's ``_rpc_lock`` (one RPC in
    flight per server) and clears in a ``finally``.
    """

    async def _stamp_origin_headers(request) -> None:
        try:
            pending = getattr(server, "_pending_origin_headers", None)
            if not pending:
                return
            if request.method.upper() != "POST":
                return
            # Only the JSON-RPC tools/call POST. A session GET or an
            # initialize handshake must not spend a single-use attestation.
            try:
                body = request.content or b""
            except Exception:
                body = b""
            if b'"tools/call"' not in body:
                return
            for name in [
                key for key in request.headers
                if key.lower().startswith(_ORIGIN_HEADER_PREFIX)
            ]:
                del request.headers[name]
            for name, value in pending.items():
                request.headers[name] = value
        except Exception as exc:  # pragma: no cover - never break a call
            logger.warning(
                "origin attestation: header injection failed (%s)"
                " — request will go out unattested",
                exc, exc_info=True,
            )

    return _stamp_origin_headers
