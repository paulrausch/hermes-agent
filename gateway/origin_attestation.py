"""Mint per-call origin attestations for outbound control-plane MCP calls.

The control plane needs a *trustworthy* answer to "which conversation caused
this dispatch". A model-composed ``origin={...}`` argument cannot be that: the
model chooses whether to pass one and which one. Hermes, however, already holds
the system-derived identity of the current turn in the gateway session
ContextVars (``gateway.session_context``), which no model output can reach.

This module turns that trusted identity into the ``X-Origin-*`` HTTP headers
described by the frozen contract (``origin-attestation.md``):

    v1|<principal>|<issued_at>|<surface>|<chat_id>|<thread_id>|<session_id>|<user_id>|<message_id>|<label>
    v2|<principal>|<issued_at>|<nonce>|<surface>|<chat_id>|<thread_id>|<session_id>|<user_id>|<message_id>|<label>

Fixed order, **fixed arity** (every field has a slot, empty string when
absent), joined with ``|``, UTF-8, HMAC-SHA256, lowercase hex. Hermes mints v2
(with a fresh nonce per call); the plane spends each attestation exactly once.

Hard rules honoured here:

* The identity comes from ContextVars only — never from tool arguments.
* Values are sanitised to bounded printable ASCII *before* signing, so the
  signed preimage and the wire value are byte-identical.
* Absent shared secret => zero headers, a debug log, and no exception. An
  unsigned origin is never sent.
* A fresh nonce every call; attestations are never cached or reused.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# Env var holding the secret shared with the control plane and the route
# consumer. 32 bytes of os.urandom, hex.
ORIGIN_SECRET_ENV = "CONTROL_PLANE_ORIGIN_SECRET"

# The plane identity our bearer token authenticates as. This is an AUDIENCE
# binding baked into the preimage, not a user id.
ORIGIN_PRINCIPAL_ENV = "CONTROL_PLANE_ORIGIN_PRINCIPAL"
DEFAULT_PRINCIPAL = "ops-full"

# Routing fields, in preimage order. Fixed arity is load-bearing: with a
# variable-length join a value could be shifted into the next field's slot and
# sign the same string.
FIELD_ORDER = (
    "surface",
    "chat_id",
    "thread_id",
    "session_id",
    "user_id",
    "message_id",
    "label",
)

# Each field is capped at 128 chars after .strip() (the wire cap is 256).
MAX_FIELD_LEN = 128

# The plane's surface allowlist. Anything outside it is refused at the door.
ALLOWED_SURFACES = frozenset(
    {"telegram", "discord", "slack", "cli", "api", "web", "system"}
)

# Hermes platform/source identities mapped onto that allowlist. Unknown
# identities fall back to "system" rather than being guessed at.
SURFACE_ALIASES = {
    "telegram": "telegram",
    "discord": "discord",
    "slack": "slack",
    "cli": "cli",
    "tui": "cli",
    "codex": "cli",
    "desktop": "cli",
    "local": "cli",
    "api": "api",
    "api_server": "api",
    "web": "web",
    "webhook": "api",
    "msgraph_webhook": "api",
    "system": "system",
    "gateway": "system",
    "cron": "system",
    "kanban": "system",
    "tool": "system",
}

# Header names, one per field. Case-insensitive on the wire; the plane refuses
# ANY other header under the X-Origin- prefix with BAD_ARGS, so this mapping is
# exhaustive by design — do not invent extras.
FIELD_HEADERS = {
    "surface": "X-Origin-Surface",
    "chat_id": "X-Origin-Chat-Id",
    "thread_id": "X-Origin-Thread-Id",
    "session_id": "X-Origin-Session-Id",
    "user_id": "X-Origin-User-Id",
    "message_id": "X-Origin-Message-Id",
    "label": "X-Origin-Label",
}
HEADER_ATTESTATION = "X-Origin-Attestation"
HEADER_ISSUED_AT = "X-Origin-Issued-At"
HEADER_NONCE = "X-Origin-Nonce"
HEADER_ASSERTED_BY = "X-Origin-Asserted-By"
HEADER_ATTESTATION_V = "X-Origin-Attestation-V"
HEADER_ATTESTATION_CLAIM = "X-Origin-Attestation-Claim"


def sanitize_field(value: Any) -> str:
    """Reduce *value* to the bounded printable ASCII the contract allows.

    Header values are latin-1 on the wire while the preimage is UTF-8 text, so
    anything non-ASCII is a mismatch waiting to happen and the plane refuses
    it. Stripping here — before signing — keeps the signed text and the wire
    value byte-identical.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = "".join(ch for ch in text if 0x20 <= ord(ch) <= 0x7E)
    text = text.strip()
    if len(text) > MAX_FIELD_LEN:
        text = text[:MAX_FIELD_LEN].strip()
    return text


def sanitize_fields(fields: Mapping[str, Any]) -> Dict[str, str]:
    """Sanitise every routing field, keeping fixed arity (empty when absent)."""
    return {name: sanitize_field(fields.get(name)) for name in FIELD_ORDER}


def normalize_surface(*candidates: Any) -> str:
    """Map Hermes platform/source identities onto the plane's allowlist."""
    for candidate in candidates:
        text = sanitize_field(candidate).lower()
        if not text:
            continue
        mapped = SURFACE_ALIASES.get(text)
        if mapped:
            return mapped
        if text in ALLOWED_SURFACES:
            return text
    return ""


def preimage(
    fields: Mapping[str, Any],
    principal: str,
    issued_at: Any,
    nonce: Optional[str] = None,
) -> str:
    """Build the exact string that gets HMAC'd. Fixed order, fixed arity."""
    if nonce:
        parts = ["v2", str(principal), str(issued_at), str(nonce)]
    else:
        parts = ["v1", str(principal), str(issued_at)]
    parts += [str(fields.get(name) or "") for name in FIELD_ORDER]
    return "|".join(parts)


def attestation_for(
    fields: Mapping[str, Any],
    principal: str,
    issued_at: Any,
    secret: str,
    nonce: Optional[str] = None,
) -> str:
    """HMAC-SHA256 of the preimage, lowercase hex."""
    return hmac.new(
        secret.encode("utf-8"),
        preimage(fields, principal, issued_at, nonce).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def attestation_claim(nonce: str) -> str:
    """``sha256(nonce)`` — the v2 claim declaration the plane cross-checks."""
    return hashlib.sha256(nonce.encode("utf-8")).hexdigest()


def new_nonce() -> str:
    """16 bytes of os.urandom, hex. Unique per submit; never reused."""
    return secrets.token_hex(16)


def _resolve_secret() -> str:
    """Read the shared secret through the profile-aware secret path.

    Falls back to ``os.environ`` when the secret-scope machinery is not
    importable (bare test processes). Any failure resolves to "" — absent
    secret means "send nothing", never an exception.
    """
    try:
        from agent.secret_scope import get_secret

        value = get_secret(ORIGIN_SECRET_ENV, "") or ""
    except Exception:
        value = os.environ.get(ORIGIN_SECRET_ENV, "") or ""
    return value.strip()


def _resolve_principal() -> str:
    try:
        from agent.secret_scope import get_secret

        value = get_secret(ORIGIN_PRINCIPAL_ENV, "") or ""
    except Exception:
        value = os.environ.get(ORIGIN_PRINCIPAL_ENV, "") or ""
    value = sanitize_field(value)
    return value or DEFAULT_PRINCIPAL


def collect_origin_fields() -> Dict[str, str]:
    """Read the current turn's identity from the gateway session ContextVars.

    This is the ONLY trusted source. Tool arguments are never consulted: the
    whole point of the transport channel is that the model is not on it.
    """
    try:
        from gateway.session_context import get_session_env
    except Exception:  # pragma: no cover - session_context always importable
        return {name: "" for name in FIELD_ORDER}

    platform = get_session_env("HERMES_SESSION_PLATFORM", "")
    source = get_session_env("HERMES_SESSION_SOURCE", "")
    raw = {
        "surface": normalize_surface(platform, source, os.environ.get("HERMES_PLATFORM", "")),
        "chat_id": get_session_env("HERMES_SESSION_CHAT_ID", ""),
        "thread_id": get_session_env("HERMES_SESSION_THREAD_ID", ""),
        "session_id": get_session_env("HERMES_SESSION_ID", ""),
        "user_id": get_session_env("HERMES_SESSION_USER_ID", ""),
        "message_id": get_session_env("HERMES_SESSION_MESSAGE_ID", ""),
        "label": get_session_env("HERMES_SESSION_CHAT_NAME", ""),
    }
    fields = sanitize_fields(raw)
    # normalize_surface already returns an allowlisted value or ""; a session
    # with no recognisable surface still has a real address, so fall back to
    # "system" rather than dropping the origin entirely.
    if not fields["surface"]:
        fields["surface"] = "system"
    return fields


def build_origin_headers(
    fields: Optional[Mapping[str, Any]] = None,
    secret: Optional[str] = None,
    principal: Optional[str] = None,
    issued_at: Optional[int] = None,
    nonce: Optional[str] = None,
) -> Dict[str, str]:
    """Mint a fresh v2 attestation and return the ``X-Origin-*`` headers.

    Returns ``{}`` — no headers at all — when the secret is absent or the
    identity cannot address a destination. Never raises: a failure to mint
    degrades the dispatch to "no origin", it does not break the tool call.

    A new nonce is generated on every call. The plane spends each attestation
    exactly once, so the result must never be cached or replayed.
    """
    try:
        resolved_secret = (secret if secret is not None else _resolve_secret()).strip()
        if not resolved_secret:
            logger.debug(
                "origin attestation: %s not set — sending no X-Origin-* headers",
                ORIGIN_SECRET_ENV,
            )
            return {}

        raw = collect_origin_fields() if fields is None else fields
        # Sanitise unconditionally: the signed text and the wire value must be
        # byte-identical, whoever supplied the fields.
        safe = sanitize_fields(raw)

        if safe["surface"] not in ALLOWED_SURFACES:
            logger.debug(
                "origin attestation: surface %r not in allowlist — no headers",
                safe["surface"],
            )
            return {}
        # At least one of chat_id / session_id is required; an origin that
        # cannot address a destination is not a route.
        if not (safe["chat_id"] or safe["session_id"]):
            logger.debug(
                "origin attestation: no chat_id or session_id in session "
                "context — no headers"
            )
            return {}

        resolved_principal = sanitize_field(
            principal if principal is not None else _resolve_principal()
        ) or DEFAULT_PRINCIPAL
        stamp = str(int(time.time()) if issued_at is None else int(issued_at))
        call_nonce = nonce or new_nonce()

        signature = attestation_for(
            safe, resolved_principal, stamp, resolved_secret, call_nonce
        )

        headers: Dict[str, str] = {}
        for name in FIELD_ORDER:
            value = safe[name]
            if value:
                headers[FIELD_HEADERS[name]] = value
        headers[HEADER_ATTESTATION] = signature
        headers[HEADER_ISSUED_AT] = stamp
        headers[HEADER_NONCE] = call_nonce
        # Declarations: cross-checked by the plane, and a mismatch names the
        # real problem instead of a bare "forged".
        headers[HEADER_ASSERTED_BY] = resolved_principal
        headers[HEADER_ATTESTATION_V] = "v2"
        headers[HEADER_ATTESTATION_CLAIM] = attestation_claim(call_nonce)
        return headers
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "origin attestation: minting failed (%s) — no headers", exc, exc_info=True
        )
        return {}
