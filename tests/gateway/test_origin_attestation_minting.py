"""Behavioural tests for origin attestation minting: ContextVar sourcing,
per-call isolation, nonce freshness, absent-secret degradation, sanitisation.
"""

from __future__ import annotations

import hashlib
import hmac
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway import origin_attestation as oa  # noqa: E402
from gateway import session_context as sc  # noqa: E402
from tools import mcp_origin_headers as moh  # noqa: E402

SECRET = "test-secret-do-not-use"
SUBMIT_ARGS = {"op": "agents.job_submit", "args": {"prompt": "hi"}}


@contextmanager
def session(**kw):
    """Set the gateway session ContextVars for the duration of the block."""
    tokens = sc.set_session_vars(**kw)
    try:
        yield
    finally:
        sc.clear_session_vars(tokens)


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv(oa.ORIGIN_SECRET_ENV, SECRET)
    return SECRET


# --------------------------------------------------------------------------
# identity comes from ContextVars, never from tool arguments
# --------------------------------------------------------------------------

def test_headers_are_built_from_contextvars(secret):
    with session(platform="telegram", chat_id="193825162", user_id="u42",
                 message_id="9001", session_id="sess-real"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert h["X-Origin-Surface"] == "telegram"
    assert h["X-Origin-Chat-Id"] == "193825162"
    assert h["X-Origin-User-Id"] == "u42"
    assert h["X-Origin-Message-Id"] == "9001"
    assert h["X-Origin-Session-Id"] == "sess-real"


def test_tool_arguments_cannot_forge_identity(secret):
    """A model-supplied `origin` in the args must not influence the headers."""
    hostile = {
        "op": "agents.job_submit",
        "origin": {"chat_id": "999666", "user_id": "attacker",
                   "surface": "slack", "session_id": "sess-evil"},
        "chat_id": "999666",
    }
    with session(platform="telegram", chat_id="193825162", user_id="u42",
                 session_id="sess-real"):
        h = moh.mint_origin_headers_for_call("control-plane", "jsbc_call", hostile)
    assert h["X-Origin-Chat-Id"] == "193825162"
    assert h["X-Origin-User-Id"] == "u42"
    assert h["X-Origin-Surface"] == "telegram"
    assert h["X-Origin-Session-Id"] == "sess-real"
    assert "999666" not in h.values()
    assert "attacker" not in h.values()


def test_no_stray_x_origin_headers(secret):
    """Any unknown X-Origin-* header is BAD_ARGS on the plane."""
    allowed = set(oa.FIELD_HEADERS.values()) | {
        oa.HEADER_ATTESTATION, oa.HEADER_ISSUED_AT, oa.HEADER_NONCE,
        oa.HEADER_ASSERTED_BY, oa.HEADER_ATTESTATION_V,
        oa.HEADER_ATTESTATION_CLAIM,
    }
    with session(platform="telegram", chat_id="1", session_id="s"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert h
    for name in h:
        assert name.lower().startswith("x-origin-")
        assert name in allowed, f"unexpected header {name}"


# --------------------------------------------------------------------------
# per-call isolation: no shared-client leakage
# --------------------------------------------------------------------------

def test_two_sessions_produce_different_header_sets(secret):
    with session(platform="telegram", chat_id="111", user_id="alice",
                 session_id="sess-a"):
        first = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    with session(platform="slack", chat_id="222", user_id="bob",
                 session_id="sess-b"):
        second = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)

    assert first != second
    assert first["X-Origin-Chat-Id"] == "111"
    assert second["X-Origin-Chat-Id"] == "222"
    assert first["X-Origin-Session-Id"] != second["X-Origin-Session-Id"]
    assert first["X-Origin-User-Id"] != second["X-Origin-User-Id"]
    assert first["X-Origin-Surface"] != second["X-Origin-Surface"]
    assert first["X-Origin-Attestation"] != second["X-Origin-Attestation"]


def test_fresh_nonce_per_call(secret):
    nonces, attestations, claims = set(), set(), set()
    with session(platform="telegram", chat_id="1", session_id="s"):
        for _ in range(25):
            h = moh.mint_origin_headers_for_call(
                "control-plane", "jsbc_call", SUBMIT_ARGS)
            nonces.add(h["X-Origin-Nonce"])
            attestations.add(h["X-Origin-Attestation"])
            claims.add(h["X-Origin-Attestation-Claim"])
    # identical identity, but every mint is single-use and distinct
    assert len(nonces) == 25
    assert len(attestations) == 25
    assert len(claims) == 25


def test_claim_is_sha256_of_nonce_and_version_is_v2(secret):
    with session(platform="telegram", chat_id="1", session_id="s"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert h["X-Origin-Attestation-V"] == "v2"
    assert h["X-Origin-Asserted-By"] == oa.DEFAULT_PRINCIPAL
    assert h["X-Origin-Attestation-Claim"] == hashlib.sha256(
        h["X-Origin-Nonce"].encode("utf-8")).hexdigest()


def test_wire_headers_verify_against_v2_preimage(secret):
    """Recompute the plane's check: the HMAC must cover the sent values."""
    with session(platform="telegram", chat_id="193825162", thread_id="77",
                 user_id="u42", message_id="9001", session_id="sess-real",
                 chat_name="Ops Room"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)

    fields = {
        field: h.get(header, "")
        for field, header in oa.FIELD_HEADERS.items()
    }
    pre = oa.preimage(
        fields,
        principal=h["X-Origin-Asserted-By"],
        issued_at=h["X-Origin-Issued-At"],
        nonce=h["X-Origin-Nonce"],
    )
    assert pre.startswith("v2|ops-full|")
    expected = hmac.new(
        SECRET.encode("utf-8"), pre.encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    assert h["X-Origin-Attestation"] == expected


# --------------------------------------------------------------------------
# absent secret => zero headers, no exception
# --------------------------------------------------------------------------

def test_absent_secret_yields_no_headers_and_no_exception(monkeypatch):
    monkeypatch.delenv(oa.ORIGIN_SECRET_ENV, raising=False)
    monkeypatch.setattr(oa, "_resolve_secret", lambda: "")
    with session(platform="telegram", chat_id="193825162", session_id="s"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert h == {}
    assert not [k for k in h if k.lower().startswith("x-origin-")]


def test_empty_secret_yields_no_headers(monkeypatch):
    monkeypatch.setenv(oa.ORIGIN_SECRET_ENV, "   ")
    monkeypatch.setattr(oa, "_resolve_secret", lambda: "   ")
    with session(platform="telegram", chat_id="1", session_id="s"):
        assert oa.build_origin_headers() == {}


def test_unaddressable_identity_yields_no_headers(secret):
    """No chat_id and no session_id is not a route -> mint nothing."""
    with session(platform="telegram", user_id="u1"):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert h == {}


# --------------------------------------------------------------------------
# scoping
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "server,tool,args,expected",
    [
        ("control-plane", "jsbc_call", {"op": "agents.job_submit"}, True),
        ("control-plane", "jsbc_call", {"op": "agents.job_status"}, False),
        ("control-plane", "jsbc_catalog", {"op": "agents.job_submit"}, False),
        ("some-other-mcp", "jsbc_call", {"op": "agents.job_submit"}, False),
        ("control-plane", "jsbc_call", None, False),
        ("control-plane", "jsbc_call", {}, False),
    ],
)
def test_minting_is_narrowly_scoped(secret, server, tool, args, expected):
    assert moh.should_mint_origin(server, tool, args) is expected
    with session(platform="telegram", chat_id="1", session_id="s"):
        h = moh.mint_origin_headers_for_call(server, tool, args)
    assert bool(h) is expected


# --------------------------------------------------------------------------
# sanitisation: signed text and wire value must be identical
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Ops Room \u2014 caf\u00e9 \U0001f600", "Ops Room  caf"),   # non-ASCII stripped
        ("  padded  ", "padded"),                                    # stripped
        ("tab\tand\nnewline", "tabandnewline"),
        (12345, "12345"),                                            # int accepted
        (None, ""),
    ],
)
def test_sanitize_field_strips_non_ascii(raw, expected):
    got = oa.sanitize_field(raw)
    assert got.isascii()
    assert all(32 <= ord(c) <= 126 for c in got)
    assert got == expected


def test_oversized_value_capped_at_128(secret):
    long_name = "N" * 500
    with session(platform="telegram", chat_id="1", session_id="s",
                 chat_name=long_name):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)
    assert len(h["X-Origin-Label"]) == oa.MAX_FIELD_LEN == 128


def test_sanitisation_is_identical_in_preimage_and_on_the_wire(secret):
    """The plane re-derives the preimage from the header values it received.

    If we signed the raw value but sent the sanitised one (or vice versa) every
    submit would fail verification. This asserts they are byte-identical.
    """
    dirty_label = "caf\u00e9 \U0001f600 \u2014 " + ("L" * 300)
    dirty_user = "u\u00ff42\t"
    with session(platform="telegram", chat_id="193825162",
                 session_id="sess-real", user_id=dirty_user,
                 chat_name=dirty_label):
        h = moh.mint_origin_headers_for_call(
            "control-plane", "jsbc_call", SUBMIT_ARGS)

    for header in h.values():
        assert header.isascii()
        assert len(header) <= 256
        assert all(32 <= ord(c) <= 126 for c in header)

    # Sanitised exactly once: re-sanitising a sent value is a no-op.
    for field, header_name in oa.FIELD_HEADERS.items():
        sent = h.get(header_name, "")
        assert oa.sanitize_field(sent) == sent

    # And the signature covers exactly those sent values.
    fields = {f: h.get(hn, "") for f, hn in oa.FIELD_HEADERS.items()}
    expected = hmac.new(
        SECRET.encode("utf-8"),
        oa.preimage(fields, principal=h["X-Origin-Asserted-By"],
                    issued_at=h["X-Origin-Issued-At"],
                    nonce=h["X-Origin-Nonce"]).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    assert h["X-Origin-Attestation"] == expected
    assert len(h["X-Origin-Label"]) == 128
