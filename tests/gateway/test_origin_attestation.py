"""Tests for the Hermes-side origin attestation minting half.

Covers the frozen contract in /root/origin-attestation.md: preimage shape,
HMAC test vectors, ContextVar sourcing, per-call nonce freshness, absent-secret
behaviour and sanitisation parity between the signed text and the wire value.
"""

from __future__ import annotations

import hashlib
import hmac
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway import origin_attestation as oa  # noqa: E402
from gateway import session_context as sc  # noqa: E402
from tools import mcp_origin_headers as moh  # noqa: E402

SECRET = "test-secret-do-not-use"
NONCE = "6f1d9c2b8a4e7f30"


# --------------------------------------------------------------------------
# preimage / HMAC
# --------------------------------------------------------------------------

def test_v1_preimage_fixed_arity_and_order():
    pre = oa.preimage(
        {"surface": "telegram", "chat_id": "193825162", "session_id": "s-abc",
         "user_id": "u1"},
        principal="ops-full", issued_at="1787900000",
    )
    assert pre == "v1|ops-full|1787900000|telegram|193825162||s-abc|u1||"
    # fixed arity: 10 slots for v1, empty string where absent
    assert len(pre.split("|")) == 10


def test_v2_preimage_inserts_nonce_after_issued_at():
    pre = oa.preimage(
        {"surface": "telegram", "chat_id": "193825162", "session_id": "s-abc",
         "user_id": "u1"},
        principal="ops-full", issued_at="1787900000", nonce=NONCE,
    )
    assert pre == (
        "v2|ops-full|1787900000|6f1d9c2b8a4e7f30|telegram|193825162||s-abc|u1||"
    )
    assert len(pre.split("|")) == 11


@pytest.mark.parametrize("nonce", [None, NONCE])
def test_attestation_matches_independent_hmac(nonce):
    fields = {"surface": "telegram", "chat_id": "193825162",
              "session_id": "s-abc", "user_id": "u1"}
    pre = oa.preimage(
        fields, principal="ops-full", issued_at="1787900000", nonce=nonce,
    )
    expected = hmac.new(
        SECRET.encode("utf-8"), pre.encode("utf-8"), hashlib.sha256,
    ).hexdigest()
    got = oa.attestation_for(
        fields, principal="ops-full", issued_at="1787900000",
        secret=SECRET, nonce=nonce,
    )
    assert got == expected
    assert got == got.lower() and len(got) == 64


def test_empty_fields_still_occupy_their_slot():
    pre = oa.preimage(
        {"surface": "cli", "session_id": "s1"},
        principal="ops-full", issued_at="1", nonce=NONCE,
    )
    assert pre == "v2|ops-full|1|6f1d9c2b8a4e7f30|cli|||s1|||"
