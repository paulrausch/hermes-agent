"""Outbound gate: actionable content must never be silently suppressed."""
import pytest

from gateway.platforms.outbound_gate import (
    DELIVER, REFORMAT, SUPPRESS, ENFORCE, DRY_RUN, GateConfig, gate,
)

ENFORCED = {"config": GateConfig(silence=ENFORCE, reformat=ENFORCE)}


def _gate(text, **ctx):
    return gate(text, event_type="orchestration", context={**ENFORCED, **ctx})


def test_marker_plus_approval_id_is_delivered_not_dropped():
    text = "Verified the deploy. Approve ap_1a2b3c4d to proceed with the restart. [SILENT]"
    d = _gate(text)
    assert d.action != SUPPRESS
    assert d.send and d.outgoing
    assert "ap_1a2b3c4d" in d.outgoing
    assert "[SILENT]" not in d.outgoing.upper()
    assert d.silence_overridden and not d.suppressed_actionable
    assert d.log_fields()["out_chars"] > 0


def test_marker_plus_needs_you_without_id_is_delivered():
    d = _gate("[SILENT] I need your decision on whether the client migration goes ahead tonight.")
    assert d.action in (DELIVER, REFORMAT)
    assert d.outgoing and "decision" in d.outgoing.lower()
    assert d.silence_overridden


def test_bare_marker_line_plus_actionable_is_delivered():
    d = _gate("SILENT\nPlease approve ap_9f8e7d6c before noon.")
    assert d.outgoing and "ap_9f8e7d6c" in d.outgoing
    assert not any(ln.strip().upper() == "SILENT" for ln in d.outgoing.splitlines())


def test_dry_run_matches_enforce_for_actionable():
    d = gate("Approve ap_1a2b3c4d now. [SILENT]", context={"config": GateConfig(DRY_RUN, DRY_RUN)})
    assert d.silence_overridden and d.action != SUPPRESS


@pytest.mark.parametrize("text", [
    "[SILENT]",
    "[silent]",
    "Checked everything, all quiet, nothing else to add. [SILENT]",
    "SILENT",
    "",
])
def test_genuinely_empty_replies_still_suppressed(text):
    d = _gate(text)
    assert d.action == SUPPRESS
    assert d.outgoing is None
    assert not d.silence_overridden and not d.suppressed_actionable


def test_nothing_new_still_suppressed():
    assert _gate("Nothing new to report.").outgoing is None
