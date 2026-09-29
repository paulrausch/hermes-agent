"""outbound-gate — the last check between an agent's reply and Paul's Telegram.

What this closes
----------------
A control-plane wake reaches the Hermes gateway's webhook route. The gateway
runs an agent turn and delivers the reply. When the turn has nothing new to
say, the wake prompt tells the model to reply with exactly ``[SILENT]``. At
least six times the model wrote several sentences of verification prose and
then added ``[SILENT]``, and the prose still reached Paul as a DM.

A prompt instruction cannot fix this, because the prompt is not what decides
delivery. Delivery is decided by a matcher in the gateway, and that matcher is
narrower than the failure. Upstream hermes-agent
(``gateway/response_filters.is_autonomous_silence_response``, read at 7b761da2
on 2026-09-25) suppresses a webhook reply only when one of these holds:

- the whole reply is a marker;
- the first or last non-blank line is a marker of at most 64 chars;
- the reply OPENS with a bracketed marker.

So ``<analysis prose> [SILENT]`` on one line is delivered: the last line is the
whole paragraph, longer than 64 chars. The same happens when the marker trails
mid-paragraph, or when the model leaves it out. Every one of those shapes
reaches Paul. See docs/findings/2026-09-25-silent-marker-has-no-gate-on-this-box.md.

The rules, in order (first match wins)
--------------------------------------
======================  ==================  ====================================
rule                    action              when
======================  ==================  ====================================
``silent_marker``       suppress            a bracketed marker (``[SILENT]``,
                                            any case, inner spaces, full-width
                                            brackets) ANYWHERE in the reply. An
                                            override, never an exact match.
``silent_line``         suppress            a bare marker (``SILENT``,
                                            ``NO_REPLY``, ``NO REPLY``) as a
                                            WHOLE LINE, edge punctuation
                                            ignored. A superset of upstream's
                                            first/last-line rule. A word
                                            inside a sentence ("silent retry")
                                            is NOT a marker.
``empty``               suppress            empty or whitespace-only reply
``nothing_new``         suppress            no marker, but the reply announces
                                            that there is nothing new ("nothing
                                            new", "nothing to report", "staying
                                            silent"), AND it neither asks for
                                            Paul nor reports a failure AND
                                            carries no actionable id.
``technical_prose``     reformat            too long, too many lines, or at
                                            least ``JARGON_THRESHOLD`` ids /
                                            shas / code spans / snake_case
                                            field names. Rewritten into Paul's
                                            short, emoji-coded format.
``technical_fallback``  reformat            technical, and no plain sentence
                                            survives the rewrite. Delivered as
                                            status + truncated plain text + a
                                            pointer, never the raw wall.
``plain``               deliver             anything else, unchanged
======================  ==================  ====================================

An ACTIONABLE id always survives a rewrite. That covers any ``ap_`` approval id
(Paul's only use for one is to paste it back), and any id that shares a sentence
with an ask ("approve", "reply with", "answer", "decide"). A reply that asks for
Paul (status ``needs_you``) is actionable too.

Actionable content is never suppressed. When ``silent_marker`` or ``silent_line``
matches a reply that is actionable, the marker is stripped and the rest goes
through the reformat / deliver rules instead, and the decision is logged with
``silence_overridden=True``. (Before, such a reply was dropped to zero chars and
logged ``suppressed_actionable=True``; that field stays in the log and is now
always False, so a non-zero count means a regression.) ``dry_run`` behaves the
same way, so it predicts enforce.

What it deliberately does NOT do
--------------------------------
- **No LLM call.** A gate that calls a model can fail open when the model is
  down, adds latency to every message, and can itself produce a wall of prose.
  The rewrite is deterministic string work. A model-written summary would be
  better prose; where it would run is proposed in hermes/README-outbound-gate.md.
- **No message text in the log.** One line per decision carries the rule,
  lengths, counts and event type. It never carries the text, which may hold
  client names.
- **No imports from Hermes or control_plane.** Stdlib only, like
  hermes/plane_job_events.py. Copy the one file onto the orchestrator box.

Modes
-----
Each lane (``wake`` = webhook-triggered turns, ``chat`` = turns Paul started)
has two knobs, one for the silence rules and one for the reformat rule. Each is
``enforce``, ``dry_run`` (decide and log, deliver the original) or ``off``. The
defaults: wake enforces both, chat is off. See ``config_from_env``.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping

log = logging.getLogger("hermes.outbound_gate")

SUPPRESS = "suppress"
DELIVER = "deliver"
REFORMAT = "reformat"

ENFORCE = "enforce"
DRY_RUN = "dry_run"
OFF = "off"
MODES = (ENFORCE, DRY_RUN, OFF)

WAKE = "wake"
CHAT = "chat"

# ---------------------------------------------------------------------------
# Silence markers
# ---------------------------------------------------------------------------
# Bracketed: matched ANYWHERE. The brackets make it a control token that
# ordinary prose does not produce, which is why it can be an override. Square
# or full-width brackets, any case, whitespace inside. The Chinese forms are
# upstream's LIVE_GATEWAY_SILENT_MARKERS translations, kept for parity.
_BRACKETED_RE = re.compile(
    r"[\[【［]\s*(?:SILENT|NO[_ ]REPLY|静默|沉默)\s*[\]】］]",
    re.IGNORECASE,
)
# Bare: only as a whole line. "Silent retry succeeded" must be delivered.
_BARE_MARKERS = frozenset({"SILENT", "NO_REPLY", "NO REPLY", "静默", "沉默"})
# What a wake turn with nothing to say SAYS when it forgets the marker. Narrow
# on purpose: whole phrases that announce the absence of news, never a word.
_NOTHING_NEW_RE = re.compile(
    r"\b(?:nothing (?:new|genuinely new|further|to report)|"
    r"nothing (?:for|to tell) (?:you|paul)|no new (?:information|news|events?)|"
    r"(?:stay|staying|remain|remaining) silent)\b",
    re.IGNORECASE,
)
# NOT in it: "no action needed". "Deploy completed, no action needed" is a
# real report, and suppressing it is the opposite failure.

# ---------------------------------------------------------------------------
# Technical-prose detection
# ---------------------------------------------------------------------------
MAX_PLAIN_CHARS = 400
MAX_PLAIN_LINES = 6
JARGON_THRESHOLD = 3
# Output bounds for a rewrite.
MAX_HEADLINE_CHARS = 140
MAX_DETAIL_CHARS = 160
MAX_FALLBACK_CHARS = 200

# Plane ids: j_20260925T081500_1a2b3c4d, oi_1a2b3c4d5e, ap_1a2b3c4d,
# ev_<hex>, wf_<hex>, or_<hex>. Prefix, underscore, then at least 6 hex.
_JOB_ID_RE = re.compile(r"\bj_\d{8}T\d{6}_[0-9a-f]{6,}\b")
_PREFIXED_ID_RE = re.compile(r"\b(?:oi|ap|ev|wf|or|fd|q|br)_[0-9a-f]{6,}\b")
# A sha: 7-40 hex with at least one digit and one letter, so a date
# (20260925) or a word (decade, facade) never counts.
_SHA_RE = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
_CODE_SPAN_RE = re.compile(r"`[^`\n]+`|```.*?```", re.DOTALL)
# A field name: lowercase words joined by underscores (target_already_delivered).
# Leading \b plus no digit start keeps ids out; ids are counted separately.
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_URL_RE = re.compile(r"https?://\S+")
# An op or module name (deploy.restart, stage_reverify.checked) and an error
# code (APPROVAL_REQUIRED). Both are words for a log, not for a reader.
_DOTTED_RE = re.compile(r"\b[a-z][a-z_]{2,}(?:\.[a-z_]{3,})+\b")
_CODE_WORD_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

_ACTIONABLE_PREFIX = "ap_"
_ASK_RE = re.compile(
    r"\b(?:approv\w*|reply with|answer|decide|decision|paste|your call|confirm)\b",
    re.IGNORECASE,
)

# Status, highest precedence first. A reply that asks for Paul outranks one
# that reports a failure, which outranks progress, which outranks done.
_STATUS = (
    ("needs_you", "⚠️", "Needs you", re.compile(
        r"\b(?:needs? (?:you|your)|approv(?:e|al)|your (?:call|decision|answer)|"
        r"waiting (?:on|for) you|please (?:confirm|decide|reply)|decision needed)\b",
        re.IGNORECASE)),
    ("failed", "❌", "Failed", re.compile(
        r"\b(?:failed|failing|failure|errored|crashed|broke(?:n)?|refused|"
        r"could not|couldn't|red)\b",
        re.IGNORECASE)),
    ("in_progress", "🔄", "In progress", re.compile(
        r"\b(?:in progress|running|dispatched|started|building|waiting for|"
        r"moving|underway|queued)\b",
        re.IGNORECASE)),
    ("done", "✅", "Done", re.compile(
        r"\b(?:merged|done|complete[d]?|landed|deployed|verified|green|"
        r"closed|shipped|fixed|passed)\b",
        re.IGNORECASE)),
)
_INFO = ("info", "ℹ️", "Update")
# "no failures", "not failed", "0 errors", "without error" negate a keyword.
_NEGATION_RE = re.compile(r"(?:\bno|\bnot|\bzero|\b0|\bwithout|\bnever)\W+(?:\w+\W+)?$",
                          re.IGNORECASE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKDOWN_RE = re.compile(r"(?m)^\s*(?:[#>]+|[-*+]|\d+[.)])\s+|\*\*|__")

DEFAULT_POINTER = "Full detail is in the orchestrator's session."


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided, and what to send.

    ``action`` is what the rule WOULD do. ``text`` is the gated text, or
    ``None`` for a suppression. ``outgoing`` is what the caller must actually
    send: ``None`` = send nothing. In ``dry_run`` it is always the original, so a
    first-day install changes nothing but the log.
    """

    action: str
    rule: str
    text: str | None
    outgoing: str | None
    mode: str
    lane: str
    event_type: str | None
    status: str | None = None
    in_chars: int = 0
    in_lines: int = 0
    jargon: int = 0
    kept_ids: int = 0
    suppressed_actionable: bool = False
    silence_overridden: bool = False

    @property
    def enforced(self) -> bool:
        return self.mode == ENFORCE

    @property
    def send(self) -> bool:
        return self.outgoing is not None

    def log_fields(self) -> dict[str, Any]:
        """Every field of the decision EXCEPT any text. Safe to log."""
        return {
            "decision": self.action,
            "rule": self.rule,
            "mode": self.mode,
            "lane": self.lane,
            "event_type": self.event_type,
            "status": self.status,
            "in_chars": self.in_chars,
            "in_lines": self.in_lines,
            "out_chars": len(self.outgoing) if self.outgoing is not None else 0,
            "jargon": self.jargon,
            "kept_ids": self.kept_ids,
            "suppressed_actionable": self.suppressed_actionable,
            "silence_overridden": self.silence_overridden,
        }


@dataclass(frozen=True)
class GateConfig:
    silence: str = ENFORCE
    reformat: str = ENFORCE


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
def _strip_edge_punct(s: str) -> str:
    start, end = 0, len(s)
    while start < end and unicodedata.category(s[start]).startswith("P") and s[start] not in "[]":
        start += 1
    while end > start and unicodedata.category(s[end - 1]).startswith("P") and s[end - 1] not in "[]":
        end -= 1
    return s[start:end].strip()


def silence_rule(text: str) -> str | None:
    """The silence rule that matches ``text``, or ``None``. Pure."""
    if not text or not text.strip():
        return "empty"
    if _BRACKETED_RE.search(text):
        return "silent_marker"
    for line in text.splitlines():
        cand = _strip_edge_punct(" ".join(line.strip().upper().split()))
        if cand in _BARE_MARKERS:
            return "silent_line"
    # The model dropped the marker but SAID there was nothing to say. Only
    # when nothing in the reply asks for Paul or reports a failure, and no id
    # in it is one he must act on - "nothing new, but the deploy failed" is
    # delivered.
    if (_NOTHING_NEW_RE.search(text) and not actionable_ids(text)
            and status_of(text)[0] not in ("needs_you", "failed")):
        return "nothing_new"
    return None


def is_actionable(text: str) -> bool:
    """True when ``text`` holds something Paul must act on. Pure."""
    return bool(actionable_ids(text)) or status_of(text)[0] == "needs_you"


def strip_silence_markers(text: str) -> str:
    """``text`` without bracketed markers or bare-marker lines. Pure."""
    kept = []
    for line in _BRACKETED_RE.sub(" ", text).splitlines():
        cand = _strip_edge_punct(" ".join(line.strip().upper().split()))
        if cand not in _BARE_MARKERS:
            kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _ids(text: str) -> list[str]:
    found = _JOB_ID_RE.findall(text) + _PREFIXED_ID_RE.findall(text)
    return list(dict.fromkeys(found))


def _jargon_tokens(text: str, keep: frozenset[str] = frozenset()) -> int:
    no_code = _CODE_SPAN_RE.sub(" ", text)
    ids = [i for i in _ids(no_code) if i not in keep]
    rest = _PREFIXED_ID_RE.sub(" ", _JOB_ID_RE.sub(" ", no_code))
    rest = _URL_RE.sub(" ", rest)
    dotted = _DOTTED_RE.findall(rest)
    rest = _DOTTED_RE.sub(" ", rest)
    return (
        len(_CODE_SPAN_RE.findall(text))
        + len(ids)
        + len(dotted)
        + len(_SHA_RE.findall(rest))
        + len(_SNAKE_RE.findall(rest))
        + len(_CODE_WORD_RE.findall(rest))
    )


def jargon_count(text: str) -> int:
    """Ids + shas + code spans + snake_case / dotted names + error codes. Pure."""
    return _jargon_tokens(text)


def is_technical(text: str) -> bool:
    stripped = text.strip()
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    return (
        len(stripped) > MAX_PLAIN_CHARS
        or len(lines) > MAX_PLAIN_LINES
        or jargon_count(stripped) >= JARGON_THRESHOLD
    )


def actionable_ids(text: str) -> list[str]:
    """Ids Paul must act on: every ``ap_`` id, and any id in a sentence that asks."""
    out: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        asks = bool(_ASK_RE.search(sentence))
        for i in _ids(sentence):
            if i.startswith(_ACTIONABLE_PREFIX) or asks:
                out.append(i)
    return list(dict.fromkeys(out))


def _unnegated(pattern: re.Pattern[str], text: str) -> bool:
    for m in pattern.finditer(text):
        if not _NEGATION_RE.search(text[max(0, m.start() - 24):m.start()]):
            return True
    return False


def status_of(text: str) -> tuple[str, str, str]:
    """(key, emoji, label). Deterministic keyword precedence, negation-aware."""
    for key, emoji, label, pattern in _STATUS:
        if _unnegated(pattern, text):
            return key, emoji, label
    return _INFO


_HOLE = "\x00"
# A word that only introduced the token just removed ("head 8c1d2e3", "as
# 3f9a1c2", "on oi_9b1abc4f3c") goes with it, or the sentence reads "head,".
_DANGLING_RE = re.compile(
    r"\b(?:at|as|on|in|of|to|head|sha|commit|job|item|row|id|ids|ref|via|with)\s+" + _HOLE,
    re.IGNORECASE,
)
# key=value, as a log line writes it (terminal=done, sha=...).
_KV_RE = re.compile(r"\b\w+=\S*")
_JARGON_PATTERNS = (_CODE_SPAN_RE, _URL_RE, _JOB_ID_RE, _PREFIXED_ID_RE, _KV_RE, _DOTTED_RE,
                    _SHA_RE, _SNAKE_RE, _CODE_WORD_RE, _BRACKETED_RE)


def _dejargon(text: str, keep: frozenset[str] = frozenset()) -> str:
    s = text
    for pat in _JARGON_PATTERNS:
        s = pat.sub(lambda m: m.group(0) if m.group(0) in keep else _HOLE, s)
    s = _MARKDOWN_RE.sub(" ", s)
    for _ in range(3):
        s = _DANGLING_RE.sub(_HOLE, s)
    s = s.replace(_HOLE, " ")
    s = re.sub(r"(?:\s*[=|/]\s*){2,}", " ", s)
    # Brackets and parentheses emptied by the removals above.
    s = re.sub(r"[(\[{][\s,;:/=]*[)\]}]", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", s)
    s = re.sub(r"[,;:]\s*([.!?])", r"\1", s)
    s = re.sub(r"\b(and|or|but)\s*([,.;:!?]|$)", r"\2", s)
    return " ".join(s.split()).strip(" ,;:-—=")


def _plain(sentence: str, keep: frozenset[str] = frozenset()) -> str:
    """One sentence with the jargon taken out, or '' if nothing readable is left."""
    s = _dejargon(sentence, keep)
    words = re.findall(r"[A-Za-z]{2,}", s)
    return s if len(words) >= 4 else ""


def _cut(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = s[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return head + "…"


def reformat(text: str, *, pointer: str = DEFAULT_POINTER) -> tuple[str, str, str, int]:
    """Rewrite technical prose into the short format. Returns (text, rule, status, kept_ids).

    Shape, at most four lines::

        ✅ Done: <the first plain sentence>
        <one more plain sentence, when there is one>
        👉 To act: ap_1a2b3c4d          (only when there is something to paste)
        <pointer>                        (fallback only)
    """
    key, emoji, label = status_of(text)
    keep = actionable_ids(text)
    kept = frozenset(keep)
    # Every readable sentence, ranked by how much jargon it carried in the
    # ORIGINAL: the ones written for a reader rather than for a log. A
    # verification wall usually ends with its one plain conclusion, so order
    # alone would headline the first bullet.
    ranked = []
    for pos, raw in enumerate(_SENTENCE_SPLIT_RE.split(text)):
        plain = _plain(raw, kept)
        if plain:
            ranked.append((_jargon_tokens(raw, kept), pos, plain))
    ranked.sort()
    lines: list[str]
    if ranked:
        rule = "technical_prose"
        lines = [f"{emoji} {label}: {_cut(ranked[0][2], MAX_HEADLINE_CHARS)}"]
        if len(ranked) > 1 and ranked[1][0] <= 1:
            lines.append(_cut(ranked[1][2], MAX_DETAIL_CHARS))
    else:
        rule = "technical_fallback"
        flat = _dejargon(text, kept)
        body = _cut(flat, MAX_FALLBACK_CHARS) if len(re.findall(r"[A-Za-z]{2,}", flat)) >= 4 \
            else "no readable summary"
        lines = [f"{emoji} {label}: {body}"]
    if keep:
        lines.append("👉 To act: " + ", ".join(keep))
    if rule == "technical_fallback":
        lines.append(pointer)
    return "\n".join(lines), rule, key, len(keep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def gate(
    text: Any,
    *,
    event_type: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> GateDecision:
    """Decide what, if anything, of ``text`` reaches the human. Pure except for one log line.

    ``context`` keys, all optional:

    - ``lane``: ``"wake"`` (default) or ``"chat"``
    - ``config``: a ``GateConfig``; default ``config_from_env(lane)``
    - ``pointer``: the fallback pointer line
    - ``chat_id``: logged (a ``webhook:<route>:<delivery_id>`` session key, not
      a person), so a decision can be matched to a delivery
    """
    ctx = dict(context or {})
    lane = ctx.get("lane") or WAKE
    cfg = ctx.get("config") if isinstance(ctx.get("config"), GateConfig) else config_from_env(lane)
    raw = text if isinstance(text, str) else ""
    stripped = raw.strip()
    in_lines = len([ln for ln in stripped.splitlines() if ln.strip()])
    base = dict(lane=lane, event_type=event_type, in_chars=len(raw), in_lines=in_lines)

    decision: GateDecision
    silent = silence_rule(raw) if cfg.silence != OFF else None
    overridden = False
    if silent is not None and silent != "empty" and is_actionable(raw):
        # Fail open: a marker never buys silence for something Paul must act on.
        overridden = True
        silent = None
        raw = stripped = strip_silence_markers(raw)
    base["silence_overridden"] = overridden
    if silent is not None:
        decision = GateDecision(
            action=SUPPRESS, rule=silent, text=None,
            outgoing=None if cfg.silence == ENFORCE else raw,
            mode=cfg.silence,
            **base,
        )
    elif cfg.reformat != OFF and is_technical(stripped):
        new, rule, status, kept = reformat(stripped, pointer=ctx.get("pointer") or DEFAULT_POINTER)
        decision = GateDecision(
            action=REFORMAT, rule=rule, text=new,
            outgoing=new if cfg.reformat == ENFORCE else raw,
            mode=cfg.reformat, status=status, jargon=jargon_count(stripped), kept_ids=kept,
            **base,
        )
    else:
        mode = cfg.reformat if cfg.silence == OFF else cfg.silence
        decision = GateDecision(
            action=DELIVER, rule="plain" if stripped else "empty_ungated", text=raw,
            outgoing=raw, mode=mode, jargon=jargon_count(stripped) if stripped else 0,
            **base,
        )

    fields = decision.log_fields()
    if ctx.get("chat_id"):
        fields["chat_id"] = str(ctx["chat_id"])
    log.info("outbound-gate: " + " ".join(f"{k}=%s" for k in fields), *fields.values())
    return decision


def gated_text(text: Any, *, event_type: str | None = None,
               context: Mapping[str, Any] | None = None) -> str | None:
    """Convenience for a send path: the text to post, or ``None`` to post nothing."""
    return gate(text, event_type=event_type, context=context).outgoing


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULTS = {WAKE: ENFORCE, CHAT: OFF}


def _mode(name: str, default: str, env: Mapping[str, str]) -> str:
    raw = (env.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw not in MODES:
        # Never fail OPEN on a typo: fall back to the lane default, loudly.
        log.warning("outbound-gate: %s=%r is not one of %s; using %s", name, raw, MODES, default)
        return default
    return raw


def config_from_env(lane: str = WAKE, env: Mapping[str, str] | None = None) -> GateConfig:
    """``OUTBOUND_GATE_<LANE>`` sets the silence rules, ``OUTBOUND_GATE_<LANE>_REFORMAT``
    the rewrite (it defaults to whatever the silence knob resolved to).

    ======================================  =========  ==========
    variable                                wake       chat
    ======================================  =========  ==========
    ``OUTBOUND_GATE_WAKE`` / ``_CHAT``      enforce    off
    ``OUTBOUND_GATE_WAKE_REFORMAT`` / ...   = above    = above
    ======================================  =========  ==========
    """
    env = os.environ if env is None else env
    lane = lane if lane in _DEFAULTS else WAKE
    prefix = f"OUTBOUND_GATE_{lane.upper()}"
    silence = _mode(prefix, _DEFAULTS[lane], env)
    return GateConfig(silence=silence, reformat=_mode(prefix + "_REFORMAT", silence, env))
