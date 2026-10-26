"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision; its free-text fallback is read here)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.

``extract_rating`` returns ``None`` when no rating can be found, and every
caller turns that into ``REVIEW`` rather than a tradeable position: a decision
nobody can read is not a Hold, and a Hold recorded in its place is quoted back to
the next run as a call that was never made (#1170).
"""

from __future__ import annotations

import re
import unicodedata

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

# Signal emitted when the model's decision has no recognizable rating. It is not
# a tradeable position: it flags output that needs a human/re-run rather than
# silently degrading to Hold. Callers that map the signal onto the 5-tier enum
# (e.g. ``PortfolioRating(signal)``) should guard with ``is_review`` first.
RATING_REVIEW = "REVIEW"

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating — **X**" — tolerates markdown
# bold wrappers and any dash or colon a model writes as the separator. "rating"
# must start a word, so "Operating margin: Sell-side" is not a label.
_RATING_LABEL_RE = re.compile(r"(?<![a-z])rating\b[^:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(\w+)",
                              re.IGNORECASE)

# The decision's own rating line, in the shape the Portfolio Manager is asked to
# open with ("- **Rating**: X", "**Final Rating**: X", "## Our rating - X"): an
# optional list marker, emphasis and heading marks, and only words naming the
# decision itself before "rating". "Consensus rating: Buy" or "Trader's rating:
# Buy" is someone else's rating.
_OWN_QUALIFIER = r"(?:(?:final|our|overall|my|recommended|updated|revised|new|current)\s+)*"
_RATING_LINE_RE = re.compile(
    r"\s*(?P<item>(?:[-+*\u2022]|\d+[.)])\s+)?[\s*_#]*" + _OWN_QUALIFIER
    + r"rating[^\w:\-\u2010-\u2015]*[:\-\u2010-\u2015][\s*]*(?P<value>\w+)",
    re.IGNORECASE,
)

# A line presenting the scale rather than a decision ("Rating Scale: Buy, ...").
_RATING_SCALE_RE = re.compile(r"rating\s*(scale|options|legend)", re.IGNORECASE)

def extract_rating(text: str) -> str | None:
    """Extract a 5-tier rating from its label, or ``None`` if there is none.

    Reads an explicit "Rating: X" label (tolerant of markdown bold) in the
    NFKC-normalized text, so fullwidth punctuation like ``Rating：Overweight``
    matches as ASCII does: the decision's opening rating line, else its own
    rating lines or, failing those, every label, when they agree.
    """
    if not text:
        return None
    norm = unicodedata.normalize("NFKC", text)

    # The decision is asked to open with its rating, so a first line (after any
    # headings) in that shape is the call. A list item there may also be a quote
    # heading a list of other parties' ratings, so it counts with the decision's
    # other rating lines, which leave out list items; failing those, every label
    # counts. Either way they must agree: ratings that differ, with nothing
    # marking which one is the call, are no call (#1170). Lines presenting the
    # scale itself are a legend the model echoed, not a call.
    lines = [line for line in norm.splitlines() if line.strip() and not _RATING_SCALE_RE.search(line)]
    first = next((line for line in lines if not line.lstrip().startswith("#")
                  or _RATING_LINE_RE.match(line)), "")
    m = _RATING_LINE_RE.match(first)
    opening = m if m and m.group("value").lower() in _RATING_SET else None
    if opening and not opening.group("item"):
        return opening.group("value").capitalize()

    own = [opening.group("value").capitalize()] if opening else []
    labels = []
    for line in lines:
        m = _RATING_LINE_RE.match(line)
        if m and not m.group("item") and m.group("value").lower() in _RATING_SET:
            own.append(m.group("value").capitalize())
        labels += [v.capitalize() for v in _RATING_LABEL_RE.findall(line) if v.lower() in _RATING_SET]
    # Without a label there is no call to read: a rating word in the prose may be
    # one the text argues against ("not a Sell"), and reading it reports a
    # direction nobody decided.
    found = own or labels
    return found[0] if found and len(set(found)) == 1 else None


def parse_rating(text: str, default: str = RATING_REVIEW) -> str:
    """Extract a 5-tier rating, or ``REVIEW`` when the decision has none.

    For callers that need a string for every decision, such as the memory log's
    entry tag. The default is the review sentinel, never a tradeable rating.
    """
    rating = extract_rating(text)
    return rating if rating is not None else default


def run_rating(final_state: dict) -> str:
    """A finished run's rating: the Portfolio Manager's own, else read from its decision.

    The fallback serves a state without ``final_rating``, such as a run an older
    version completed and a checkpoint hands back unchanged.
    """
    return final_state.get("final_rating") or parse_rating(final_state.get("final_trade_decision", ""))


def is_review(signal: str) -> bool:
    """Whether a signal is the non-tradeable REVIEW sentinel (#1170)."""
    return signal == RATING_REVIEW


# --- confidence, and the cap data coverage puts on a decision ----------------------------

CONFIDENCE_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# Written when a decision states no confidence the program can read (the free-text
# path): unknown, not low, so it is never mistaken for a coverage cap.
CONFIDENCE_UNPARSED = "unparsed"

_CONFIDENCE_RE = re.compile(r"confidence[\s*_]*[:：][\s*_]*(low|medium|high)\b", re.IGNORECASE)


_RECOMMENDATION_RE = re.compile(r"recommendation[\s*_]*[:\uff1a][\s*_]*(\w+)", re.IGNORECASE)


def parse_recommendation(text: str) -> str:
    """The research plan's ``Recommendation: X`` (a 5-tier rating), or ``REVIEW``."""
    m = _RECOMMENDATION_RE.search(unicodedata.normalize("NFKC", text or ""))
    value = m.group(1).capitalize() if m else ""
    return value if value.lower() in _RATING_SET else RATING_REVIEW


def parse_confidence(text: str) -> str:
    """The decision's ``Confidence: low|medium|high`` line, or ``unparsed``."""
    m = _CONFIDENCE_RE.search(unicodedata.normalize("NFKC", text or ""))
    return m.group(1).lower() if m else CONFIDENCE_UNPARSED


def confidence_ceiling(coverage: dict | None, settings: dict | None = None) -> tuple[str, bool, list[str]]:
    """``(highest confidence allowed, whether the rating is capped, why)`` for a run's coverage.

    The rating is capped when coverage is low or a critical check was not fully
    served (``capped_by``); confidence is then at most low. Medium coverage with
    nothing critical missing allows at most medium and leaves the rating alone.
    No coverage (a run without the Data Coverage step) caps nothing.
    """
    settings = settings or {}
    if not coverage:
        return "high", False, []
    reasons = []
    if coverage.get("level") == "low" and settings.get("cap_on_low", True):
        reasons.append(f"data coverage is low (score {coverage.get('score')})")
    if coverage.get("capped_by") and settings.get("cap_on_critical", True):
        reasons.append("critical data not fully served: " + ", ".join(coverage["capped_by"]))
    if reasons:
        return "low", True, reasons
    if coverage.get("level") == "medium":
        return "medium", False, [f"data coverage is medium (score {coverage.get('score')})"]
    return "high", False, []


def apply_coverage_cap(rating: str, confidence: str, coverage: dict | None, settings: dict | None = None) -> dict:
    """What the coverage cap makes of a decision's rating and confidence.

    Rules (config ``data_coverage``): coverage low or a critical check missing
    moves Buy to Overweight and Sell to Underweight (``rating_cap``) and holds
    confidence at low; medium coverage holds confidence at medium. An
    ``unparsed`` confidence stays unparsed: it is unknown, and the cap is not
    written over it. ``enforce`` (default false) only records the change as a
    shadow; true applies it. Returns the model's and the capped values, whether
    the cap changed anything, whether it was applied, the reasons, and a note
    for the decision text.
    """
    settings = settings or {}
    ceiling, cap_rating, reasons = confidence_ceiling(coverage, settings)
    mapping = settings.get("rating_cap") or {"Buy": "Overweight", "Sell": "Underweight"}
    capped_rating = mapping.get(rating, rating) if cap_rating else rating
    capped_confidence = confidence
    if confidence in CONFIDENCE_LEVELS and CONFIDENCE_LEVELS.index(confidence) > CONFIDENCE_LEVELS.index(ceiling):
        capped_confidence = ceiling
    changed = (capped_rating, capped_confidence) != (rating, confidence)
    enforced = bool(settings.get("enforce", False))
    result = {
        "model_rating": rating, "model_confidence": confidence,
        "capped_rating": capped_rating, "capped_confidence": capped_confidence,
        "ceiling": ceiling, "rating_capped": cap_rating, "changed": changed,
        "enforced": enforced, "reasons": reasons,
    }
    result["rating"] = capped_rating if enforced else rating
    result["confidence"] = capped_confidence if enforced else confidence
    result["note"] = _cap_note(result)
    return result


def cap_moves(cap: dict) -> str:
    """What the cap changes, in words ("the call from Sell to Underweight and confidence
    from high to low"), or an empty string when it changes nothing."""
    moves = []
    if cap["capped_rating"] != cap["model_rating"]:
        moves.append(f"the call from {cap['model_rating']} to {cap['capped_rating']}")
    if cap["capped_confidence"] != cap["model_confidence"]:
        moves.append(f"confidence from {cap['model_confidence']} to {cap['capped_confidence']}")
    return " and ".join(moves)


def _cap_note(cap: dict) -> str:
    """One line for the decision text. It never writes "rating" before a colon or dash,
    so the rating parser cannot read it as the decision's rating."""
    why = "; ".join(cap["reasons"]) or "data coverage allows any call"
    change = cap_moves(cap)
    if not change:
        return f"**Coverage Cap**: no change ({why}; confidence ceiling {cap['ceiling']})"
    if cap["enforced"]:
        return f"**Coverage Cap**: applied, moving {change} ({why})"
    return f"**Coverage Cap**: not applied (shadow mode); if applied it would move {change} ({why})"
