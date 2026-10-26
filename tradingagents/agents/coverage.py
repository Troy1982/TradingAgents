"""What data a run actually had: every tool result classified, and a coverage score.

The analysts are told to say when data is missing, but whether they do is up to
the model. This module reads the tool results themselves: each analyst's
results are classified (``classify_tool_output``) into data checks, Taiwan
fundamentals are checked for freshness against the filing deadlines, and the
Data Coverage step after the analysts turns the checks into one score and level
that later steps can be held to.

A check is a plain dict, so it survives checkpointing:
``{"analyst", "tool", "status", "detail"}``, plus ``"scored": False`` on a
check that is listed but left out of the score.

Statuses, and what each counts for in the score:
- ``ok`` (1): data was served
- ``empty`` (1): the source was observed and holds nothing (e.g. no news that week)
- ``partial`` (0.5): served, but older than it should be on the run's date
- ``insufficient`` (0.5): the analyst reported, but from too little data to
  read (a Taiwan sentiment with too few independent stories); it stands for the
  analyst's result, so the source checks it was read from are not scored
- ``not_applicable`` (left out): the source does not cover this market
- ``late_publication`` (left out): served, but filed later than expected (a
  monthly revenue ``create_time`` past its cut-off); listed so the timing is seen
- ``withheld``, ``unavailable``, ``no_data``, ``not_offered``, ``error``,
  ``missing_report`` (0): the data was not there
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import NOT_FOR_TAIWAN, is_taiwan_symbol, taiwan_stock_id

STATUS_SCORE = {
    "ok": 1.0,
    "empty": 1.0,
    "partial": 0.5,
    "insufficient": 0.5,
    "withheld": 0.0,
    "unavailable": 0.0,
    "no_data": 0.0,
    "not_offered": 0.0,
    "error": 0.0,
    "missing_report": 0.0,
}
GAP_STATUSES = frozenset(STATUS_SCORE) - {"ok", "empty"} | {"not_applicable", "late_publication"}

# How much each analyst's data weighs in the score; an analyst not in the run is left out.
ANALYST_WEIGHTS = {
    "fundamentals": 0.30,
    "market": 0.25,
    "chips": 0.20,
    "news": 0.15,
    "social": 0.10,
}


def _first_line(text: str, limit: int = 200) -> str:
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def classify_tool_output(text) -> tuple[str, str]:
    """``(status, detail)`` for one tool result, read from the sentinels the data layer writes."""
    text = text if isinstance(text, str) else str(text or "")
    stripped = text.strip()
    if not stripped:
        return "no_data", "the tool returned nothing"
    if stripped.startswith("DATA_UNAVAILABLE:"):
        return "unavailable", _first_line(stripped)
    if stripped.startswith("NO_DATA_AVAILABLE:"):
        return "no_data", _first_line(stripped)
    if stripped.startswith("Error"):
        return "error", _first_line(stripped)
    if NOT_FOR_TAIWAN in stripped[:300]:
        return "not_applicable", _first_line(stripped)
    if "withheld for this date" in stripped:
        return "withheld", next(
            (ln.strip() for ln in stripped.splitlines() if "withheld for this date" in ln), _first_line(stripped))
    if stripped.startswith("<") and "unavailable" in stripped.split(">", 1)[0]:
        return "unavailable", _first_line(stripped)
    if stripped.startswith(("No news found", "No global news found", "No insider transactions")):
        return "empty", _first_line(stripped)
    return "ok", ""


def check(analyst: str, tool: str, status: str, detail: str = "", scored: bool = True) -> dict:
    """A data check; ``scored=False`` lists it without counting it (see ``insufficient``)."""
    entry = {"analyst": analyst, "tool": tool, "status": status, "detail": detail}
    if not scored:
        entry["scored"] = False
    return entry


# One line naming every status's score, for the coverage block wherever it is shown.
SCORING = ("Scoring: ok, empty 1; partial, insufficient 0.5; withheld, unavailable, no_data, not_offered, "
           "error, missing_report 0; not_applicable, late_publication and checks marked not scored are left out.")


# Tools that answer for a Taiwan listing without covering it: whatever they return
# says nothing about the company, so it is not counted for or against the run.
NOT_FOR_TAIWAN_TOOLS = {
    "get_insider_transactions": "Yahoo and Alpha Vantage hold no Taiwan insider (董監) filings",
}


def tool_checks(analyst: str, messages: Iterable, ticker: str = "") -> list[dict]:
    """One check per tool result in an analyst's message history."""
    taiwan = is_taiwan_symbol(ticker) if ticker else False
    checks = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        tool = getattr(message, "name", None) or "tool"
        if taiwan and tool in NOT_FOR_TAIWAN_TOOLS:
            checks.append(check(analyst, tool, "not_applicable", f"{NOT_FOR_TAIWAN}: {NOT_FOR_TAIWAN_TOOLS[tool]}"))
            continue
        status, detail = classify_tool_output(message.content)
        checks.append(check(analyst, tool, status, detail))
    return checks


# --- freshness (Taiwan fundamentals) -------------------------------------------------

_REVENUE_MONTH = re.compile(r"^Latest Monthly Revenue \((\d{4}-\d{2})\)", re.M)
_STATEMENT_PERIOD = re.compile(r"^Latest Statement Period: (\d{4}-\d{2}-\d{2})", re.M)

def expected_revenue_month(trade_date: str, financial: bool = False) -> pd.Period:
    """The latest month whose revenue had to be public by ``trade_date`` (``publication``, as the data layer uses)."""
    from tradingagents.dataflows.vendors.finmind.publication import latest_due_revenue_month

    return latest_due_revenue_month(trade_date, financial)


def expected_statement_period(trade_date: str, financial: bool) -> pd.Timestamp:
    """The latest quarter end whose statements had to be public by ``trade_date`` (``publication``)."""
    from tradingagents.dataflows.vendors.finmind.publication import latest_due_statement

    return latest_due_statement(trade_date, financial)


# Tools whose results carry a quarter's statement: the summary names its latest
# period, the statements list theirs as CSV columns.
STATEMENT_TOOLS = ("get_fundamentals", "get_income_statement", "get_balance_sheet", "get_cashflow")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


_SECTION_UNAVAILABLE = re.compile(r"^SECTION_UNAVAILABLE (\w+): (.*)$", re.M)


def section_checks(fundamentals_output: str | None) -> list[dict]:
    """An ``unavailable`` check for each section the fundamentals summary could not serve.

    The summary is served whole when one dataset fails, naming the section it
    lost (``SECTION_UNAVAILABLE monthly_revenue: ...``); each such section is a
    gap of its own, under the section's name.
    """
    return [check("fundamentals", name, "unavailable", detail)
            for name, detail in _SECTION_UNAVAILABLE.findall(fundamentals_output or "")]


_REVENUE_LATE = re.compile(r"^REVENUE_LATE (\d{4}-\d{2}): (.*)$", re.M)


def revenue_timing_checks(fundamentals_output: str | None) -> list[dict]:
    """A ``late_publication`` note for each monthly revenue filed later than expected."""
    return [check("fundamentals", f"revenue_timing {month}", "late_publication", detail)
            for month, detail in _REVENUE_LATE.findall(fundamentals_output or "")]


def latest_statement_period(output: str | None) -> pd.Timestamp | None:
    """The newest statement period a tool result shows, or None when it shows none."""
    if not output or classify_tool_output(output)[0] != "ok":
        return None
    found = _STATEMENT_PERIOD.search(output)
    if found:
        return pd.Timestamp(found.group(1))
    # A statement table: the CSV header row (the first line not a "#" comment) names the periods.
    header = next((ln for ln in output.splitlines() if ln.strip() and not ln.startswith("#")), "")
    periods = [pd.Timestamp(d) for d in _DATE.findall(header)]
    return max(periods) if periods else None


def freshness_checks(
    ticker: str, trade_date: str, fundamentals_output: str | None, financial: bool | None = None,
    statement_outputs: Iterable[str | None] = (),
) -> list[dict]:
    """Whether a Taiwan listing's monthly revenue and latest statement are as new as the deadlines require.

    Monthly revenue is read from ``get_fundamentals``; the latest statement from
    whichever result shows the newest period, the summary or any statement
    (``statement_outputs``), so a failed summary does not hide a served
    statement. ``financial`` picks the statement deadline (financial and
    insurance companies file later); None looks it up on FinMind.
    """
    if not is_taiwan_symbol(ticker) or not trade_date:
        return []
    statement_outputs = [o for o in statement_outputs if o]
    if not fundamentals_output and not statement_outputs:
        return [check("fundamentals", "freshness", "partial",
                      "no fundamentals tool was called, so monthly revenue and statement freshness "
                      "could not be checked")]
    checks = []

    if financial is None:
        from tradingagents.dataflows.vendors.finmind.publication import is_financial
        financial = is_financial(taiwan_stock_id(ticker))
    expected_month = expected_revenue_month(trade_date, financial)
    found = _REVENUE_MONTH.search(fundamentals_output or "")
    lost = {name for name, _ in _SECTION_UNAVAILABLE.findall(fundamentals_output or "")}
    if "monthly_revenue" in lost:
        pass  # section_checks reports it as unavailable, which says more than "behind"
    elif found is None:
        served = "get_fundamentals was not called" if not fundamentals_output else "no monthly revenue served"
        checks.append(check("fundamentals", "monthly_revenue", "partial",
                            f"{served}; {expected_month} was due by {trade_date}"))
    elif pd.Period(found.group(1), "M") < expected_month:
        checks.append(check("fundamentals", "monthly_revenue", "partial",
                            f"latest monthly revenue is {found.group(1)}; {expected_month} was due by {trade_date}"))
    else:
        checks.append(check("fundamentals", "monthly_revenue", "ok"))

    expected_period = expected_statement_period(trade_date, financial)
    periods = [p for p in map(latest_statement_period, [fundamentals_output, *statement_outputs]) if p is not None]
    latest = max(periods) if periods else None
    if latest is None:
        checks.append(check("fundamentals", "quarterly_statement", "partial",
                            f"no statement period served; the quarter ending {expected_period:%Y-%m-%d} "
                            f"was due by {trade_date}"))
    elif latest < expected_period:
        checks.append(check("fundamentals", "quarterly_statement", "partial",
                            f"latest statement is for {latest:%Y-%m-%d}; the quarter ending "
                            f"{expected_period:%Y-%m-%d} was due by {trade_date}"))
    else:
        checks.append(check("fundamentals", "quarterly_statement", "ok"))
    return checks


def latest_tool_output(messages: Iterable, tool: str) -> str | None:
    """The last result ``tool`` returned in a message history, or None if it was never called."""
    outputs = [m.content for m in messages if getattr(m, "type", None) == "tool" and getattr(m, "name", None) == tool]
    return str(outputs[-1]) if outputs else None


# --- the run's coverage ----------------------------------------------------------------

# Checks a decision cannot stand without: prices and the company's own figures.
# One that is not fully served caps the level at medium, however the rest score,
# since many served checks would otherwise outweigh it.
CRITICAL_CHECKS = (
    "get_stock_data", "get_verified_market_snapshot", "get_fundamentals",
    "monthly_revenue", "quarterly_statement",
)


def _settings() -> tuple[float, float, tuple[str, ...]]:
    settings = get_config().get("data_coverage") or {}
    return (float(settings.get("high", 0.8)), float(settings.get("medium", 0.5)),
            tuple(settings.get("critical", CRITICAL_CHECKS)))


def coverage_summary(checks: Iterable[Mapping], analysts: Iterable[str]) -> dict:
    """Score (0–1), level and gaps over the run's checks, weighted by analyst.

    An analyst's score is the mean of its checks' scores, sources that do not
    apply left out; an analyst with no scored check counts in full. Weights are
    renormalized over the analysts in the run. A critical check (``CRITICAL_CHECKS``,
    config ``data_coverage.critical``) scoring below 1 caps the level at medium;
    ``capped_by`` names those checks.
    """
    checks = list(checks)
    analysts = [a for a in analysts if a in ANALYST_WEIGHTS]
    total_weight = sum(ANALYST_WEIGHTS[a] for a in analysts)
    by_analyst = {}
    for analyst in analysts:
        scores = [STATUS_SCORE[c["status"]] for c in checks
                  if c["analyst"] == analyst and c["status"] in STATUS_SCORE and c.get("scored", True)]
        by_analyst[analyst] = sum(scores) / len(scores) if scores else 1.0
    score = (sum(ANALYST_WEIGHTS[a] * s for a, s in by_analyst.items()) / total_weight) if total_weight else 1.0
    high, medium, critical = _settings()
    level = "high" if score >= high else "medium" if score >= medium else "low"
    capped_by = sorted({c["tool"] for c in checks if c["analyst"] in by_analyst and c["tool"] in critical
                        and c.get("scored", True) and STATUS_SCORE.get(c["status"], 1.0) < 1.0})
    if capped_by and level == "high":
        level = "medium"
    return {
        "score": round(score, 2),
        "level": level,
        "capped_by": capped_by,
        "by_analyst": {a: round(s, 2) for a, s in by_analyst.items()},
        "gaps": [dict(c) for c in checks if c["status"] in GAP_STATUSES],
    }


def render_coverage_report(summary: Mapping) -> str:
    """The summary as markdown for the saved report, in the run's output language (``i18n``).

    The prompts read ``render_coverage``, in English with each check's own detail.
    """
    from tradingagents.agents import i18n

    lines = [
        i18n.phrase("coverage_title", level=i18n.level(summary["level"]), score=f"{summary['score']:.2f}"),
        i18n.phrase("per_analyst", items=("、" if i18n.report_language() == i18n.ZH else ", ").join(
            f"{i18n.analyst(a)} {s:.2f}" for a, s in summary["by_analyst"].items())),
    ]
    if summary.get("capped_by"):
        lines.append(i18n.phrase("capped", tools=("、" if i18n.report_language() == i18n.ZH else ", ").join(
            i18n.tool(t) for t in summary["capped_by"])))
    lines.append(i18n.phrase("scoring"))
    gaps = summary["gaps"]
    if gaps:
        lines += ["", i18n.phrase("table_head"), "|---|---|---|---|"]
        for gap in gaps:
            detail = i18n.detail(gap.get("detail", "")).replace("|", "\\|")
            lines.append(f"| {i18n.analyst(gap['analyst'])} | {i18n.tool(gap['tool'])} | "
                         f"{i18n.status(gap['status'], gap.get('scored', True))} | {detail} |")
    else:
        lines.append(i18n.phrase("no_gaps"))
    return "\n".join(lines)


def render_coverage(summary: Mapping) -> str:
    """The summary as markdown, in English with each check's detail, for the prompts that read it."""
    lines = [
        f"**Data Coverage**: {summary['level']} (score {summary['score']:.2f})",
        "Per analyst: " + ", ".join(f"{a} {s:.2f}" for a, s in summary["by_analyst"].items()),
    ]
    if summary.get("capped_by"):
        lines.append("Capped at medium: critical data not fully served (" + ", ".join(summary["capped_by"]) + ")")
    lines.append(SCORING)
    gaps = summary["gaps"]
    if gaps:
        lines += ["", "| Analyst | Tool | Status | Detail |", "|---|---|---|---|"]
        for gap in gaps:
            detail = str(gap.get("detail", "")).replace("|", "\\|")
            status = gap["status"] if gap.get("scored", True) else f"{gap['status']} (not scored)"
            lines.append(f"| {gap['analyst']} | {gap['tool']} | {status} | {detail} |")
    else:
        lines.append("No data gaps.")
    return "\n".join(lines)
