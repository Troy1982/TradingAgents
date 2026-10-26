"""Data coverage: tool results classified, Taiwan freshness checked, and the run scored."""

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tradingagents.agents import coverage
from tradingagents.agents.analysts import news_analyst
from tradingagents.dataflows.symbols import NOT_FOR_TAIWAN

# --- classification -------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("text, status", [
    ("# Stock data for 2383\nDate,Open\n2026-10-16,5165", "ok"),
    ("DATA_UNAVAILABLE: no configured vendor could serve get_news right now (x).", "unavailable"),
    ("DATA_UNAVAILABLE: optional macro_data could not be retrieved (no key).", "unavailable"),
    ("NO_DATA_AVAILABLE: No usable market data for 'ZZZZ' from any configured vendor.", "no_data"),
    ("<Yahoo Finance global news unavailable for 2026-09-27..2026-10-04: it only serves recent items>",
     "unavailable"),
    ("# Insider Transactions for 2383.TW\n# Point-in-time as of: 2026-10-04\n\n"
     "Insider transactions are withheld for this date. This vendor dates a trade by when it happened.",
     "withheld"),
    (f"<StockTwits: {NOT_FOR_TAIWAN} (not applicable to Taiwan-listed stocks).>", "not_applicable"),
    ("No news found for 2330.TW between 2026-09-01 and 2026-09-02", "empty"),
    ("Error: ValueError('start_date is not a date')\n Please fix your mistakes.", "error"),
    ("", "no_data"),
    (None, "no_data"),
])
def test_tool_outputs_are_classified_by_their_sentinels(text, status):
    assert coverage.classify_tool_output(text)[0] == status


@pytest.mark.unit
def test_a_sentinel_quoted_deep_in_data_is_not_read_as_a_gap():
    article = "## 2383.TW News (FinMind)\n\n" + "x\n" * 200 + f"a headline mentioning {NOT_FOR_TAIWAN}"
    assert coverage.classify_tool_output(article)[0] == "ok"


@pytest.mark.unit
def test_only_tool_results_become_checks():
    messages = [
        HumanMessage("2383.TW"),
        AIMessage("", tool_calls=[{"name": "get_news", "id": "1", "args": {}}]),
        ToolMessage("DATA_UNAVAILABLE: down.", name="get_news", tool_call_id="1"),
        ToolMessage("# data", name="get_stock_data", tool_call_id="2"),
        AIMessage("the report"),
    ]
    assert coverage.tool_checks("news", messages) == [
        {"analyst": "news", "tool": "get_news", "status": "unavailable", "detail": "DATA_UNAVAILABLE: down."},
        {"analyst": "news", "tool": "get_stock_data", "status": "ok", "detail": ""},
    ]


# --- freshness ------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("day, month", [
    # 2026-10-10 is a Saturday: September revenue is due on Monday the 12th.
    ("2026-10-09", "2026-08"), ("2026-10-10", "2026-08"), ("2026-10-12", "2026-09"),
    ("2026-10-31", "2026-09"), ("2026-09-10", "2026-08"),
    ("2026-01-05", "2025-11"), ("2026-01-12", "2025-12"),
])
def test_the_revenue_month_due_follows_the_10th(day, month):
    assert str(coverage.expected_revenue_month(day)) == month


@pytest.mark.unit
@pytest.mark.parametrize("day, financial, period", [
    ("2026-08-13", False, "2026-03-31"), ("2026-08-14", False, "2026-06-30"),
    ("2026-08-30", True, "2026-03-31"), ("2026-08-31", True, "2026-06-30"),
    ("2026-05-14", False, "2025-12-31"), ("2026-05-15", False, "2026-03-31"),
    ("2026-03-30", False, "2025-09-30"), ("2026-03-31", False, "2025-12-31"),
    # 2026-11-14 is a Saturday: Q3 is due on Monday the 16th.
    ("2026-10-17", False, "2026-06-30"), ("2026-11-14", False, "2026-06-30"),
    ("2026-11-16", False, "2026-09-30"),
])
def test_the_statement_due_follows_the_filing_deadlines(day, financial, period):
    assert coverage.expected_statement_period(day, financial) == pd.Timestamp(period)


def _fundamentals(revenue_month="2026-09", period="2026-06-30"):
    lines = ["# Company Fundamentals for 2383.TW", ""]
    if revenue_month:
        lines.append(f"Latest Monthly Revenue ({revenue_month}): 21,532,230,000 TWD (215.32 億)")
    if period:
        lines.append(f"Latest Statement Period: {period}")
    return "\n".join(lines)


def _statuses(checks):
    return {c["tool"]: c["status"] for c in checks}


@pytest.mark.unit
def test_current_fundamentals_pass_the_freshness_checks():
    checks = coverage.freshness_checks("2383.TW", "2026-10-17", _fundamentals(), financial=False)
    assert _statuses(checks) == {"monthly_revenue": "ok", "quarterly_statement": "ok"}


@pytest.mark.unit
def test_stale_revenue_and_a_missing_quarter_are_partial():
    checks = coverage.freshness_checks(
        "2383.TW", "2026-11-20", _fundamentals("2026-09", "2026-06-30"), financial=False)
    assert _statuses(checks) == {"monthly_revenue": "partial", "quarterly_statement": "partial"}
    details = " ".join(c["detail"] for c in checks)
    assert "2026-10 was due by 2026-11-20" in details
    assert "quarter ending 2026-09-30 was due by 2026-11-20" in details


@pytest.mark.unit
def test_a_financial_company_is_not_late_before_its_later_deadline():
    out = _fundamentals("2026-07", "2026-03-31")
    assert _statuses(coverage.freshness_checks("2881.TW", "2026-08-20", out, financial=True)) == {
        "monthly_revenue": "ok", "quarterly_statement": "ok"}
    assert _statuses(coverage.freshness_checks("2330.TW", "2026-08-20", out, financial=False))[
        "quarterly_statement"] == "partial"


@pytest.mark.unit
def test_fundamentals_without_revenue_or_a_call_are_partial():
    assert _statuses(coverage.freshness_checks(
        "2383.TW", "2026-10-17", _fundamentals(revenue_month=None), financial=False))["monthly_revenue"] == "partial"
    assert _statuses(coverage.freshness_checks("2383.TW", "2026-10-17", None)) == {"freshness": "partial"}


@pytest.mark.unit
def test_freshness_is_checked_for_taiwan_listings_only():
    assert coverage.freshness_checks("NVDA", "2026-10-17", None) == []


# --- the run's score ------------------------------------------------------------------


def _check(analyst, status, tool="t"):
    return coverage.check(analyst, tool, status)


@pytest.mark.unit
def test_a_run_with_every_source_served_is_high():
    checks = [_check(a, "ok") for a in ("market", "news", "fundamentals")]
    summary = coverage.coverage_summary(checks, ["market", "news", "fundamentals"])
    assert summary["score"] == 1.0 and summary["level"] == "high" and summary["gaps"] == []


@pytest.mark.unit
def test_sources_that_do_not_apply_cost_nothing_but_are_listed():
    checks = [_check("social", "ok", "news"), _check("social", "not_applicable", "stocktwits"),
              _check("social", "not_applicable", "reddit")]
    summary = coverage.coverage_summary(checks, ["social"])
    assert summary["score"] == 1.0
    assert [g["tool"] for g in summary["gaps"]] == ["stocktwits", "reddit"]


@pytest.mark.unit
def test_the_score_is_weighted_by_analyst():
    # Fundamentals (0.30) lost entirely, market (0.25) served: 0.25 / 0.55.
    checks = [_check("fundamentals", "unavailable"), _check("market", "ok")]
    summary = coverage.coverage_summary(checks, ["fundamentals", "market"])
    assert summary["score"] == 0.45 and summary["level"] == "low"
    assert summary["by_analyst"] == {"fundamentals": 0.0, "market": 1.0}


@pytest.mark.unit
def test_partial_counts_half_and_the_levels_follow_the_config(monkeypatch):
    checks = [_check("market", "ok"), _check("market", "partial")]
    assert coverage.coverage_summary(checks, ["market"])["score"] == 0.75
    assert coverage.coverage_summary(checks, ["market"])["level"] == "medium"
    monkeypatch.setattr(coverage, "get_config", lambda: {"data_coverage": {"high": 0.7, "medium": 0.4}})
    assert coverage.coverage_summary(checks, ["market"])["level"] == "high"


@pytest.mark.unit
def test_an_analyst_outside_the_run_is_not_scored():
    summary = coverage.coverage_summary([_check("chips", "unavailable")], ["market"])
    assert summary["score"] == 1.0 and "chips" not in summary["by_analyst"]


@pytest.mark.unit
def test_the_summary_renders_its_gaps_as_a_table():
    summary = coverage.coverage_summary([coverage.check("news", "get_news", "unavailable", "a | b")], ["news"])
    text = coverage.render_coverage(summary)
    assert text.startswith("**Data Coverage**: low (score 0.00)")
    assert "| news | get_news | unavailable | a \\| b |" in text


# --- what the news analyst leaves out -------------------------------------------------


@pytest.mark.unit
def test_the_news_analyst_names_the_tools_it_withholds(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    assert _statuses(news_analyst.withheld_checks("2383.TW")) == {
        "get_macro_indicators": "not_offered", "get_prediction_markets": "not_applicable"}
    # With FinMind, Taiwan's indicators are offered and only FRED's half is missing.
    monkeypatch.setenv("FINMIND_TOKEN", "t")
    assert _statuses(news_analyst.withheld_checks("2383.TW")) == {
        "fred_macro": "not_offered", "get_prediction_markets": "not_applicable"}
    assert _statuses(news_analyst.withheld_checks("NVDA")) == {"get_macro_indicators": "not_offered"}
    monkeypatch.setenv("FRED_API_KEY", "k")
    assert news_analyst.withheld_checks("NVDA") == []


@pytest.mark.unit
def test_insider_trades_for_a_taiwan_listing_do_not_apply():
    messages = [ToolMessage("# Insider Transactions ... withheld for this date.", name="get_insider_transactions",
                            tool_call_id="1")]
    assert coverage.tool_checks("fundamentals", messages, "2383.TW")[0]["status"] == "not_applicable"
    assert coverage.tool_checks("fundamentals", messages, "NVDA")[0]["status"] == "withheld"


@pytest.mark.unit
def test_a_past_runs_missing_global_news_is_a_gap():
    out = ("<Yahoo Finance global news unavailable for 2026-09-27..2026-10-04: it only serves recent "
           "items, so this is not an absence of market news>")
    assert coverage.classify_tool_output(out)[0] == "unavailable"
    assert coverage.classify_tool_output("No global news found between 2026-10-10 and 2026-10-17")[0] == "empty"


@pytest.mark.unit
@pytest.mark.parametrize("day, latest, status", [
    ("2026-10-17", "2026-09", "ok"),        # September was due 10-12 (10-10 is a Saturday)
    ("2026-10-04", "2026-08", "ok"),        # September not due yet: August is the latest owed
    ("2026-10-17", "2026-08", "partial"),   # September owed and missing
])
def test_monthly_revenue_freshness_on_the_acceptance_dates(day, latest, status):
    checks = coverage.freshness_checks("2383.TW", day, _fundamentals(latest, "2026-06-30"), financial=False)
    assert _statuses(checks)["monthly_revenue"] == status


@pytest.mark.unit
def test_a_failed_revenue_tool_lowers_the_level_and_is_listed():
    served = coverage.freshness_checks("2383.TW", "2026-10-17", _fundamentals(), financial=False)
    failed = [coverage.check("fundamentals", "get_fundamentals", "unavailable", "DATA_UNAVAILABLE: down")]
    failed += coverage.freshness_checks("2383.TW", "2026-10-17", "DATA_UNAVAILABLE: down", financial=False)
    ok = coverage.coverage_summary([*served, coverage.check("fundamentals", "get_fundamentals", "ok")],
                                   ["fundamentals"])
    down = coverage.coverage_summary(failed, ["fundamentals"])
    assert down["score"] < ok["score"]
    assert ok["level"] == "high" and down["level"] != "high"
    assert {(g["tool"], g["status"]) for g in down["gaps"]} >= {
        ("get_fundamentals", "unavailable"), ("monthly_revenue", "partial")}


@pytest.mark.unit
def test_a_missing_critical_check_caps_a_high_score_at_medium():
    # Many served checks outweigh a lost revenue series in the score, not in the level.
    checks = [_check("fundamentals", "ok", f"t{i}") for i in range(9)]
    checks.append(coverage.check("fundamentals", "monthly_revenue", "partial"))
    summary = coverage.coverage_summary(checks, ["fundamentals"])
    assert summary["score"] == 0.95
    assert summary["level"] == "medium" and summary["capped_by"] == ["monthly_revenue"]
    assert "Capped at medium" in coverage.render_coverage(summary)


@pytest.mark.unit
def test_the_critical_checks_follow_the_config(monkeypatch):
    checks = [_check("market", "ok", "a"), coverage.check("market", "get_stock_data", "partial")]
    monkeypatch.setattr(coverage, "get_config", lambda: {"data_coverage": {"high": 0.7, "critical": []}})
    assert coverage.coverage_summary(checks, ["market"])["level"] == "high"


_INCOME_STATEMENT = ("# Income Statement data for 2383.TW (quarterly)\n# Point-in-time as of: 2026-10-17\n"
                     "# Currency: TWD\n,2026-06-30,2026-03-31,2025-12-31\nRevenue,47275190000.0,33067000000.0,1\n")


@pytest.mark.unit
def test_a_served_statement_is_fresh_when_the_summary_failed():
    checks = coverage.freshness_checks(
        "2383.TW", "2026-10-17", "DATA_UNAVAILABLE: no configured vendor could serve get_fundamentals (down).",
        financial=False, statement_outputs=[_INCOME_STATEMENT])
    assert _statuses(checks) == {"monthly_revenue": "partial", "quarterly_statement": "ok"}


@pytest.mark.unit
def test_the_newest_period_any_statement_shows_decides():
    stale = _INCOME_STATEMENT.replace(",2026-06-30,", ",2026-03-31,", 1)
    checks = coverage.freshness_checks("2383.TW", "2026-10-17", _fundamentals(period="2026-03-31"),
                                       financial=False, statement_outputs=[stale, _INCOME_STATEMENT])
    assert _statuses(checks)["quarterly_statement"] == "ok"
    assert coverage.latest_statement_period("# x\n# withheld for this date") is None


@pytest.mark.unit
def test_statements_alone_still_check_freshness():
    checks = coverage.freshness_checks("2383.TW", "2026-10-17", None, financial=False,
                                       statement_outputs=[_INCOME_STATEMENT])
    assert _statuses(checks) == {"monthly_revenue": "partial", "quarterly_statement": "ok"}


@pytest.mark.unit
def test_a_lost_revenue_section_is_an_unavailable_critical_gap():
    out = (_fundamentals(revenue_month=None) + "\n\n## Unavailable sections\n"
           "SECTION_UNAVAILABLE monthly_revenue: FinMind could not serve TaiwanStockMonthRevenue: 503. "
           "Report it as unavailable; do not estimate it.")
    checks = coverage.section_checks(out) + coverage.freshness_checks("2383.TW", "2026-10-17", out, financial=False)
    assert _statuses(checks) == {"monthly_revenue": "unavailable", "quarterly_statement": "ok"}
    summary = coverage.coverage_summary([*checks, coverage.check("fundamentals", "get_fundamentals", "ok")],
                                        ["fundamentals"])
    assert summary["capped_by"] == ["monthly_revenue"]


@pytest.mark.unit
def test_an_insufficient_read_scores_half_and_stands_for_its_analyst():
    checks = [coverage.check("social", "news", "ok", scored=False),
              coverage.check("social", "stocktwits", "not_applicable"),
              coverage.check("social", "sentiment", "insufficient", "Insufficient Data: 1 headline")]
    summary = coverage.coverage_summary(checks, ["social"])
    assert summary["by_analyst"] == {"social": 0.5}
    assert {(g["tool"], g["status"]) for g in summary["gaps"]} == {
        ("stocktwits", "not_applicable"), ("sentiment", "insufficient")}


@pytest.mark.unit
def test_insufficient_and_no_data_are_scored_apart():
    assert coverage.STATUS_SCORE["insufficient"] == 0.5 and coverage.STATUS_SCORE["no_data"] == 0.0
    summary = coverage.coverage_summary([coverage.check("social", "sentiment", "insufficient")], ["social"])
    assert summary["score"] == 0.5


@pytest.mark.unit
def test_the_coverage_block_names_every_status_and_marks_unscored_checks():
    summary = coverage.coverage_summary(
        [coverage.check("social", "news", "unavailable", "down", scored=False),
         coverage.check("social", "sentiment", "insufficient")], ["social"])
    text = coverage.render_coverage(summary)
    assert "partial, insufficient 0.5" in text and "no_data" in text and "not_applicable" in text
    assert "| social | news | unavailable (not scored) | down |" in text
    assert "| social | sentiment | insufficient |  |" in text
