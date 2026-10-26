"""Point-in-time rules on recorded FinMind data for 2383 (台光電) and 2881 (富邦金), with no request made.

The files under tests/data/finmind are FinMind's own answers, trimmed to the
rows these cases need: monthly revenue (with the empty and re-stamped
``create_time`` values FinMind serves), the income statement, the listing, the
news days around 2026-10-04 and TWSE's 2026 trading calendar.
"""

import copy
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors import finmind
from tradingagents.dataflows.vendors.finmind import client
from tradingagents.dataflows.vendors.finmind.fundamentals import _monthly_revenue

DATA = Path(__file__).parent / "data" / "finmind"


def _load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


NEWS = _load("2383_TaiwanStockNews")


def _recorded(params, token):
    dataset = params["dataset"]
    if dataset == "TaiwanStockTradingDate":
        return _load("TaiwanStockTradingDate_2026") if params["start_date"].startswith("2026") else []
    if dataset == "TaiwanStockNews":
        return NEWS.get(params["start_date"], [])
    name = f"{params.get('data_id')}_{dataset}"
    rows = _load(name) if (DATA / f"{name}.json").exists() else []
    end = params.get("end_date")
    return [r for r in rows if not end or r["date"] <= end]


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("FINMIND_TOKEN", "test-token-not-real")
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    set_config(config)
    stub = mock.Mock(side_effect=_recorded)
    monkeypatch.setattr(client, "_request", stub)
    yield stub
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _latest_month(day, stock="2383"):
    return str(_monthly_revenue(stock, day).dropna().index[-1])


# --- monthly revenue ------------------------------------------------------------------

@pytest.mark.unit
def test_the_recorded_rows_are_the_ones_these_cases_are_about():
    rows = {(r["revenue_year"], r["revenue_month"]): r["create_time"] for r in _load("2383_TaiwanStockMonthRevenue")}
    assert rows[(2026, 2)] == rows[(2026, 3)] == "2026-04-21"   # re-stamped after their deadlines
    assert rows[(2026, 9)] == "2026-10-05"
    assert rows[(2025, 6)] in (None, "")                           # older rows carry none


@pytest.mark.unit
def test_february_revenue_re_stamped_in_april_is_public_by_its_march_deadline(recorded):
    assert _latest_month("2026-03-15") == "2026-02"
    assert _latest_month("2026-03-09") == "2026-01"


@pytest.mark.unit
def test_september_revenue_filed_on_october_5th_is_not_public_on_the_4th(recorded):
    assert _latest_month("2026-10-04") == "2026-08"
    assert _latest_month("2026-10-05") == "2026-09"


@pytest.mark.unit
def test_a_row_with_no_create_time_is_public_by_its_deadline(recorded):
    # June 2025 revenue: no create_time, due 2025-07-10.
    assert _latest_month("2025-07-10") == "2025-06"
    assert _latest_month("2025-07-09") == "2025-05"


# --- quarterly statements -------------------------------------------------------------

@pytest.mark.unit
def test_q2_is_not_served_before_its_august_14_deadline(recorded):
    before = finmind.get_income_statement("2383.TW", "quarterly", "2026-08-10")
    assert "2026-06-30" not in before and "2026-03-31" in before
    assert "2026-06-30" in finmind.get_income_statement("2383.TW", "quarterly", "2026-08-14")


# --- news -----------------------------------------------------------------------------

@pytest.mark.unit
def test_a_headline_past_taipei_midnight_is_not_news_on_the_4th(recorded):
    # Stamped 2026-10-04 16:22 UTC: 00:22 on 10-05 in Taipei, the day the revenue was filed.
    stamps = [a["date"] for a in NEWS["2026-10-04"] if "215.32" in a["title"]]
    assert stamps and all(s >= "2026-10-04 16:00:00" for s in stamps)
    out = finmind.get_news("2383.TW", "2026-09-27", "2026-10-04")
    assert "215.32" not in out and "9月營收" not in out
    assert out.startswith("## 2383.TW News (FinMind)")


@pytest.mark.unit
def test_the_cutoff_follows_the_configured_as_of_time(recorded):
    # Published after 13:30 in Taipei (05:30 UTC) but before midnight (16:00 UTC).
    day, article = next((d, a) for d, arts in sorted(NEWS.items()) for a in arts
                        if "05:30:00" < a["date"][11:] < "16:00:00" and a["date"][:10] == d)
    assert article["title"] in finmind.get_news("2383.TW", day, day)
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["as_of_time"] = "13:30:00"
    set_config(config)
    assert article["title"] not in finmind.get_news("2383.TW", day, day)


# --- a financial company: published after the 10th ----------------------------------------

@pytest.mark.unit
def test_the_recorded_2881_rows_are_late_filings():
    rows = {(r["revenue_year"], r["revenue_month"]): r["create_time"] for r in _load("2881_TaiwanStockMonthRevenue")}
    assert rows[(2026, 9)] == "2026-10-16"       # due 2026-10-12; 富邦金 published 10-15
    assert rows[(2026, 2)] == "2026-04-21"       # the market-wide re-entry


@pytest.mark.unit
@pytest.mark.parametrize("day, month", [
    ("2026-10-12", "2026-08"), ("2026-10-13", "2026-08"), ("2026-10-14", "2026-08"), ("2026-10-15", "2026-08"),
    ("2026-10-16", "2026-09"),
])
def test_2881_september_revenue_is_read_when_published_not_on_the_10th(recorded, day, month):
    assert _latest_month(day, "2881") == month


@pytest.mark.unit
def test_2881_february_revenue_re_entered_in_april_is_public_ten_days_after_its_deadline(recorded):
    # Due 2026-03-10; financial allowance 10 days -> 2026-03-20 (a Friday); re-stamped 04-21.
    assert _latest_month("2026-03-19", "2881") == "2026-01"
    assert _latest_month("2026-03-20", "2881") == "2026-02"


@pytest.mark.unit
def test_a_general_companys_re_entry_still_counts_on_its_deadline(recorded):
    assert _latest_month("2026-03-15") == "2026-02"      # 2383: due 03-10, re-stamped 04-21


@pytest.mark.unit
def test_the_allowance_follows_the_config(recorded):
    from tradingagents.dataflows.config import get_config, set_config
    set_config({**get_config(), "revenue_publication": {"financial_allowance_days": 0}})
    assert _latest_month("2026-10-12", "2881") == "2026-09"   # no allowance: back to the deadline


# --- filings later than expected are named -------------------------------------------------

def _rows(stock, created_by_month):
    """Revenue rows for 2026-01..09, each with the create_time given (by covered month)."""
    rows = []
    for m in range(1, 10):
        published = f"2026-{m + 1:02d}-01"
        rows.append({"date": published, "stock_id": stock, "revenue": 100 + m, "revenue_month": m,
                     "revenue_year": 2026, "create_time": created_by_month.get(m, f"2026-{m + 1:02d}-08")})
    return rows


def _serve(monkeypatch, stock, rows, industry):
    def answer(params, token):
        if params["dataset"] == "TaiwanStockTradingDate":
            return _load("TaiwanStockTradingDate_2026")
        if params["dataset"] == "TaiwanStockMonthRevenue":
            return rows
        if params["dataset"] == "TaiwanStockInfo":
            return [{"stock_id": stock, "stock_name": "x", "industry_category": industry, "type": "twse"}]
        return []
    monkeypatch.setattr(client, "_request", mock.Mock(side_effect=answer))


@pytest.mark.unit
def test_late_filings_beyond_what_is_expected_are_named_and_logged(recorded, monkeypatch, caplog):
    from tradingagents.dataflows.vendors.finmind.fundamentals import revenue_timing_notes
    # A financial company 3 days late (within 10) and one 14 days late (past it).
    _serve(monkeypatch, "2888", _rows("2888", {8: "2026-09-13", 9: "2026-10-26"}), "金融保險")
    with caplog.at_level("WARNING"):
        notes = revenue_timing_notes("2888", "2026-10-30")
    assert notes == ["REVENUE_LATE 2026-09: create_time 2026-10-26 is 14 days after its deadline 2026-10-12, "
                     "past the 10 expected; read as published on 2026-10-26"]
    assert "REVENUE_LATE 2026-09" in caplog.text


@pytest.mark.unit
def test_a_general_company_one_day_late_or_re_entered_is_not_named(recorded, monkeypatch):
    from tradingagents.dataflows.vendors.finmind.fundamentals import revenue_timing_notes
    # 1 day late (FinMind's next-morning stamp), a known re-entry date, and 3 days late.
    _serve(monkeypatch, "1402", _rows("1402", {7: "2026-08-11", 2: "2026-04-21", 9: "2026-10-15"}), "紡織纖維")
    notes = revenue_timing_notes("1402", "2026-10-30")
    assert [n.split(":")[0] for n in notes] == ["REVENUE_LATE 2026-09"]
    assert "is 3 days after its deadline 2026-10-12, past the 1 expected" in notes[0]


@pytest.mark.unit
def test_the_summary_and_the_coverage_name_a_late_filing(recorded, monkeypatch):
    from tradingagents.agents import coverage
    _serve(monkeypatch, "2888", _rows("2888", {9: "2026-10-26"}), "金融保險")
    out = finmind.get_fundamentals("2888.TW", "2026-10-30")
    assert "## Revenue timing notes\nREVENUE_LATE 2026-09:" in out
    [note] = coverage.revenue_timing_checks(out)
    assert note["status"] == "late_publication" and note["tool"] == "revenue_timing 2026-09"
    summary = coverage.coverage_summary([note, coverage.check("fundamentals", "get_fundamentals", "ok")],
                                        ["fundamentals"])
    assert summary["score"] == 1.0 and summary["gaps"] == [note]      # listed, not scored


# --- one rule: the data served and the freshness expected agree -------------------------------

def _freshness(stock, day, financial):
    from tradingagents.agents import coverage
    served = str(_monthly_revenue(stock, day).dropna().index[-1])
    out = f"Latest Monthly Revenue ({served}): 1 TWD\nLatest Statement Period: 2026-06-30"
    checks = coverage.freshness_checks(f"{stock}.TW", day, out, financial=financial)
    return served, str(coverage.expected_revenue_month(day, financial)), {c["tool"]: c["status"] for c in checks}


@pytest.mark.unit
def test_2881_on_10_14_has_august_and_is_fresh(recorded):
    served, due, status = _freshness("2881", "2026-10-14", financial=True)
    assert (served, due, status["monthly_revenue"]) == ("2026-08", "2026-08", "ok")


@pytest.mark.unit
def test_2881_on_10_16_has_september_once_published(recorded):
    # Filed 10-16: served from then; owed only from its cut-off (10-12 + 10 days).
    served, due, status = _freshness("2881", "2026-10-16", financial=True)
    assert (served, due, status["monthly_revenue"]) == ("2026-09", "2026-08", "ok")
    from tradingagents.agents import coverage
    assert str(coverage.expected_revenue_month("2026-10-22", True)) == "2026-09"


@pytest.mark.unit
def test_2383_on_10_14_owes_and_has_september(recorded):
    served, due, status = _freshness("2383", "2026-10-14", financial=False)
    assert (served, due, status["monthly_revenue"]) == ("2026-09", "2026-09", "ok")


@pytest.mark.unit
@pytest.mark.parametrize("industry, financial", [("金融保險", True), ("紡織纖維", False)])
def test_without_a_create_time_what_is_served_is_exactly_what_is_owed(recorded, monkeypatch, industry, financial):
    """The data layer and the freshness check read one rule: on every day, the latest
    month served when no row has a create_time is the latest month owed."""
    from tradingagents.agents import coverage
    from tradingagents.dataflows.vendors.finmind import calendar
    _serve(monkeypatch, "2999", [dict(r, create_time="") for r in _rows("2999", {})], industry)
    calendar.clear_cache()
    # The rows run to September revenue, so up to the day before October's is due (11-10).
    for day in pd.date_range("2026-09-01", "2026-11-09").strftime("%Y-%m-%d"):
        served = str(_monthly_revenue("2999", day).dropna().index[-1])
        assert served == str(coverage.expected_revenue_month(day, financial)), day
