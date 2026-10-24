"""Point-in-time rules on recorded FinMind data for 2383 (台光電), with no request made.

The files under tests/data/finmind are FinMind's own answers, trimmed to the
rows these cases need: monthly revenue (with the empty and re-stamped
``create_time`` values FinMind serves), the income statement, the listing, the
news days around 2026-10-04 and TWSE's 2026 trading calendar.
"""

import copy
import json
from pathlib import Path
from unittest import mock

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
    if params.get("data_id") != "2383":
        return []
    rows = _load(f"2383_{dataset}") if (DATA / f"2383_{dataset}.json").exists() else []
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


def _latest_month(day):
    return str(_monthly_revenue("2383", day).dropna().index[-1])


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
