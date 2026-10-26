"""FinMind vendor (Taiwan): symbol handling, routing fallback, output shape,
statement point-in-time rules, news windowing and the request cache.

HTTP is stubbed at ``client._request``, so these run offline and without a token.
"""

import copy
from unittest import mock

import pandas as pd
import pytest
import requests

import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.vendors import finmind
from tradingagents.dataflows.vendors.finmind import client

TOKEN = "test-token-not-real"

# Prices end on the latest weekday, so the staleness check passes on any day the suite runs.
_DATES = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=300)
END = _DATES[-1].strftime("%Y-%m-%d")
START5 = _DATES[-5].strftime("%Y-%m-%d")
_D = lambda i: _DATES[i].strftime("%Y-%m-%d")  # noqa: E731
_PRICES = [
    {"date": d.strftime("%Y-%m-%d"), "stock_id": "2330", "Trading_Volume": 1000 + i,
     "Trading_money": 0, "open": 100.0 + i, "max": 101.0 + i, "min": 99.0 + i,
     "close": 100.5 + i, "spread": 0.0, "Trading_turnover": 0}
    for i, d in enumerate(_DATES)
]


def _statement(rows):
    return [{"date": d, "stock_id": "2330", "type": t, "value": v, "origin_name": t} for d, t, v in rows]


_INCOME = _statement(
    [(f"2025-{m}", "Revenue", 100.0 * q) for q, m in enumerate(["03-31", "06-30", "09-30", "12-31"], 1)]
    + [(f"2025-{m}", "EPS", 1.0) for m in ["03-31", "06-30", "09-30", "12-31"]]
    + [(f"2025-{m}", "IncomeAfterTaxes", 10.0) for m in ["03-31", "06-30", "09-30", "12-31"]]
    + [("2026-03-31", "Revenue", 500.0), ("2026-06-30", "Revenue", 600.0)]
    + [("2026-03-31", "IncomeAfterTaxes", 50.0), ("2026-06-30", "IncomeAfterTaxes", 60.0)]
    + [(f"2025-{m}", "EquityAttributableToOwnersOfParent", 9.0) for m in ["09-30", "12-31"]]
    + [("2026-03-31", "EquityAttributableToOwnersOfParent", 45.0),
       ("2026-06-30", "EquityAttributableToOwnersOfParent", 54.0)]
)
# Year-to-date, as FinMind reports the cash-flow statement.
_CASHFLOW = _statement([
    ("2026-03-31", "CashFlowsFromOperatingActivities", 70.0),
    ("2026-06-30", "CashFlowsFromOperatingActivities", 150.0),
    ("2026-03-31", "CashBalancesBeginningOfPeriod", 1000.0),
    ("2026-03-31", "CashBalancesEndOfPeriod", 1070.0),
    ("2026-06-30", "CashBalancesBeginningOfPeriod", 1000.0),
    ("2026-06-30", "CashBalancesEndOfPeriod", 1150.0),
])
_BALANCE = _statement([
    ("2026-06-30", "TotalAssets", 900.0), ("2026-06-30", "TotalAssets_per", 100.0),
    ("2026-06-30", "Liabilities", 300.0), ("2026-06-30", "Equity", 600.0),
    ("2026-06-30", "ShorttermBorrowings", 60.0), ("2026-06-30", "LongtermBorrowings", 30.0),
])


def _revenue_row(i):
    """Month i of 2025-01..2026-09: revenue 100 + i, with a dip in 2025-12.

    FinMind dates a row by the month it is published (the next one) and leaves
    create_time empty on older rows.
    """
    covered = pd.Period("2025-01", "M") + i
    published = covered + 1
    return {
        "date": f"{published}-01", "stock_id": "2330", "country": "Taiwan",
        "revenue": 50 if str(covered) == "2025-12" else 100 + i,
        "revenue_month": covered.month, "revenue_year": covered.year,
        "create_time": None if covered.year == 2025 else f"{published}-08",
    }


_REVENUE = [_revenue_row(i) for i in range(21)]
_NEWS = {
    "2026-10-15": [
        {"date": "2026-10-15 05:37:00", "stock_id": "2330", "link": "https://a", "source": "A", "title": "Closed up"},
        {"date": "2026-10-15 06:08:01", "stock_id": "2330", "link": "https://a", "source": "A", "title": "Closed up"},
    ],
    "2026-10-16": [
        {"date": "2026-10-16 01:00:15", "stock_id": "2330", "link": "https://b", "source": "B", "title": "Targets raised"},
    ],
}


def _fake_api(params, token):
    assert token == TOKEN
    dataset = params["dataset"]
    if dataset == "TaiwanStockPrice":
        return _PRICES if params.get("data_id") == "2330" else []
    if dataset == "TaiwanStockFinancialStatements":
        return _INCOME
    if dataset == "TaiwanStockCashFlowsStatement":
        return _CASHFLOW
    if dataset == "TaiwanStockBalanceSheet":
        return _BALANCE
    if dataset == "TaiwanStockDividendResult":
        return [
            {"date": _D(-2), "stock_id": "2330", "before_price": 110.0, "after_price": 105.0,
             "stock_and_cache_dividend": 5.0, "stock_or_cache_dividend": "息"},
            {"date": _D(-1), "stock_id": "2330", "before_price": 110.0, "after_price": 100.0,
             "stock_and_cache_dividend": 10.0, "stock_or_cache_dividend": "權"},
        ]
    if dataset == "TaiwanStockMonthRevenue":
        return _REVENUE
    if dataset == "TaiwanStockNews":
        assert "end_date" not in params  # FinMind refuses a range for news
        return _NEWS.get(params["start_date"], [])
    if dataset == "TaiwanStockInfo":
        listing = {"2330": "twse", "8069": "tpex", "2881": "twse"}.get(params.get("data_id"))
        industry = "金融保險業" if params.get("data_id") == "2881" else "Semis"
        return [{"stock_id": params["data_id"], "stock_name": "TSMC", "industry_category": industry,
                 "type": listing}] if listing else []
    return []


@pytest.fixture
def api(monkeypatch, tmp_path):
    """FinMind answers from the tables above, into a cache of the test's own."""
    monkeypatch.setenv("FINMIND_TOKEN", TOKEN)
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    set_config(config)
    stub = mock.Mock(side_effect=_fake_api)
    monkeypatch.setattr(client, "_request", stub)
    yield stub
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


# --- symbols -------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("2330", "2330"), ("2330.TW", "2330"), ("2330.tw", "2330"), ("8069.TWO", "8069"),
    ("0050", "0050"), ("00632R.TW", "00632R"),
    ("AAPL", None), ("0700.HK", None), ("600519.SS", None), ("BTC-USD", None), ("", None),
])
def test_taiwan_symbols_are_recognised_in_every_spelling(raw, expected):
    assert client.taiwan_stock_id(raw) == expected


@pytest.mark.unit
def test_a_bare_code_resolves_to_its_exchange(api):
    assert client.resolve_taiwan_symbol("2330") == "2330.TW"
    assert client.resolve_taiwan_symbol("8069") == "8069.TWO"
    assert client.resolve_taiwan_symbol("2330.TW") is None   # already suffixed
    assert client.resolve_taiwan_symbol("AAPL") is None


@pytest.mark.unit
def test_without_finmind_only_a_four_digit_code_is_assumed_twse(monkeypatch):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    assert client.resolve_taiwan_symbol("2330") == "2330.TW"
    assert client.resolve_taiwan_symbol("600519") is None  # may be a mainland listing


@pytest.mark.unit
def test_the_cli_gives_a_bare_taiwan_code_its_suffix(monkeypatch):
    from cli.prompts import normalize_ticker_symbol

    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    assert normalize_ticker_symbol("2330") == "2330.TW"
    assert normalize_ticker_symbol("2330.tw") == "2330.TW"
    assert normalize_ticker_symbol("aapl") == "AAPL"


# --- routing -------------------------------------------------------------------

@pytest.mark.unit
def test_a_us_symbol_is_passed_on_without_a_request_or_a_token(monkeypatch):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    with (mock.patch.object(client, "_request", side_effect=AssertionError("requested")),
          pytest.raises(NoMarketDataError, match="not a Taiwan-listed symbol")):
        finmind.get_stock_data("AAPL", START5, END)


@pytest.mark.unit
def test_the_default_chain_sends_taiwan_to_finmind_and_the_rest_to_yahoo(api):
    yahoo = mock.Mock(return_value="yahoo prices")
    with mock.patch.dict(router.VENDOR_METHODS["get_stock_data"], {"yfinance": yahoo}):
        taiwan = router.route_to_vendor("get_stock_data", "2330.TW", START5, END)
        us = router.route_to_vendor("get_stock_data", "AAPL", START5, END)
    assert taiwan.startswith("# Stock data for 2330.TW")
    assert us == "yahoo prices"
    yahoo.assert_called_once_with("AAPL", START5, END)


@pytest.mark.unit
def test_a_missing_token_hands_a_taiwan_symbol_to_yahoo(monkeypatch):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    yahoo = mock.Mock(return_value="yahoo prices")
    with mock.patch.dict(router.VENDOR_METHODS["get_stock_data"], {"yfinance": yahoo}):
        out = router.route_to_vendor("get_stock_data", "2330.TW", START5, END)
    assert out == "yahoo prices"


@pytest.mark.unit
def test_a_rate_limited_finmind_hands_the_call_to_yahoo(api):
    api.side_effect = client.FinMindRateLimitError("limit")
    yahoo = mock.Mock(return_value="yahoo prices")
    with mock.patch.dict(router.VENDOR_METHODS["get_stock_data"], {"yfinance": yahoo}):
        assert router.route_to_vendor("get_stock_data", "2330.TW", START5, END) == "yahoo prices"


# --- HTTP errors -----------------------------------------------------------------

def _response(status, body):
    response = mock.Mock(status_code=status)
    response.json.return_value = body
    return response


@pytest.mark.unit
@pytest.mark.parametrize("status, body, error", [
    (402, {"msg": "Requests reach the upper limit", "status": 402}, client.FinMindRateLimitError),
    (400, {"msg": "Token is illegal.", "status": 400}, client.FinMindNotConfiguredError),
    (400, {"msg": "Your level is free.", "status": 400}, VendorUnavailableError),
    (503, {}, VendorUnavailableError),
])
def test_finmind_errors_map_to_router_reactions(status, body, error):
    with (mock.patch.object(client.requests, "get", return_value=_response(status, body)),
          pytest.raises(error) as raised):
        client._request({"dataset": "TaiwanStockPrice"}, TOKEN)
    assert TOKEN not in str(raised.value)


@pytest.mark.unit
def test_the_token_travels_in_a_header_not_the_url():
    with mock.patch.object(client.requests, "get",
                           return_value=_response(200, {"status": 200, "data": []})) as get:
        client._request({"dataset": "TaiwanStockPrice"}, TOKEN)
    assert get.call_args.kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert TOKEN not in str(get.call_args.kwargs["params"])


@pytest.mark.unit
def test_a_network_failure_is_an_outage_and_does_not_leak_the_token():
    with (mock.patch.object(client.requests, "get", side_effect=requests.ConnectionError(f"x?token={TOKEN}")),
          pytest.raises(VendorUnavailableError) as raised):
        client._request({"dataset": "TaiwanStockPrice"}, TOKEN)
    assert TOKEN not in str(raised.value) and raised.value.__cause__ is None


# --- cache -----------------------------------------------------------------------

@pytest.mark.unit
def test_answers_are_served_from_the_cache(api):
    finmind.get_stock_data("2330.TW", START5, END)
    finmind.get_stock_data("2330.TW", START5, END)
    finmind.get_indicators("2330.TW", "rsi", END, 5)
    assert api.call_count == 2   # prices once, ex-rights results once


@pytest.mark.unit
def test_a_closed_past_range_never_expires(api, monkeypatch):
    client.fetch("TaiwanStockPER", "2330", "2025-01-01", "2025-02-01", ttl=0)
    client.fetch("TaiwanStockPER", "2330", "2025-01-01", "2025-02-01", ttl=0)
    client.fetch("TaiwanStockPER", "2330", "2025-01-01", None, ttl=0)  # open range, expired
    client.fetch("TaiwanStockPER", "2330", "2025-01-01", None, ttl=0)
    assert api.call_count == 3


# --- prices and indicators -------------------------------------------------------

@pytest.mark.unit
def test_stock_data_matches_the_yahoo_report_shape(api):
    out = finmind.get_stock_data("2330.TW", START5, END)
    header, csv = out.split("\n\n", 1)
    assert header == f"# Stock data for 2330.TW from {START5} to {END}\n# Total records: 5"
    assert csv.splitlines()[0] == "Date,Open,High,Low,Close,Volume,Dividends,Stock Splits"
    assert csv.splitlines()[-1].startswith(f"{END},")


@pytest.mark.unit
def test_ex_dividend_and_ex_rights_days_fill_yahoos_action_columns(api):
    out = finmind.get_stock_data("2330.TW", START5, END)
    rows = {line.split(",")[0]: line.split(",")[-2:] for line in out.splitlines()[4:]}
    assert rows[_D(-2)] == ["5.0", "0.0"]   # cash dividend
    assert rows[_D(-1)] == ["0.0", "1.1"]   # stock dividend, as a split ratio
    assert rows[_D(-3)] == ["0.0", "0.0"]


@pytest.mark.unit
def test_an_unlisted_code_is_no_data(api):
    with pytest.raises(NoMarketDataError):
        finmind.get_stock_data("9999.TW", START5, END)


@pytest.mark.unit
def test_indicators_use_the_shared_report_and_no_later_prices(api):
    as_of = _D(-3)
    out = finmind.get_indicators("2330", "close_10_ema", as_of, 3)
    assert out.startswith(f"## close_10_ema values from {(_DATES[-3] - pd.Timedelta(days=3)):%Y-%m-%d} to {as_of}:")
    assert END not in out
    assert "10 EMA: A responsive short-term average." in out
    with pytest.raises(ValueError, match="not supported"):
        finmind.get_indicators("2330", "bogus", END, 3)


# --- statements ------------------------------------------------------------------

@pytest.mark.unit
def test_year_to_date_cash_flows_are_reported_per_quarter(api):
    out = finmind.get_cashflow("2330.TW", "quarterly")
    frame = pd.read_csv(pd.io.common.StringIO(out.split("\n", 2)[2]), index_col=0)
    assert frame.at["CashFlowsFromOperatingActivities", "2026-06-30"] == 80.0   # 150 - 70
    assert frame.at["CashFlowsFromOperatingActivities", "2026-03-31"] == 70.0
    assert frame.at["CashBalancesBeginningOfPeriod", "2026-06-30"] == 1070.0    # Q1's close
    assert frame.at["CashBalancesEndOfPeriod", "2026-06-30"] == 1150.0


@pytest.mark.unit
def test_annual_income_sums_complete_years_only(api):
    out = finmind.get_income_statement("2330.TW", "annual")
    assert "2025-12-31" in out and "2026-12-31" not in out
    csv = "\n".join(line for line in out.splitlines() if not line.startswith("#"))
    frame = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
    assert frame.at["Revenue", "2025-12-31"] == 1000.0
    assert frame.at["EPS", "2025-12-31"] == 4.0


@pytest.mark.unit
def test_balance_sheet_drops_share_of_total_rows(api):
    out = finmind.get_balance_sheet("2330.TW")
    assert out.startswith("# Balance Sheet data for 2330.TW (quarterly)\n")
    assert "TotalAssets," in out and "_per" not in out


@pytest.mark.unit
def test_a_past_run_sees_only_periods_past_their_filing_deadline(api):
    before = finmind.get_income_statement("2330.TW", "quarterly", "2026-08-13")
    after = finmind.get_income_statement("2330.TW", "quarterly", "2026-08-14")
    assert "2026-06-30" not in before and "2026-03-31" in before
    assert "2026-06-30" in after
    assert "# Point-in-time as of: 2026-08-13" in before


@pytest.mark.unit
def test_a_financial_company_is_served_on_the_later_deadline(api):
    # 2881 is listed under 金融保險業: its Q2 is due 08-31, not 08-14.
    assert "2026-06-30" not in finmind.get_income_statement("2881.TW", "quarterly", "2026-08-30")
    assert "2026-06-30" in finmind.get_income_statement("2881.TW", "quarterly", "2026-08-31")


@pytest.mark.unit
@pytest.mark.parametrize("period, financial, due", [
    ("2026-03-31", False, "2026-05-15"), ("2026-06-30", False, "2026-08-14"),
    ("2026-09-30", False, "2026-11-14"), ("2025-12-31", False, "2026-03-31"),
    ("2026-03-31", True, "2026-05-31"), ("2026-06-30", True, "2026-08-31"),
    ("2026-09-30", True, "2026-11-30"), ("2025-12-31", True, "2026-03-31"),
])
def test_filing_deadlines(period, financial, due):
    from tradingagents.dataflows.vendors.finmind.publication import statement_deadline
    assert statement_deadline(pd.Timestamp(period), financial) == pd.Timestamp(due)


@pytest.mark.unit
def test_an_unreadable_listing_takes_the_later_deadline(api, monkeypatch):
    from tradingagents.dataflows.vendors.finmind.publication import is_financial
    monkeypatch.setattr(client, "_request", mock.Mock(side_effect=VendorUnavailableError("down")))
    assert is_financial("2330")
    assert is_financial("9999")    # not listed


@pytest.mark.unit
def test_fundamentals_report_trailing_figures_and_withhold_todays_profile_from_the_past(api):
    live = finmind.get_fundamentals("2330.TW")
    assert "Name: TSMC" in live
    assert "TTM Period: 2025 Q3～2026 Q2 (the four latest published quarters, summed)" in live
    assert "Revenue (TTM 2025 Q3～2026 Q2): 1,800 TWD" in live   # Q3, Q4 2025 + Q1, Q2 2026
    assert "EPS (TTM" not in live   # EPS exists for 2025 only, so the span is incomplete
    assert "Debt Ratio (Total Liabilities / Total Assets): 33.3%" in live
    assert "Interest-bearing Debt to Equity ((Short-term Borrowings + Long-term Borrowings" \
           " + Bonds Payable) / Equity): 15.0% (debt 90 TWD)" in live
    assert "Debt to Equity:" not in live
    past = finmind.get_fundamentals("2330.TW", "2026-08-30")
    assert "Name:" not in past and "# Point-in-time as of: 2026-08-30" in past


@pytest.mark.unit
def test_parent_net_income_is_reported_apart_from_net_income_with_minorities(api):
    out = finmind.get_fundamentals("2330.TW")
    # The four latest quarters: 2025 Q3 to 2026 Q2.
    assert "Net Income Attributable to Owners of the Parent (歸母淨利, TTM 2025 Q3～2026 Q2): 117 TWD" in out
    assert "Net Income incl. Non-controlling Interests (稅後淨利, TTM 2025 Q3～2026 Q2): 130 TWD" in out
    statement = finmind.get_income_statement("2330.TW", "quarterly")
    assert "EquityAttributableToOwnersOfParent = net income attributable to owners of the parent (歸母淨利)" in statement


# --- news ------------------------------------------------------------------------

@pytest.mark.unit
def test_news_is_fetched_per_day_deduplicated_and_windowed(api):
    out = finmind.get_news("2330.TW", "2026-10-15", "2026-10-16")
    assert out.startswith("## 2330.TW News (FinMind), from 2026-10-15 to 2026-10-16:")
    assert out.count("### Closed up (source: A)") == 1
    assert out.index("Targets raised") < out.index("Closed up")   # newest first
    dates = sorted(c.args[0]["start_date"] for c in api.call_args_list)
    # The UTC day before the window holds the first Taipei day's early hours.
    assert dates == ["2026-10-14", "2026-10-15", "2026-10-16"]


@pytest.mark.unit
def test_the_news_window_is_cut_on_taipei_days(api, monkeypatch):
    news = {
        # 2026-10-15 01:30 in Taipei: inside a window that starts on 10-15.
        "2026-10-14": [{"date": "2026-10-14 17:30:00", "link": "https://early", "source": "A",
                        "title": "Early on the first day"}],
        # 2026-10-17 00:22 in Taipei: past a window that ends on 10-16.
        "2026-10-16": [{"date": "2026-10-16 16:22:00", "link": "https://late", "source": "B",
                        "title": "September revenue filed"},
                       {"date": "2026-10-16 15:59:00", "link": "https://last", "source": "C",
                        "title": "Last minute of the day"}],
    }
    monkeypatch.setattr(client, "_request", mock.Mock(
        side_effect=lambda params, token: news.get(params["start_date"], [])))
    out = finmind.get_news("2330.TW", "2026-10-15", "2026-10-16")
    assert "Early on the first day" in out
    assert "Last minute of the day" in out
    assert "September revenue filed" not in out


@pytest.mark.unit
def test_a_window_with_no_articles_is_reported_as_none_found(api):
    assert finmind.get_news("2330.TW", "2026-09-01", "2026-09-02").startswith(
        "No news found for 2330.TW between 2026-09-01 and 2026-09-02")


# --- monthly revenue ---------------------------------------------------------------

def _revenue_table(out):
    csv = out.split("## Monthly Revenue (last 15 months, by the month covered; TWD)\n", 1)[1]
    return pd.read_csv(pd.io.common.StringIO(csv), index_col="Revenue Month", dtype=str)


@pytest.mark.unit
def test_monthly_revenue_covers_15_months_by_the_month_covered(api):
    out = finmind.get_fundamentals("2330.TW")
    table = _revenue_table(out)
    assert list(table.index) == [str(pd.Period("2025-07", "M") + i) for i in range(15)]
    # Older rows have no create_time and must still be served (they were dropped).
    assert table.at["2025-07", "Revenue (TWD)"] == "106"
    assert table.at["2026-09", "MoM"] == "+0.84%"          # 120 / 119
    assert table.at["2026-09", "YoY"] == "+11.11%"         # 120 / 108
    assert table.at["2025-12", "Record High"] == "False" and table.at["2026-01", "Record High"] == "True"


@pytest.mark.unit
def test_monthly_revenue_summary_ytd_streak_and_derived_quarter(api):
    out = finmind.get_fundamentals("2330.TW")
    assert "Latest Monthly Revenue (2026-09): 120 TWD (0.00 億), MoM +0.84%, YoY +11.11%" in out
    # Jan-Sep 2026: 112..120 = 1044; Jan-Sep 2025: 100..108 = 936
    assert "YTD Revenue (2026-01 to 2026-09): 1,044 TWD (0.00 億), YoY +11.54%" in out
    assert "Consecutive Record-High Months (vs. all months since 2025-01): 9" in out
    # Q3 2026 has all three months but no income statement yet: 118 + 119 + 120.
    assert ("Q3 2026 Revenue (derived from monthly revenue; income statement not yet published): "
            "357 TWD (0.00 億), YoY +11.21%") in out


@pytest.mark.unit
def test_a_past_run_sees_only_revenue_published_by_its_date(api):
    # September 2026 revenue was published 2026-10-08.
    before = finmind.get_fundamentals("2330.TW", "2026-10-07")
    assert "Latest Monthly Revenue (2026-08)" in before
    assert "Q3 2026 Revenue" not in before
    # An older row with no create_time counts as published on the 10th of its month.
    assert "2025-08," in _revenue_table(finmind.get_fundamentals("2330.TW", "2025-09-10")).to_csv()
    assert "Latest Monthly Revenue (2025-07)" in finmind.get_fundamentals("2330.TW", "2025-09-09")


@pytest.mark.unit
def test_revenue_counts_as_public_on_its_create_time_or_its_deadline_whichever_is_first(api, monkeypatch):
    rows = copy.deepcopy(_REVENUE)
    for row in rows:
        covered = (row["revenue_year"], row["revenue_month"])
        if covered in ((2026, 2), (2026, 3)):      # due 03-10 and 04-10, re-stamped 04-21
            row["create_time"] = "2026-04-21"
        if covered == (2026, 7):                   # due 08-10, filed early on 08-03
            row["create_time"] = "2026-08-03"
    monkeypatch.setattr(client, "_request", mock.Mock(
        side_effect=lambda params, token: rows if params["dataset"] == "TaiwanStockMonthRevenue"
        else _fake_api(params, token)))
    assert "Latest Monthly Revenue (2026-02)" in finmind.get_fundamentals("2330.TW", "2026-03-10")
    assert "Latest Monthly Revenue (2026-01)" in finmind.get_fundamentals("2330.TW", "2026-03-09")
    assert "Latest Monthly Revenue (2026-07)" in finmind.get_fundamentals("2330.TW", "2026-08-03")
    assert "Latest Monthly Revenue (2026-06)" in finmind.get_fundamentals("2330.TW", "2026-08-02")


@pytest.mark.unit
def test_a_revenue_deadline_on_a_weekend_moves_to_monday(api, monkeypatch):
    rows = [dict(r, create_time=None) for r in _REVENUE]   # no create_time: the deadline alone decides
    monkeypatch.setattr(client, "_request", mock.Mock(
        side_effect=lambda params, token: rows if params["dataset"] == "TaiwanStockMonthRevenue"
        else _fake_api(params, token)))
    # 2026-10-10 is a Saturday, so September revenue is due on Monday 2026-10-12.
    assert "Latest Monthly Revenue (2026-08)" in finmind.get_fundamentals("2330.TW", "2026-10-11")
    assert "Latest Monthly Revenue (2026-09)" in finmind.get_fundamentals("2330.TW", "2026-10-12")


@pytest.mark.unit
def test_a_quarter_with_its_income_statement_out_is_not_derived(api, monkeypatch):
    rows = [r for r in _REVENUE if (r["revenue_year"], r["revenue_month"]) <= (2026, 6)]
    monkeypatch.setattr(client, "_request", mock.Mock(
        side_effect=lambda params, token: rows if params["dataset"] == "TaiwanStockMonthRevenue"
        else _fake_api(params, token)))
    out = finmind.get_fundamentals("2330.TW")
    assert "Latest Monthly Revenue (2026-06)" in out
    assert "derived from monthly revenue" not in out   # Q2 2026 statement is published


# --- a section that cannot be had ------------------------------------------------------

_PER = [{"date": "2026-10-06", "stock_id": "2330", "PER": 24.5, "PBR": 6.1, "dividend_yield": 1.8}]


def _with(failing_dataset, error):
    def answer(params, token):
        if params["dataset"] == failing_dataset:
            raise error
        if params["dataset"] == "TaiwanStockPER":
            return _PER
        return _fake_api(params, token)
    return mock.Mock(side_effect=answer)


@pytest.mark.unit
def test_a_failed_revenue_dataset_costs_only_its_section(api, monkeypatch):
    monkeypatch.setattr(client, "_request", _with(
        "TaiwanStockMonthRevenue", VendorUnavailableError("FinMind could not serve TaiwanStockMonthRevenue: 503")))
    out = finmind.get_fundamentals("2330.TW", "2026-10-07")
    assert "PE Ratio (TWSE-published, 2026-10-06): 24.5" in out                 # valuation still served
    assert "Revenue (TTM 2025 Q3～2026 Q2): 1,800 TWD" in out                     # and the statements
    assert "Net Income Attributable to Owners of the Parent (歸母淨利" in out
    assert "Last Close:" in out                                                  # and prices
    assert "Latest Monthly Revenue" not in out
    assert "SECTION_UNAVAILABLE monthly_revenue: FinMind could not serve TaiwanStockMonthRevenue: 503" in out


@pytest.mark.unit
def test_failed_valuation_costs_only_its_section(api, monkeypatch):
    monkeypatch.setattr(client, "_request", _with("TaiwanStockPER", VendorUnavailableError("PER down")))
    out = finmind.get_fundamentals("2330.TW", "2026-10-07")
    assert "SECTION_UNAVAILABLE valuation: PER down" in out and "Latest Monthly Revenue (2026-08)" in out


@pytest.mark.unit
def test_failed_statements_still_raise_for_the_router(api, monkeypatch):
    monkeypatch.setattr(client, "_request", _with(
        "TaiwanStockFinancialStatements", VendorUnavailableError("statements down")))
    with pytest.raises(VendorUnavailableError, match="statements down"):
        finmind.get_fundamentals("2330.TW", "2026-10-07")


# --- the hourly request budget ----------------------------------------------------------

@pytest.mark.unit
def test_requests_past_the_hourly_budget_are_refused_unsent(api):
    from tradingagents.dataflows.config import get_config
    set_config({**get_config(), "finmind_requests_per_hour": 2})
    client.fetch("TaiwanStockPER", "2330", "2026-01-01", "2026-01-31")
    client.fetch("TaiwanStockPER", "2330", "2026-02-01", "2026-02-28")
    client.fetch("TaiwanStockPER", "2330", "2026-01-01", "2026-01-31")      # cached: not counted
    with pytest.raises(client.FinMindRateLimitError, match="budget of 2 requests per hour"):
        client.fetch("TaiwanStockPER", "2330", "2026-03-01", "2026-03-31")
    assert api.call_count == 2


@pytest.mark.unit
def test_an_old_request_frees_its_slot(api, monkeypatch):
    from tradingagents.dataflows.config import get_config
    set_config({**get_config(), "finmind_requests_per_hour": 1})
    now = [1_000_000.0]
    monkeypatch.setattr(client.time, "time", lambda: now[0])
    client.fetch("TaiwanStockPER", "2330", "2026-01-01", "2026-01-31")
    now[0] += 3601
    client.fetch("TaiwanStockPER", "2330", "2026-02-01", "2026-02-28")
    assert api.call_count == 2


@pytest.mark.unit
def test_one_failed_revenue_dataset_is_one_scored_gap(api, monkeypatch):
    from tradingagents.agents import coverage
    monkeypatch.setattr(client, "_request", _with(
        "TaiwanStockMonthRevenue", VendorUnavailableError("FinMind could not serve TaiwanStockMonthRevenue: 503")))
    out = finmind.get_fundamentals("2330.TW", "2026-10-07")
    assert out.count("SECTION_UNAVAILABLE") == 1 and "revenue_timing" not in out
    checks = (coverage.section_checks(out)
              + coverage.revenue_timing_checks(out)
              + coverage.freshness_checks("2330.TW", "2026-10-07", out, financial=False))
    scored_gaps = [c for c in checks if c["status"] not in ("ok", "empty", "not_applicable", "late_publication")
                   and c.get("scored", True)]
    assert [(c["tool"], c["status"]) for c in scored_gaps] == [("monthly_revenue", "unavailable")]
