"""Taiwan market indicators from FinMind: point-in-time by publication, and handing the rest to FRED."""

import copy
from unittest import mock

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import UnsupportedSymbolError
from tradingagents.dataflows.vendors.finmind import client, macro

DAYS = ["2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06"]   # 10-03 and 10-04 are a weekend


def _answer(params, token):
    dataset, rows = params["dataset"], []
    for i, day in enumerate(DAYS):
        if params.get("end_date") and day > params["end_date"]:
            continue   # FinMind honours the end date it is sent
        if dataset == "TaiwanExchangeRate":
            rows.append({"date": day, "currency": "USD", "cash_buy": 31.4, "cash_sell": 32.0,
                         "spot_buy": 31.7 + i / 100, "spot_sell": 31.8 + i / 100})
        elif dataset == "TaiwanStockPrice":
            rows.append({"date": day, "stock_id": params["data_id"], "close": 48000.0 + 100 * i,
                         "spread": 100.0, "Trading_money": 9e11})
        elif dataset == "TaiwanStockTotalInstitutionalInvestors":
            rows += [{"date": day, "name": "Foreign_Investor", "buy": 5e10, "sell": 4e10 + 1e9 * i},
                     {"date": day, "name": "total", "buy": 9e10, "sell": 8e10}]
        elif dataset == "TaiwanStockTotalMarginPurchaseShortSale":
            rows += [{"date": day, "name": "MarginPurchase", "TodayBalance": 9_300_000 + i},
                     {"date": day, "name": "ShortSale", "TodayBalance": 230_000},
                     {"date": day, "name": "MarginPurchaseMoney", "TodayBalance": 6.3e11}]
    return rows


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("FINMIND_TOKEN", "test-token-not-real")
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    set_config(config)
    stub = mock.Mock(side_effect=_answer)
    monkeypatch.setattr(client, "_request", stub)
    yield stub
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _with_as_of_time(value):
    from tradingagents.dataflows.config import get_config
    set_config({**get_config(), "as_of_time": value})


@pytest.mark.unit
@pytest.mark.parametrize("indicator", list(macro.INDICATORS))
def test_a_run_dated_on_a_weekend_reads_no_later_day(api, indicator):
    out = macro.get_macro_data(indicator, "2026-10-04", 10)
    assert "2026-10-02" in out
    assert "2026-10-05" not in out and "2026-10-06" not in out
    assert "point-in-time: rows up to 2026-10-04" in out


@pytest.mark.unit
def test_the_run_date_counts_only_once_its_row_is_published(api):
    # 2026-10-05 with as_of_time 13:30: the TAIEX close (14:00) and the evening
    # margin balances (21:30) are not out yet; with 23:59:59 both are.
    _with_as_of_time("13:30:00")
    for indicator in ("taiex", "market_margin", "market_institutional", "twd_usd"):
        out = macro.get_macro_data(indicator, "2026-10-05", 10)
        assert "2026-10-05" not in out.split("\n\n", 1)[1], indicator
        assert "so the series ends on 2026-10-04" in out
    _with_as_of_time("23:59:59")
    assert "2026-10-05" in macro.get_macro_data("market_margin", "2026-10-05", 10).split("\n\n", 1)[1]


@pytest.mark.unit
def test_tables_are_in_the_units_they_name(api):
    inst = macro.get_macro_data("market_institutional", "2026-10-02", 5)
    assert "date,Foreign_Investor,total" in inst and "2026-10-02,90.0,100.0" in inst      # 億, net buy
    margin = macro.get_macro_data("market_margin", "2026-10-02", 5)
    assert "margin_balance_張,short_balance_張,margin_money_億" in margin and "6300.0" in margin
    taiex = macro.get_macro_data("taiex", "2026-10-02", 5)
    assert "2026-10-02,48100.0,100.0,9000.0" in taiex
    assert "change 100.0 (" not in taiex                    # a day's change has no change over the window


@pytest.mark.unit
def test_aliases_and_an_indicator_finmind_does_not_serve(api):
    assert "USD/TWD" in macro.get_macro_data("usd_twd", "2026-10-02", 5)
    with pytest.raises(UnsupportedSymbolError):
        macro.get_macro_data("cpi", "2026-10-02")
    assert api.call_count == 1                              # the unsupported one sent nothing


@pytest.mark.unit
def test_the_router_serves_taiwan_indicators_and_hands_the_rest_to_fred(api, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert router.route_to_vendor("get_macro_indicators", "taiex", "2026-10-02", 5).startswith("## FinMind: TAIEX")
    out = router.route_to_vendor("get_macro_indicators", "cpi", "2026-10-02", 30)
    assert out.startswith("DATA_UNAVAILABLE: optional macro_data could not be retrieved")
    assert "FRED_API_KEY" in out                            # the reason, not "no data for 'cpi'"
    monkeypatch.setitem(router.VENDOR_METHODS["get_macro_indicators"], "fred", lambda *a, **k: "FRED series")
    assert router.route_to_vendor("get_macro_indicators", "cpi", "2026-10-02", 30) == "FRED series"


@pytest.mark.unit
def test_a_us_indicator_needs_no_finmind_token(monkeypatch):
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    with pytest.raises(UnsupportedSymbolError):
        macro.get_macro_data("fed_funds_rate", "2026-10-02")
