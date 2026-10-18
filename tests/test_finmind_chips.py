"""Taiwan chip data (籌碼面), peer valuation, the chip analyst and its CLI gating.

The fixture is 2383 (台光電) around its 2026-10-13 and 2026-10-15 high-volume
drops, as FinMind reported them. HTTP is stubbed at ``client._request``, so
these run offline and without a token.
"""

import copy
import json
from collections import Counter
from unittest import mock

import pandas as pd
import pytest

import tradingagents.default_config as default_config
from cli.models import AnalystType, AssetType
from cli.prompts import filter_analysts_for_asset_type, parse_analysts
from tradingagents.agents.analysts import chip_analyst, fundamentals_analyst
from tradingagents.agents.managers.research_manager import peer_valuation_block
from tradingagents.dataflows import router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.vendors.finmind import chips, client, peers

TOKEN = "test-token-not-real"

# (date, close, volume in shares): 25 quiet days, then 2383 as traded.
_TRADED = [
    ("2026-09-15", 5115.0, 1525730), ("2026-09-16", 5085.0, 1686031), ("2026-09-17", 4695.0, 5813349),
    ("2026-09-18", 4890.0, 3783127), ("2026-09-21", 4895.0, 2394262), ("2026-09-22", 4860.0, 2101406),
    ("2026-09-23", 5020.0, 2450930), ("2026-09-24", 5050.0, 2346653), ("2026-09-29", 4920.0, 2146878),
    ("2026-09-30", 4935.0, 1549084), ("2026-10-01", 4975.0, 1363430), ("2026-10-02", 5185.0, 2397410),
    ("2026-10-05", 5700.0, 1316051), ("2026-10-06", 5995.0, 5086859), ("2026-10-07", 5950.0, 2968881),
    ("2026-10-08", 6000.0, 1859243), ("2026-10-12", 6150.0, 2345344), ("2026-10-13", 5545.0, 4882149),
    ("2026-10-14", 5520.0, 2325865), ("2026-10-15", 5120.0, 5904207), ("2026-10-16", 5165.0, 2773138),
]
_QUIET = [(d.strftime("%Y-%m-%d"), 5100.0, 2000000) for d in pd.bdate_range("2026-08-01", periods=25)]
_DAYS = _QUIET + _TRADED
_PRICES = [
    {"date": d, "stock_id": "2383", "Trading_Volume": v, "Trading_money": 0, "open": c, "max": c,
     "min": c, "close": c, "spread": 0.0, "Trading_turnover": 0}
    for d, c, v in _DAYS
]

# (buy, sell) in shares, by investor, on the two drops; every other day is flat.
_FLOWS = {
    "2026-10-13": {"Foreign_Investor": (1485400, 2411933), "Foreign_Dealer_Self": (0, 0),
                   "Dealer_self": (118871, 91309), "Dealer_Hedging": (252366, 209660),
                   "Investment_Trust": (354100, 523454)},
    "2026-10-15": {"Foreign_Investor": (3012078, 3974573), "Foreign_Dealer_Self": (0, 0),
                   "Dealer_self": (70006, 160270), "Dealer_Hedging": (439666, 287637),
                   "Investment_Trust": (308000, 40000)},
}
_INSTITUTIONAL = [
    {"date": d, "stock_id": "2383", "name": name, "buy": buy, "sell": sell}
    for d, _, _ in _DAYS
    for name, (buy, sell) in _FLOWS.get(d, dict.fromkeys(_FLOWS["2026-10-13"], (100000, 100000))).items()
]

# Margin balance in lots: +226 on 10-13 and +55 on 10-15, as reported.
_MARGIN_BALANCE = {"2026-10-12": 2520, "2026-10-13": 2746, "2026-10-14": 2789, "2026-10-15": 2844, "2026-10-16": 2903}


def _margin_rows():
    rows, previous = [], 2500
    for d, _, _ in _DAYS:
        balance = _MARGIN_BALANCE.get(d, previous)
        rows.append({"date": d, "stock_id": "2383", "MarginPurchaseTodayBalance": balance,
                     "MarginPurchaseYesterdayBalance": previous, "MarginPurchaseLimit": 89580,
                     "ShortSaleTodayBalance": 6, "ShortSaleYesterdayBalance": 6, "OffsetLoanAndShort": 0})
        previous = balance
    return rows


_SHAREHOLDING = [
    {"date": d, "stock_id": "2383", "ForeignInvestmentSharesRatio": 44.0 - 0.02 * i}
    for i, (d, _, _) in enumerate(_DAYS)
]
_LENDING = [
    {"date": "2026-10-13", "stock_id": "2383", "transaction_type": "議借", "volume": 54, "fee_rate": 0.01},
    {"date": "2026-10-13", "stock_id": "2383", "transaction_type": "議借", "volume": 70, "fee_rate": 0.8},
    {"date": "2026-10-13", "stock_id": "2383", "transaction_type": "議借", "volume": 11, "fee_rate": 1.0},
    {"date": "2026-10-14", "stock_id": "2383", "transaction_type": "競價", "volume": 455, "fee_rate": 0.01},
]
_PER = {
    "2383": {"PER": 80.77, "PBR": 31.73, "dividend_yield": 0.48},
    "6274": {"PER": 82.02, "PBR": 20.01, "dividend_yield": 0.46},
    "6213": {"PER": 120.0, "PBR": 12.46, "dividend_yield": 0.39},
}
_NAMES = {"2383": "台光電", "6274": "台燿", "6213": "聯茂"}


def _fake_api(params, token):
    assert token == TOKEN
    dataset, data_id = params["dataset"], params.get("data_id")
    if data_id != "2383" and dataset not in ("TaiwanStockPER", "TaiwanStockInfo"):
        return []
    if dataset == "TaiwanStockPrice":
        return _PRICES
    if dataset == "TaiwanStockInstitutionalInvestorsBuySell":
        return _INSTITUTIONAL
    if dataset == "TaiwanStockMarginPurchaseShortSale":
        return _margin_rows()
    if dataset == "TaiwanStockShareholding":
        return _SHAREHOLDING
    if dataset == "TaiwanStockSecuritiesLending":
        return _LENDING
    if dataset == "TaiwanStockPER":
        row = _PER.get(data_id)
        return [{"date": "2026-10-16", "stock_id": data_id, **row}] if row else []
    if dataset == "TaiwanStockInfo":
        name = _NAMES.get(data_id)
        return [{"stock_id": data_id, "stock_name": name, "type": "twse"}] if name else []
    return []


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("FINMIND_TOKEN", TOKEN)
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["data_cache_dir"] = str(tmp_path / "cache")
    set_config(config)
    # Only the peer lists a test writes are read.
    monkeypatch.setattr(peers, "BUILTIN_PEERS_PATH", tmp_path / "builtin_peers.json")
    monkeypatch.setattr(peers, "USER_PEERS_PATH", tmp_path / "user_peers.json")
    stub = mock.Mock(side_effect=_fake_api)
    monkeypatch.setattr(client, "_request", stub)
    yield stub
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _datasets(stub) -> Counter:
    return Counter(call.args[0]["dataset"] for call in stub.call_args_list)


# --- chip data -----------------------------------------------------------------------

@pytest.mark.unit
def test_institutional_flows_are_grouped_and_reported_in_lots(api):
    net = chips.institutional_net("2383", "2026-10-16")
    assert round(net.at["2026-10-13", chips.FOREIGN]) == -927
    assert round(net.at["2026-10-13", chips.TRUST]) == -169
    assert round(net.at["2026-10-13", chips.DEALER]) == 70     # self and hedging together
    report = chips.get_institutional_flows("2383.TW", "2026-10-16", 5)
    assert "Unit: lots" in report and "Foreign (外資),-1,889," in report  # 5-day total of the two drops


@pytest.mark.unit
def test_margin_report_gives_changes_and_the_short_to_margin_ratio(api):
    table = chips.margin_table("2383", "2026-10-16")
    assert table.at["2026-10-13", "Margin Change (融資增減)"] == 226
    assert table.at["2026-10-16", "Short/Margin Ratio % (券資比)"] == round(6 / 2903 * 100, 2)
    assert "券資比" in chips.get_margin_short("2383.TW", "2026-10-16")


@pytest.mark.unit
def test_summary_flags_the_high_volume_drops_and_reads_them_out(api):
    report = chips.get_chip_summary("2383.TW", "2026-10-16", 20)
    flagged = [line for line in report.splitlines() if line.startswith("### ")]
    assert [line[4:14] for line in flagged] == ["2026-09-17", "2026-10-13", "2026-10-15"]
    drop = report.split("### 2026-10-13")[1].split("###")[0]
    assert "Largest institutional seller: Foreign (外資) (-927 lots)" in drop
    assert "Margin balance rose +226 lots" in drop and "散戶接刀" in drop
    assert "Securities lending traded: 135 lots" in drop


@pytest.mark.unit
def test_the_rule_follows_the_configured_thresholds(api):
    set_config({"chip_analysis": {"volume_spike_ratio": 2.2, "down_day_pct": 3.0}})
    report = chips.get_chip_summary("2383.TW", "2026-10-16", 20)
    assert [line[4:14] for line in report.splitlines() if line.startswith("### ")] == ["2026-09-17", "2026-10-15"]


@pytest.mark.unit
def test_every_chip_tool_of_a_run_shares_one_request_per_dataset(api):
    for tool in (chips.get_chip_summary, chips.get_institutional_flows, chips.get_margin_short,
                 chips.get_foreign_holding, chips.get_securities_lending):
        tool("2383.TW", "2026-10-16")
    counts = _datasets(api)
    assert counts["TaiwanStockInstitutionalInvestorsBuySell"] == 1
    assert counts["TaiwanStockMarginPurchaseShortSale"] == 1
    assert counts["TaiwanStockShareholding"] == 1
    assert counts["TaiwanStockSecuritiesLending"] == 1


@pytest.mark.unit
def test_a_past_run_sees_nothing_after_its_date(api):
    report = chips.get_chip_summary("2383.TW", "2026-10-14", 20)
    assert "2026-10-15" not in report and "Point-in-time as of: 2026-10-14" in report
    assert "2026-10-15" not in chips.get_foreign_holding("2383.TW", "2026-10-14")


@pytest.mark.unit
def test_a_us_symbol_gets_no_chip_data_and_no_request(api):
    assert router.route_to_vendor("get_chip_summary", "AAPL", "2026-10-16", 20).startswith("NO_DATA_AVAILABLE")
    assert router.route_to_vendor("get_peer_valuation", "AAPL", "2026-10-16").startswith("NO_DATA_AVAILABLE")
    api.assert_not_called()


# --- peer valuation --------------------------------------------------------------------

@pytest.mark.unit
def test_peer_valuation_compares_with_the_peer_median(api):
    peers.BUILTIN_PEERS_PATH.write_text(json.dumps({"2383": ["6274", "6213"]}), encoding="utf-8")
    report = peers.get_peer_valuation("2383.TW", "2026-10-16")
    assert "Peer,6274,台燿,2026-10-16,82.02,20.01,0.46" in report
    assert "PER: subject 80.77, peer median 101.01, premium/discount -20.0%" in report


@pytest.mark.unit
def test_peer_lists_are_layered_and_codes_lose_their_suffix(api):
    peers.BUILTIN_PEERS_PATH.write_text(json.dumps({"2383": ["6274"], "2330": ["2303"]}), encoding="utf-8")
    peers.USER_PEERS_PATH.write_text(json.dumps({"2383.TW": ["6213.TW", "2383"]}), encoding="utf-8")
    assert peers.load_peer_groups() == {"2383": ["6213"], "2330": ["2303"]}
    set_config({"peer_groups": {"2330": ["2454.TW"]}})
    assert peers.load_peer_groups() == {"2383": ["6213"], "2330": ["2454"]}


@pytest.mark.unit
def test_an_unreadable_peer_file_is_skipped(api):
    peers.BUILTIN_PEERS_PATH.write_text(json.dumps({"2383": ["6274"]}), encoding="utf-8")
    peers.USER_PEERS_PATH.write_text("{not json", encoding="utf-8")
    assert peers.load_peer_groups() == {"2383": ["6274"]}


@pytest.mark.unit
def test_a_peer_without_data_is_marked_and_the_rest_are_compared(api):
    set_config({"peer_groups": {"2383": ["6274", "9999"]}})
    report = peers.get_peer_valuation("2383.TW", "2026-10-16")
    assert "Peer,9999,,no data (無資料),n/a,n/a,n/a" in report
    assert "peer median 82.02" in report


@pytest.mark.unit
def test_no_peer_group_is_said_not_raised(api):
    assert "No peer group is configured for 2383" in peers.get_peer_valuation("2383.TW", "2026-10-16")


@pytest.mark.unit
def test_the_shipped_peer_list_names_2383():
    assert peers._read(peers.BUILTIN_PEERS_PATH)["2383"] == ["6274", "6213"]


# --- agents ----------------------------------------------------------------------------

@pytest.mark.unit
def test_the_chip_analyst_files_a_note_for_a_us_stock_without_calling_the_model():
    llm = mock.Mock()
    node = chip_analyst.create_chip_analyst(llm)
    out = node({"company_of_interest": "AAPL", "trade_date": "2026-10-16", "messages": []})
    assert "Taiwan-listed" in out["chip_report"] and "AAPL" in out["chip_report"]
    llm.assert_not_called()
    llm.bind_tools.assert_not_called()


@pytest.mark.unit
def test_peer_valuation_is_offered_to_the_fundamentals_analyst_for_taiwan_only():
    names = lambda tools: {t.name for t in tools}  # noqa: E731
    assert "get_peer_valuation" in names(fundamentals_analyst.offered_tools("2383.TW"))
    assert "get_peer_valuation" not in names(fundamentals_analyst.offered_tools("AAPL"))


@pytest.mark.unit
def test_the_research_manager_gets_the_peer_table_for_taiwan_only(api):
    set_config({"peer_groups": {"2383": ["6274"]}})
    block = peer_valuation_block({"company_of_interest": "2383.TW", "trade_date": "2026-10-16"})
    assert "Peer Valuation (TaiwanStockPER)" in block and "台燿" in block
    assert peer_valuation_block({"company_of_interest": "AAPL", "trade_date": "2026-10-16"}) == ""


# --- CLI -------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_chip_analyst_is_offered_for_taiwan_tickers_only():
    every = list(AnalystType)
    assert AnalystType.CHIPS in filter_analysts_for_asset_type(every, AssetType.STOCK, "2383.TW")
    assert AnalystType.CHIPS not in filter_analysts_for_asset_type(every, AssetType.STOCK, "AAPL")
    assert AnalystType.CHIPS not in filter_analysts_for_asset_type(every, AssetType.CRYPTO)
    # A backtest grid may mix markets: the analyst stays and skips what it cannot serve.
    assert AnalystType.CHIPS in filter_analysts_for_asset_type(every, AssetType.STOCK)


@pytest.mark.unit
def test_naming_the_chip_analyst_for_a_us_ticker_is_refused():
    assert parse_analysts("market,chips", AssetType.STOCK, "2383.TW") == [AnalystType.MARKET, AnalystType.CHIPS]
    with pytest.raises(ValueError, match="Taiwan-listed stocks only"):
        parse_analysts("chips", AssetType.STOCK, "AAPL")
