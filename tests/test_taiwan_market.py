"""Taiwan symbols across the pipeline: one set of prices for every agent, no
US-only sources, stated data gaps, and a log of every vendor fallback.

Offline: vendors and fetchers are stubbed.
"""

import copy
import logging
from unittest import mock

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import tradingagents.default_config as default_config
from tradingagents.agents.analysts import news_analyst, sentiment_analyst
from tradingagents.agents.context import NEWS_NUMBERS_RULE
from tradingagents.dataflows import ohlcv, router
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError, UnsupportedSymbolError
from tradingagents.dataflows.symbols import is_taiwan_symbol
from tradingagents.dataflows.vendors.yahoo import snapshot


@pytest.fixture(autouse=True)
def defaults():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    yield
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _prices(close=100.0):
    return pd.DataFrame({
        "Date": pd.bdate_range(end="2026-10-16", periods=260),
        "Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000,
    })


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("2383.TW", True), ("8069.TWO", True), ("2330", True), ("AAPL", False), ("0700.HK", False),
])
def test_taiwan_symbols_are_recognised(raw, expected):
    assert is_taiwan_symbol(raw) is expected


# --- problem 1: one set of prices ---------------------------------------------------

def _loaders(finmind_close, yahoo_close):
    def finmind(symbol, as_of_date, fill_gaps=True):
        if not is_taiwan_symbol(symbol):
            raise UnsupportedSymbolError(symbol, symbol, "not Taiwan")
        return _prices(finmind_close)

    return {"finmind": finmind, "yfinance": lambda *a, **k: _prices(yahoo_close)}


@pytest.mark.unit
def test_the_snapshot_prices_a_taiwan_symbol_from_finmind(monkeypatch):
    monkeypatch.setattr(ohlcv, "LOADERS", _loaders(finmind_close=200.0, yahoo_close=150.0))
    out = snapshot.build_verified_market_snapshot("2383.TW", "2026-10-16", 5)
    assert "| Close | 200.00 |" in out
    assert "Price source: FinMind TaiwanStockPrice, traded prices (not dividend-adjusted)" in out


@pytest.mark.unit
def test_the_snapshot_still_prices_a_us_symbol_from_yahoo(monkeypatch):
    monkeypatch.setattr(ohlcv, "LOADERS", _loaders(finmind_close=200.0, yahoo_close=150.0))
    out = snapshot.build_verified_market_snapshot("AAPL", "2026-10-16", 5)
    assert "| Close | 150.00 |" in out
    assert "Price source: Yahoo Finance, dividend- and split-adjusted prices" in out


@pytest.mark.unit
def test_a_taiwan_snapshot_falls_back_to_yahoo_and_logs_why(monkeypatch, caplog):
    def finmind_down(*a, **k):
        raise NoMarketDataError("2383.TW", "2383", "no price rows on FinMind")

    monkeypatch.setattr(ohlcv, "LOADERS", {"finmind": finmind_down, "yfinance": lambda *a, **k: _prices(150.0)})
    with caplog.at_level(logging.INFO, logger="tradingagents.dataflows"):
        out = snapshot.build_verified_market_snapshot("2383.TW", "2026-10-16", 5)
    assert "| Close | 150.00 |" in out
    assert "Vendor 'finmind' could not serve 2383.TW prices" in caplog.text


# --- problem 2: fallback log ----------------------------------------------------------

@pytest.mark.unit
def test_a_taiwan_fallback_is_logged_and_a_us_hand_over_is_not(caplog):
    def finmind(symbol, *a, **k):
        if is_taiwan_symbol(symbol):
            raise NoMarketDataError(symbol, symbol, "no rows on FinMind")
        raise UnsupportedSymbolError(symbol, symbol, "not a Taiwan-listed symbol")

    stubs = {"finmind": finmind, "yfinance": lambda *a, **k: "yahoo prices"}
    with mock.patch.dict(router.VENDOR_METHODS["get_stock_data"], stubs), \
         caplog.at_level(logging.INFO, logger="tradingagents.dataflows"):
        router.route_to_vendor("get_stock_data", "2383.TW", "2026-10-01", "2026-10-16")
        router.route_to_vendor("get_stock_data", "AAPL", "2026-10-01", "2026-10-16")
    assert "Vendor 'finmind' has no data for get_stock_data(2383.TW): " in caplog.text
    assert "get_stock_data(2383.TW) served by yfinance" in caplog.text
    assert "get_stock_data(AAPL) served by yfinance" in caplog.text
    assert "finmind' has no data for get_stock_data(AAPL)" not in caplog.text


# --- problem 5: no SEC EDGAR for Taiwan ------------------------------------------------

@pytest.mark.unit
def test_sec_edgar_is_not_asked_about_a_taiwan_symbol():
    served = []

    def no_data(name):
        def impl(*a, **k):
            served.append(name)
            raise NoMarketDataError("x", "x", "none")
        return impl

    chain = {name: no_data(name) for name in router.VENDOR_METHODS["get_balance_sheet"]}
    with mock.patch.dict(router.VENDOR_METHODS, {"get_balance_sheet": chain}):
        router.route_to_vendor("get_balance_sheet", "2383.TW", "quarterly", None)
        assert served == ["finmind", "yfinance"]
        served.clear()
        router.route_to_vendor("get_balance_sheet", "AAPL", "quarterly", None)
        assert served == ["finmind", "sec_edgar", "yfinance"]


@pytest.mark.unit
def test_a_chain_of_us_only_vendors_reports_no_data_for_taiwan():
    set_config({"tool_vendors": {"get_balance_sheet": "sec_edgar"}})
    sec = mock.Mock(side_effect=AssertionError("asked"))
    with mock.patch.dict(router.VENDOR_METHODS["get_balance_sheet"], {"sec_edgar": sec}):
        out = router.route_to_vendor("get_balance_sheet", "2383.TW", "quarterly", None)
    assert out.startswith("NO_DATA_AVAILABLE") and "Taiwan-listed" in out


# --- problems 2, 3, 8: sentiment analyst -----------------------------------------------

class _CapturingLLM:
    def __init__(self):
        self.prompts = []

    def with_structured_output(self, *a, **k):
        raise NotImplementedError

    def invoke(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        return AIMessage(content="report")


def _run_sentiment(monkeypatch, ticker, news):
    calls = []
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: news)
    monkeypatch.setattr(sentiment_analyst, "jev_screen", lambda t: calls.append("jev"))
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages",
                        lambda *a, **k: calls.append("stocktwits") or "st posts")
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts",
                        lambda *a, **k: calls.append("reddit") or "rd posts")
    llm = _CapturingLLM()
    sentiment_analyst.create_sentiment_analyst(llm)(
        {"company_of_interest": ticker, "trade_date": "2026-10-16", "messages": []})
    return calls, llm.prompts[-1]


# Six different stories, enough for a Taiwan sentiment read.
_TW_NEWS = "## 2383.TW News (FinMind), from x:\n\n" + "".join(
    f"### {title} (source: S)\nLink: l\n\n" for title in (
        "台光電擴產泰國新廠", "外資調升台光電目標價", "CCL漲價潮延續", "AI伺服器拉貨強勁",
        "台光電九月營收創高", "銅箔基板族群走弱"))


@pytest.mark.unit
def test_taiwan_sentiment_skips_us_social_sources_and_says_so(monkeypatch):
    calls, prompt = _run_sentiment(monkeypatch, "2383.TW", _TW_NEWS)
    assert calls == []
    assert "6 headlines in the window, 6 independent after de-duplication" in prompt
    assert prompt.count("此來源不適用台股") >= 3          # both blocks and the rule
    assert "Do not infer any sentiment" in prompt
    assert "News headlines — FinMind" in prompt and "Yahoo Finance, past 7 days" not in prompt
    assert NEWS_NUMBERS_RULE in prompt


@pytest.mark.unit
def test_taiwan_sentiment_with_too_few_stories_calls_no_model(monkeypatch):
    news = "## 2383.TW News (FinMind), from x:\n\n" + "".join(
        f"### {title} (source: S)\nLink: l\n\n" for title in (
            "台光電擴產泰國新廠 - 經濟日報", "台光電擴產泰國新廠 - money.udn.com",   # one story, twice
            "2383 台光電 - 股市爆料同學會", "2383 台光電 - -202 / -66 / +9 - 股",      # entry pages: one
            "外資調升台光電目標價"))
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: news)
    llm = _CapturingLLM()
    out = sentiment_analyst.create_sentiment_analyst(llm)(
        {"company_of_interest": "2383.TW", "trade_date": "2026-10-16", "messages": []})
    assert llm.prompts == []
    report = out["sentiment_report"]
    assert report.startswith("**Overall Sentiment:** **Insufficient Data** (Score: n/a)")
    assert "5 headlines in the window, 3 independent after de-duplication (2 entry/discussion pages" in report
    assert "not a neutral reading" in report
    gap = [c for c in out["data_checks"] if c["tool"] == "sentiment"]
    assert gap == [{"analyst": "social", "tool": "sentiment", "status": "insufficient",
                    "detail": gap[0]["detail"]}] and "Insufficient Data" in gap[0]["detail"]
    news = [c for c in out["data_checks"] if c["tool"] == "news"]
    assert news[0]["status"] == "ok" and news[0]["scored"] is False


@pytest.mark.unit
def test_the_story_minimum_follows_the_config(monkeypatch):
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["sentiment_min_items"] = 7
    set_config(config)
    monkeypatch.setattr(sentiment_analyst.get_news, "func", lambda *a: _TW_NEWS)
    llm = _CapturingLLM()
    out = sentiment_analyst.create_sentiment_analyst(llm)(
        {"company_of_interest": "2383.TW", "trade_date": "2026-10-16", "messages": []})
    assert llm.prompts == [] and "Insufficient Data" in out["sentiment_report"]


@pytest.mark.unit
def test_us_sentiment_still_reads_stocktwits_and_reddit(monkeypatch):
    calls, prompt = _run_sentiment(monkeypatch, "NVDA", "## NVDA News, from x:\n\n### t")
    assert calls == ["jev", "stocktwits", "reddit"]
    assert "st posts" in prompt and "rd posts" in prompt
    assert "independent after de-duplication" not in prompt     # US runs are not counted
    assert "此來源不適用台股" not in prompt
    assert "News headlines — Yahoo Finance, past 7 days" in prompt
    assert NEWS_NUMBERS_RULE in prompt


# --- problems 4, 8: news analyst and FRED -----------------------------------------------

def _news_prompt(monkeypatch):
    seen = {}

    def take_turn(prompt, llm, tools, messages):
        seen["tools"] = [t.name for t in tools]
        seen["prompt"] = "\n".join(str(m.content) for m in prompt.format_messages(messages=messages))
        return AIMessage(content="r"), "r"

    monkeypatch.setattr(news_analyst, "take_turn", take_turn)
    news_analyst.create_news_analyst(object())(
        {"company_of_interest": "2383.TW", "trade_date": "2026-10-16", "messages": []})
    return seen


@pytest.mark.unit
def test_without_a_fred_key_or_a_finmind_token_the_macro_tool_is_withheld_and_the_gap_stated(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("FINMIND_TOKEN", raising=False)
    seen = _news_prompt(monkeypatch)
    assert "get_macro_indicators" not in seen["tools"]
    assert "macroeconomic data (FRED) is missing for this run" in seen["prompt"]
    assert NEWS_NUMBERS_RULE in seen["prompt"]


@pytest.mark.unit
def test_a_taiwan_run_with_finmind_gets_taiwan_macro_and_is_told_fred_is_missing(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("FINMIND_TOKEN", "t")
    seen = _news_prompt(monkeypatch)
    assert "get_macro_indicators" in seen["tools"]
    assert "Taiwan macro data (FinMind) is available while US macro data (FRED) is missing" in seen["prompt"]
    assert "'taiex', 'tpex', 'twd_usd'" in seen["prompt"] and "'cpi'" not in seen["prompt"]
    assert "macroeconomic data (FRED) is missing for this run" not in seen["prompt"]


@pytest.mark.unit
def test_with_a_fred_key_the_macro_tool_is_offered(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    seen = _news_prompt(monkeypatch)
    assert "get_macro_indicators" in seen["tools"]
    assert "is missing for this run" not in seen["prompt"]


@pytest.mark.unit
def test_the_cli_warns_before_a_run_without_a_fred_key(monkeypatch):
    from cli import run
    from cli.models import AnalystType

    printed = []
    monkeypatch.setattr(run.console, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    run._warn_missing_fred_key({"analysts": [AnalystType.NEWS]}, config)
    assert printed and "FRED_API_KEY is not set" in printed[0]
    assert "https://fred.stlouisfed.org/docs/api/api_key.html" in printed[0]

    printed.clear()
    run._warn_missing_fred_key({"analysts": [AnalystType.MARKET]}, config)   # no news analyst
    monkeypatch.setenv("FRED_API_KEY", "k")
    run._warn_missing_fred_key({"analysts": [AnalystType.NEWS]}, config)
    assert printed == []


@pytest.mark.unit
def test_the_cli_writes_vendor_choices_to_the_run_directory(tmp_path):
    from cli import run

    run._log_vendors_to(tmp_path / "data_vendors.log")
    logging.getLogger("tradingagents.dataflows.router").info("get_news(2383.TW) served by finmind")
    run._vendor_log_handler.flush()
    assert "served by finmind" in (tmp_path / "data_vendors.log").read_text(encoding="utf-8")
    run._log_vendors_to(tmp_path / "second.log")   # a later run replaces the handler
    logging.getLogger("tradingagents.dataflows.router").info("second run")
    run._vendor_log_handler.flush()
    assert "second run" not in (tmp_path / "data_vendors.log").read_text(encoding="utf-8")
    logging.getLogger("tradingagents.dataflows").removeHandler(run._vendor_log_handler)
    run._vendor_log_handler.close()
    run._vendor_log_handler = None
    logging.getLogger("tradingagents.dataflows").setLevel(logging.NOTSET)


# --- report header shows the chain actually used ----------------------------------------

@pytest.mark.unit
def test_the_chain_for_a_taiwan_symbol_leaves_out_us_only_vendors():
    assert router.vendor_chain_for("finmind,sec_edgar,yfinance", "2383.TW") == ["finmind", "yfinance"]
    assert router.vendor_chain_for("polymarket", "2383.TW") == []
    assert router.vendor_chain_for("finmind,sec_edgar,yfinance", "AAPL") == ["finmind", "sec_edgar", "yfinance"]
    assert router.vendor_chain_for("polymarket", "AAPL") == ["polymarket"]


@pytest.mark.unit
@pytest.mark.parametrize("ticker, fundamentals, prediction", [
    ("2383.TW", "fundamental_data finmind,yfinance", "prediction_markets none (此來源不適用台股)"),
    ("AAPL", "fundamental_data finmind,sec_edgar,yfinance", "prediction_markets polymarket"),
])
def test_the_markdown_report_header_shows_the_chain_used_for_the_ticker(ticker, fundamentals, prediction):
    from tradingagents.reporting import _header

    settings = {"data_vendors": default_config.DEFAULT_CONFIG["data_vendors"], "tool_vendors": {}}
    line = next(row for row in _header(ticker, {}, settings).splitlines() if row.startswith("- Data vendors:"))
    assert fundamentals in line and prediction in line


@pytest.mark.unit
def test_the_html_report_header_shows_the_chain_used_for_the_ticker():
    import re

    from tradingagents.report_html import _run_details

    settings = {"data_vendors": default_config.DEFAULT_CONFIG["data_vendors"], "tool_vendors": {}}
    taiwan = re.sub(r"<[^>]+>", " ", dict(_run_details({}, settings, "2383.TW"))["Data vendors"])
    us = re.sub(r"<[^>]+>", " ", dict(_run_details({}, settings, "AAPL"))["Data vendors"])
    assert "sec_edgar" not in taiwan and "polymarket" not in taiwan and "此來源不適用台股" in taiwan
    assert "sec_edgar" in us and "polymarket" in us


# --- Polymarket is not offered for Taiwan -----------------------------------------------

def _news_run(monkeypatch, ticker):
    seen = {}

    def take_turn(prompt, llm, tools, messages):
        seen["tools"] = [t.name for t in tools]
        seen["prompt"] = "\n".join(str(m.content) for m in prompt.format_messages(messages=messages))
        return AIMessage(content="r"), "r"

    monkeypatch.setattr(news_analyst, "take_turn", take_turn)
    news_analyst.create_news_analyst(object())(
        {"company_of_interest": ticker, "trade_date": "2026-10-16", "messages": []})
    return seen


@pytest.mark.unit
def test_polymarket_is_withheld_for_a_taiwan_symbol_and_the_report_told_why(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    seen = _news_run(monkeypatch, "2383.TW")
    assert "get_prediction_markets" not in seen["tools"]
    assert "get_prediction_markets(topic" not in seen["prompt"]
    assert "Polymarket 此來源不適用台股" in seen["prompt"]


@pytest.mark.unit
def test_polymarket_is_still_offered_for_a_us_symbol(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    seen = _news_run(monkeypatch, "AAPL")
    assert seen["tools"] == ["get_news", "get_global_news", "get_macro_indicators", "get_prediction_markets"]
    assert "get_prediction_markets(topic, limit)" in seen["prompt"]
    assert "此來源不適用台股" not in seen["prompt"]
