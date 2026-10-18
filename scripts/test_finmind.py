"""Live check of the FinMind (Taiwan) vendor against the real APIs.

    .venv/bin/python scripts/test_finmind.py

1. Calls every FinMind function for 2330 (TWSE) and 8069 (TPEx) and prints the
   first lines of each answer.
2. Routes AAPL through the default vendor chain and checks it is FinMind that
   declines and yfinance that answers.
3. Puts FinMind's 2330 prices next to Yahoo's 2330.TW and checks columns and values.
4. Checks the 2383 report fixes against figures recomputed by hand from FinMind:
   the snapshot and get_indicators agree on traded-price indicators, the debt
   metrics, and the monthly revenue analysis.

Uses FINMIND_TOKEN when it is set. Without it, this script (and only this
script) calls FinMind anonymously, which FinMind allows at a lower hourly limit;
the vendor itself still requires the token. The cache goes to a temporary
directory, so the run does not touch your own cache. Exits 1 if any check fails.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yfinance as yf  # noqa: E402

from tradingagents.dataflows import router  # noqa: E402
from tradingagents.dataflows.config import get_config, set_config  # noqa: E402
from tradingagents.dataflows.errors import NoMarketDataError  # noqa: E402
from tradingagents.dataflows.vendors import finmind  # noqa: E402
from tradingagents.dataflows.vendors.finmind import client  # noqa: E402

PREVIEW_LINES = 8
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def preview(title: str, text: str, lines: int = PREVIEW_LINES) -> None:
    print(f"\n--- {title} ---")
    body = text.splitlines()
    print("\n".join(body[:lines]) + (f"\n  ... ({len(body) - lines} more lines)" if len(body) > lines else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def use_anonymous_finmind() -> None:
    """Send FinMind requests without a token (test script only)."""
    real_get = requests.get

    def anonymous_get(url, params=None, headers=None, timeout=None):
        return real_get(url, params=params, timeout=timeout)

    client.get_token = lambda: "anonymous"
    client.requests.get = anonymous_get


def last_trading_day() -> str:
    """The latest day FinMind has a 2330 close for (today's may not be in yet)."""
    start = (date.today() - timedelta(days=14)).isoformat()
    prices = client.fetch("TaiwanStockPrice", "2330", start, None, ttl=client.PRICE_TTL_SECONDS)
    return str(prices["date"].max())


# --- 1. every function, 2330 and 8069 --------------------------------------------

def run_functions(end: str) -> None:
    start = (date.fromisoformat(end) - timedelta(days=10)).isoformat()
    past = (date.fromisoformat(end) - timedelta(days=90)).isoformat()
    for symbol in ("2330", "8069"):
        section(f"1. FinMind functions for {symbol}")
        calls = [
            ("get_stock_data", lambda s=symbol: finmind.get_stock_data(s, start, end)),
            ("get_indicators rsi", lambda s=symbol: finmind.get_indicators(s, "rsi", end, 5)),
            ("get_indicators macd", lambda s=symbol: finmind.get_indicators(s, "macd", end, 5)),
            ("get_fundamentals", lambda s=symbol: finmind.get_fundamentals(s)),
            (f"get_fundamentals as of {past}", lambda s=symbol: finmind.get_fundamentals(s, past)),
            ("get_balance_sheet quarterly", lambda s=symbol: finmind.get_balance_sheet(s, "quarterly")),
            ("get_balance_sheet annual", lambda s=symbol: finmind.get_balance_sheet(s, "annual")),
            ("get_cashflow quarterly", lambda s=symbol: finmind.get_cashflow(s, "quarterly")),
            ("get_income_statement quarterly", lambda s=symbol: finmind.get_income_statement(s, "quarterly")),
            ("get_income_statement annual", lambda s=symbol: finmind.get_income_statement(s, "annual")),
            ("get_news", lambda s=symbol: finmind.get_news(s, start, end)),
        ]
        for name, call in calls:
            try:
                out = call()
                preview(f"{symbol} {name}", out)
                check(f"{symbol} {name}", isinstance(out, str) and bool(out.strip()))
            except Exception as exc:
                traceback.print_exc()
                check(f"{symbol} {name}", False, f"{type(exc).__name__}: {exc}")

    section("1b. ticker spellings")
    for raw, expected in [("2330", "2330.TW"), ("8069", "8069.TWO"), ("0050", "0050.TW")]:
        got = client.resolve_taiwan_symbol(raw)
        check(f"resolve {raw}", got == expected, f"{got}")
    try:
        same = {s: finmind.get_stock_data(s, start, end).split("\n", 2)[2]
                for s in ("2330", "2330.TW", "2330.tw")}
        check("2330 / 2330.TW / 2330.tw give the same rows", len(set(same.values())) == 1)
        tpex = {s: finmind.get_stock_data(s, start, end).split("\n", 2)[2] for s in ("8069", "8069.TWO")}
        check("8069 / 8069.TWO give the same rows", len(set(tpex.values())) == 1)
    except Exception as exc:
        check("ticker spellings", False, f"{type(exc).__name__}: {exc}")


# --- 2. AAPL goes to yfinance -------------------------------------------------------

def run_us_fallback(end: str) -> None:
    section("2. AAPL through the default vendor chain")
    start = (date.fromisoformat(end) - timedelta(days=10)).isoformat()
    print("  core_stock_apis =", get_config()["data_vendors"]["core_stock_apis"])

    finmind_impl = router.VENDOR_METHODS["get_stock_data"]["finmind"]
    yahoo_impl = router.VENDOR_METHODS["get_stock_data"]["yfinance"]
    seen = {}

    def spy_finmind(*args, **kwargs):
        try:
            return finmind_impl(*args, **kwargs)
        except Exception as exc:
            seen["finmind"] = exc
            raise

    def spy_yahoo(*args, **kwargs):
        seen["yfinance"] = True
        return yahoo_impl(*args, **kwargs)

    with mock.patch.dict(router.VENDOR_METHODS["get_stock_data"],
                         {"finmind": spy_finmind, "yfinance": spy_yahoo}), \
         mock.patch.object(client, "_request", side_effect=AssertionError("FinMind was called for AAPL")):
        try:
            out = router.route_to_vendor("get_stock_data", "AAPL", start, end)
        except Exception as exc:
            check("AAPL routed", False, f"{type(exc).__name__}: {exc}")
            return
    preview("AAPL get_stock_data (via router)", out)
    check("FinMind declined AAPL with NoMarketDataError, without a request",
          isinstance(seen.get("finmind"), NoMarketDataError), repr(seen.get("finmind")))
    check("yfinance answered AAPL", seen.get("yfinance") is True and out.startswith("# Stock data for AAPL"))

    out = router.route_to_vendor("get_indicators", "AAPL", "rsi", end, 3)
    check("AAPL indicators answered by yfinance", out.startswith("## rsi values"))

    section("2b. 2330.TW with no FINMIND_TOKEN goes to yfinance")
    token = os.environ.pop("FINMIND_TOKEN", None)
    try:
        with mock.patch.object(client, "get_token",
                               side_effect=client.FinMindNotConfiguredError("FINMIND_TOKEN environment variable is not set.")):
            out = router.route_to_vendor("get_stock_data", "2330.TW", start, end)
        preview("2330.TW without a token (via router)", out, 4)
        check("2330.TW falls back to yfinance without a token", out.startswith("# Stock data for 2330.TW"))
    finally:
        if token is not None:
            os.environ["FINMIND_TOKEN"] = token


# --- 3. FinMind 2330 vs Yahoo 2330.TW ----------------------------------------------

def run_price_comparison(end: str) -> None:
    section("3. 2330 prices: FinMind vs Yahoo 2330.TW")
    start = (date.fromisoformat(end) - timedelta(days=21)).isoformat()
    ours = pd.read_csv(pd.io.common.StringIO(finmind.get_stock_data("2330", start, end).split("\n\n", 1)[1]),
                       index_col="Date")

    raw = yf.Ticker("2330.TW").history(start=start, end=(date.fromisoformat(end) + timedelta(days=1)).isoformat(),
                                        auto_adjust=False, actions=False)
    raw.index = raw.index.tz_localize(None).strftime("%Y-%m-%d")
    yahoo = raw[["Open", "High", "Low", "Close", "Volume"]]
    ours = ours[["Open", "High", "Low", "Close", "Volume"]]

    vendor_out = router.VENDOR_METHODS["get_stock_data"]["yfinance"]("2330.TW", start, end)
    vendor_columns = vendor_out.split("\n\n", 1)[1].splitlines()[0]
    our_columns = finmind.get_stock_data("2330", start, end).split("\n\n", 1)[1].splitlines()[0]
    check("same CSV columns as the yfinance vendor", our_columns == vendor_columns,
          f"{our_columns!r} vs {vendor_columns!r}")
    common = ours.index.intersection(yahoo.index)
    only_ours, only_yahoo = sorted(set(ours.index) - set(common)), sorted(set(yahoo.index) - set(common))
    check("same trading days", not only_ours and not only_yahoo,
          f"{len(common)} shared; only FinMind {only_ours}; only Yahoo {only_yahoo}")

    side = pd.concat({"FinMind": ours.loc[common], "Yahoo": yahoo.loc[common]}, axis=1)
    side = side.swaplevel(axis=1)[["Open", "High", "Low", "Close", "Volume"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(side.to_string())

    for col in ["Open", "High", "Low", "Close"]:
        diff = (ours.loc[common, col] - yahoo.loc[common, col]).abs().max()
        check(f"{col} identical", diff < 0.01, f"max difference {diff:.4f}")
    # Volume is not compared for equality: Yahoo's 2330.TW volume leaves out part
    # of the day's trading (it is 10-30% below TWSE's total, and the value traded
    # divided by Yahoo's volume lands above the day's high). FinMind reports TWSE's
    # full-day shares, so it is checked against the value traded that day instead.
    raw_rows = client.fetch("TaiwanStockPrice", "2330", start, end).set_index("date")
    vwap = raw_rows.loc[common, "Trading_money"] / ours.loc[common, "Volume"]
    inside = (vwap >= ours.loc[common, "Low"] - 0.01) & (vwap <= ours.loc[common, "High"] + 0.01)
    check("FinMind Volume matches TWSE value traded (implied average price within Low..High)",
          bool(inside.all()), f"{int(inside.sum())}/{len(inside)} days")
    short = (yahoo.loc[common, "Volume"] / ours.loc[common, "Volume"])
    check("Yahoo Volume never exceeds FinMind's full-day total", bool((short <= 1.0001).all()),
          f"Yahoo reports {short.min():.0%}–{short.max():.0%} of TWSE's shares")

    year_start = (date.fromisoformat(end) - timedelta(days=365)).isoformat()
    ours_year = pd.read_csv(pd.io.common.StringIO(
        finmind.get_stock_data("2330", year_start, end).split("\n\n", 1)[1]), index_col="Date")
    ours_div = ours_year.loc[ours_year["Dividends"] != 0, "Dividends"]
    y_hist = yf.Ticker("2330.TW").history(start=year_start, end=end)
    y_div = y_hist.loc[y_hist["Dividends"] != 0, "Dividends"]
    y_div.index = y_div.index.tz_localize(None).strftime("%Y-%m-%d")
    print("\n  ex-dividend days in the past year (FinMind | Yahoo):")
    for day in sorted(set(ours_div.index) | set(y_div.index)):
        print(f"    {day}: {ours_div.get(day, '-')} | {y_div.get(day, '-')}")
    same_days = set(ours_div.index) == set(y_div.index)
    same_amounts = same_days and all(abs(ours_div[d] - y_div[d]) < 0.01 for d in ours_div.index)
    check("Dividends: same ex-dividend days and amounts as Yahoo", same_amounts)

    # What the yfinance vendor itself reports is dividend-adjusted (auto_adjust),
    # so it can differ from traded prices before an ex-dividend date.
    adjusted = router.VENDOR_METHODS["get_stock_data"]["yfinance"]("2330.TW", start, end)
    adj = pd.read_csv(pd.io.common.StringIO(adjusted.split("\n\n", 1)[1]), index_col=0)
    adj.index = pd.to_datetime(adj.index).strftime("%Y-%m-%d")
    shared = adj.index.intersection(ours.index)
    gap = (adj.loc[shared, "Close"] - ours.loc[shared, "Close"]).abs().max()
    print(f"\n  note: yfinance vendor (dividend-adjusted) vs FinMind (traded) Close, max gap {gap:.2f} TWD")


# --- 4. 2383 report fixes, as of 2026-10-16 ------------------------------------------

def run_2383_acceptance(_end: str) -> None:
    import re

    from tradingagents.dataflows.vendors.yahoo.snapshot import build_verified_market_snapshot

    as_of = "2026-10-16"
    section(f"4. 2383.TW report fixes as of {as_of}")
    snap = build_verified_market_snapshot("2383.TW", as_of, 5)
    preview("2383 verified snapshot", snap, 30)
    check("snapshot prices come from FinMind traded prices", "Price source: FinMind TaiwanStockPrice" in snap)

    # Recomputed by hand from FinMind TaiwanStockPrice closes. SMA200 is
    # 4039.525 exactly, so either rounding (4039.52 / 4039.53) is right.
    expected = {"close_50_sma": 5451.00, "close_200_sma": 4039.525, "rsi": 44.97, "macdh": -11.22,
                "close_10_ema": 5453.43, "boll": 5282.75}
    for name, want in expected.items():
        got = float(re.search(rf"\| {name} \| (-?[\d.]+) \|", snap).group(1))
        line = router.route_to_vendor("get_indicators", "2383.TW", name, as_of, 0).splitlines()[2]
        tool = float(line.split(": ")[1])
        check(f"{name}: snapshot {got} = tool {tool:.4f} = hand-computed {want}",
              abs(got - want) <= 0.006 and abs(tool - want) <= 0.006 and abs(got - round(tool, 2)) < 1e-9)

    fundamentals = finmind.get_fundamentals("2383.TW", as_of)
    preview("2383 fundamentals", fundamentals, 50)
    check("interest-bearing debt to equity is 43.9%",
          "/ Equity): 43.9% (debt 25,603,624,000 TWD)" in fundamentals)
    check("debt ratio is 63.2%", "Debt Ratio (Total Liabilities / Total Assets): 63.2%" in fundamentals)
    check("total liabilities / equity is no longer called debt to equity", "Debt to Equity:" not in fundamentals)
    check("Q3 2026 revenue derived from monthly revenue is 608.85 億",
          "Q3 2026 Revenue (derived from monthly revenue; income statement not yet published): "
          "60,884,517,000 TWD (608.85 億)" in fundamentals)

    heading = "## Monthly Revenue (last 15 months, by the month covered; TWD)\n"
    table_csv = fundamentals.split(heading, 1)[1] if heading in fundamentals else "Revenue Month\n"
    table = pd.read_csv(pd.io.common.StringIO(table_csv), index_col="Revenue Month", dtype=str)
    raw = client.fetch("TaiwanStockMonthRevenue", "2383", "2025-01-01", None)
    raw_by_month = {f"{int(r.revenue_year)}-{int(r.revenue_month):02d}": f"{float(r.revenue):.0f}"
                    for r in raw.itertuples()}
    mismatched = [m for m in table.index if raw_by_month.get(m) != table.at[m, "Revenue (TWD)"]]
    check("15 months of revenue, each matching FinMind's raw month",
          len(table) == 15 and not mismatched, f"{len(table)} months; mismatched {mismatched}")


def main() -> int:
    set_config({"data_cache_dir": tempfile.mkdtemp(prefix="finmind-check-")})
    if os.getenv("FINMIND_TOKEN"):
        print("FinMind: using FINMIND_TOKEN")
    else:
        print("FinMind: FINMIND_TOKEN is not set; this script calls FinMind anonymously")
        use_anonymous_finmind()

    end = last_trading_day()
    print(f"Latest FinMind trading day: {end}")
    for step in (run_functions, run_us_fallback, run_price_comparison, run_2383_acceptance):
        try:
            step(end)
        except Exception as exc:
            traceback.print_exc()
            check(step.__name__, False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if not r[1]]
    section(f"Summary: {len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
