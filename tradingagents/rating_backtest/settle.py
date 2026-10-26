"""Excess returns after an analysis date, from FinMind, with no model.

Entry is the close of the first trading day after the analysis date (the run
reads the market as of that day's close), exit the close N trading days later.
A stock's return adds back what it paid: on each ex-dividend day in the holding
period the price relative is multiplied by ``before_price / after_price`` from
``TaiwanStockDividendResult`` (``TaiwanStockPriceAdj`` is not on the free plan).
The benchmark is the total return index of the stock's market, TAIEX for TWSE
and TPEx for TPEx, so both sides include dividends.

A daily move past the 10% price limit that no dividend explains is another
corporate action (a capital reduction, a split): the cell is flagged, not fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tradingagents.dataflows.vendors.finmind.client import fetch

HORIZONS = (5, 20, 60)
BENCHMARK = {"twse": "TAIEX", "tpex": "TPEx"}
PRICE_LIMIT = 0.105
# Indexed by date strings like every other series here, so it can be compared with a day.
_NO_DIVIDENDS = pd.Series(dtype=float, index=pd.Index([], dtype=object))


@dataclass
class Series:
    """Closes by trading day, and each ex-dividend day's ``before / after`` factor."""
    close: pd.Series
    factor: pd.Series


def trading_days(start: str, end: str) -> list[str]:
    days = fetch("TaiwanStockTradingDate", None, start, end)
    return sorted(d for d in days["date"].astype(str) if start <= d <= end)


def stock_series(stock_id: str, start: str, end: str) -> Series:
    prices = fetch("TaiwanStockPrice", stock_id, start, end)
    prices = prices[prices["Trading_Volume"].astype(float) > 0] if not prices.empty else prices
    close = (pd.Series(prices["close"].astype(float).to_numpy(), index=prices["date"].astype(str)).sort_index()
             if not prices.empty else pd.Series(dtype=float, index=pd.Index([], dtype=object)))
    dividends = fetch("TaiwanStockDividendResult", stock_id, start, end)
    factor = _NO_DIVIDENDS
    if not dividends.empty:
        factor = pd.Series((dividends["before_price"].astype(float) / dividends["after_price"].astype(float)).to_numpy(),
                           index=dividends["date"].astype(str))
    return Series(close, factor)


def index_series(market: str, start: str, end: str) -> pd.Series:
    rows = fetch("TaiwanStockTotalReturnIndex", BENCHMARK[market], start, end)
    return pd.Series(rows["price"].astype(float).to_numpy(), index=rows["date"].astype(str)).sort_index()


def _close_on(close: pd.Series, day: str) -> tuple[float | None, str | None]:
    """The last close on or before ``day`` (a suspended stock keeps its last price)."""
    known = close[close.index <= day]
    return (float(known.iloc[-1]), str(known.index[-1])) if not known.empty else (None, None)


def holding_return(series: Series, entry: str, exit_: str) -> tuple[float | None, list[str]]:
    """The total return from the close of ``entry`` to that of ``exit_``, and flags."""
    start, start_day = _close_on(series.close, entry)
    end, end_day = _close_on(series.close, exit_)
    if start is None or start_day != entry:
        return None, ["no_entry_price"]
    flags = [] if end_day == exit_ else ["stale_exit"]
    held = series.factor[(series.factor.index > entry) & (series.factor.index <= exit_)]
    window = series.close[(series.close.index >= entry) & (series.close.index <= exit_)]
    moves = window.pct_change().dropna()
    explained = set(held.index)
    if any(abs(m) > PRICE_LIMIT and day not in explained for day, m in moves.items()):
        flags.append("unexplained_jump")
    return end / start * float(held.prod()) - 1, flags


def momentum(series: Series, index: pd.Series, known_days: list[str], days_back: int = 20) -> float | None:
    """The excess return over the ``days_back`` trading days up to the last of ``known_days`` (known then)."""
    if len(known_days) <= days_back:
        return None
    begin, end = known_days[-days_back - 1], known_days[-1]
    stock, _ = holding_return(series, begin, end)
    if stock is None or begin not in index.index or end not in index.index:
        return None
    return stock - float(index[end] / index[begin] - 1)


def settle_cells(cells: list[tuple[str, str]], markets: dict[str, str], as_of: str,
                 horizons=HORIZONS) -> list[dict]:
    """Per (ticker, analysis date): entry day, 20-day momentum up to the date, and per horizon
    the stock, benchmark and excess return.

    A horizon whose exit day is after ``as_of`` is left empty (pending).
    """
    first = min(d for _, d in cells)
    start = (pd.Timestamp(first) - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    days = trading_days(start, as_of)
    indices = {m: index_series(m, start, as_of) for m in set(markets.values())}
    series = {}
    rows = []
    for ticker, date in cells:
        stock_id, market = ticker.split(".")[0], markets[ticker]
        if ticker not in series:
            series[ticker] = stock_series(stock_id, start, as_of)
        later = [d for d in days if d > date]
        row = {"ticker": ticker, "date": date, "market": market, "entry": later[0] if later else None,
               "momentum_20": momentum(series[ticker], indices[market], [d for d in days if d <= date])}
        for n in horizons:
            row.update({f"ret_{n}": None, f"bench_{n}": None, f"excess_{n}": None, f"flags_{n}": ""})
            if len(later) <= n:
                continue
            entry, exit_ = later[0], later[n]
            stock, flags = holding_return(series[ticker], entry, exit_)
            index = indices[market]
            if stock is None or entry not in index.index or exit_ not in index.index:
                row[f"flags_{n}"] = ";".join(flags or ["no_benchmark"])
                continue
            bench = float(index[exit_] / index[entry] - 1)
            row.update({f"ret_{n}": stock, f"bench_{n}": bench, f"excess_{n}": stock - bench,
                        f"flags_{n}": ";".join(flags)})
        rows.append(row)
    return rows
