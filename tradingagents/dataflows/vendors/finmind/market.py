"""Taiwan daily prices and technical indicators from FinMind (TaiwanStockPrice).

Prices are as traded (not dividend-adjusted): ``TaiwanStockPriceAdj`` is not
served to every account level, and the traded price is the one a Taiwan
broker shows. Output matches the Yahoo vendor's, so the analysts read the same
report whichever vendor answered.
"""

from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.indicators import (
    check_indicator,
    indicator_values,
    render_indicator_window,
    window_lines,
)
from tradingagents.dataflows.vendors.finmind.client import (
    PRICE_TTL_SECONDS,
    SLOW_TTL_SECONDS,
    fetch,
    require_taiwan_stock,
)
from tradingagents.dataflows.vendors.yahoo.ohlcv import _assert_ohlcv_not_stale

# FinMind column -> the OHLCV name every price vendor reports.
_COLUMNS = {
    "date": "Date",
    "open": "Open",
    "max": "High",
    "min": "Low",
    "close": "Close",
    "Trading_Volume": "Volume",
}

# History fetched for indicators: enough for a 200-day average before any
# window an analyst asks for. Anchored to January 1st so the cached request is
# the same one all year rather than a new one every day.
_HISTORY_YEARS = 5


def load_prices(stock_id: str, symbol: str, earliest: str) -> pd.DataFrame:
    """Daily OHLCV for ``stock_id`` from at least ``earliest`` to today, date-sorted.

    Days with no trade (FinMind reports a zero close) are dropped.
    """
    anchor = f"{date.today().year - _HISTORY_YEARS}-01-01"
    raw = fetch("TaiwanStockPrice", stock_id, min(anchor, earliest), None, ttl=PRICE_TTL_SECONDS)
    if raw.empty or not set(_COLUMNS) <= set(raw.columns):
        raise NoMarketDataError(symbol, stock_id, "no price rows on FinMind")
    data = raw[list(_COLUMNS)].rename(columns=_COLUMNS)
    data["Date"] = pd.to_datetime(data["Date"])
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data[data["Close"] > 0]
    return data.sort_values("Date").reset_index(drop=True)


def _corporate_actions(stock_id: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Yahoo's ``Dividends`` and ``Stock Splits`` columns from FinMind's ex-rights results.

    A cash dividend (息) is its amount per share in TWD; a stock dividend (權) is
    the ratio of the reference prices before and after, as Yahoo reports a
    split. A day with both (權息) is reported whole as ``Dividends``: FinMind
    gives one combined value. 0.0 on every other day, as Yahoo reports them.
    """
    actions = pd.DataFrame(0.0, index=dates, columns=["Dividends", "Stock Splits"])
    events = fetch("TaiwanStockDividendResult", stock_id, f"{dates.min():%Y-%m-%d}",
                   f"{dates.max():%Y-%m-%d}", ttl=SLOW_TTL_SECONDS)
    for row in events.to_dict("records"):
        day = pd.Timestamp(row.get("date"))
        if day not in actions.index:
            continue
        kind = str(row.get("stock_or_cache_dividend") or "")
        before, after = row.get("before_price"), row.get("after_price")
        if kind == "權" and before and after:
            actions.at[day, "Stock Splits"] = round(float(before) / float(after), 6)
        else:
            actions.at[day, "Dividends"] = float(row.get("stock_and_cache_dividend") or 0.0)
    return actions


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """OHLCV for a Taiwan symbol over ``[start_date, end_date]``, as CSV."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    stock_id = require_taiwan_stock(symbol)

    data = load_prices(stock_id, symbol, start_date)
    data = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
    if data.empty:
        raise NoMarketDataError(symbol, stock_id, f"no rows between {start_date} and {end_date}")
    _assert_ohlcv_not_stale(data, end_date, symbol, stock_id)

    data = data.set_index("Date")
    for col in ["Open", "High", "Low", "Close"]:
        data[col] = data[col].round(2)
    data[["Dividends", "Stock Splits"]] = _corporate_actions(stock_id, data.index)

    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n\n"
    return header + data.to_csv()


def load_ohlcv(symbol: str, as_of_date: str, fill_gaps: bool = True) -> pd.DataFrame:
    """Traded daily OHLCV up to ``as_of_date``, shaped like the Yahoo vendor's
    ``load_ohlcv`` (a ``Date`` column, date-sorted).

    FinMind reports no row for a day without a trade, so there are no gaps to
    fill; ``fill_gaps`` is accepted for the same call signature.
    """
    stock_id = require_taiwan_stock(symbol)
    earliest = (datetime.strptime(as_of_date, "%Y-%m-%d") - relativedelta(years=3)).strftime("%Y-%m-%d")
    data = load_prices(stock_id, symbol, earliest)
    data = data[data["Date"] <= as_of_date].reset_index(drop=True)  # no look-ahead
    if data.empty:
        raise NoMarketDataError(symbol, stock_id, f"no price rows on or before {as_of_date}")
    _assert_ohlcv_not_stale(data, as_of_date, symbol, stock_id)
    return data


def get_indicators(symbol: str, indicator: str, as_of_date: str, look_back_days: int) -> str:
    """One technical indicator for a Taiwan symbol over the look-back window."""
    check_indicator(indicator)
    data = load_ohlcv(symbol, as_of_date)
    values = indicator_values(data.copy(), indicator)
    return render_indicator_window(
        indicator, as_of_date, look_back_days, window_lines(values, as_of_date, look_back_days)
    )
