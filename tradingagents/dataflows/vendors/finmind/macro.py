"""Taiwan market and macro indicators from FinMind, for the news analyst's macro tool.

Each indicator is a daily series dated by its Taipei trading day, served up to
the analysis date. A day's row is published after the market closes, at
``published`` (Taipei): the run's own date counts only when ``as_of_time`` is at
or after it, otherwise the series ends the day before, so a run held to the
13:30 close cannot read that evening's margin balances.

Only series a run can date are served. Left out: policy rates (FinMind's
InterestRate has no Taiwan central bank, and its Fed rows lag), government bond
yields (US-dated, published after Taipei's day ends) and the business
indicator (a paid dataset, published a month late). An indicator this vendor
does not serve is handed to the next one (FRED).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from tradingagents.dataflows.date_window import as_of_time, is_historical
from tradingagents.dataflows.errors import NoMarketDataError, UnsupportedSymbolError
from tradingagents.dataflows.vendors.finmind.client import PRICE_TTL_SECONDS, fetch

DEFAULT_LOOKBACK_DAYS = 60
MAX_ROWS = 40


@dataclass(frozen=True)
class Indicator:
    title: str
    dataset: str
    data_id: str | None
    published: time        # Taipei time a day's row is out
    unit: str


INDICATORS = {
    "twd_usd": Indicator("USD/TWD exchange rate (Bank of Taiwan board rate)", "TaiwanExchangeRate", "USD",
                         time(16, 0), "TWD per USD"),
    "taiex": Indicator("TAIEX (TWSE Capitalization Weighted Stock Index)", "TaiwanStockPrice", "TAIEX",
                       time(14, 0), "index points; turnover in TWD 100 million (億)"),
    "tpex": Indicator("TPEx index (Taipei Exchange)", "TaiwanStockPrice", "TPEx",
                      time(14, 0), "index points; turnover in TWD 100 million (億)"),
    "market_institutional": Indicator("TWSE market-wide institutional net buy/sell", "TaiwanStockTotalInstitutionalInvestors",
                                      None, time(15, 30), "TWD 100 million (億), net buy positive"),
    "market_margin": Indicator("TWSE market-wide margin purchase and short sale balances",
                               "TaiwanStockTotalMarginPurchaseShortSale", None, time(21, 30),
                               "lots (張); margin money in TWD 100 million (億)"),
}

# Names a model may reach for, mapped to the indicators above.
ALIASES = {
    "usd_twd": "twd_usd", "twd": "twd_usd", "exchange_rate": "twd_usd",
    "taiwan_index": "taiex", "twse": "taiex", "otc": "tpex", "tpex_index": "tpex",
    "institutional": "market_institutional", "foreign_investors": "market_institutional",
    "margin": "market_margin", "margin_balance": "market_margin",
}


def _resolve(indicator: str) -> str:
    key = str(indicator or "").strip().lower()
    key = ALIASES.get(key, key)
    if key not in INDICATORS:
        raise UnsupportedSymbolError(
            indicator, indicator, f"not a FinMind Taiwan indicator ({', '.join(INDICATORS)})")
    return key


def last_known_day(as_of_date: str, published: time) -> date:
    """The latest trading day a run dated ``as_of_date`` can read: the date itself
    once ``as_of_time`` reaches the day's publication, else the day before."""
    day = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    return day if as_of_time() >= published else day - timedelta(days=1)


def _rows(spec: Indicator, start: str, as_of_date: str) -> pd.DataFrame:
    historical = is_historical(as_of_date)
    rows = fetch(spec.dataset, spec.data_id, start, as_of_date if historical else None,
                 ttl=PRICE_TTL_SECONDS)
    if rows.empty or "date" not in rows:
        return rows
    cutoff = last_known_day(as_of_date, spec.published).isoformat()
    return rows[rows["date"].astype(str).str[:10] <= cutoff]


def _hundred_million(series: pd.Series) -> pd.Series:
    return (series.astype(float) / 1e8).round(2)


def _table(key: str, rows: pd.DataFrame) -> pd.DataFrame:
    """The indicator's daily table, oldest first."""
    if key == "twd_usd":
        table = rows[["date", "spot_buy", "spot_sell", "cash_buy", "cash_sell"]].copy()
    elif key in ("taiex", "tpex"):
        table = rows[["date", "close", "spread"]].copy()
        table["turnover_億"] = _hundred_million(rows["Trading_money"])
        table = table.rename(columns={"spread": "change"})
    elif key == "market_institutional":
        net = rows.assign(net=rows["buy"].astype(float) - rows["sell"].astype(float))
        table = net.pivot_table(index="date", columns="name", values="net", aggfunc="sum")
        table = (table / 1e8).round(2).reset_index()
        table.columns.name = None
    else:  # market_margin
        wide = rows.pivot_table(index="date", columns="name", values="TodayBalance", aggfunc="sum")
        wide.columns.name = None
        table = pd.DataFrame({"date": wide.index})
        for name, label, scale in (("MarginPurchase", "margin_balance_張", 1),
                                   ("ShortSale", "short_balance_張", 1),
                                   ("MarginPurchaseMoney", "margin_money_億", 1e8)):
            if name in wide:
                table[label] = (wide[name].to_numpy() / scale).round(2)
    return table.sort_values("date").reset_index(drop=True)


def get_macro_data(indicator: str, as_of_date: str, look_back_days: int | None = None) -> str:
    """A Taiwan market indicator as a markdown report, as known by ``as_of_date``."""
    key = _resolve(indicator)
    spec = INDICATORS[key]
    look_back_days = DEFAULT_LOOKBACK_DAYS if look_back_days is None else int(look_back_days)
    start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    rows = _rows(spec, start, as_of_date)
    if rows.empty:
        raise NoMarketDataError(key, key, f"no {spec.dataset} rows from {start} to {as_of_date}")
    table = _table(key, rows)
    cutoff = last_known_day(as_of_date, spec.published)
    first, last = table.iloc[0], table.iloc[-1]
    # A day's own change is not a level: it has no change over the window.
    value_cols = [c for c in table.columns if c not in ("date", "change")]
    changes = ", ".join(
        f"{c} {last[c]} ({float(last[c]) - float(first[c]):+,.2f} since {first['date']})"
        for c in value_cols if pd.notna(last[c]) and pd.notna(first[c]))
    note = ("" if cutoff.isoformat() == as_of_date else
            f" {as_of_date}'s row is published at {spec.published:%H:%M} Taipei, after as_of_time, so the "
            f"series ends on {cutoff}.")
    return (
        f"## FinMind: {spec.title} ({key})\n"
        f"- Source: FinMind {spec.dataset}" + (f" ({spec.data_id})" if spec.data_id else "") + "\n"
        f"- Units: {spec.unit}\n"
        f"- Frequency: daily, by Taipei trading day; a day's row is out at {spec.published:%H:%M} Taipei\n"
        f"- Window: {start} to {as_of_date} (point-in-time: rows up to {cutoff}).{note}\n"
        f"- Latest ({last['date']}): {changes}\n\n"
        + table.tail(MAX_ROWS).to_csv(index=False)
    )
