"""Taiwan chip data (籌碼面) from FinMind: who is buying and selling a listing.

- ``TaiwanStockInstitutionalInvestorsBuySell``: the three institutional
  investor groups' daily buys and sells, reported in shares;
- ``TaiwanStockMarginPurchaseShortSale``: margin purchase and short sale
  balances, reported in lots (張);
- ``TaiwanStockShareholding``: foreign investors' share of the listing;
- ``TaiwanStockSecuritiesLending``: securities lending trades, one row per trade.

Every report is in lots (1 lot = 1,000 shares), so the institutional flows and
the margin balances read on one scale. Each dataset is asked for over the same
window ending on the run's date (``_window``), whichever tool asks first, so the
tools of one run share one cached answer per dataset and FinMind's hourly cap is
spent once.

The data is published after each day's close, as the closing price is, so a
run dated on a day sees that day's figures, as it sees that day's close.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.finmind.client import (
    PRICE_TTL_SECONDS,
    fetch,
    require_taiwan_stock,
)
from tradingagents.dataflows.vendors.finmind.market import load_prices

# Calendar days asked for: about 80 trading days, room for the longest view.
_WINDOW_DAYS = 120
MAX_DAYS = 60

# FinMind investor name -> the group it is reported under.
_GROUPS = {
    "Foreign_Investor": "Foreign (外資)",
    "Foreign_Dealer_Self": "Foreign (外資)",
    "Investment_Trust": "Investment Trust (投信)",
    "Dealer_self": "Dealer (自營商)",
    "Dealer_Hedging": "Dealer (自營商)",
}
FOREIGN, TRUST, DEALER = "Foreign (外資)", "Investment Trust (投信)", "Dealer (自營商)"
_GROUP_ORDER = [FOREIGN, TRUST, DEALER]
TOTAL = "Total (三大法人合計)"

_DEFAULTS = {"lookback_days": 20, "volume_spike_ratio": 1.5, "down_day_pct": 3.0}

_LOTS_NOTE = "# Unit: lots (張, 1 lot = 1,000 shares). Positive = net buy / increase.\n"


def _settings() -> dict:
    return {**_DEFAULTS, **(get_config().get("chip_analysis") or {})}


def _day(as_of_date: str | None) -> str:
    return as_of_date if is_historical(as_of_date) else date.today().isoformat()


def _days(days) -> int:
    try:
        return max(1, min(int(days), MAX_DAYS))
    except (TypeError, ValueError):
        return int(_settings()["lookback_days"])


def _rows(dataset: str, stock_id: str, as_of_date: str | None) -> pd.DataFrame:
    """``dataset`` rows for the shared window ending on the run's date, ``date`` as YYYY-MM-DD."""
    day = _day(as_of_date)
    start = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=_WINDOW_DAYS)).strftime("%Y-%m-%d")
    frame = fetch(dataset, stock_id, start, day if is_historical(as_of_date) else None, ttl=PRICE_TTL_SECONDS)
    if frame.empty or "date" not in frame:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = frame["date"].astype(str).str[:10]
    return frame[frame["date"] <= day]  # no look-ahead


def _header(title: str, symbol: str, as_of_date: str | None) -> str:
    header = f"# {title} for {symbol.upper()}\n"
    if is_historical(as_of_date):
        header += f"# Point-in-time as of: {as_of_date}\n"
    return header


def _lots(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:+,.0f}"


# --- institutional investors ------------------------------------------------------

def institutional_net(stock_id: str, as_of_date: str | None) -> pd.DataFrame:
    """Net buys (lots) by date and investor group, with their total; empty when FinMind has none."""
    rows = _rows("TaiwanStockInstitutionalInvestorsBuySell", stock_id, as_of_date)
    if rows.empty or not {"name", "buy", "sell"} <= set(rows.columns):
        return pd.DataFrame()
    rows = rows.assign(group=rows["name"].map(_GROUPS)).dropna(subset=["group"])
    rows["net"] = (pd.to_numeric(rows["buy"], errors="coerce")
                   - pd.to_numeric(rows["sell"], errors="coerce")) / 1000
    net = rows.pivot_table(index="date", columns="group", values="net", aggfunc="sum")
    net = net.reindex(columns=_GROUP_ORDER).fillna(0.0).sort_index()
    net[TOTAL] = net[_GROUP_ORDER].sum(axis=1)
    net.columns.name = None
    return net


def _streak(series: pd.Series) -> str:
    """How many latest days in a row the series was a net buy, or a net sell."""
    values = [v for v in series.tolist() if v != 0]
    if not values:
        return "no net trades"
    sign = values[-1] > 0
    count = 0
    for value in reversed(series.tolist()):
        if value == 0 or (value > 0) != sign:
            break
        count += 1
    return f"net {'buy' if sign else 'sell'} {count} trading day(s) in a row"


def get_institutional_flows(symbol: str, as_of_date: str | None = None, days: int = 20) -> str:
    """The three institutional groups' daily net buys, with 5-day and N-day totals."""
    stock_id = require_taiwan_stock(symbol)
    days = _days(days)
    net = institutional_net(stock_id, as_of_date)
    if net.empty:
        raise NoMarketDataError(symbol, stock_id, "no institutional investor rows on FinMind")
    shown = net.tail(days)
    lines = [
        f"Window: {shown.index[0]} to {shown.index[-1]} ({len(shown)} trading days)",
        f"Foreign (外資) streak: {_streak(net[FOREIGN])}",
        f"Investment Trust (投信) streak: {_streak(net[TRUST])}",
        "",
        "## Cumulative net (累計買賣超)",
        f"Group,Last 5 days,Window ({len(shown)} days)",
    ]
    lines += [f"{col},{_lots(net[col].tail(5).sum())},{_lots(shown[col].sum())}" for col in [*_GROUP_ORDER, TOTAL]]
    table = shown.round(0).astype(int)
    table.index.name = "Date"
    return (
        _header("Institutional investors net buy/sell (三大法人買賣超)", symbol, as_of_date)
        + _LOTS_NOTE
        + "# FinMind reports shares; converted to lots here. Foreign includes foreign dealers;"
        " Dealer includes hedging.\n\n"
        + "\n".join(lines)
        + f"\n\n## Daily net buy/sell (oldest first)\n{table.to_csv()}"
    )


# --- margin purchase and short sale ------------------------------------------------

_MARGIN_COLUMNS = {
    "MarginPurchaseTodayBalance", "MarginPurchaseYesterdayBalance",
    "ShortSaleTodayBalance", "ShortSaleYesterdayBalance",
}


def margin_table(stock_id: str, as_of_date: str | None) -> pd.DataFrame:
    """Margin and short balances (lots) and their daily changes by date; empty when FinMind has none."""
    rows = _rows("TaiwanStockMarginPurchaseShortSale", stock_id, as_of_date)
    if rows.empty or not set(rows.columns) >= _MARGIN_COLUMNS:
        return pd.DataFrame()
    rows = rows.drop_duplicates("date", keep="last").set_index("date").sort_index()
    num = lambda col: pd.to_numeric(rows[col], errors="coerce")  # noqa: E731
    table = pd.DataFrame({
        "Margin Balance (融資餘額)": num("MarginPurchaseTodayBalance"),
        "Margin Change (融資增減)": num("MarginPurchaseTodayBalance") - num("MarginPurchaseYesterdayBalance"),
        "Short Balance (融券餘額)": num("ShortSaleTodayBalance"),
        "Short Change (融券增減)": num("ShortSaleTodayBalance") - num("ShortSaleYesterdayBalance"),
    })
    margin = table["Margin Balance (融資餘額)"]
    table["Short/Margin Ratio % (券資比)"] = (table["Short Balance (融券餘額)"] / margin.where(margin > 0) * 100).round(2)
    if "OffsetLoanAndShort" in rows:
        table["Offset (資券互抵)"] = num("OffsetLoanAndShort")
    if "MarginPurchaseLimit" in rows:
        limit = num("MarginPurchaseLimit")
        table["Margin Utilization % (融資使用率)"] = (margin / limit.where(limit > 0) * 100).round(2)
    table.index.name = "Date"
    return table


def get_margin_short(symbol: str, as_of_date: str | None = None, days: int = 20) -> str:
    """Margin purchase and short sale balances, their changes and the short/margin ratio."""
    stock_id = require_taiwan_stock(symbol)
    days = _days(days)
    table = margin_table(stock_id, as_of_date)
    if table.empty:
        raise NoMarketDataError(symbol, stock_id, "no margin purchase / short sale rows on FinMind")
    shown = table.tail(days)
    last = shown.iloc[-1]
    lines = [
        f"Window: {shown.index[0]} to {shown.index[-1]} ({len(shown)} trading days)",
        f"Latest margin balance (融資餘額): {last['Margin Balance (融資餘額)']:,.0f} lots",
        f"Margin change over the window: {_lots(shown['Margin Change (融資增減)'].sum())} lots;"
        f" last 5 days: {_lots(shown['Margin Change (融資增減)'].tail(5).sum())} lots",
        f"Latest short balance (融券餘額): {last['Short Balance (融券餘額)']:,.0f} lots",
        f"Short change over the window: {_lots(shown['Short Change (融券增減)'].sum())} lots",
        f"Latest short/margin ratio (券資比): {last['Short/Margin Ratio % (券資比)']}%",
    ]
    return (
        _header("Margin purchase and short sale (融資融券)", symbol, as_of_date)
        + _LOTS_NOTE + "\n"
        + "\n".join(lines)
        + f"\n\n## Daily (oldest first)\n{shown.to_csv()}"
    )


# --- foreign shareholding ----------------------------------------------------------

def foreign_ratio(stock_id: str, as_of_date: str | None) -> pd.Series:
    """Foreign investors' share of the listing (%) by date; empty when FinMind has none."""
    rows = _rows("TaiwanStockShareholding", stock_id, as_of_date)
    if rows.empty or "ForeignInvestmentSharesRatio" not in rows:
        return pd.Series(dtype=float)
    rows = rows.drop_duplicates("date", keep="last").set_index("date").sort_index()
    return pd.to_numeric(rows["ForeignInvestmentSharesRatio"], errors="coerce").dropna()


def _points(series: pd.Series, n: int) -> str:
    if len(series) < 2:
        return "n/a"
    window = series.tail(n + 1)
    return f"{window.iloc[-1] - window.iloc[0]:+.2f} pp (from {window.iloc[0]:.2f}% on {window.index[0]})"


def get_foreign_holding(symbol: str, as_of_date: str | None = None, days: int = 60) -> str:
    """Foreign investors' shareholding ratio over the window and its change."""
    stock_id = require_taiwan_stock(symbol)
    days = _days(days)
    ratio = foreign_ratio(stock_id, as_of_date)
    if ratio.empty:
        raise NoMarketDataError(symbol, stock_id, "no foreign shareholding rows on FinMind")
    shown = ratio.tail(days)
    lines = [
        f"Latest foreign shareholding ratio (外資持股比例): {shown.iloc[-1]:.2f}% ({shown.index[-1]})",
        f"Change over the last 5 trading days: {_points(ratio, 5)}",
        f"Change over the last 20 trading days: {_points(ratio, 20)}",
        f"Change over the window ({len(shown)} trading days): {_points(ratio, len(shown) - 1)}",
        f"Window high / low: {shown.max():.2f}% / {shown.min():.2f}%",
    ]
    table = shown.rename("Foreign Shareholding %").to_frame()
    table.index.name = "Date"
    return (
        _header("Foreign shareholding (外資持股)", symbol, as_of_date)
        + "# Unit: percent of issued shares held by foreign investors; pp = percentage points.\n\n"
        + "\n".join(lines)
        + f"\n\n## Daily (oldest first)\n{table.to_csv()}"
    )


# --- securities lending ------------------------------------------------------------

def lending_daily(stock_id: str, as_of_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lending trades summed by date, and by date and transaction type; empty when none."""
    rows = _rows("TaiwanStockSecuritiesLending", stock_id, as_of_date)
    if rows.empty or "volume" not in rows:
        return pd.DataFrame(), pd.DataFrame()
    rows = rows.assign(
        volume=pd.to_numeric(rows["volume"], errors="coerce").fillna(0.0),
        fee_rate=pd.to_numeric(rows["fee_rate"], errors="coerce") if "fee_rate" in rows else float("nan"),
        transaction_type=rows["transaction_type"].astype(str) if "transaction_type" in rows else "",
    )
    rows["fee_x_volume"] = rows["fee_rate"] * rows["volume"]
    daily = rows.groupby("date").agg(
        volume=("volume", "sum"), trades=("volume", "size"), fee_x_volume=("fee_x_volume", "sum"))
    daily["Weighted Fee Rate %"] = (daily["fee_x_volume"] / daily["volume"].where(daily["volume"] > 0)).round(2)
    daily = daily.drop(columns="fee_x_volume").rename(
        columns={"volume": "Lending Volume (借券成交張數)", "trades": "Trades"})
    daily.index.name = "Date"
    by_type = rows.pivot_table(index="date", columns="transaction_type", values="volume", aggfunc="sum").fillna(0.0)
    return daily.sort_index(), by_type.sort_index()


def get_securities_lending(symbol: str, as_of_date: str | None = None, days: int = 20) -> str:
    """Securities lending trades by day: volume, number of trades and fee rate."""
    stock_id = require_taiwan_stock(symbol)
    days = _days(days)
    daily, by_type = lending_daily(stock_id, as_of_date)
    header = _header("Securities lending trades (借券成交)", symbol, as_of_date) + "# Unit: lots (張).\n\n"
    if daily.empty:
        # A day without a lending trade has no row, so no rows is an answer, not missing data.
        return header + "No securities lending trades on file in the window."
    shown = daily.tail(days)
    volume = shown["Lending Volume (借券成交張數)"]
    types = by_type.loc[by_type.index.isin(shown.index)].sum()
    lines = [
        f"Window: {shown.index[0]} to {shown.index[-1]} ({len(shown)} days with lending trades)",
        f"Total lent over the window: {volume.sum():,.0f} lots in {int(shown['Trades'].sum())} trades",
        f"Daily average: {volume.mean():,.0f} lots; largest day: {volume.idxmax()} ({volume.max():,.0f} lots)",
        "By transaction type: " + ", ".join(f"{k} {v:,.0f} lots" for k, v in types.items()),
    ]
    return header + "\n".join(lines) + f"\n\n## Daily (oldest first)\n{shown.to_csv()}"


# --- the joined view -----------------------------------------------------------------

def get_chip_summary(symbol: str, as_of_date: str | None = None, days: int = 20) -> str:
    """Price, volume, institutional flows, margin and lending joined by day, with
    high-volume down days flagged and read out."""
    stock_id = require_taiwan_stock(symbol)
    days = _days(days)
    settings = _settings()
    ratio_cut, drop_cut = float(settings["volume_spike_ratio"]), float(settings["down_day_pct"])
    day = _day(as_of_date)

    earliest = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=_WINDOW_DAYS)).strftime("%Y-%m-%d")
    prices = load_prices(stock_id, symbol, earliest)
    prices = prices[prices["Date"] <= day].copy()
    if prices.empty:
        raise NoMarketDataError(symbol, stock_id, f"no price rows on or before {day}")
    prices["date"] = prices["Date"].dt.strftime("%Y-%m-%d")
    prices["Change %"] = (prices["Close"].pct_change() * 100).round(2)
    prices["Volume Ratio (x 20d avg)"] = (prices["Volume"] / prices["Volume"].shift(1).rolling(20).mean()).round(2)
    view = prices.set_index("date")[["Close", "Change %", "Volume", "Volume Ratio (x 20d avg)"]]
    view["Volume"] = (view["Volume"] / 1000).round(0)
    view = view.rename(columns={"Volume": "Volume (lots)"})

    net = institutional_net(stock_id, as_of_date)
    margin = margin_table(stock_id, as_of_date)
    lending, _ = lending_daily(stock_id, as_of_date)
    missing = []
    for frame, columns, label in (
        (net, [*_GROUP_ORDER, TOTAL], "institutional flows"),
        (margin, ["Margin Change (融資增減)", "Short Change (融券增減)"], "margin / short"),
    ):
        if frame.empty:
            missing.append(label)
        else:
            view = view.join(frame[columns])
    if not lending.empty:
        view = view.join(lending[["Lending Volume (借券成交張數)"]])
        view["Lending Volume (借券成交張數)"] = view["Lending Volume (借券成交張數)"].fillna(0.0)

    shown = view.tail(days).copy()
    flagged = shown[(shown["Volume Ratio (x 20d avg)"] >= ratio_cut) & (shown["Change %"] <= -drop_cut)]

    lines = [
        f"Window: {shown.index[0]} to {shown.index[-1]} ({len(shown)} trading days)",
        f"High-volume down day (爆量下跌日) rule: volume >= {ratio_cut:g}x the prior 20-day average"
        f" and close down >= {drop_cut:g}%",
    ]
    if missing:
        lines.append(f"Not available from FinMind for this window: {', '.join(missing)}")
    if flagged.empty:
        lines.append("High-volume down days in the window: none")
    else:
        lines.append(f"High-volume down days in the window: {len(flagged)}")
    for when, row in flagged.iterrows():
        lines.append(
            f"\n### {when}: close {row['Close']} ({row['Change %']:+.2f}%), volume {row['Volume (lots)']:,.0f} lots"
            f" ({row['Volume Ratio (x 20d avg)']:.2f}x 20-day average)"
        )
        if not net.empty and when in net.index:
            groups = {g: row.get(g) for g in _GROUP_ORDER}
            sellers = sorted((v, g) for g, v in groups.items() if pd.notna(v) and v < 0)
            lines.append("- Institutional net: " + ", ".join(f"{g} {_lots(v)}" for g, v in groups.items())
                         + f"; total {_lots(row.get(TOTAL))} lots")
            lines.append(f"- Largest institutional seller: {sellers[0][1]} ({_lots(sellers[0][0])} lots)"
                         if sellers else "- No institutional group was a net seller")
        else:
            lines.append("- Institutional net: not on file for this day")
        change = row.get("Margin Change (融資增減)")
        if change is None or pd.isna(change):
            lines.append("- Margin change: not on file for this day")
        elif change > 0:
            lines.append(f"- Margin balance rose {_lots(change)} lots on the drop:"
                         " margin buyers added (融資增加，散戶接刀)")
        else:
            lines.append(f"- Margin balance {_lots(change)} lots: margin buyers did not add on the drop")
        if "Lending Volume (借券成交張數)" in row:
            lines.append(f"- Securities lending traded: {row['Lending Volume (借券成交張數)']:,.0f} lots")

    lots = [c for c in shown.columns if c not in ("Close", "Change %", "Volume Ratio (x 20d avg)")]
    shown[lots] = shown[lots].round(0)
    shown.index.name = "Date"
    return (
        _header("Chip summary (籌碼總覽)", symbol, as_of_date)
        + _LOTS_NOTE
        + "# Institutional columns are net buys by group; Volume Ratio compares the day's volume"
        " with the average of the 20 trading days before it.\n\n"
        + "\n".join(lines)
        + f"\n\n## Daily (oldest first)\n{shown.to_csv()}"
    )
