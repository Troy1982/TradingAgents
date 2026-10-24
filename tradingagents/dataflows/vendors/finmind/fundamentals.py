"""Taiwan fundamentals and financial statements from FinMind.

FinMind dates a statement by the period it covers and records no filing date,
but Taiwan sets a filing deadline for every period, so a past run is served
exactly the periods whose deadline had passed (``public_by``), instead of having
statements withheld. With no filing date to go on, a statement counts as public
on its deadline even when the company filed it earlier. Financial and insurance
companies file later than everyone else, so the deadline follows the company's
FinMind industry; when that cannot be read the later, financial deadline is used,
so a period is never served early. A deadline on a day the market is closed
moves to the next day it is open.

Monthly revenue is due by the 10th of the following month (moved the same
way), and counts as public on the earlier of its ``create_time`` and that
deadline: FinMind re-stamps rows it re-enters (2383 has February and March 2026
both stamped 2026-04-21), and older rows carry no ``create_time`` at all.

FinMind's three statements are not shaped alike:
- the income statement is per quarter;
- the cash-flow statement is year-to-date, so quarters are differenced
  (Q2 = H1 - Q1) to report per quarter, as the Yahoo vendor does;
- the balance sheet carries a ``<item>_per`` share-of-total row per item,
  which is dropped.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.vendors.finmind.calendar import due_by, next_business_day
from tradingagents.dataflows.vendors.finmind.client import fetch, require_taiwan_stock
from tradingagents.dataflows.vendors.finmind.market import load_prices

_QUARTERS_SHOWN = 5
_YEARS_SHOWN = 4

# Quarter-end month -> (months after the quarter end, day) of the filing deadline:
# 5/15, 8/14, 11/14 and 3/31 for most companies, 5/31, 8/31, 11/30 and 3/31 for
# financial and insurance companies.
_DEADLINES = {3: (2, 15), 6: (2, 14), 9: (2, 14), 12: (3, 31)}
_FINANCIAL_DEADLINES = {3: (2, 31), 6: (2, 31), 9: (2, 30), 12: (3, 31)}

# Monthly revenue is due by this day of the following month.
REVENUE_DEADLINE_DAY = 10

# FinMind industry_category words that mark a financial or insurance company.
_FINANCIAL_INDUSTRY = ("金融", "保險")

# Rows of the cash-flow statement that are balances, not flows over the period.
_CASH_OPENING = "CashBalancesBeginningOfPeriod"
_CASH_CLOSING = "CashBalancesEndOfPeriod"


def is_financial(stock_id: str) -> bool:
    """Whether FinMind lists ``stock_id`` as a financial or insurance company.

    True too when the listing cannot be read, so the later deadline applies.
    The industry is today's; companies seldom move into or out of finance.
    """
    try:
        info = fetch("TaiwanStockInfo", stock_id, "2000-01-01", None)
    except Exception:  # noqa: BLE001 - the later deadline is the safe answer
        return True
    if info.empty or "industry_category" not in info:
        return True
    return any(word in str(industry) for industry in info["industry_category"] for word in _FINANCIAL_INDUSTRY)


def filing_deadline(period_end: pd.Timestamp, financial: bool) -> pd.Timestamp:
    """The statutory date by which a company had to file the period ending ``period_end``."""
    months, day = (_FINANCIAL_DEADLINES if financial else _DEADLINES)[period_end.month]
    month_end = period_end + pd.offsets.MonthEnd(months)
    return month_end.replace(day=min(day, month_end.day))


def public_by(period_end: pd.Timestamp, financial: bool) -> pd.Timestamp:
    """The day a statement counts as public: its deadline, or the next day the market is open."""
    return next_business_day(filing_deadline(period_end, financial))


def revenue_public_by(month: pd.Period) -> pd.Timestamp:
    """The day revenue for ``month`` was due: the 10th of the next month, or the next day the market is open."""
    return next_business_day(f"{month + 1}-{REVENUE_DEADLINE_DAY:02d}")


def _known_on(as_of_date: str | None) -> str:
    return as_of_date if is_historical(as_of_date) else date.today().isoformat()


def _statement_frame(dataset: str, stock_id: str, as_of_date: str | None) -> pd.DataFrame:
    """Line items (rows) by period end (columns, oldest first) public on the run's date."""
    historical = is_historical(as_of_date)
    year = int((as_of_date if historical else date.today().isoformat())[:4])
    rows = fetch(dataset, stock_id, f"{year - 6}-01-01", as_of_date if historical else None)
    if rows.empty or not {"date", "type", "value"} <= set(rows.columns):
        return pd.DataFrame()
    rows = rows[~rows["type"].astype(str).str.endswith("_per")]
    wide = rows.pivot_table(index="type", columns="date", values="value", aggfunc="first")
    wide.columns = pd.to_datetime(wide.columns)
    wide.index.name = None
    if historical:
        cutoff = pd.Timestamp(as_of_date)
        financial = is_financial(stock_id)
        wide = wide[[c for c in wide.columns
                     if c.month in _DEADLINES and due_by(filing_deadline(c, financial), cutoff)]]
    return wide.sort_index(axis=1)


def _quarterly_cashflow(ytd: pd.DataFrame) -> pd.DataFrame:
    """Year-to-date cash flows differenced into single quarters.

    A quarter whose previous quarter of the same year is missing cannot be
    differenced and is left out rather than reported as a year-to-date figure.
    """
    out = {}
    for col in ytd.columns:
        if col.month == 3:
            out[col] = ytd[col]
            continue
        prev = col - pd.offsets.QuarterEnd(1)
        if prev not in ytd.columns:
            continue
        quarter = ytd[col] - ytd[prev]
        if _CASH_OPENING in ytd.index:
            quarter[_CASH_OPENING] = ytd.at[_CASH_CLOSING, prev] if _CASH_CLOSING in ytd.index else pd.NA
        if _CASH_CLOSING in ytd.index:
            quarter[_CASH_CLOSING] = ytd.at[_CASH_CLOSING, col]
        out[col] = quarter
    return pd.DataFrame(out, index=ytd.index)


def _annual_income(quarters: pd.DataFrame) -> pd.DataFrame:
    """Fiscal years with all four quarters, summed (EPS is the sum of quarterly EPS)."""
    years = {}
    for year, cols in quarters.T.groupby(quarters.columns.year):
        if len(cols) == 4:
            years[pd.Timestamp(year=year, month=12, day=31)] = cols.T.sum(axis=1, min_count=1)
    return pd.DataFrame(years, index=quarters.index)


def _year_ends(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[c for c in frame.columns if c.month == 12]]


def _shaped(kind: str, frame: pd.DataFrame, annual: bool) -> pd.DataFrame:
    if kind == "income":
        return _annual_income(frame) if annual else frame
    if kind == "cashflow":
        return _year_ends(frame) if annual else _quarterly_cashflow(frame)
    return _year_ends(frame) if annual else frame


# FinMind names both net income rows after what they belong to.
_INCOME_NOTE = (
    "# EquityAttributableToOwnersOfParent = net income attributable to owners of the parent (歸母淨利);"
    " IncomeAfterTaxes = net income incl. non-controlling interests (稅後淨利). EPS is on the parent's share.\n"
)

_STATEMENTS = {
    "balance": ("TaiwanStockBalanceSheet", "Balance Sheet"),
    "income": ("TaiwanStockFinancialStatements", "Income Statement"),
    "cashflow": ("TaiwanStockCashFlowsStatement", "Cash Flow"),
}


def _statement(kind: str, ticker: str, freq: str, as_of_date: str | None) -> str:
    stock_id = require_taiwan_stock(ticker)
    dataset, title = _STATEMENTS[kind]
    annual = str(freq).lower() == "annual"
    frame = _statement_frame(dataset, stock_id, as_of_date)
    if not frame.empty:
        frame = _shaped(kind, frame, annual)
    frame = frame.dropna(axis=1, how="all")
    if frame.empty:
        raise NoMarketDataError(ticker, stock_id, f"no {title.lower()} data")

    shown = frame.iloc[:, ::-1].iloc[:, : _YEARS_SHOWN if annual else _QUARTERS_SHOWN]  # newest first
    shown = shown.dropna(how="all")  # items this company does not report
    shown.columns = [c.strftime("%Y-%m-%d") for c in shown.columns]
    label = ticker.upper()
    header = f"# {title} data for {label} ({freq})\n"
    if is_historical(as_of_date):
        header += f"# Point-in-time as of: {as_of_date} (periods past their filing deadline)\n"
    header += "# Currency: TWD\n"
    if kind == "income":
        header += _INCOME_NOTE
    return header + shown.to_csv()


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Balance sheet for a Taiwan symbol from FinMind."""
    return _statement("balance", ticker, freq, as_of_date)


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Cash-flow statement for a Taiwan symbol from FinMind, per quarter or per year."""
    return _statement("cashflow", ticker, freq, as_of_date)


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: str | None = None) -> str:
    """Income statement for a Taiwan symbol from FinMind."""
    return _statement("income", ticker, freq, as_of_date)


def _quarter(period_end: pd.Timestamp) -> str:
    return f"{period_end.year} Q{(period_end.month - 1) // 3 + 1}"


def _trailing_four_quarters(income: pd.DataFrame) -> tuple[pd.Series, str | None]:
    """The latest four quarters summed and the span they cover (``2025 Q3～2026 Q2``),
    or an empty series and None when they are not four consecutive quarters."""
    last = list(income.columns[-4:])
    if len(last) < 4 or any(b - pd.offsets.QuarterEnd(1) != a for a, b in zip(last, last[1:], strict=False)):
        return pd.Series(dtype=float), None
    return income[last].sum(axis=1, min_count=4), f"{_quarter(last[0])}～{_quarter(last[-1])}"


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0) or pd.isna(numerator) or pd.isna(denominator):
        return None
    return round(float(numerator) / float(denominator), 4)


def _twd(value):
    return None if value is None or pd.isna(value) else f"{value:,.0f} TWD"


def _latest_before(frame: pd.DataFrame, column: str, day: str) -> pd.Series | None:
    if frame.empty or column not in frame:
        return None
    frame = frame[frame[column].astype(str).str[:10] <= day]
    return None if frame.empty else frame.sort_values(column).iloc[-1]


def _hundred_million(value: float) -> str:
    """A TWD amount with its 億元 reading, as Taiwan reports quote revenue."""
    return f"{value:,.0f} TWD ({value / 1e8:,.2f} 億)"


def _pct(value) -> str:
    return "" if value is None or pd.isna(value) else f"{value:+.2%}"


def _monthly_revenue(stock_id: str, day: str) -> pd.Series:
    """Monthly revenue published by ``day``, indexed by the month it covers.

    FinMind's ``date`` is the month of publication; the month covered is
    ``revenue_year`` / ``revenue_month``. A row counts as published on the
    earlier of its ``create_time`` and its deadline, the 10th of its ``date``
    month moved past any closure (see the module docstring). The index runs
    over every calendar month, so a month never filed reads as missing, not
    skipped.
    """
    rows = fetch("TaiwanStockMonthRevenue", stock_id, "2000-01-01", None)
    if rows.empty or not {"date", "revenue", "revenue_year", "revenue_month"} <= set(rows.columns):
        return pd.Series(dtype=float)
    deadline = rows["date"].astype(str).str[:7] + f"-{REVENUE_DEADLINE_DAY:02d}"
    created = (rows["create_time"].astype("string").str.strip().str[:10].fillna("")
               if "create_time" in rows else pd.Series("", index=rows.index))
    filed = (created != "") & (created <= day)
    rows = rows[filed | deadline.map(lambda d: due_by(d, day))]
    if rows.empty:
        return pd.Series(dtype=float)
    months = pd.PeriodIndex.from_fields(
        year=rows["revenue_year"].astype(int), month=rows["revenue_month"].astype(int), freq="M")
    revenue = pd.Series(rows["revenue"].astype(float).to_numpy(), index=months)
    revenue = revenue[~revenue.index.duplicated(keep="last")].sort_index()
    return revenue.reindex(pd.period_range(revenue.index.min(), revenue.index.max(), freq="M"))


def _monthly_revenue_report(revenue: pd.Series, income: pd.DataFrame) -> tuple[list, str]:
    """Summary fields and a last-15-months table for the monthly revenue series."""
    if revenue.dropna().empty:
        return [], ""
    year = revenue.index.year
    ytd = revenue.groupby(year).transform(lambda s: s.cumsum(skipna=False))
    table = pd.DataFrame({
        "Revenue (TWD)": revenue,
        "MoM": revenue / revenue.shift(1) - 1,
        "YoY": revenue / revenue.shift(12) - 1,
        "YTD Revenue (TWD)": ytd,
        "YTD YoY": ytd / ytd.shift(12) - 1,
        # A record month beats every earlier month on file.
        "Record High": revenue > revenue.cummax().shift(1),
    })

    streak = 0
    for is_record in reversed(table["Record High"].tolist()):
        if not is_record:
            break
        streak += 1

    last = table.dropna(subset=["Revenue (TWD)"]).iloc[-1]
    month = table.dropna(subset=["Revenue (TWD)"]).index[-1]
    fields = [
        (f"Latest Monthly Revenue ({month})",
         f"{_hundred_million(last['Revenue (TWD)'])}, MoM {_pct(last['MoM'])}, YoY {_pct(last['YoY'])}"),
        (f"YTD Revenue ({month.year}-01 to {month})",
         None if pd.isna(last["YTD Revenue (TWD)"])
         else f"{_hundred_million(last['YTD Revenue (TWD)'])}, YoY {_pct(last['YTD YoY'])}"),
        (f"Consecutive Record-High Months (vs. all months since {revenue.index.min()})", streak),
    ]

    # The latest quarter whose three months are all filed but whose income
    # statement is not yet public: its revenue, summed from the months.
    quarter_end = month if month.month % 3 == 0 else month - month.month % 3
    quarter = pd.period_range(quarter_end - 2, quarter_end, freq="M")
    published = {c.to_period("M") for c in income.columns} if not income.empty else set()
    if quarter_end not in published and revenue.reindex(quarter).notna().all():
        total = revenue.reindex(quarter).sum()
        prior = revenue.reindex(quarter - 12)
        yoy = total / prior.sum() - 1 if prior.notna().all() else None
        q = (quarter_end.month - 1) // 3 + 1
        fields.append((
            f"Q{q} {quarter_end.year} Revenue (derived from monthly revenue; income statement not yet published)",
            f"{_hundred_million(total)}" + ("" if yoy is None else f", YoY {_pct(yoy)}"),
        ))

    shown = table.iloc[-15:].copy()
    shown.index = shown.index.astype(str)
    shown.index.name = "Revenue Month"
    for col in ("MoM", "YoY", "YTD YoY"):
        shown[col] = shown[col].map(_pct)
    for col in ("Revenue (TWD)", "YTD Revenue (TWD)"):
        shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    section = (
        "\n\n## Monthly Revenue (last 15 months, by the month covered; TWD)\n"
        + shown.to_csv()
    )
    return fields, section


def get_fundamentals(ticker: str, as_of_date: str | None = None) -> str:
    """Valuation, trailing results and balance-sheet ratios for a Taiwan symbol, as of the run's date.

    Every figure is dated, so a past run gets the values known on its date.
    Name and industry are present-day only and are left out of a past run.
    Market cap, beta and forward estimates are not served by FinMind's free
    datasets and are left out rather than estimated.
    """
    stock_id = require_taiwan_stock(ticker)
    historical = is_historical(as_of_date)
    day = _known_on(as_of_date)
    day_dt = datetime.strptime(day, "%Y-%m-%d")
    fields: list[tuple[str, object]] = []

    if not historical:
        info = fetch("TaiwanStockInfo", stock_id, "2000-01-01", None)
        if not info.empty:
            row = info.iloc[-1]
            fields += [("Name", row.get("stock_name")), ("Industry", row.get("industry_category"))]

    per_start = (day_dt - relativedelta(days=30)).strftime("%Y-%m-%d")
    per = _latest_before(
        fetch("TaiwanStockPER", stock_id, per_start, day if historical else None), "date", day
    )
    if per is not None:
        fields += [
            # TWSE's published figure, on its own trailing-EPS basis, not the TTM span below.
            (f"PE Ratio (TWSE-published, {per.get('date')})", per.get("PER")),
            ("Price to Book", per.get("PBR")),
            ("Dividend Yield", None if pd.isna(per.get("dividend_yield")) else f"{per.get('dividend_yield')}%"),
        ]

    income = _statement_frame("TaiwanStockFinancialStatements", stock_id, as_of_date)
    balance = _statement_frame("TaiwanStockBalanceSheet", stock_id, as_of_date)
    ttm, span = _trailing_four_quarters(income)
    if span:
        fields.append(("TTM Period", f"{span} (the four latest published quarters, summed)"))
    ttm_label = f"TTM {span}" if span else "TTM"
    latest = balance.iloc[:, -1] if not balance.empty else pd.Series(dtype=float)

    def item(series, key):
        return series.get(key) if key in series.index else None

    revenue, net_income = item(ttm, "Revenue"), item(ttm, "IncomeAfterTaxes")
    parent_income = item(ttm, "EquityAttributableToOwnersOfParent")
    equity, parent_equity = item(latest, "Equity"), item(latest, "EquityAttributableToOwnersOfParent")
    liabilities = item(latest, "Liabilities")
    eps = item(ttm, "EPS")
    debt_ratio = _ratio(liabilities, item(latest, "TotalAssets"))
    # Interest-bearing debt only; accounts payable and other operating
    # liabilities are not debt (total liabilities / equity is not D/E).
    borrowings = [item(latest, k) for k in ("ShorttermBorrowings", "LongtermBorrowings", "BondsPayable")]
    borrowings = [b for b in borrowings if b is not None and not pd.isna(b)]
    debt = sum(borrowings) if borrowings else None
    debt_to_equity = _ratio(debt, equity)
    balance_date = balance.columns[-1].strftime("%Y-%m-%d") if not balance.empty else "latest"
    fields += [
        (f"EPS ({ttm_label}, on net income attributable to owners of the parent)",
         None if eps is None or pd.isna(eps) else f"{eps:.2f} TWD"),
        (f"Revenue ({ttm_label})", _twd(revenue)),
        (f"Gross Profit ({ttm_label})", _twd(item(ttm, "GrossProfit"))),
        (f"Operating Income ({ttm_label})", _twd(item(ttm, "OperatingIncome"))),
        # 歸母淨利 is the parent's share; IncomeAfterTaxes adds non-controlling interests.
        (f"Net Income Attributable to Owners of the Parent (歸母淨利, {ttm_label})", _twd(parent_income)),
        (f"Net Income incl. Non-controlling Interests (稅後淨利, {ttm_label})", _twd(net_income)),
        (f"Gross Margin ({ttm_label})", _ratio(item(ttm, "GrossProfit"), revenue)),
        (f"Operating Margin ({ttm_label})", _ratio(item(ttm, "OperatingIncome"), revenue)),
        (f"Profit Margin ({ttm_label} net income incl. non-controlling interests / revenue)",
         _ratio(net_income, revenue)),
        (f"Return on Equity ({ttm_label} parent net income / parent equity at {balance_date})",
         _ratio(parent_income, parent_equity)),
        (f"Return on Assets ({ttm_label} net income incl. non-controlling interests / total assets"
         f" at {balance_date})",
         _ratio(net_income, item(latest, "TotalAssets"))),
        ("Debt Ratio (Total Liabilities / Total Assets)",
         None if debt_ratio is None else f"{debt_ratio:.1%}"),
        ("Interest-bearing Debt to Equity ((Short-term Borrowings + Long-term Borrowings"
         " + Bonds Payable) / Equity)",
         None if debt_to_equity is None else f"{debt_to_equity:.1%} (debt {_twd(debt)})"),
        ("Current Ratio", _ratio(item(latest, "CurrentAssets"), item(latest, "CurrentLiabilities"))),
    ]
    if not balance.empty:
        fields.append(("Latest Statement Period", balance.columns[-1].strftime("%Y-%m-%d")))

    revenue_fields, revenue_section = _monthly_revenue_report(_monthly_revenue(stock_id, day), income)
    fields += revenue_fields

    closes = load_prices(stock_id, ticker, (day_dt - relativedelta(years=1)).strftime("%Y-%m-%d"))
    closes = closes[closes["Date"] <= day]
    year = closes[closes["Date"] > day_dt - relativedelta(years=1)]
    if not year.empty:
        fields += [
            ("Last Close", f"{closes['Close'].iloc[-1]} TWD ({closes['Date'].iloc[-1]:%Y-%m-%d})"),
            ("52 Week High", f"{year['High'].max()} TWD"),
            ("52 Week Low", f"{year['Low'].min()} TWD"),
            ("50 Day Average", f"{closes['Close'].tail(50).mean():.2f} TWD"),
            ("200 Day Average", f"{closes['Close'].tail(200).mean():.2f} TWD"),
        ]

    lines = [f"{label}: {v}" for label, v in fields if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not lines:
        raise NoMarketDataError(ticker, stock_id, "no fundamental fields")

    header = f"# Company Fundamentals for {ticker.upper()}\n"
    if historical:
        header += f"# Point-in-time as of: {as_of_date}\n"
    return header + "\n" + "\n".join(lines) + revenue_section
