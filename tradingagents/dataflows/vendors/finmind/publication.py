"""When a Taiwan company's monthly revenue and quarterly statements count as public.

The one place these rules live: the data layer trims what a run is served with
them, and the coverage step judges freshness with them, so the two can never
disagree about what a run on a given date should have.

Monthly revenue for a month is due by the 10th of the next month, moved to the
next trading day, and its cut-off is that deadline, plus an allowance for
financial and insurance companies (config
``revenue_publication.financial_allowance_days``, 10), whose holding companies
and insurers publish their monthly results two to seven days after the 10th.
A month counts as public on the earlier of its ``create_time`` and its
cut-off; with no ``create_time``, on its cut-off.

A quarter's statements are due by its filing deadline (5/15, 8/14, 11/14, 3/31;
5/31, 8/31, 11/30, 3/31 for financial and insurance companies), moved to the
next trading day. FinMind records no filing date, so a quarter counts as public
on its deadline.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.dataflows.vendors.finmind.calendar import due_by, next_business_day
from tradingagents.dataflows.vendors.finmind.client import fetch

# Monthly revenue is due by this day of the following month.
REVENUE_DEADLINE_DAY = 10

# Quarter-end month -> (months after the quarter end, day) of the filing deadline.
_DEADLINES = {3: (2, 15), 6: (2, 14), 9: (2, 14), 12: (3, 31)}
_FINANCIAL_DEADLINES = {3: (2, 31), 6: (2, 31), 9: (2, 30), 12: (3, 31)}
QUARTER_END_MONTHS = frozenset(_DEADLINES)

# FinMind industry_category words that mark a financial or insurance company.
_FINANCIAL_INDUSTRY = ("金融", "保險")


def is_financial(stock_id: str) -> bool:
    """Whether FinMind lists ``stock_id`` as a financial or insurance company.

    True too when the listing cannot be read, so the later deadlines apply.
    The industry is today's; companies seldom move into or out of finance.
    """
    try:
        info = fetch("TaiwanStockInfo", stock_id, "2000-01-01", None)
    except Exception:  # noqa: BLE001 - the later deadline is the safe answer
        return True
    if info.empty or "industry_category" not in info:
        return True
    return any(word in str(industry) for industry in info["industry_category"] for word in _FINANCIAL_INDUSTRY)


def revenue_settings() -> dict:
    from tradingagents.dataflows.config import get_config

    return {"financial_allowance_days": 10, "general_late_warning_days": 1, "known_backfill_dates": [],
            **(get_config().get("revenue_publication") or {})}


# --- monthly revenue ------------------------------------------------------------------------

def revenue_deadline(month: pd.Period) -> pd.Timestamp:
    """The statutory day ``month``'s revenue is due: the 10th of the next month, unmoved."""
    return pd.Timestamp(f"{month + 1}-{REVENUE_DEADLINE_DAY:02d}")


def revenue_allowance(financial: bool) -> int:
    return int(revenue_settings()["financial_allowance_days"]) if financial else 0


def revenue_cut_off(month: pd.Period, financial: bool) -> pd.Timestamp:
    """The day ``month``'s revenue counts as public without a ``create_time``."""
    due = next_business_day(revenue_deadline(month))
    allowance = revenue_allowance(financial)
    return next_business_day(due + pd.Timedelta(days=allowance)) if allowance else due


def revenue_cut_off_passed(month: pd.Period, financial: bool, day) -> bool:
    """Whether ``month``'s cut-off falls on or before ``day``.

    Only a cut-off near ``day`` is looked up on the trading calendar, so a run
    reads the calendar for the years around its date, not every year it has.
    """
    allowance = revenue_allowance(financial)
    if not allowance:
        return due_by(revenue_deadline(month), day)
    if revenue_deadline(month) + pd.Timedelta(days=allowance + 30) <= pd.Timestamp(day):
        return True   # long past, however it moved
    return due_by(next_business_day(revenue_deadline(month)) + pd.Timedelta(days=allowance), day)


def revenue_visible(month: pd.Period, create_time: str, financial: bool, day) -> bool:
    """Whether ``month``'s revenue is public by ``day``: filed by then, or past its cut-off."""
    if create_time and create_time <= str(day)[:10]:
        return True
    return revenue_cut_off_passed(month, financial, day)


def latest_due_revenue_month(day, financial: bool) -> pd.Period:
    """The latest month whose revenue had to be public by ``day``, filed or not."""
    month = pd.Period(pd.Timestamp(day), "M") - 1
    while not revenue_cut_off_passed(month, financial, day):
        month -= 1
    return month


# --- quarterly statements ----------------------------------------------------------------------

def statement_deadline(period_end: pd.Timestamp, financial: bool) -> pd.Timestamp:
    """The statutory date by which a company had to file the quarter ending ``period_end``."""
    months, day = (_FINANCIAL_DEADLINES if financial else _DEADLINES)[pd.Timestamp(period_end).month]
    month_end = pd.Timestamp(period_end) + pd.offsets.MonthEnd(months)
    return month_end.replace(day=min(day, month_end.day))


def statement_visible(period_end: pd.Timestamp, financial: bool, day) -> bool:
    """Whether the quarter ending ``period_end`` is public by ``day``: its deadline, moved, has passed."""
    return due_by(statement_deadline(period_end, financial), day)


def latest_due_statement(day, financial: bool) -> pd.Timestamp:
    """The latest quarter end whose statements had to be public by ``day``."""
    quarter_end = (pd.Timestamp(day) - pd.offsets.QuarterEnd(1)).normalize()
    while not statement_visible(quarter_end, financial, day):
        quarter_end = (quarter_end - pd.offsets.QuarterEnd(1)).normalize()
    return quarter_end
