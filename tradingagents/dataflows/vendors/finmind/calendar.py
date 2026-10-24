"""Taiwan business days, for filing deadlines that fall on a weekend or holiday.

A deadline that falls on a day the market is closed moves to the next day it
is open. The open days are TWSE's trading calendar (``TaiwanStockTradingDate``),
published a year ahead. TWSE closes on a few days government offices keep
open (the days before Lunar New Year), so a deadline may move a day or two
later than the law requires, never earlier. When the calendar cannot be read
(no token, an outage) only weekends are skipped.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from tradingagents.dataflows.vendors.finmind.client import fetch

logger = logging.getLogger(__name__)


# Calendars read this process, by year; a year that could not be read is asked again.
_CALENDARS: dict[int, frozenset[date]] = {}


def _trading_days(year: int) -> frozenset[date] | None:
    """TWSE's trading days in ``year``, or None when the calendar cannot be read."""
    if year in _CALENDARS:
        return _CALENDARS[year]
    try:
        rows = fetch("TaiwanStockTradingDate", None, f"{year}-01-01", f"{year}-12-31")
    except Exception as exc:  # noqa: BLE001 - a deadline is still due; weekends are skipped instead
        logger.info("Taiwan trading calendar for %d unavailable (%s); skipping weekends only", year, exc)
        return None
    if rows.empty or "date" not in rows:
        return None
    _CALENDARS[year] = frozenset(pd.to_datetime(rows["date"]).dt.date)
    return _CALENDARS[year]


def clear_cache() -> None:
    _CALENDARS.clear()


# No closure lasts this long (Lunar New Year is the longest, about ten days), so a
# deadline this far before a date had passed by it however it moved.
_LONGEST_CLOSURE = timedelta(days=15)


def due_by(deadline, day) -> bool:
    """Whether ``deadline``, moved past any closure, falls on or before ``day``.

    Only a deadline close to ``day`` is looked up, so a run reads the calendar
    for the years around its date, not for every year a dataset covers.
    """
    deadline, day = pd.Timestamp(deadline), pd.Timestamp(day)
    if deadline > day:
        return False
    if deadline <= day - _LONGEST_CLOSURE:
        return True
    return next_business_day(deadline) <= day


def next_business_day(day) -> pd.Timestamp:
    """``day`` itself when the market is open on it, else the next day it is."""
    current = pd.Timestamp(day).date()
    for _ in range(31):
        days = _trading_days(current.year)
        is_open = current in days if days else current.weekday() < 5
        if is_open:
            return pd.Timestamp(current)
        current += timedelta(days=1)
    return pd.Timestamp(day)  # a month with no open day: a broken calendar, not a holiday
