"""Taiwan company news from FinMind (TaiwanStockNews).

FinMind serves one day of a stock's news per request and stamps each article
in UTC. Each day of the window is fetched (and cached for good once it is over),
so unlike Yahoo's feed the window is observed in full and an empty result is a
real absence of news. Articles carry a title, source and link, but no summary.

The window is read on the Taipei clock, as the run's date is, ending at the
configured ``as_of_time`` on the analysis date (``in_market_window``). A Taipei
day starts at 16:00 UTC the day before, so the UTC day before the window is
fetched too. Cut in UTC, a run dated 2026-10-04 read a 00:22 2026-10-05
headline announcing the September revenue filed that day.
"""

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import TAIPEI, coverage_gap, in_market_window
from tradingagents.dataflows.vendors.finmind.client import (
    PRICE_TTL_SECONDS,
    fetch,
    require_taiwan_stock,
)
from tradingagents.dataflows.vendors.yahoo.news import _format_articles

# Requests per call are one per day; a longer window keeps its latest days.
MAX_DAYS = 31


def _parse_utc(value) -> datetime | None:
    try:
        return datetime.strptime(str(value)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """News about a Taiwan symbol published in ``[start_date, end_date]``."""
    stock_id = require_taiwan_stock(ticker)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    today = date.today()

    last_day = min(end_dt.date(), today)
    first_day = max(start_dt.date(), last_day - timedelta(days=MAX_DAYS - 1))
    articles, seen = [], set()
    day = last_day
    # The UTC day before the first Taipei day holds that day's first eight hours.
    while day >= first_day - timedelta(days=1):
        rows = fetch(
            "TaiwanStockNews", stock_id, day.isoformat(), None,
            ttl=PRICE_TTL_SECONDS, final=day < today - timedelta(days=1),
        )
        for row in rows.to_dict("records"):
            key = (row.get("link") or "", row.get("title") or "")
            if key in seen:
                continue
            seen.add(key)
            articles.append({
                "title": row.get("title") or "No title",
                "summary": "",
                "publisher": row.get("source") or "Unknown",
                "link": row.get("link") or "",
                "pub_date": _parse_utc(row.get("date")),
            })
        day -= timedelta(days=1)

    in_range = [a for a in articles if in_market_window(a["pub_date"], start_date, end_date, TAIPEI)]
    in_range.sort(key=lambda a: a["pub_date"] or datetime.min.replace(tzinfo=UTC), reverse=True)
    in_range = in_range[: get_config()["news_article_limit"]]

    note = ""
    if first_day > start_dt.date():
        note = f"(only the latest {MAX_DAYS} days, from {first_day}, were searched)\n\n"
    if in_range:
        return f"## {ticker} News (FinMind), from {start_date} to {end_date}:\n\n{note}{_format_articles(in_range)}"

    subject = f"news for {ticker}"
    gap = coverage_gap([pd.Timestamp(first_day).to_pydatetime()], start_date, end_date, "FinMind news", subject)
    return gap or f"No news found for {ticker} between {first_day} and {end_date}"
