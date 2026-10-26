"""The stocks a backtest covers, drawn from what was listed on the start date.

Hindsight is kept out by using only what was known on ``as_of``: the listing
(``TaiwanStockInfo`` keeps delisted companies, and each row is the last day a
company was seen in a market) and 40 trading days of turnover before it. Stocks
are drawn in a seeded random order and taken while their market's quota and
their industry's cap allow, so the sample is a stratified random draw from the
liquid listed stocks of the day, not a choice made knowing what came after.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import pandas as pd

from tradingagents.dataflows.vendors.finmind.client import fetch

# Not common stocks, or too thin a market to analyze.
EXCLUDED_INDUSTRIES = {"存託憑證", "創新版股票", "創新板股票"}
# A company listed under both a broad and a specific category is filed under the specific one.
BROAD_INDUSTRIES = {"電子工業"}
SUFFIX = {"twse": ".TW", "tpex": ".TWO"}


@dataclass(frozen=True)
class Stock:
    ticker: str
    stock_id: str
    market: str
    industry: str
    name: str
    mean_turnover: float


def industry_key(name: str) -> str:
    """One name for categories FinMind spells two ways (其他電子類 / 其他電子業)."""
    return name[:-1] if len(name) > 2 and name[-1] in "類業" else name


def listed_on(info: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Common stocks listed on TWSE or TPEx on ``as_of``: stock_id, market, industry, name.

    A company's market is that of the first row seen on or after ``as_of`` (a
    move from TPEx to TWSE later shows as two rows); one last seen before it had
    left the market by then.
    """
    rows = info[info["stock_id"].astype(str).str.fullmatch(r"[1-9]\d{3}")]
    rows = rows[rows["date"].astype(str) >= as_of]
    listed = []
    for stock_id, group in rows.groupby("stock_id"):
        group = group.sort_values("date")
        market = str(group["type"].iloc[0]).lower()
        if market not in SUFFIX:
            continue
        categories = [c for c in group["industry_category"].astype(str) if c]
        specific = [c for c in categories if c not in BROAD_INDUSTRIES] or categories
        if not specific or EXCLUDED_INDUSTRIES & set(categories):
            continue
        listed.append({"stock_id": stock_id, "market": market, "industry": industry_key(specific[0]),
                       "name": str(group["stock_name"].iloc[0])})
    return pd.DataFrame(listed, columns=["stock_id", "market", "industry", "name"])


def trading_days_before(as_of: str, count: int) -> list[str]:
    """The last ``count`` trading days on or before ``as_of``."""
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=count * 2 + 30)).strftime("%Y-%m-%d")
    days = fetch("TaiwanStockTradingDate", None, start, as_of)
    days = sorted(d for d in days["date"].astype(str) if d <= as_of)
    return days[-count:]


def liquidity(stock_id: str, as_of: str, window: list[str], listed_since: str) -> tuple[bool, float]:
    """Whether ``stock_id`` traded through ``window`` and was listed by ``listed_since``; its mean turnover."""
    start = min(listed_since, window[0])
    prices = fetch("TaiwanStockPrice", stock_id, start, as_of)
    if prices.empty or str(prices["date"].min()) > listed_since:
        return False, 0.0
    traded = prices[prices["date"].astype(str).isin(window) & (prices["Trading_Volume"].astype(float) > 0)]
    mean = float(traded["Trading_money"].astype(float).sum()) / len(window)
    return len(traded) >= 0.9 * len(window), mean


def select_universe(as_of: str, quotas: dict[str, int], seed: int, max_per_industry: int,
                    min_turnover: float, listed_since: str, window_days: int = 40) -> list[Stock]:
    """Draw ``quotas`` ({"twse": n, "tpex": m}) stocks listed on ``as_of``, ``max_per_industry`` per industry.

    A stock qualifies with prices since ``listed_since``, trades on 90% of the
    ``window_days`` trading days up to ``as_of``, and a mean daily turnover of
    ``min_turnover`` TWD over them. Each candidate costs one FinMind request.
    """
    listed = listed_on(fetch("TaiwanStockInfo", None, "2000-01-01"), as_of)
    window = trading_days_before(as_of, window_days)
    order = sorted(listed["stock_id"])
    random.Random(seed).shuffle(order)
    by_id = listed.set_index("stock_id")
    taken: list[Stock] = []
    per_industry: dict[str, int] = {}
    for stock_id in order:
        row = by_id.loc[stock_id]
        if sum(s.market == row["market"] for s in taken) >= quotas.get(row["market"], 0):
            continue
        if per_industry.get(row["industry"], 0) >= max_per_industry:
            continue
        ok, mean = liquidity(stock_id, as_of, window, listed_since)
        if not ok or mean < min_turnover:
            continue
        taken.append(Stock(stock_id + SUFFIX[row["market"]], stock_id, row["market"], row["industry"],
                           row["name"], round(mean)))
        per_industry[row["industry"]] = per_industry.get(row["industry"], 0) + 1
        if len(taken) == sum(quotas.values()):
            break
    return taken


def as_rows(stocks: list[Stock]) -> list[dict]:
    return [asdict(s) for s in stocks]
