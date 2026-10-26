
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import (
    market_timezone,
    market_window,
    withhold_undisclosed_trades,
)
from tradingagents.dataflows.vendors.alpha_vantage.common import (
    _make_api_request,
    format_datetime_for_api,
)


def _time_range(start_date: str, end_date: str, tz: str | None) -> tuple[str, str]:
    """``time_from`` / ``time_to`` for the window, in UTC on ``tz``'s clock when given."""
    if tz is None:
        return format_datetime_for_api(start_date), format_datetime_for_api(end_date, end_of_day=True)
    start, cutoff = market_window(start_date, end_date, tz)
    return format_datetime_for_api(start), format_datetime_for_api(cutoff)


def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    # Without a limit the endpoint returns 50 articles, each with per-ticker
    # sentiment arrays, and all of it reaches the prompt.
    time_from, time_to = _time_range(start_date, end_date, market_timezone(ticker))
    params = {
        "tickers": ticker,
        "time_from": time_from,
        "time_to": time_to,
        "limit": str(get_config()["news_article_limit"]),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_global_news(
    as_of_date, look_back_days: int | None = None, limit: int | None = None, tz: str | None = None,
    queries: list[str] | None = None,
) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        as_of_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back; ``None`` uses
            ``global_news_lookback_days`` from the active config.
        limit: Maximum number of articles; ``None`` uses
            ``global_news_article_limit`` from the active config.
        tz: The run's market clock (``market_timezone``); None cuts the window in UTC.
        queries: Accepted for the router's other vendors; Alpha Vantage filters
            by its fixed market topics, not by search terms.

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    from datetime import datetime, timedelta

    config = get_config()
    if look_back_days is None:
        look_back_days = config["global_news_lookback_days"]
    if limit is None:
        limit = config["global_news_article_limit"]

    curr_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    time_from, time_to = _time_range(start_date, as_of_date, tz)
    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": time_from,
        "time_to": time_to,
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(symbol: str, as_of_date: str | None = None) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".
        as_of_date: The analysis date (yyyy-mm-dd); a past one withholds them.

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    withheld = withhold_undisclosed_trades(as_of_date, symbol)
    if withheld:
        return withheld
    return _make_api_request("INSIDER_TRANSACTIONS", {"symbol": symbol})
