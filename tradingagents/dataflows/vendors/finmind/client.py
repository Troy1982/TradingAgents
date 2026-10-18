"""FinMind requests: the token, the symbol check, the error mapping and the cache.

FinMind serves Taiwan-listed (TWSE/TPEx) symbols. Every public function checks
the symbol before anything else, so a US ticker is handed to the next vendor
without a request, and without a missing-token warning, while a Taiwan ticker
without ``FINMIND_TOKEN`` is reported as not configured and the router moves on.

FinMind caps requests per hour, so every answer is kept under
``data_cache_dir/finmind``: a range that ends before today never changes and is
kept for good; one that reaches today is refetched after its TTL.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date

import pandas as pd
import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import (
    UnsupportedSymbolError,
    VendorNotConfiguredError,
    VendorUnavailableError,
)
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import safe_ticker_component, taiwan_stock_id

logger = logging.getLogger(__name__)

API_URL = "https://api.finmindtrade.com/api/v4/data"
REQUEST_TIMEOUT = 30

# How long an answer that reaches today may be served from the cache.
PRICE_TTL_SECONDS = 900          # same as Yahoo's same-day OHLCV cache
SLOW_TTL_SECONDS = 6 * 3600      # statements, valuation, revenue, listing info

class FinMindNotConfiguredError(VendorNotConfiguredError):
    """FinMind was selected for a Taiwan symbol but FINMIND_TOKEN is missing or rejected."""


class FinMindRateLimitError(VendorUnavailableError):
    """FinMind's hourly request cap was reached."""


def require_taiwan_stock(symbol: str) -> str:
    """The FinMind ``data_id`` for ``symbol``, or UnsupportedSymbolError so the router tries the next vendor."""
    stock_id = taiwan_stock_id(symbol)
    if stock_id is None:
        raise UnsupportedSymbolError(symbol, symbol, "not a Taiwan-listed symbol; FinMind serves TWSE/TPEx only")
    return stock_id


def get_token() -> str:
    token = os.getenv("FINMIND_TOKEN")
    if not token:
        raise FinMindNotConfiguredError("FINMIND_TOKEN environment variable is not set.")
    return token


def _cache_path(dataset: str, data_id: str | None, start_date: str, end_date: str | None) -> str:
    folder = os.path.join(get_config()["data_cache_dir"], "finmind", dataset)
    os.makedirs(folder, exist_ok=True)
    name = f"{safe_ticker_component(data_id) if data_id else 'all'}_{start_date}_{end_date or 'open'}.json"
    return os.path.join(folder, name)


def _cache_is_fresh(path: str, end_date: str | None, ttl: int, final: bool) -> bool:
    if final or (end_date is not None and end_date < date.today().isoformat()):
        return True  # a closed past range never changes
    return time.time() - os.path.getmtime(path) <= ttl


def _request(params: dict, token: str) -> list[dict]:
    try:
        response = requests.get(
            API_URL,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"FinMind request failed: {type(exc).__name__}") from None

    try:
        body = response.json()
    except ValueError:
        body = {}
    message = str(body.get("msg", ""))
    status = body.get("status", response.status_code)

    if response.status_code == 402 or status == 402 or "upper limit" in message.lower():
        raise FinMindRateLimitError(f"FinMind request limit reached: {message or response.status_code}")
    if "token" in message.lower() and status != 200:
        raise FinMindNotConfiguredError(f"FinMind rejected FINMIND_TOKEN: {message}")
    if response.status_code != 200 or status != 200:
        raise VendorUnavailableError(
            f"FinMind could not serve {params.get('dataset')}: {message or response.status_code}"
        )
    return body.get("data") or []


def fetch(
    dataset: str,
    data_id: str | None,
    start_date: str,
    end_date: str | None = None,
    *,
    ttl: int = SLOW_TTL_SECONDS,
    final: bool = False,
) -> pd.DataFrame:
    """Rows of ``dataset`` for ``data_id`` from ``start_date`` (to ``end_date``), cached.

    ``final`` marks an answer that can no longer change although no ``end_date``
    is sent (a single-day dataset asked about a past day), so it is kept for good.

    An empty frame is a real answer (FinMind returns no rows for a code it does
    not list) and is cached like any other; callers decide what it means.
    """
    token = get_token()
    path = _cache_path(dataset, data_id, start_date, end_date)
    if os.path.exists(path) and _cache_is_fresh(path, end_date, ttl, final):
        try:
            with open(path, encoding="utf-8") as handle:
                return pd.DataFrame(json.load(handle))
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable FinMind cache %s", path)

    params = {"dataset": dataset, "start_date": start_date}
    if data_id:
        params["data_id"] = data_id
    if end_date:
        params["end_date"] = end_date
    rows = _request(params, token)

    def write(temp: str) -> None:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False)

    replace_file(path, write)
    return pd.DataFrame(rows)


def listing_suffix(stock_id: str) -> str | None:
    """``.TWO`` for a TPEx listing, ``.TW`` for TWSE, None when FinMind does not
    list the code; raises when FinMind cannot be asked."""
    info = fetch("TaiwanStockInfo", stock_id, "2000-01-01", None)
    if info.empty or "type" not in info:
        return None
    types = set(info["type"].astype(str).str.lower())
    return ".TW" if "twse" in types else ".TWO" if "tpex" in types else None


def resolve_taiwan_symbol(raw: str) -> str | None:
    """A bare Taiwan code resolved to its Yahoo symbol (``2330`` -> ``2330.TW``,
    ``8069`` -> ``8069.TWO``); a suffixed or non-Taiwan symbol returns None.

    Without FinMind (no token, outage) a plain four-digit code is taken as TWSE,
    and a longer bare code (which may be a mainland listing such as ``600519``)
    is left alone.
    """
    if not isinstance(raw, str) or "." in raw:
        return None
    stock_id = taiwan_stock_id(raw)
    if stock_id is None:
        return None
    try:
        suffix = listing_suffix(stock_id)
    except Exception as exc:  # resolution is a convenience; never block the run on it
        logger.info("Could not resolve the listing of %s on FinMind (%s)", stock_id, exc)
        suffix = ".TW" if re.fullmatch(r"\d{4}", stock_id) else None
    return stock_id + suffix if suffix else None
