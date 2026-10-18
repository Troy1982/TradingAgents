"""Daily OHLCV frames from the configured price vendor, for code that computes on prices.

The verified market snapshot computes indicators from a frame, not a report, so
it cannot go through ``route_to_vendor``. It loads the frame here instead, along
the same ``core_stock_apis`` chain as ``get_stock_data`` and ``get_indicators``,
so every agent reads one set of prices: FinMind's traded prices for a Taiwan
symbol, Yahoo's adjusted prices otherwise.
"""

from __future__ import annotations

import logging

import pandas as pd

from tradingagents.dataflows.errors import UnsupportedSymbolError, VendorError
from tradingagents.dataflows.router import get_vendor
from tradingagents.dataflows.vendors.finmind.market import load_ohlcv as load_finmind_ohlcv
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv as load_yahoo_ohlcv

logger = logging.getLogger(__name__)

# Vendors that can serve a frame; others in the chain (alpha_vantage) are skipped.
LOADERS = {
    "finmind": load_finmind_ohlcv,
    "yfinance": load_yahoo_ohlcv,
}

# How each vendor's prices are stated, for the reader of a computed figure.
PRICE_BASIS = {
    "finmind": "FinMind TaiwanStockPrice, traded prices (not dividend-adjusted)",
    "yfinance": "Yahoo Finance, dividend- and split-adjusted prices",
}


def _chain() -> list[str]:
    configured = [v.strip() for v in get_vendor("core_stock_apis", "get_stock_data").split(",")]
    if any(v in ("", "default") for v in configured):
        return list(LOADERS)
    chain = [v for v in configured if v in LOADERS]
    return chain or ["yfinance"]


def load_ohlcv(symbol: str, as_of_date: str, fill_gaps: bool = True) -> pd.DataFrame:
    """OHLCV up to ``as_of_date`` from the first configured vendor that has it.

    The frame's ``attrs["vendor"]`` names the vendor that served it. Raises the
    last vendor's error when none could.
    """
    last: VendorError | None = None
    for vendor in _chain():
        try:
            data = LOADERS[vendor](symbol, as_of_date, fill_gaps=fill_gaps)
        except UnsupportedSymbolError as exc:
            logger.debug("Vendor %r does not cover %s prices: %s", vendor, symbol, exc)
            last = exc
            continue
        except VendorError as exc:
            logger.info("Vendor %r could not serve %s prices: %s; trying next vendor.", vendor, symbol, exc)
            last = exc
            continue
        logger.info("OHLCV frame for %s served by %s", symbol, vendor)
        data.attrs["vendor"] = vendor
        return data
    if last is None:
        raise RuntimeError("No price vendor is configured for OHLCV frames")
    raise last
