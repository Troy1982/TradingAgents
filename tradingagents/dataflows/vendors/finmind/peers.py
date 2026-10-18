"""Peer valuation for a Taiwan listing from FinMind (TaiwanStockPER).

Peers are not inferred: they are listed by hand, as a JSON object mapping a
stock code to its peers' codes (``{"2383": ["6274", "6213"]}``). Three layers
are read, each replacing the lists of the codes it names:

1. ``tradingagents/data/tw_peers.json``, shipped with the package;
2. ``~/.tradingagents/tw_peers.json``, the user's own list;
3. the ``peer_groups`` config key, for a run set up in code.

Codes may carry their exchange suffix (``2383.TW``); it is dropped.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.symbols import taiwan_stock_id
from tradingagents.dataflows.vendors.finmind.client import fetch, require_taiwan_stock
from tradingagents.dataflows.vendors.finmind.fundamentals import _known_on, _latest_before

logger = logging.getLogger(__name__)

BUILTIN_PEERS_PATH = Path(__file__).resolve().parents[3] / "data" / "tw_peers.json"
USER_PEERS_PATH = Path(os.path.expanduser("~")) / ".tradingagents" / "tw_peers.json"


def _normalized(groups) -> dict[str, list[str]]:
    """``groups`` with every code bare and checked; anything else is dropped."""
    if not isinstance(groups, dict):
        return {}
    out = {}
    for key, peers in groups.items():
        stock_id = taiwan_stock_id(str(key))
        if stock_id is None or not isinstance(peers, list):
            continue
        ids = [taiwan_stock_id(str(p)) for p in peers]
        out[stock_id] = list(dict.fromkeys(p for p in ids if p and p != stock_id))
    return out


def _read(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    try:
        return _normalized(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable peer list %s: %s", path, exc)
        return {}


def load_peer_groups() -> dict[str, list[str]]:
    """Every configured peer group, by bare stock code."""
    groups = _read(BUILTIN_PEERS_PATH)
    groups.update(_read(USER_PEERS_PATH))
    groups.update(_normalized(get_config().get("peer_groups")))
    return groups


def _name(stock_id: str) -> str:
    try:
        info = fetch("TaiwanStockInfo", stock_id, "2000-01-01", None)
    except (NoMarketDataError, VendorUnavailableError):
        return ""
    if info.empty or "stock_name" not in info:
        return ""
    return str(info["stock_name"].iloc[-1])


def _number(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def _valuation(stock_id: str, day: str, end: str | None) -> dict | None:
    """The latest TaiwanStockPER row on or before ``day``, asked for as the fundamentals report asks."""
    start = (datetime.strptime(day, "%Y-%m-%d") - relativedelta(days=30)).strftime("%Y-%m-%d")
    row = _latest_before(fetch("TaiwanStockPER", stock_id, start, end), "date", day)
    if row is None:
        return None
    return {"date": str(row.get("date"))[:10], "PER": _number(row.get("PER")),
            "PBR": _number(row.get("PBR")), "Yield": _number(row.get("dividend_yield"))}


def _cell(value, suffix="") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _versus(own, median) -> str:
    if own is None or median is None or median == 0:
        return "n/a"
    return f"{own / median - 1:+.1%}"


def get_peer_valuation(symbol: str, as_of_date: str | None = None) -> str:
    """PER, PBR and dividend yield of a Taiwan listing next to its configured peers'."""
    stock_id = require_taiwan_stock(symbol)
    historical = is_historical(as_of_date)
    day = _known_on(as_of_date)
    end = day if historical else None
    header = f"# Peer valuation (同業估值比較) for {symbol.upper()}\n"
    if historical:
        header += f"# Point-in-time as of: {as_of_date}\n"
    header += "# Source: TaiwanStockPER (TWSE/TPEx-published PER, PBR, dividend yield %)\n\n"

    peers = load_peer_groups().get(stock_id, [])
    if not peers:
        return header + (
            f"No peer group is configured for {stock_id}, so no peer comparison is available."
            " Peers are listed by hand in tw_peers.json (~/.tradingagents/tw_peers.json)."
        )

    own = _valuation(stock_id, day, end)   # the instrument's own failure is the tool's
    rows = [(stock_id, own)]
    for peer in peers:
        try:
            rows.append((peer, _valuation(peer, day, end)))
        except (NoMarketDataError, VendorUnavailableError) as exc:
            logger.info("No valuation for peer %s of %s: %s", peer, stock_id, exc)
            rows.append((peer, None))

    table = ["Role,Code,Name,Date,PER,PBR,Dividend Yield %"]
    for i, (code, val) in enumerate(rows):
        role = "Subject" if i == 0 else "Peer"
        if val is None:
            table.append(f"{role},{code},{_name(code)},no data (無資料),n/a,n/a,n/a")
        else:
            table.append(f"{role},{code},{_name(code)},{val['date']},{_cell(val['PER'])},"
                         f"{_cell(val['PBR'])},{_cell(val['Yield'])}")

    def median(key):
        # A PER of 0 is how TWSE publishes a loss-making company: not a multiple.
        values = [v[key] for _, v in rows[1:] if v and v[key] is not None and (key != "PER" or v[key] > 0)]
        return float(pd.Series(values).median()) if values else None

    lines = ["", "## Subject versus the peer median"]
    for key, label in (("PER", "PER"), ("PBR", "PBR"), ("Yield", "Dividend Yield %")):
        med = median(key)
        mine = own[key] if own else None
        lines.append(f"{label}: subject {_cell(mine)}, peer median {_cell(med)}, premium/discount {_versus(mine, med)}")
    lines.append("A PER of 0 is a loss-making period as TWSE publishes it and is left out of the median.")
    return header + "\n".join(table) + "\n" + "\n".join(lines)
