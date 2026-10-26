"""Run the cells of a plan, each as a CLI run into a results directory of its own.

A run is exactly what a user gets from the CLI: the same entry point, flags and
environment, with ``--no-memory-log``. Each run writes under
``<out>/runs/<ticker>_<date>_r<run>`` (its state log and report), so runs never
share a file and nothing outside ``<out>`` is written. A stream runs every
plan cell whose ticker falls to it, run 1 of all its cells before any run 2.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from tradingagents.rating_backtest.store import ResultsStore, run_row

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RUN_TIMEOUT_S = 3600


def cli_command(plan: dict, ticker: str, date: str) -> list[str]:
    cli = Path(sys.executable).with_name("tradingagents")
    return [str(cli), "--ticker", ticker, "--date", date, "--analysts", ",".join(plan["analysts"]),
            "--save", "--no-show", "--html", "--no-memory-log"]


def run_environment(plan: dict, run_dir: Path) -> dict:
    """The CLI's environment: the plan's settings, and results kept in ``run_dir``."""
    env = {**os.environ, **{k: str(v) for k, v in plan["env"].items()}}
    env["TRADINGAGENTS_RESULTS_DIR"] = str(run_dir)
    env["TRADINGAGENTS_MEMORY_LOG_ENABLED"] = "false"
    # A blank key would stop .env from supplying it (load_dotenv never overrides).
    if not env.get("FRED_API_KEY"):
        env.pop("FRED_API_KEY", None)
    return env


def stream_cells(plan: dict, stream: int, streams: int) -> list[tuple[str, str, int]]:
    """This stream's cells in run order: tickers dealt round-robin, run 1 of every cell first."""
    tickers = [s["ticker"] for s in plan["universe"]][stream::streams]
    return [(ticker, date, run) for run in range(1, int(plan["runs"]) + 1)
            for ticker in tickers for date in plan["dates"]]


def run_cell(plan: dict, out: Path, ticker: str, date: str, run: int) -> dict:
    run_dir = out / "runs" / f"{ticker}_{date}_r{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with open(run_dir / "cli.log", "w", encoding="utf-8") as log:
        try:
            exit_code = subprocess.run(
                cli_command(plan, ticker, date), cwd=REPO, env=run_environment(plan, run_dir),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_S,
            ).returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
    elapsed = time.time() - started
    state_path = run_dir / ticker / "TradingAgentsStrategy_logs" / f"full_states_log_{date}.json"
    if exit_code != 0 or not state_path.exists():
        return {"ticker": ticker, "date": date, "run": run, "status": "error", "exit": exit_code,
                "elapsed_s": round(elapsed), "run_dir": str(run_dir)}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return run_row(state, ticker, date, run, elapsed, str(run_dir), plan.get("commit", ""))


def run_stream(plan: dict, out: Path, stream: int, streams: int) -> None:
    store = ResultsStore(out)
    cells = stream_cells(plan, stream, streams)
    for index, (ticker, date, run) in enumerate(cells, 1):
        if (ticker, date, run) in store.completed():
            continue
        logger.info("[%d/%d] %s %s run %d", index, len(cells), ticker, date, run)
        row = run_cell(plan, out, ticker, date, run)
        store.append(row)
        logger.info("  -> %s %s (%ss)", row["status"], row.get("final_rating", row.get("exit")), row["elapsed_s"])
