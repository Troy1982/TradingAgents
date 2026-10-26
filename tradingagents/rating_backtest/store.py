"""``runs.jsonl``: one line per run, what it decided and what it had to decide on."""

from __future__ import annotations

import fcntl
import json
import re
from collections import Counter
from pathlib import Path

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")
SCORE = {"Buy": 2, "Overweight": 1, "Hold": 0, "Underweight": -1, "Sell": -2}


class ResultsStore:
    """Rows appended by any number of streams at once (each line written under a lock)."""

    def __init__(self, directory):
        self.path = Path(directory) / "runs.jsonl"

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:  # a line cut short by a crash is not a run
                continue
        return rows

    def completed(self) -> set[tuple[str, str, int]]:
        """Cells already run to the end; a failed run is tried again."""
        return {(r["ticker"], r["date"], int(r["run"])) for r in self.rows() if r.get("status") == "ok"}

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            fcntl.flock(handle, fcntl.LOCK_UN)


def _sentiment(report: str) -> str:
    first = (report or "").splitlines()[0] if report else ""
    found = re.search(r"\*\*Overall Sentiment:\*\*\s*\**([^*(]+)", first)
    return found.group(1).strip() if found else ""


def run_row(state: dict, ticker: str, date: str, run: int, elapsed: float, run_dir: str, commit: str) -> dict:
    """The line for one finished run, from its final state log."""
    coverage = state.get("data_coverage") or {}
    cap = state.get("coverage_cap") or {}
    research = state.get("research_coverage_cap") or {}
    settings = state.get("run_settings") or {}
    return {
        "ticker": ticker, "date": date, "run": run, "status": "ok",
        "final_rating": state.get("final_rating"), "final_confidence": state.get("final_confidence"),
        "model_rating": cap.get("model_rating"), "model_confidence": cap.get("model_confidence"),
        "cap_applied": bool(cap.get("changed")) and bool(cap.get("enforced")),
        "research_rating": research.get("model_rating"),
        "coverage_score": coverage.get("score"), "coverage_level": coverage.get("level"),
        "capped_by": coverage.get("capped_by", []),
        "sentiment": _sentiment(state.get("sentiment_report", "")),
        "elapsed_s": round(elapsed), "run_dir": run_dir, "commit": commit,
        "deep_llm": settings.get("deep_think_llm"), "quick_llm": settings.get("quick_think_llm"),
        "debate_rounds": settings.get("max_debate_rounds"), "risk_rounds": settings.get("max_risk_discuss_rounds"),
    }


def cell_ratings(rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Each cell's final ratings in run order (the last row of a run counts)."""
    latest: dict[tuple[str, str, int], str] = {}
    for row in rows:
        if row.get("status") == "ok":
            latest[(row["ticker"], row["date"], int(row["run"]))] = row.get("final_rating")
    cells: dict[tuple[str, str], list[str]] = {}
    for (ticker, date, _run), rating in sorted(latest.items()):
        cells.setdefault((ticker, date), []).append(rating)
    return cells


def majority(ratings: list[str]) -> tuple[str | None, float]:
    """The cell's rating and the share of runs that gave it.

    The rating most runs gave; with no rating given by more than one run, the
    median on the five-point scale. Runs without a parseable rating are left out.
    """
    valid = [r for r in ratings if r in SCORE]
    if not valid:
        return None, 0.0
    rating, count = Counter(valid).most_common(1)[0]
    if count == 1 and len(valid) > 1:
        ordered = sorted(valid, key=SCORE.get)
        rating = ordered[(len(ordered) - 1) // 2]
    return rating, round(Counter(valid)[rating] / len(valid), 3)
