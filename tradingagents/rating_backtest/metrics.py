"""The rating's predictive power, set against simple baselines on the same cells.

Overlapping holding periods make cells on nearby dates move together, so the
cross-sectional statistics are computed within each analysis date and then
averaged over dates, and their spread over dates gives the t-value and the
bootstrap interval. A cell's rating is the majority of its runs (``store.majority``).
"""

from __future__ import annotations

import math
import random

import pandas as pd

from tradingagents.rating_backtest.store import SCORE, cell_ratings, majority

UP = {"Buy", "Overweight"}
DOWN = {"Sell", "Underweight"}


def revenue_yoy(stock_id: str, date: str) -> float | None:
    """Year-over-year growth of the latest monthly revenue public on ``date`` (``publication.py`` decides)."""
    from tradingagents.dataflows.vendors.finmind.fundamentals import _monthly_revenue

    revenue = _monthly_revenue(stock_id, date).dropna()
    if revenue.empty:
        return None
    month = revenue.index[-1]
    previous = revenue.get(month - 12)
    if previous is None or pd.isna(previous) or previous <= 0:
        return None
    return float(revenue[month] / previous - 1)


def cell_table(runs: list[dict], settled: list[dict], universe: list[dict], revenue=revenue_yoy) -> pd.DataFrame:
    """One row per cell: the majority rating, agreement, run details and what followed."""
    info = {s["ticker"]: s for s in universe}
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in runs:
        if row.get("status") == "ok":
            by_cell.setdefault((row["ticker"], row["date"]), []).append(row)
    outcome = {(r["ticker"], r["date"]): r for r in settled}
    rows = []
    for (ticker, date), ratings in sorted(cell_ratings(runs).items()):
        rating, agreement = majority(ratings)
        details = by_cell[(ticker, date)]
        rows.append({
            "ticker": ticker, "date": date, "market": info.get(ticker, {}).get("market"),
            "industry": info.get(ticker, {}).get("industry"), "runs": len(ratings), "ratings": "/".join(ratings),
            "rating": rating, "score": SCORE.get(rating), "agreement": agreement,
            "confidence": details[-1].get("final_confidence"),
            "confidences": "/".join(str(d.get("final_confidence")) for d in details),
            "cap_applied": sum(bool(d.get("cap_applied")) for d in details),
            "coverage_score": details[-1].get("coverage_score"),
            "capped_by": ";".join(sorted({c for d in details for c in d.get("capped_by") or []})),
            "revenue_yoy": revenue(ticker.split(".")[0], date),
            **{k: v for k, v in outcome.get((ticker, date), {}).items() if k not in ("ticker", "date", "market")},
        })
    return pd.DataFrame(rows)


def _per_date(cells: pd.DataFrame, value: str, func) -> pd.Series:
    out = {}
    for date, group in cells.dropna(subset=[value]).groupby("date"):
        result = func(group)
        if result is not None and not (isinstance(result, float) and math.isnan(result)):
            out[date] = result
    return pd.Series(out, dtype=float)


def _summary(per_date: pd.Series) -> dict:
    n = len(per_date)
    mean = float(per_date.mean()) if n else None
    sd = float(per_date.std(ddof=1)) if n > 1 else None
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd else None
    return {"mean": mean, "dates": n, "t": t, "ci95": bootstrap(per_date)}


def bootstrap(per_date: pd.Series, draws: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """A 95% interval for the mean over dates, resampling dates."""
    values = list(per_date)
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def spread(cells: pd.DataFrame, excess: str, label: str = "rating") -> dict:
    """Mean excess of the up calls minus that of the down calls, within each date."""
    def one(group):
        up, down = group[group[label].isin(UP)], group[group[label].isin(DOWN)]
        return float(up[excess].mean() - down[excess].mean()) if len(up) and len(down) else None
    return _summary(_per_date(cells, excess, one))


def rank_ic(cells: pd.DataFrame, excess: str, signal: str = "score") -> dict:
    """Spearman correlation of ``signal`` with the excess return, within each date."""
    def one(group):
        group = group.dropna(subset=[signal])
        if len(group) < 4 or group[signal].nunique() < 2 or group[excess].nunique() < 2:
            return None
        return float(group[signal].rank().corr(group[excess].rank()))
    return _summary(_per_date(cells, excess, one))


def hit_rate(cells: pd.DataFrame, excess: str, score: str = "score") -> tuple[float | None, int]:
    """Share of directional calls (Hold left out) whose sign matched the excess return's."""
    calls = cells.dropna(subset=[excess, score])
    calls = calls[calls[score] != 0]
    if calls.empty:
        return None, 0
    right = ((calls[score] > 0) == (calls[excess] > 0)).sum()
    return float(right / len(calls)), len(calls)


def by_rating(cells: pd.DataFrame, excess: str) -> pd.DataFrame:
    settled = cells.dropna(subset=[excess])
    return (settled.groupby("rating")[excess].agg(["count", "mean", "median"])
            .reindex([r for r in SCORE if r in set(settled["rating"])]))


def by_group(cells: pd.DataFrame, excess: str, column: str) -> pd.DataFrame:
    """Per value of ``column`` (agreement, confidence): count, hit rate, mean signed excess."""
    settled = cells.dropna(subset=[excess, "score"]).copy()
    settled["signed"] = settled["score"].clip(-1, 1) * settled[excess]
    rows = []
    for value, group in settled.groupby(column):
        rate, calls = hit_rate(group, excess)
        rows.append({column: value, "cells": len(group), "directional": calls, "hit_rate": rate,
                     "mean_signed_excess": float(group["signed"].mean())})
    return pd.DataFrame(rows)


def matched_labels(cells: pd.DataFrame, signal: str) -> pd.Series:
    """A baseline's calls in the model's proportions: per date, the top share Overweight, the bottom Underweight."""
    rated = cells.dropna(subset=["rating"])
    up_share = rated["rating"].isin(UP).mean() if len(rated) else 0
    down_share = rated["rating"].isin(DOWN).mean() if len(rated) else 0
    labels = pd.Series("Hold", index=cells.index)
    for _date, group in cells.dropna(subset=[signal]).groupby("date"):
        order = group[signal].rank(method="first", ascending=False)
        n = len(group)
        labels[order[order <= round(up_share * n)].index] = "Overweight"
        labels[order[order > n - round(down_share * n)].index] = "Underweight"
    return labels.where(cells[signal].notna())


def evaluate(cells: pd.DataFrame, horizons=(5, 20, 60)) -> dict:
    """Every metric, per horizon, for the model and the baselines."""
    cells = cells.copy()
    for signal in ("momentum_20", "revenue_yoy"):
        cells[f"{signal}_call"] = matched_labels(cells, signal) if signal in cells else None
        cells[f"{signal}_score"] = cells[f"{signal}_call"].map(SCORE)
    results = {}
    for n in horizons:
        excess = f"excess_{n}"
        if excess not in cells or cells[excess].notna().sum() == 0:
            continue
        settled = cells.dropna(subset=[excess])
        model_hit = hit_rate(settled, excess)
        results[n] = {
            "cells": len(settled), "dates": settled["date"].nunique(),
            "by_rating": by_rating(settled, excess),
            "spread": spread(settled, excess), "rank_ic": rank_ic(settled, excess),
            "hit_rate": model_hit,
            "by_agreement": by_group(settled, excess, "agreement"),
            "by_confidence": by_group(settled, excess, "confidence"),
            "baselines": {
                "always_overweight": {"mean_excess": float(settled[excess].mean())},
                **{name: {"spread": spread(settled, excess, f"{name}_call"),
                          "rank_ic": rank_ic(settled, excess, name),
                          "hit_rate": hit_rate(settled, excess, f"{name}_score")}
                   for name in ("momentum_20", "revenue_yoy")},
            },
        }
    return results
