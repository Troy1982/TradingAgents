"""``python -m tradingagents.rating_backtest {universe,run,settle,report} OUT ...``"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import date as Date
from pathlib import Path

import pandas as pd

from tradingagents.rating_backtest import metrics, runner, settle, universe
from tradingagents.rating_backtest.store import ResultsStore

# The settings of every run; the FRED key comes from .env.
DEFAULT_ENV = {
    "TRADINGAGENTS_LLM_PROVIDER": "chatgpt",
    "TRADINGAGENTS_QUICK_THINK_LLM": "gpt-6-luna",
    "TRADINGAGENTS_DEEP_THINK_LLM": "gpt-6.1-sol",
    "TRADINGAGENTS_OUTPUT_LANGUAGE": "Traditional Chinese",
    "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1",
    "TRADINGAGENTS_MAX_RISK_ROUNDS": "1",
}
ANALYSTS = ["market", "sentiment", "news", "fundamentals", "chips"]


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=runner.REPO,
                          capture_output=True, text=True).stdout.strip()


def cmd_universe(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stocks = universe.select_universe(
        args.as_of, {"twse": args.twse, "tpex": args.tpex}, args.seed, args.max_per_industry,
        args.min_turnover, args.listed_since)
    plan = {
        "as_of": args.as_of, "seed": args.seed, "quotas": {"twse": args.twse, "tpex": args.tpex},
        "max_per_industry": args.max_per_industry, "min_turnover": args.min_turnover,
        "listed_since": args.listed_since, "dates": args.dates.split(","), "runs": args.runs,
        "analysts": ANALYSTS, "env": DEFAULT_ENV, "commit": _commit(),
        "universe": universe.as_rows(stocks),
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in stocks:
        print(f"{s.ticker:10} {s.market:5} {s.industry:10} {s.name:8} {s.mean_turnover / 1e8:8.1f} 億")


def _plan(out: Path) -> dict:
    return json.loads((out / "plan.json").read_text(encoding="utf-8"))


def cmd_run(args) -> None:
    out = Path(args.out)
    stream, streams = (int(x) for x in args.stream.split("/"))
    runner.run_stream(_plan(out), out, stream, streams)


def cmd_settle(args) -> None:
    out = Path(args.out)
    plan = _plan(out)
    markets = {s["ticker"]: s["market"] for s in plan["universe"]}
    cells = [(s["ticker"], d) for s in plan["universe"] for d in plan["dates"]]
    rows = settle.settle_cells(cells, markets, args.as_of or Date.today().isoformat())
    pd.DataFrame(rows).to_csv(out / "settled.csv", index=False)
    print(f"settled {len(rows)} cells into {out / 'settled.csv'}")


def cmd_report(args) -> None:
    out = Path(args.out)
    plan = _plan(out)
    runs = ResultsStore(out).rows()
    settled = pd.read_csv(out / "settled.csv").to_dict("records")
    cells = metrics.cell_table(runs, settled, plan["universe"])
    cells.to_csv(out / "cells.csv", index=False, encoding="utf-8-sig")
    (out / "report.md").write_text(render(plan, runs, cells, metrics.evaluate(cells)), encoding="utf-8")
    print(f"wrote {out / 'cells.csv'} and {out / 'report.md'}")


def _pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{100 * value:+.2f}%"


def _stat(summary: dict, pct: bool = True) -> str:
    fmt = _pct if pct else (lambda v: "—" if v is None else f"{v:+.3f}")
    ci = summary.get("ci95")
    text = f"{fmt(summary['mean'])}（{summary['dates']} 日"
    text += f"，t {summary['t']:.2f}" if summary.get("t") is not None else ""
    text += f"，95% 區間 {fmt(ci[0])} ~ {fmt(ci[1])}" if ci else ""
    return text + "）"


def render(plan: dict, runs: list[dict], cells: pd.DataFrame, results: dict) -> str:
    ok = [r for r in runs if r.get("status") == "ok"]
    failed = [r for r in runs if r.get("status") != "ok"]
    lines = [
        "# 評級預測力回測", "",
        f"- 程式版本 {plan.get('commit')}；股票 {len(plan['universe'])} 檔 × 日期 {len(plan['dates'])} 個 × "
        f"每樣本 {plan['runs']} 次",
        f"- 執行成功 {len(ok)} 次、失敗 {len(failed)} 次；平均耗時 "
        f"{sum(r['elapsed_s'] for r in ok) / max(len(ok), 1) / 60:.1f} 分鐘",
        f"- 評級分布：{cells['rating'].value_counts().to_dict()}",
        f"- 上限套用次數：{sum(bool(r.get('cap_applied')) for r in ok)}；覆蓋等級："
        f"{pd.Series([r.get('coverage_level') for r in ok]).value_counts().to_dict()}", "",
    ]
    for n, result in results.items():
        hit, calls = result["hit_rate"]
        lines += [
            f"## {n} 個交易日（{result['cells']} 個樣本、{result['dates']} 個日期）", "",
            "| 評級 | 樣本 | 平均超額報酬 | 中位數 |", "|---|---|---|---|",
            *[f"| {rating} | {int(row['count'])} | {_pct(row['mean'])} | {_pct(row['median'])} |"
              for rating, row in result["by_rating"].iterrows()], "",
            f"- 看多減看空報酬差：{_stat(result['spread'])}",
            f"- 評級與超額報酬的等級相關：{_stat(result['rank_ic'], pct=False)}",
            f"- 方向命中率：{_pct(hit) if hit is not None else '—'}（{calls} 個有方向的評級）", "",
            "| 基準 | 報酬差 | 等級相關 | 命中率 |", "|---|---|---|---|",
            f"| 一律 Overweight | 平均超額 {_pct(result['baselines']['always_overweight']['mean_excess'])} | — | — |",
        ]
        for name, label in (("momentum_20", "20 日動能"), ("revenue_yoy", "月營收年增率")):
            b = result["baselines"][name]
            lines.append(f"| {label} | {_stat(b['spread'])} | {_stat(b['rank_ic'], pct=False)} | "
                         f"{_pct(b['hit_rate'][0]) if b['hit_rate'][0] is not None else '—'} |")
        for key, title in (("by_agreement", "一致性"), ("by_confidence", "信心")):
            table = result[key]
            if not table.empty:
                lines += ["", f"| {title} | 樣本 | 有方向 | 命中率 | 平均方向超額 |", "|---|---|---|---|---|"]
                lines += [f"| {row.iloc[0]} | {row['cells']} | {row['directional']} | "
                          f"{_pct(row['hit_rate']) if row['hit_rate'] is not None else '—'} | "
                          f"{_pct(row['mean_signed_excess'])} |" for _, row in table.iterrows()]
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m tradingagents.rating_backtest")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("universe", help="draw the stocks and write plan.json")
    p.add_argument("out")
    p.add_argument("--as-of", required=True)
    p.add_argument("--dates", required=True, help="comma-separated analysis dates")
    p.add_argument("--twse", type=int, default=18)
    p.add_argument("--tpex", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260603)
    p.add_argument("--max-per-industry", type=int, default=4)
    p.add_argument("--min-turnover", type=float, default=1e8)
    p.add_argument("--listed-since", default="2026-03-02")
    p.add_argument("--runs", type=int, default=3)
    p.set_defaults(func=cmd_universe)

    p = commands.add_parser("run", help="run this stream's cells not yet in runs.jsonl")
    p.add_argument("out")
    p.add_argument("--stream", default="0/1", help="k/n: this is stream k of n")
    p.set_defaults(func=cmd_run)

    p = commands.add_parser("settle", help="excess returns of every cell")
    p.add_argument("out")
    p.add_argument("--as-of", help="last day of prices to use (default: today)")
    p.set_defaults(func=cmd_settle)

    p = commands.add_parser("report", help="metrics and baselines into report.md")
    p.add_argument("out")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
