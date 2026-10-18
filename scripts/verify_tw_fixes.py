"""Verify the Taiwan-market fixes and that nothing else broke.

    .venv/bin/python scripts/verify_tw_fixes.py

1. Live checks of the Taiwan fixes, on 2383.TW against AAPL:
   - the report header shows the vendor chain a ticker actually uses
     (no sec_edgar / polymarket for Taiwan), and routing matches it;
   - Polymarket is not offered for a Taiwan symbol, and the report is told so;
   - TTM figures name the quarters they cover (2383: 2025 Q3～2026 Q2).
2. The offline test suite (every test; the three Windows-only lock tests are
   deselected because they cannot pass on macOS or Linux).
3. scripts/test_finmind.py: every FinMind function live, the AAPL fallback,
   the Yahoo price comparison, and the 2383 report-fix acceptance.

Needs FINMIND_TOKEN for step 1 (falls back to anonymous FinMind access like
test_finmind.py). Exits 1 if any check fails.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tradingagents.default_config as default_config  # noqa: E402
from tradingagents.dataflows import router  # noqa: E402
from tradingagents.dataflows.config import set_config  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def live_fixes() -> None:
    import os

    from tradingagents.agents.analysts import news_analyst
    from tradingagents.dataflows.vendors import finmind
    from tradingagents.reporting import _header

    section("1. Taiwan fixes, live (2383.TW vs AAPL)")
    config = default_config.build_default_config()
    config["data_cache_dir"] = tempfile.mkdtemp(prefix="tw-fixes-")
    set_config(config)
    if not os.getenv("FINMIND_TOKEN"):
        sys.path.insert(0, str(ROOT / "scripts"))
        import test_finmind
        print("  FINMIND_TOKEN is not set; using anonymous FinMind access for this check")
        test_finmind.use_anonymous_finmind()

    # Fix 1: the header shows the chain actually used, and routing agrees.
    settings = {"data_vendors": config["data_vendors"], "tool_vendors": config["tool_vendors"]}
    for ticker, must, must_not in [
        ("2383.TW", ["fundamental_data finmind,yfinance", "prediction_markets none (此來源不適用台股)"],
         ["sec_edgar", "polymarket"]),
        ("AAPL", ["fundamental_data finmind,sec_edgar,yfinance", "prediction_markets polymarket"], []),
    ]:
        line = next(row for row in _header(ticker, {}, settings).splitlines() if row.startswith("- Data vendors:"))
        print(f"  {ticker}: {line}")
        check(f"{ticker} report header shows the chain it uses",
              all(m in line for m in must) and not any(m in line for m in must_not))

    tried = []
    real = dict(router.VENDOR_METHODS["get_balance_sheet"])
    spies = {name: (lambda *a, _n=name, _f=fn, **k: (tried.append(_n), _f(*a, **k))[1])
             for name, fn in real.items()}
    with mock.patch.dict(router.VENDOR_METHODS["get_balance_sheet"], spies):
        out = router.route_to_vendor("get_balance_sheet", "2383.TW", "quarterly", "2026-10-16")
    check("2383 balance sheet is routed without sec_edgar", "sec_edgar" not in tried,
          f"vendors tried: {tried}; answer starts {out.splitlines()[0]!r}")

    # Fix 2: Polymarket is not offered for Taiwan.
    tw_tools = [t.name for t in news_analyst.offered_tools("2383.TW")]
    us_tools = [t.name for t in news_analyst.offered_tools("AAPL")]
    check("Polymarket not offered for 2383.TW", "get_prediction_markets" not in tw_tools, f"{tw_tools}")
    check("Polymarket still offered for AAPL", "get_prediction_markets" in us_tools, f"{us_tools}")
    check("the Taiwan notice names the source as not applicable",
          "Polymarket 此來源不適用台股" in news_analyst.PREDICTION_MARKETS_NOT_FOR_TAIWAN)

    # Fix 3: TTM names its quarters.
    fundamentals = finmind.get_fundamentals("2383.TW", "2026-10-16")
    ttm_lines = [row for row in fundamentals.splitlines() if "TTM" in row]
    print("  " + "\n  ".join(ttm_lines))
    check("2383 TTM period is 2025 Q3～2026 Q2",
          "TTM Period: 2025 Q3～2026 Q2 (the four latest published quarters, summed)" in fundamentals)
    unlabeled = [row for row in ttm_lines if "(TTM)" in row]
    check("every TTM figure names its span", not unlabeled, f"unlabeled: {unlabeled}")


def offline_suite() -> None:
    section("2. Offline test suite")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "not test_on_windows"],
        cwd=ROOT, capture_output=True, text=True,
    )
    summary = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"  {summary}")
    if proc.returncode != 0:
        print("\n".join(line for line in proc.stdout.splitlines() if line.startswith("FAILED")))
    check("offline test suite passes (Windows-only lock tests deselected)", proc.returncode == 0, summary)


def live_finmind_script() -> None:
    section("3. scripts/test_finmind.py (live)")
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "test_finmind.py")],
                          cwd=ROOT, capture_output=True, text=True)
    lines = proc.stdout.splitlines()
    summary = next((line for line in reversed(lines) if line.startswith("Summary:")), "(no summary)")
    print(f"  {summary}")
    for line in lines:
        if "[FAIL]" in line:
            print(line)
    check("test_finmind.py passes", proc.returncode == 0, summary)


def main() -> int:
    for step in (live_fixes, offline_suite, live_finmind_script):
        try:
            step()
        except Exception as exc:
            traceback.print_exc()
            check(step.__name__, False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if not r[1]]
    section(f"Summary: {len(results) - len(failed)}/{len(results)} checks passed")
    for name, _, detail in failed:
        print(f"  FAIL {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
