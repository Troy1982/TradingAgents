"""The rating backtest: universe draw, results store, settlement and metrics, on made-up data."""

import json
import subprocess

import pandas as pd
import pytest

from tradingagents.rating_backtest import metrics, runner, settle, universe
from tradingagents.rating_backtest.store import ResultsStore, cell_ratings, majority, run_row

DAYS = [f"2026-06-{d:02d}" for d in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17)]


def _fake_fetch(tables):
    def fetch(dataset, data_id, start, end=None, **_):
        frame = tables[(dataset, data_id)]
        if "date" in frame and dataset != "TaiwanStockInfo":
            frame = frame[(frame["date"] >= start) & ((end is None) | (frame["date"] <= (end or "9999")))]
        return frame.reset_index(drop=True)
    return fetch


# --- universe -----------------------------------------------------------------------------------

INFO = pd.DataFrame([
    {"stock_id": "1101", "type": "twse", "industry_category": "水泥工業", "stock_name": "A", "date": "2026-10-25"},
    {"stock_id": "2330", "type": "twse", "industry_category": "電子工業", "stock_name": "B", "date": "2026-10-25"},
    {"stock_id": "2330", "type": "twse", "industry_category": "半導體業", "stock_name": "B", "date": "2026-10-25"},
    {"stock_id": "8908", "type": "tpex", "industry_category": "油電燃氣業", "stock_name": "C", "date": "2026-10-18"},
    {"stock_id": "8908", "type": "twse", "industry_category": "油電燃氣業", "stock_name": "C", "date": "2026-10-25"},
    {"stock_id": "2888", "type": "twse", "industry_category": "金融保險", "stock_name": "D", "date": "2025-07-03"},
    {"stock_id": "5371", "type": "tpex", "industry_category": "光電業", "stock_name": "E", "date": "2026-08-26"},
    {"stock_id": "9103", "type": "twse", "industry_category": "存託憑證", "stock_name": "F", "date": "2026-10-25"},
    {"stock_id": "0050", "type": "twse", "industry_category": "ETF", "stock_name": "G", "date": "2026-10-25"},
    {"stock_id": "6666", "type": "emerging", "industry_category": "其他", "stock_name": "H", "date": "2026-10-25"},
])


@pytest.mark.unit
def test_listed_on_keeps_what_was_listed_that_day():
    listed = universe.listed_on(INFO, "2026-06-02").set_index("stock_id")
    # Delisted later (5371) stays; delisted before (2888), DRs, ETFs and emerging stocks go.
    assert sorted(listed.index) == ["1101", "2330", "5371", "8908"]
    assert listed.loc["8908", "market"] == "tpex"  # it moved to TWSE after the date
    assert listed.loc["2330", "industry"] == "半導體"  # the specific category, not 電子工業
    assert universe.industry_key("其他電子類") == universe.industry_key("其他電子業") == "其他電子"


def _prices(stock_id, money, first="2026-06-01", gaps=()):
    days = [d for d in DAYS if d >= first and d not in gaps]
    return pd.DataFrame({"date": days, "stock_id": stock_id, "Trading_Volume": 1000.0, "Trading_money": money,
                         "close": 10.0})


@pytest.mark.unit
def test_select_universe_draws_by_quota_cap_and_liquidity(monkeypatch):
    info = pd.DataFrame([
        {"stock_id": f"{1000 + i}", "type": "twse" if i < 6 else "tpex",
         "industry_category": "甲業" if i % 2 else "乙業", "stock_name": str(i), "date": "2026-10-25"}
        for i in range(10)])
    tables = {("TaiwanStockInfo", None): info,
              ("TaiwanStockTradingDate", None): pd.DataFrame({"date": DAYS})}
    for i in range(10):
        sid = f"{1000 + i}"
        tables[("TaiwanStockPrice", sid)] = (_prices(sid, 5e7) if i == 2 else       # too thin
                                              _prices(sid, 2e8, first="2026-06-10") if i == 3 else  # too new
                                              _prices(sid, 2e8, gaps=DAYS[2:5]) if i == 4 else      # halted
                                              _prices(sid, 2e8))
    monkeypatch.setattr(universe, "fetch", _fake_fetch(tables))
    args = ("2026-06-17", {"twse": 2, "tpex": 2}, 7, 2, 1e8, "2026-06-01", 10)
    drawn = universe.select_universe(*args)
    assert drawn == universe.select_universe(*args)  # the seed fixes the draw
    assert sum(s.market == "twse" for s in drawn) == 2 and sum(s.market == "tpex" for s in drawn) == 2
    assert not {"1002", "1003", "1004"} & {s.stock_id for s in drawn}
    assert max(pd.Series([s.industry for s in drawn]).value_counts()) <= 2
    assert all(s.ticker.endswith(".TW" if s.market == "twse" else ".TWO") for s in drawn)


# --- store --------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("ratings, expected", [
    (["Overweight", "Overweight", "Hold"], ("Overweight", 0.667)),
    (["Buy", "Hold", "Sell"], ("Hold", 0.333)),
    (["Underweight", "Overweight", "Hold"], ("Hold", 0.333)),
    (["Hold"], ("Hold", 1.0)),
    (["REVIEW", "Hold", "REVIEW"], ("Hold", 1.0)),
    (["REVIEW"], (None, 0.0)),
])
def test_majority(ratings, expected):
    assert majority(ratings) == expected


@pytest.mark.unit
def test_store_resumes_and_retries_failures(tmp_path):
    store = ResultsStore(tmp_path)
    store.append({"ticker": "2313.TW", "date": "2026-06-03", "run": 1, "status": "ok", "final_rating": "Hold"})
    store.append({"ticker": "2313.TW", "date": "2026-06-17", "run": 1, "status": "error"})
    with open(store.path, "a", encoding="utf-8") as handle:
        handle.write('{"ticker": "cut sh')  # a crash mid-line
    assert store.completed() == {("2313.TW", "2026-06-03", 1)}
    assert cell_ratings(store.rows()) == {("2313.TW", "2026-06-03"): ["Hold"]}


STATE = {
    "final_rating": "Overweight", "final_confidence": "medium",
    "coverage_cap": {"model_rating": "Buy", "model_confidence": "high", "changed": True, "enforced": True},
    "research_coverage_cap": {"model_rating": "Overweight"},
    "data_coverage": {"score": 0.82, "level": "high", "capped_by": ["monthly_revenue"]},
    "sentiment_report": "**Overall Sentiment:** **Insufficient Data** (Score: n/a)\n...",
    "run_settings": {"deep_think_llm": "gpt-6.1-sol", "quick_think_llm": "gpt-6-luna",
                     "max_debate_rounds": 1, "max_risk_discuss_rounds": 1},
}


@pytest.mark.unit
def test_run_row_reads_the_state_log():
    row = run_row(STATE, "2313.TW", "2026-06-03", 2, 301.4, "/x", "abc1234")
    assert row["final_rating"] == "Overweight" and row["model_rating"] == "Buy" and row["cap_applied"]
    assert row["capped_by"] == ["monthly_revenue"] and row["sentiment"] == "Insufficient Data"
    assert row["research_rating"] == "Overweight" and row["elapsed_s"] == 301 and row["run"] == 2


# --- runner -------------------------------------------------------------------------------------

PLAN = {"universe": [{"ticker": t} for t in ("A.TW", "B.TW", "C.TWO")], "dates": ["d1", "d2"], "runs": 2,
        "analysts": ["market", "chips"], "env": {"TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1"}, "commit": "abc"}


@pytest.mark.unit
def test_streams_split_tickers_and_run_every_cell_once_before_a_second_time():
    cells = runner.stream_cells(PLAN, 0, 2)
    assert {t for t, _, _ in cells} == {"A.TW", "C.TWO"}
    assert [r for _, _, r in cells] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert sorted(runner.stream_cells(PLAN, 1, 2) + cells) == sorted(runner.stream_cells(PLAN, 0, 1))


@pytest.mark.unit
def test_a_run_keeps_its_results_and_leaves_the_memory_log(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "")
    env = runner.run_environment(PLAN, tmp_path)
    assert env["TRADINGAGENTS_RESULTS_DIR"] == str(tmp_path) and env["TRADINGAGENTS_MEMORY_LOG_ENABLED"] == "false"
    assert "FRED_API_KEY" not in env  # left for .env to supply
    assert "--no-memory-log" in runner.cli_command(PLAN, "A.TW", "d1")

    def fake_run(cmd, env, **_):
        logs = tmp_path / "out" / "runs" / "A.TW_2026-06-03_r1" / "A.TW" / "TradingAgentsStrategy_logs"
        assert env["TRADINGAGENTS_RESULTS_DIR"] == str(logs.parents[1])
        logs.mkdir(parents=True)
        (logs / "full_states_log_2026-06-03.json").write_text(json.dumps(STATE), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    row = runner.run_cell(PLAN, tmp_path / "out", "A.TW", "2026-06-03", 1)
    assert row["status"] == "ok" and row["final_rating"] == "Overweight" and row["commit"] == "abc"

    monkeypatch.setattr(runner.subprocess, "run", lambda cmd, **_: subprocess.CompletedProcess(cmd, 1))
    assert runner.run_cell(PLAN, tmp_path / "out", "A.TW", "2026-06-04", 1)["status"] == "error"


# --- settlement ---------------------------------------------------------------------------------

def _series(closes, factors=None):
    return settle.Series(pd.Series(closes, index=DAYS[:len(closes)]), pd.Series(factors, dtype=float) if factors else settle._NO_DIVIDENDS)


@pytest.mark.unit
def test_holding_return_adds_back_dividends_and_flags_what_it_cannot_explain():
    # Ex-dividend on DAYS[2]: price drops from 100 to 95 after a 5 dividend.
    total, flags = settle.holding_return(_series([100, 100, 95, 95], {DAYS[2]: 100 / 95}), DAYS[0], DAYS[3])
    assert total == pytest.approx(0.0) and flags == []
    _, flags = settle.holding_return(_series([100, 100, 50, 50]), DAYS[0], DAYS[3])
    assert flags == ["unexplained_jump"]
    total, flags = settle.holding_return(_series([100, 110]), DAYS[0], DAYS[3])  # halted after DAYS[1]
    assert total == pytest.approx(0.1) and flags == ["stale_exit"]
    assert settle.holding_return(_series([100]), DAYS[1], DAYS[3]) == (None, ["no_entry_price"])


@pytest.mark.unit
def test_settle_enters_the_day_after_and_compares_with_the_stocks_market(monkeypatch):
    closes = [100 + i for i in range(len(DAYS))]
    tables = {
        ("TaiwanStockTradingDate", None): pd.DataFrame({"date": DAYS}),
        ("TaiwanStockPrice", "2313"): pd.DataFrame({"date": DAYS, "close": closes, "Trading_Volume": 1.0}),
        ("TaiwanStockDividendResult", "2313"): pd.DataFrame(columns=["date", "before_price", "after_price"]),
        ("TaiwanStockTotalReturnIndex", "TAIEX"): pd.DataFrame({"date": DAYS, "price": 1000.0}),
        ("TaiwanStockTotalReturnIndex", "TPEx"): pd.DataFrame({"date": DAYS, "price": 500.0}),
    }
    monkeypatch.setattr(settle, "fetch", _fake_fetch(tables))
    (row,) = settle.settle_cells([("2313.TW", DAYS[2])], {"2313.TW": "twse"}, DAYS[-1], horizons=(5, 20))
    assert row["entry"] == DAYS[3]
    assert row["ret_5"] == pytest.approx(closes[8] / closes[3] - 1) and row["bench_5"] == 0
    assert row["excess_5"] == row["ret_5"]
    assert row["excess_20"] is None  # not yet settled
    assert row["momentum_20"] is None  # not enough history before the date


# --- metrics ------------------------------------------------------------------------------------

def _cells():
    rows = []
    for date, shift in (("d1", 0.0), ("d2", 0.01), ("d3", -0.01), ("d4", 0.02)):
        for rating, excess, momentum in (("Overweight", 0.05, 0.3), ("Overweight", 0.03, 0.2),
                                         ("Hold", 0.0, 0.1), ("Hold", 0.01, -0.1), ("Underweight", -0.04, -0.2)):
            rows.append({"date": date, "rating": rating, "score": {"Overweight": 1, "Hold": 0, "Underweight": -1}[rating],
                         "excess_20": excess + shift, "momentum_20": momentum, "revenue_yoy": -momentum,
                         "agreement": 1.0 if rating != "Hold" else 0.667, "confidence": "medium"})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_the_metrics_read_a_rating_that_ranks_returns():
    cells = _cells()
    assert metrics.spread(cells, "excess_20")["mean"] == pytest.approx(0.08)
    assert metrics.rank_ic(cells, "excess_20")["mean"] > 0.8
    assert metrics.hit_rate(cells, "excess_20") == (1.0, 12)
    labels = metrics.matched_labels(cells, "momentum_20")
    assert (labels == "Overweight").sum() == 8 and (labels == "Underweight").sum() == 4
    result = metrics.evaluate(cells, horizons=(20,))[20]
    assert result["baselines"]["momentum_20"]["spread"]["mean"] == pytest.approx(0.08)
    assert result["baselines"]["revenue_yoy"]["spread"]["mean"] == pytest.approx(-0.065)
    assert result["baselines"]["always_overweight"]["mean_excess"] == pytest.approx(cells["excess_20"].mean())
    assert set(result["by_agreement"]["agreement"]) == {0.667, 1.0}


@pytest.mark.unit
def test_cell_table_takes_the_majority_of_runs():
    runs = [{"ticker": "A.TW", "date": "d1", "run": r, "status": "ok", "final_rating": rating,
             "final_confidence": "low", "cap_applied": r == 3, "capped_by": ["monthly_revenue"] if r == 3 else []}
            for r, rating in ((1, "Hold"), (2, "Overweight"), (3, "Overweight"))]
    settled = [{"ticker": "A.TW", "date": "d1", "market": "twse", "excess_20": 0.02}]
    (row,) = metrics.cell_table(runs, settled, [{"ticker": "A.TW", "market": "twse", "industry": "X"}],
                                revenue=lambda *_: 0.1).to_dict("records")
    assert row["rating"] == "Overweight" and row["agreement"] == 0.667 and row["ratings"] == "Hold/Overweight/Overweight"
    assert row["cap_applied"] == 1 and row["capped_by"] == "monthly_revenue" and row["excess_20"] == 0.02
