"""The coverage cap on the final decision: the rule, the Portfolio Manager, and every reader of the result."""

import copy
from unittest.mock import MagicMock

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.rating import apply_coverage_cap, parse_confidence, parse_rating
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating
from tradingagents.backtest import summarize
from tradingagents.dataflows.config import set_config
from tradingagents.memory import TradingMemoryLog
from tradingagents.reporting import write_report_tree

HIGH = {"score": 0.9, "level": "high", "capped_by": [], "by_analyst": {}, "gaps": []}
MEDIUM = {"score": 0.7, "level": "medium", "capped_by": [], "by_analyst": {}, "gaps": []}
LOW = {"score": 0.4, "level": "low", "capped_by": [], "by_analyst": {}, "gaps": []}
CRITICAL = {"score": 0.86, "level": "medium", "capped_by": ["monthly_revenue"], "by_analyst": {},
            "gaps": [{"analyst": "fundamentals", "tool": "monthly_revenue", "status": "unavailable",
                      "detail": "TaiwanStockMonthRevenue down"}]}
ENFORCE = {"enforce": True}


@pytest.fixture(autouse=True)
def _config():
    yield
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _enforce(on: bool):
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["data_coverage"]["enforce"] = on
    set_config(config)


# --- the rule -------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("coverage, rating, confidence, expected", [
    (HIGH, "Sell", "high", ("Sell", "high")),                  # nothing to cap
    (MEDIUM, "Sell", "high", ("Sell", "medium")),              # medium: confidence only
    (MEDIUM, "Buy", "low", ("Buy", "low")),
    (LOW, "Buy", "high", ("Overweight", "low")),               # low: the call and confidence
    (CRITICAL, "Sell", "high", ("Underweight", "low")),        # critical gap: the same
    (CRITICAL, "Hold", "medium", ("Hold", "low")),             # only Buy and Sell move
    (CRITICAL, "Overweight", "low", ("Overweight", "low")),
    (CRITICAL, "Sell", "unparsed", ("Underweight", "unparsed")),   # unknown confidence stays unknown
    (CRITICAL, "REVIEW", "unparsed", ("REVIEW", "unparsed")),
    (None, "Sell", "high", ("Sell", "high")),                  # no coverage measured
])
def test_the_cap_rule(coverage, rating, confidence, expected):
    cap = apply_coverage_cap(rating, confidence, coverage, ENFORCE)
    assert (cap["rating"], cap["confidence"]) == expected


@pytest.mark.unit
def test_shadow_mode_records_the_change_and_keeps_the_call():
    cap = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": False})
    assert (cap["rating"], cap["confidence"]) == ("Sell", "high")
    assert (cap["capped_rating"], cap["capped_confidence"]) == ("Underweight", "low")
    assert cap["note"] == ("**Coverage Cap**: not applied (shadow mode); if applied it would move the call from "
                           "Sell to Underweight and confidence from high to low (critical data not fully served: "
                           "monthly_revenue)")


@pytest.mark.unit
def test_the_triggers_and_mapping_follow_the_config():
    assert apply_coverage_cap("Sell", "high", CRITICAL, {**ENFORCE, "cap_on_critical": False})["rating"] == "Sell"
    assert apply_coverage_cap("Buy", "high", LOW, {**ENFORCE, "cap_on_low": False})["rating"] == "Buy"
    assert apply_coverage_cap("Sell", "high", LOW, {**ENFORCE, "rating_cap": {"Sell": "Hold"}})["rating"] == "Hold"


@pytest.mark.unit
@pytest.mark.parametrize("enforce", [False, True])
def test_the_cap_note_is_never_read_as_the_rating(enforce):
    cap = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": enforce})
    assert parse_rating(f"**Rating**: Sell\n\nThesis.\n\n{cap['note']}") == "Sell"


@pytest.mark.unit
@pytest.mark.parametrize("text, confidence", [
    ("**Confidence**: High", "high"), ("Confidence：medium", "medium"), ("- **Confidence:** low", "low"),
    ("No stated confidence.", "unparsed"), ("High confidence in the thesis", "unparsed"),
])
def test_confidence_is_read_from_its_label(text, confidence):
    assert parse_confidence(text) == confidence


# --- the Portfolio Manager, with a fake model -------------------------------------------

def _pm_state(coverage):
    return {
        "company_of_interest": "2383.TW", "past_context": "", "data_coverage": coverage,
        "risk_debate_state": {"history": "h", "aggressive_history": "", "conservative_history": "",
                              "neutral_history": "", "current_aggressive_response": "",
                              "current_conservative_response": "", "current_neutral_response": "", "count": 1},
        "investment_plan": "plan", "trader_investment_plan": "proposal",
    }


def _structured_llm(rating, confidence):
    decision = PortfolioDecision(rating=PortfolioRating(rating), executive_summary="Exit.",
                                 investment_thesis="Valuation stretched.", confidence=confidence,
                                 data_limitations="Monthly revenue unavailable; growth unconfirmed.")
    llm, prompts = MagicMock(), []
    llm.with_structured_output.return_value.invoke.side_effect = lambda p: prompts.append(p) or decision
    return llm, prompts


@pytest.mark.unit
def test_a_sell_with_high_confidence_and_a_critical_gap_becomes_underweight_low():
    _enforce(True)
    llm, prompts = _structured_llm("Sell", "high")
    out = create_portfolio_manager(llm)(_pm_state(CRITICAL))

    assert (out["final_rating"], out["final_confidence"]) == ("Underweight", "low")
    assert parse_rating(out["final_trade_decision"]) == "Underweight"
    assert ("**Coverage Cap**: applied, moving the call from Sell to Underweight and confidence from high to low"
            in out["final_trade_decision"])
    assert out["coverage_cap"]["model_rating"] == "Sell" and out["coverage_cap"]["enforced"] is True
    # The prompt carried the coverage and its ceiling.
    assert "**Data Coverage**" in prompts[0] and "at most **low**" in prompts[0]
    assert "| fundamentals | monthly_revenue | unavailable |" in prompts[0]


@pytest.mark.unit
def test_in_shadow_mode_the_call_stands_and_the_change_is_recorded():
    llm, _ = _structured_llm("Sell", "high")
    out = create_portfolio_manager(llm)(_pm_state(CRITICAL))
    assert (out["final_rating"], out["final_confidence"]) == ("Sell", "high")
    assert parse_rating(out["final_trade_decision"]) == "Sell"
    assert "not applied (shadow mode); if applied it would move the call from Sell to Underweight" in (
        out["final_trade_decision"])


@pytest.mark.unit
def test_the_free_text_path_meets_the_same_cap():
    _enforce(True)
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = MagicMock(content="**Rating**: Sell\n\n**Confidence**: high\n\nExit.")
    out = create_portfolio_manager(llm)(_pm_state(CRITICAL))
    assert (out["final_rating"], out["final_confidence"]) == ("Underweight", "low")
    assert parse_rating(out["final_trade_decision"]) == "Underweight"


@pytest.mark.unit
def test_free_text_with_no_confidence_is_unparsed_not_low():
    _enforce(True)
    llm = MagicMock()
    llm.with_structured_output.side_effect = NotImplementedError
    llm.invoke.return_value = MagicMock(content="**Rating**: Hold\n\nWait.")
    out = create_portfolio_manager(llm)(_pm_state(CRITICAL))
    assert (out["final_rating"], out["final_confidence"]) == ("Hold", "unparsed")


# --- the readers: memory log, backtest, report, CLI ------------------------------------

def _final_state(confidence="unparsed", enforce=True):
    cap = apply_coverage_cap("Sell", confidence, CRITICAL, {"enforce": enforce})
    rating = cap["rating"]
    decision = (f"**Rating**: {rating}\n\n" if cap["enforced"] and rating != "Sell" else "") + (
        f"**Rating**: Sell\n\n**Executive Summary**: Exit.\n\n{cap['note']}")
    return {"company_of_interest": "2383.TW", "trade_date": "2026-10-17", "final_trade_decision": decision,
            "final_rating": rating, "final_confidence": cap["confidence"], "coverage_cap": cap,
            "data_coverage": CRITICAL}


@pytest.mark.unit
@pytest.mark.parametrize("confidence", ["unparsed", "high"])
def test_the_memory_log_and_backtest_read_a_capped_decision(tmp_path, confidence):
    state = _final_state(confidence)
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "log.md")})
    log.store_decision("2383.TW", "2026-10-17", state["final_trade_decision"], rating=state["final_rating"])
    log.update_with_outcome("2383.TW", "2026-10-17", -0.02, 0.01, 5, "lesson", resolution_date="2026-10-24")
    [entry] = log.load_entries()
    assert entry["rating"] == "Underweight" and "Coverage Cap" in entry["decision"]
    assert "Underweight" in summarize(tmp_path / "log.md").by_rating


@pytest.mark.unit
def test_the_report_shows_the_coverage_and_what_the_cap_did(tmp_path):
    path = write_report_tree(_final_state(enforce=False), "2383.TW", tmp_path, html=True)
    text = path.read_text(encoding="utf-8")
    assert "- Confidence: unparsed" in text and "- Data coverage: medium (score 0.86)" in text
    assert "## 0. Data Coverage" in text and "partial, insufficient 0.5" in text
    assert ("Portfolio Manager: the model's call was Sell, confidence unparsed. "
            "Final: Sell, confidence unparsed.") in text
    assert "not applied (shadow mode)" in text
    assert "Data Coverage" in (tmp_path / "complete_report.html").read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_cli_shows_a_decision_with_an_unparsed_confidence(capsys):
    from cli.display import display_complete_report
    display_complete_report(_final_state())
    out = capsys.readouterr().out
    assert "Coverage Cap" in out and "Underweight" in out


# --- the Research Manager's plan: recorded, never capped ----------------------------------

@pytest.mark.unit
def test_the_research_plan_is_never_capped_but_the_report_says_what_the_cap_would_do(tmp_path):
    from tradingagents.agents.managers.research_manager import create_research_manager
    from tradingagents.agents.schemas import ResearchPlan
    _enforce(True)                     # even when the final decision's cap is applied
    plan = ResearchPlan(recommendation=PortfolioRating.SELL, rationale="Bear case.", strategic_actions="Exit.",
                        confidence="high", data_limitations="Monthly revenue unavailable.")
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = plan
    state = {"company_of_interest": "NVDA", "trade_date": "2026-10-17", "data_coverage": CRITICAL,
             "investment_debate_state": {"history": "h", "bull_history": "", "bear_history": "",
                                         "current_response": "", "count": 2}}
    out = create_research_manager(llm)(state)

    assert "**Recommendation**: Sell" in out["investment_plan"] and "**Confidence**: high" in out["investment_plan"]
    cap = out["research_coverage_cap"]
    assert (cap["model_rating"], cap["capped_rating"], cap["enforced"]) == ("Sell", "Underweight", False)

    report = {**_final_state(), "research_coverage_cap": cap}
    text = write_report_tree(report, "2383.TW", tmp_path, html=False).read_text(encoding="utf-8")
    assert ("Research Manager: recommended Sell, confidence high. The coverage cap is never applied to the "
            "research plan; if applied, it would move the call from Sell to Underweight and confidence from "
            "high to low (critical data not fully served: monthly_revenue).") in text


@pytest.mark.unit
@pytest.mark.parametrize("text, rating", [
    ("**Recommendation**: Overweight\n\n**Rationale**: r", "Overweight"),
    ("Recommendation：sell", "Sell"), ("No call here.", "REVIEW"),
])
def test_the_plans_recommendation_is_read_from_its_label(text, rating):
    from tradingagents.agents.rating import parse_recommendation
    assert parse_recommendation(text) == rating
