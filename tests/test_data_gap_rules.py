"""The data-gap rules: in the debate and decision prompts only when critical data is missing."""

import importlib
from unittest.mock import MagicMock

import pytest

from tradingagents.agents import context
from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating, ResearchPlan

HIGH = {"score": 0.96, "level": "high", "capped_by": [], "by_analyst": {}, "gaps": []}
CRITICAL = {"score": 0.89, "level": "medium", "capped_by": ["monthly_revenue"], "by_analyst": {},
            "gaps": [{"analyst": "fundamentals", "tool": "monthly_revenue", "status": "unavailable", "detail": "down"}]}

DEBATERS = [
    ("tradingagents.agents.researchers.bull_researcher", "create_bull_researcher"),
    ("tradingagents.agents.researchers.bear_researcher", "create_bear_researcher"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
]


def _state(coverage):
    state = {
        "company_of_interest": "2368.TW", "trade_date": "2026-10-17", "instrument_context": "",
        "market_report": "m", "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
        "trader_investment_plan": "plan", "investment_plan": "rp", "past_context": "",
        "investment_debate_state": {"history": "h", "bull_history": "", "bear_history": "",
                                    "current_response": "", "count": 0},
        "risk_debate_state": {"history": "h", "aggressive_history": "", "conservative_history": "",
                              "neutral_history": "", "latest_speaker": "", "current_aggressive_response": "",
                              "current_conservative_response": "", "current_neutral_response": "", "count": 0},
    }
    if coverage is not None:
        state["data_coverage"] = coverage
    return state


def _debater_prompt(module, factory, state):
    seen = []

    class LLM:
        def invoke(self, prompt):
            seen.append(prompt)
            return type("R", (), {"content": "argument"})()

    getattr(importlib.import_module(module), factory)(LLM())(state)
    return seen[0]


def _manager_prompt(module, factory, schema_value, state):
    llm, seen = MagicMock(), []
    llm.with_structured_output.return_value.invoke.side_effect = lambda p: seen.append(p) or schema_value
    getattr(importlib.import_module(module), factory)(llm)(state)
    return seen[0]


_PLAN = ResearchPlan(recommendation=PortfolioRating.HOLD, rationale="r", strategic_actions="a",
                     confidence="low", data_limitations="d")
_DECISION = PortfolioDecision(rating=PortfolioRating.HOLD, executive_summary="s", investment_thesis="t",
                              confidence="low", data_limitations="d")
MANAGERS = [
    ("tradingagents.agents.managers.research_manager", "create_research_manager", _PLAN),
    ("tradingagents.agents.managers.portfolio_manager", "create_portfolio_manager", _DECISION),
]


def _prompts(state):
    return ([_debater_prompt(m, f, state) for m, f in DEBATERS]
            + [_manager_prompt(m, f, v, state) for m, f, v in MANAGERS])


@pytest.mark.unit
@pytest.mark.parametrize("coverage", [HIGH, None], ids=["no-critical-gap", "no-coverage"])
def test_without_a_critical_gap_every_prompt_is_unchanged(monkeypatch, coverage):
    with_rules = _prompts(_state(coverage))
    monkeypatch.setattr(context, "data_gap_rules", lambda *a, **k: "")
    for module in [m for m, _ in DEBATERS] + [m for m, _, _ in MANAGERS]:
        monkeypatch.setattr(importlib.import_module(module), "data_gap_rules", lambda *a, **k: "")
    without = _prompts(_state(coverage))
    assert with_rules == without
    assert not any("Critical data is missing" in p for p in with_rules)


@pytest.mark.unit
def test_with_a_critical_gap_every_debater_and_manager_reads_the_rules():
    for prompt in _prompts(_state(CRITICAL)):
        assert "**Critical data is missing for this run** (monthly_revenue)" in prompt
        assert "A data gap is uncertainty, not evidence" in prompt
        assert "The default rating is Hold." in prompt
        assert "does not apply to an advantage one side has only because the other side's evidence is missing" \
            in prompt


@pytest.mark.unit
def test_only_the_portfolio_manager_is_asked_how_the_gaps_bear_on_its_rating():
    prompts = _prompts(_state(CRITICAL))
    asked = [("say whether this rating was affected by the missing data" in p) for p in prompts]
    assert asked == [False] * (len(DEBATERS) + 1) + [True]


@pytest.mark.unit
def test_the_rules_are_written_once():
    import inspect
    for module in [m for m, _ in DEBATERS] + [m for m, _, _ in MANAGERS]:
        src = inspect.getsource(importlib.import_module(module))
        assert "data_gap_rules(state" in src and "A data gap is uncertainty" not in src
