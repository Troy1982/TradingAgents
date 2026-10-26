"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call. Its rating is the run's
``final_rating``, and the decision is rendered to markdown as
``final_trade_decision`` for the memory log, CLI display and saved reports.
When a provider does not expose structured output, the agent falls back to
free-text generation and the rating is read from that text, and its confidence
too (``unparsed`` when it states none).

Either way the decision then meets the coverage cap (``apply_coverage_cap``):
the run's data coverage may move the rating and hold the confidence down. In
shadow mode (``data_coverage.enforce`` false, the default) the decision only
records what the cap would change; ``final_rating`` stays the model's.
"""

from __future__ import annotations

from tradingagents.agents.context import (
    coverage_context,
    data_gap_rules,
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.language_check import in_output_language, rewriter
from tradingagents.agents.rating import apply_coverage_cap, parse_confidence, parse_rating
from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.config import get_config


def _with_cap(decision_text: str, cap: dict) -> str:
    """The decision text with the cap's note, and the capped rating leading it when the cap was applied.

    The rating parser reads a decision's opening rating line, so an applied cap
    that moved the call puts its rating first; the model's own line stays below,
    named as the model's in the note.
    """
    text = f"{decision_text}\n\n{cap['note']}"
    if cap["enforced"] and cap["capped_rating"] != cap["model_rating"]:
        text = f"**Rating**: {cap['rating']}\n\n{text}"
    return text


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

{portfolio_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

{coverage_context(state)}{data_gap_rules(state, decision=True)}
Ground every conclusion in specific evidence from the analysts. The risk debate always contains conflicting stances; deciding which is stronger is the job, so conflict alone is not a reason to Hold. Commit to the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not force a direction to appear decisive. Weigh the analysts on their merits, independent of speaking order.

## Output

Write these sections, in this order, starting with the rating on its own line:

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Investment Thesis**: the evidence that decided it, and what would change it
- **Confidence**: exactly one of low / medium / high, no higher than the data coverage allows
- **Data Limitations**: which data the call lacked and how each gap limits it

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        # The typed rating is the decision; the rendered text only carries it.
        # Read back from text, a rating the thesis quotes could replace it.
        decision = invoke_structured(structured_llm, prompt, "Portfolio Manager")
        if decision is not None:
            decision_text = render_pm_decision(decision)
            model_rating, model_confidence = decision.rating.value, decision.confidence
        else:
            decision_text = llm.invoke(prompt).content
            model_rating, model_confidence = parse_rating(decision_text), parse_confidence(decision_text)

        decision_text = in_output_language(decision_text, rewriter(llm), "Portfolio Manager")
        cap = apply_coverage_cap(model_rating, model_confidence, state.get("data_coverage"),
                                 get_config().get("data_coverage") or {})
        final_trade_decision = _with_cap(decision_text, cap)
        final_rating, final_confidence = cap["rating"], cap["confidence"]

        new_risk_debate_state = {
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "final_rating": final_rating,
            "final_confidence": final_confidence,
            "coverage_cap": cap,
        }

    return portfolio_manager_node
