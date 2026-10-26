"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.context import (
    coverage_context,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.rating import apply_coverage_cap, parse_confidence, parse_recommendation
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.tools import get_peer_valuation
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.symbols import is_taiwan_symbol


def peer_valuation_block(state) -> str:
    """The peer valuation table for a Taiwan listing, as a prompt section; nothing otherwise.

    Fetched here rather than offered as a tool: the manager weighs the debate and
    calls no tools. It is the same request the fundamentals analyst makes, so it
    is served from the FinMind cache.
    """
    ticker = str(state.get("company_of_interest", ""))
    if not is_taiwan_symbol(ticker):
        return ""
    trade_date = state.get("trade_date", "")
    try:
        table = get_peer_valuation.func(ticker, trade_date, trade_date)
    except Exception as exc:  # noqa: BLE001 - context for the manager; never end the run on it
        table = f"DATA_UNAVAILABLE: peer valuation could not be retrieved ({exc})."
    return (
        "\n**Peer Valuation (TaiwanStockPER)** - reference data, not a debate argument; use it to judge"
        " whether the valuation case either side made holds against listed peers:\n"
        f"{table}\n\n---\n"
    )


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]
        peers = peer_valuation_block(state)

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis; recommend taking or growing the position
- **Overweight**: Constructive view; recommend gradually increasing exposure
- **Hold**: Balanced view; recommend maintaining the current position
- **Underweight**: Cautious view; recommend trimming exposure
- **Sell**: Strong conviction in the bear thesis; recommend exiting or avoiding the position

The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger case, sized by how decisively it wins. Choose Hold only when the evidence is still balanced after that weighing, or too thin to support a call; do not manufacture a direction to appear decisive. Weigh the bull and bear cases on their merits, independent of which side spoke first or last.

---
{peers}
**Debate History:**
{history}

---

{coverage_context(state)}
## Output

Write these sections, in this order, starting with the recommendation on its own line:

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments decided it
- **Strategic Actions**: concrete steps for the trader, sized against a standard allocation
- **Confidence**: exactly one of low / medium / high, no higher than the data coverage allows
- **Data Limitations**: which data the plan lacked and how each gap limits it

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        # What the coverage cap would make of the plan, for the report only: the
        # cap is never applied to the plan, which the trader reads as written.
        research_cap = apply_coverage_cap(
            parse_recommendation(investment_plan), parse_confidence(investment_plan),
            state.get("data_coverage"), {**(get_config().get("data_coverage") or {}), "enforce": False})

        new_investment_debate_state = {
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "research_coverage_cap": research_cap,
        }

    return research_manager_node
