"""Chip analyst (籌碼面): who is buying and selling a Taiwan listing.

Its data (institutional flows, margin, foreign holding, securities lending) is
published for TWSE/TPEx listings only. For any other symbol the node files a
one-line report saying so and calls no model, so a run that mixes markets
(a backtest grid, a programmatic caller) can keep it selected.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_chip_summary,
    get_foreign_holding,
    get_institutional_flows,
    get_margin_short,
    get_securities_lending,
)
from tradingagents.dataflows.symbols import is_taiwan_symbol

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_chip_summary,
    get_institutional_flows,
    get_margin_short,
    get_foreign_holding,
    get_securities_lending,
)

NOT_TAIWAN = (
    "Chip analysis (籌碼面) covers Taiwan-listed (TWSE/TPEx) stocks only; "
    "{ticker} is not one, so it was not analyzed."
)


def create_chip_analyst(llm):
    def chip_analyst_node(state):
        ticker = str(state["company_of_interest"])
        if not is_taiwan_symbol(ticker):
            note = NOT_TAIWAN.format(ticker=ticker)
            return {"messages": [AIMessage(content=note)], "chip_report": note}

        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        system_message = (
            "You are a Taiwan equity chip analyst (籌碼面分析師). Your job is to report who has been buying and"
            " selling this stock over roughly the last 20 trading days, from institutional flows, margin and short"
            " balances, foreign shareholding and securities lending."
            " Call `get_chip_summary` first: it joins price, volume and every chip series by day and flags the"
            " high-volume down days (爆量下跌日). Then use `get_institutional_flows`, `get_margin_short`,"
            " `get_foreign_holding` and `get_securities_lending` for the detail you need."
            " Your report must answer, with dates and figures from the tools:"
            " (1) on each high-volume down day, which institutional group (foreign investors, investment trusts,"
            " dealers) was the net seller and by how many lots;"
            " (2) whether margin balances rose on those days, i.e. retail margin buyers catching the falling knife"
            " (融資增加，散戶接刀), or fell as margin positions were cut;"
            " (3) the trend in foreign shareholding (外資持股比例) and the 20-day cumulative institutional flows."
            " Also assess whether foreign investors and investment trusts are moving together or against each other,"
            " the short/margin ratio (券資比), and whether securities lending has picked up, which can precede short"
            " selling. If the tools flag no high-volume down day, say so and describe the flows instead."
            " Every figure must come from the tool output, in the units it gives (lots, 張); if a tool reports data"
            " as unavailable, say so rather than estimating."
            " Make sure to append a Markdown table at the end of the report to organize key points in the report,"
            " organized and easy to read."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "chip_report": report,
        }

    return chip_analyst_node
