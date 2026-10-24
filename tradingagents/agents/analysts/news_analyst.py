from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import (
    NEWS_NUMBERS_RULE,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.coverage import check
from tradingagents.agents.tools import (
    get_global_news,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.dataflows.symbols import NOT_FOR_TAIWAN, is_taiwan_symbol
from tradingagents.dataflows.vendors import fred

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)

MACRO_UNAVAILABLE = (
    " Macroeconomic indicators from FRED are unavailable for this run because FRED_API_KEY "
    "is not set, so get_macro_indicators is not offered. State in the report, in its own "
    "line and in the summary table, that macroeconomic data (FRED) is missing for this run, "
    "and do not quote or estimate macro figures such as CPI, rates or unemployment."
)


PREDICTION_MARKETS_NOT_FOR_TAIWAN = (
    f" Prediction markets (Polymarket) are not offered for this Taiwan-listed stock: "
    f"{NOT_FOR_TAIWAN}. Polymarket lists no market on Taiwan companies, and a keyword search "
    f"returns unrelated contracts. State in the report, in its own line and in the summary "
    f"table, that Polymarket {NOT_FOR_TAIWAN} (this source does not apply to Taiwan stocks), "
    f"and draw no probabilities or conclusions from prediction markets."
)


def offered_tools(ticker: str = "") -> tuple:
    """The tools bound for this run. A tool that cannot serve it is left out and
    the gap is stated instead: the macro tool without a FRED key (it would only
    return 'unavailable'), prediction markets for a Taiwan symbol (Polymarket has
    no Taiwan markets, so it would return unrelated contracts)."""
    withheld = set()
    if not fred.is_configured():
        withheld.add(get_macro_indicators.name)
    if is_taiwan_symbol(ticker):
        withheld.add(get_prediction_markets.name)
    return tuple(t for t in TOOLS if t.name not in withheld)


def withheld_checks(ticker: str = "") -> list[dict]:
    """A data check for each tool ``offered_tools`` leaves out, saying why."""
    checks = []
    if not fred.is_configured():
        checks.append(check("news", get_macro_indicators.name, "not_offered", "FRED_API_KEY is not set"))
    if is_taiwan_symbol(ticker):
        checks.append(check("news", get_prediction_markets.name, "not_applicable",
                            f"Polymarket lists no Taiwan markets ({NOT_FOR_TAIWAN})"))
    return checks


def _tool_guide(tools: tuple, asset_label: str) -> str:
    """How to use each offered tool, in one sentence listing only those offered."""
    uses = {
        get_news.name: f"get_news(start_date, end_date) for news about the {asset_label} under analysis",
        get_global_news.name: "get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news",
        get_macro_indicators.name: "get_macro_indicators(indicator, curr_date, look_back_days) to ground macro "
        "commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', "
        "'10y_treasury', 'yield_curve')",
        get_prediction_markets.name: "get_prediction_markets(topic, limit) for live market-implied probabilities "
        "of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events)",
    }
    parts = [uses[t.name] for t in tools]
    listed = parts[0] if len(parts) == 1 else (
        " and ".join(parts) if len(parts) == 2 else ", ".join(parts[:-1]) + ", and " + parts[-1])
    return f"Use the available tools: {listed}."


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = offered_tools(state.get("company_of_interest", ""))
        offered = {t.name for t in tools}
        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. "
            + _tool_guide(tools, asset_label)
            + " Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + (MACRO_UNAVAILABLE if get_macro_indicators.name not in offered else "")
            + (PREDICTION_MARKETS_NOT_FOR_TAIWAN if get_prediction_markets.name not in offered else "")
            + " " + NEWS_NUMBERS_RULE
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
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
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, tools, state["messages"])

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
