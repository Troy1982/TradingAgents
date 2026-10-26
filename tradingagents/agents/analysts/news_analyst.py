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
from tradingagents.dataflows.vendors.finmind import client as finmind_client
from tradingagents.dataflows.vendors.finmind.macro import INDICATORS as TAIWAN_INDICATORS

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_news,
    get_global_news,
    get_macro_indicators,
    get_prediction_markets,
)

# Each of these asks for a line in the report, said in the report's own language
# and words: given a sentence to state, the model copied it into Chinese
# reports in English.
MACRO_UNAVAILABLE = (
    " Macroeconomic indicators from FRED are unavailable for this run because FRED_API_KEY "
    "is not set, so get_macro_indicators is not offered. In your own words and in the report's "
    "language (do not copy this instruction), say on a line of its own and in the summary table that "
    "this run has no macroeconomic data from FRED, and do not quote or estimate macro figures such as "
    "CPI, rates or unemployment."
)


TAIWAN_MACRO_ONLY = (
    " Taiwan market indicators from FinMind are available through get_macro_indicators ("
    + ", ".join(TAIWAN_INDICATORS) + "), but US and global series from FRED are not, because "
    "FRED_API_KEY is not set. In your own words and in the report's language (do not copy this "
    "instruction), say on a line of its own and in the summary table that Taiwan's macro data was read "
    "and US macro data was not, and do not quote or estimate US figures such as CPI, the fed funds rate "
    "or Treasury yields."
)


def _taiwan_macro(ticker: str) -> bool:
    """Whether FinMind can serve this run Taiwan's market indicators."""
    return is_taiwan_symbol(ticker) and finmind_client.is_configured()


PREDICTION_MARKETS_NOT_FOR_TAIWAN = (
    f" Prediction markets (Polymarket) are not offered for this Taiwan-listed stock: Polymarket lists "
    f"no market on Taiwan companies, and a keyword search returns unrelated contracts. In the report's "
    f"language, say on a line of its own and in the summary table that Polymarket {NOT_FOR_TAIWAN} "
    f"(this source does not apply to Taiwan stocks), and draw no probabilities or conclusions from "
    f"prediction markets."
)


def offered_tools(ticker: str = "") -> tuple:
    """The tools bound for this run. A tool that cannot serve it is left out and
    the gap is stated instead: the macro tool when neither FRED (no key) nor, for
    a Taiwan listing, FinMind (no token) can answer it, prediction markets for a
    Taiwan symbol (Polymarket has no Taiwan markets, so it would return
    unrelated contracts)."""
    withheld = set()
    if not fred.is_configured() and not _taiwan_macro(ticker):
        withheld.add(get_macro_indicators.name)
    if is_taiwan_symbol(ticker):
        withheld.add(get_prediction_markets.name)
    return tuple(t for t in TOOLS if t.name not in withheld)


def withheld_checks(ticker: str = "") -> list[dict]:
    """A data check for each tool ``offered_tools`` leaves out, saying why."""
    checks = []
    if not fred.is_configured() and _taiwan_macro(ticker):
        # The tool is offered for Taiwan's indicators; the US half is what is missing.
        checks.append(check("news", "fred_macro", "not_offered",
                            "FRED_API_KEY is not set: US macro series are missing; Taiwan indicators "
                            "(FinMind) are offered"))
    elif not fred.is_configured():
        checks.append(check("news", get_macro_indicators.name, "not_offered", "FRED_API_KEY is not set"))
    if is_taiwan_symbol(ticker):
        checks.append(check("news", get_prediction_markets.name, "not_applicable",
                            f"Polymarket lists no Taiwan markets ({NOT_FOR_TAIWAN})"))
    return checks


def _tool_guide(tools: tuple, asset_label: str, taiwan: bool = False) -> str:
    """How to use each offered tool, in one sentence listing only those offered."""
    market = ("Taiwan market, rate, currency and supply-chain news (it searches Taiwan's market for a Taiwan "
              "listing; a past date has no source for it)") if taiwan else "broader macroeconomic news"
    uses = {
        get_news.name: f"get_news(start_date, end_date) for news about the {asset_label} under analysis",
        get_global_news.name: f"get_global_news(curr_date, look_back_days, limit) for {market}",
        get_macro_indicators.name: (
            "get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual "
            "data: Taiwan's market from FinMind ('taiex', 'tpex', 'twd_usd', 'market_institutional', "
            "'market_margin')" + (" and US/global series from FRED (e.g. 'cpi', 'fed_funds_rate', "
                                   "'10y_treasury')" if fred.is_configured() else "")
            if taiwan else
            "get_macro_indicators(indicator, curr_date, look_back_days) to ground macro "
            "commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', "
            "'10y_treasury', 'yield_curve')"),
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
            + _tool_guide(tools, asset_label, is_taiwan_symbol(state.get("company_of_interest", "")))
            + " Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + (MACRO_UNAVAILABLE if get_macro_indicators.name not in offered else "")
            + (TAIWAN_MACRO_ONLY if get_macro_indicators.name in offered and not fred.is_configured() else "")
            + (PREDICTION_MARKETS_NOT_FOR_TAIWAN if get_prediction_markets.name not in offered else "")
            + " " + NEWS_NUMBERS_RULE
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction(labelled_lines=False)
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
