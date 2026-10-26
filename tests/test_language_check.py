"""An agent's output held to the run's language: rewritten once, then flagged, never stopped."""

import copy

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.language_check import cjk_share, in_output_language
from tradingagents.dataflows.config import set_config

ENGLISH = ("Bull Analyst: The bull case for 2881.TW rests on resilient insurance income and a recovering "
           "investment book. Valuation remains reasonable at current levels, and the dividend policy supports "
           "holders through volatility. ") * 3
CHINESE = ("多方論點：2881.TW 的保險收入穩健，投資部位回升，目前估值仍屬合理，股利政策也有助於持有人度過波動。" * 6)


@pytest.fixture
def chinese_run():
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["output_language"] = "Traditional Chinese"
    set_config(config)
    yield
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


@pytest.mark.unit
def test_the_share_counts_chinese_against_latin_letters():
    assert cjk_share(ENGLISH) == 0.0 and cjk_share(CHINESE) > 0.9
    assert cjk_share("台積電 TSMC 2330.TW 的毛利率") == pytest.approx(7 / 13)
    assert cjk_share("見 https://example.com/a-very-long-english-url 說明") == 1.0


@pytest.mark.unit
def test_an_english_turn_in_a_chinese_run_is_rewritten_once(chinese_run):
    asked = []
    out = in_output_language(ENGLISH, lambda prompt: asked.append(prompt) or CHINESE, "Bull Analyst")
    assert out == CHINESE and len(asked) == 1
    assert "Rewrite the text below entirely in Traditional Chinese" in asked[0] and ENGLISH in asked[0]


@pytest.mark.unit
def test_a_rewrite_still_in_english_is_flagged_and_the_original_kept(chinese_run):
    asked = []
    out = in_output_language(ENGLISH, lambda prompt: asked.append(prompt) or ENGLISH, "Bull Analyst")
    assert len(asked) == 1
    assert out.startswith("> ⚠️ 語言異常：此段（Bull Analyst）未以設定的語言（Traditional Chinese）撰寫")
    assert out.endswith(ENGLISH)


@pytest.mark.unit
def test_a_rewrite_that_changes_a_rating_line_is_not_taken(chinese_run):
    decision = "**Rating**: Sell\n\n" + ENGLISH
    out = in_output_language(decision, lambda prompt: "**Rating**: Buy\n\n" + CHINESE, "Portfolio Manager")
    assert "語言異常" in out and "**Rating**: Sell" in out
    kept = in_output_language(decision, lambda prompt: "**Rating**: Sell\n\n" + CHINESE, "Portfolio Manager")
    assert kept == "**Rating**: Sell\n\n" + CHINESE


@pytest.mark.unit
def test_a_failed_rewrite_does_not_stop_the_run(chinese_run):
    def broken(prompt):
        raise RuntimeError("provider down")
    assert "語言異常" in in_output_language(ENGLISH, broken, "Trader")


@pytest.mark.unit
def test_chinese_output_and_english_runs_are_left_alone(chinese_run):
    calls = []
    assert in_output_language(CHINESE, lambda p: calls.append(p), "Bull Analyst") == CHINESE
    assert in_output_language("Too short.", lambda p: calls.append(p), "Bull Analyst") == "Too short."
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))     # English
    assert in_output_language(ENGLISH, lambda p: calls.append(p), "Bull Analyst") == ENGLISH
    assert calls == []


@pytest.mark.unit
def test_a_bull_researcher_answering_in_english_is_rewritten(chinese_run):
    from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
    replies = iter([ENGLISH, CHINESE])

    class LLM:
        def invoke(self, prompt):
            return type("R", (), {"content": next(replies)})()

    state = {"company_of_interest": "2881.TW", "trade_date": "2026-08-31", "instrument_context": "",
             "market_report": "m", "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
             "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "",
                                         "current_response": "", "count": 0}}
    out = create_bull_researcher(LLM())(state)
    assert out["investment_debate_state"]["current_response"] == f"Bull Analyst: {CHINESE}"
