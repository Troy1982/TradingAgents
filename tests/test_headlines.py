"""Counting independent news stories, and keeping an unread sentiment out of the debate."""

import pytest

from tradingagents.agents import headlines
from tradingagents.agents.analysts.sentiment_analyst import insufficient_report
from tradingagents.agents.context import sentiment_report_for_debate
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.schemas import SentimentBand


def _block(*titles):
    return "## 2383.TW News (FinMind), from x:\n\n" + "".join(f"### {t} (source: S)\nLink: l\n\n" for t in titles)


@pytest.mark.unit
@pytest.mark.parametrize("a, b", [
    ("台光電：9月營收再飆新高！ 客戶排隊爭搶產能 - 自由時報", "台光電：9月營收再飆新高！客戶排隊爭搶產能 - 自由時報"),
    ("台光電(2383)、聯茂(6213)、台燿(6274)CCL三雄擴產盤點！AI高階材料 - 經濟日報",
     "台光電（2383）、聯茂（6213）、台燿（6274）CCL三雄擴產盤點！AI高階材料 - money.udn.com"),
    ("台光電、台燿 四檔看旺 - 經濟日報", "台光電、台燿 四檔看旺 - money.udn.com"),
])
def test_reposts_of_one_story_count_once(a, b):
    assert headlines.count_independent(_block(a, b)).independent == 1


@pytest.mark.unit
def test_different_stories_are_kept_apart():
    count = headlines.count_independent(_block(
        "台光電擴產泰國新廠", "外資調升台光電目標價", "銅箔基板族群走弱"))
    assert (count.headlines, count.independent) == (3, 3)


@pytest.mark.unit
@pytest.mark.parametrize("title", [
    "2383 台光電- 建滔再漲價15% - 股市爆料同學會 - CMoney",
    "2383 台光電 - -202 / -66 / +9 - 股",
    "台光電 討論區",
    "台光電(2383) 今日股價與討論 - 股市爆料同學會 - CMoney",
])
def test_entry_pages_are_not_stories(title):
    assert headlines.is_entry_page(title)


@pytest.mark.unit
@pytest.mark.parametrize("title", [
    "期貨商論壇／台光電期 動能強 - UDN",          # a newspaper column
    "台光電擴產引發市場熱烈討論",
])
def test_a_story_that_mentions_a_forum_is_still_a_story(title):
    assert not headlines.is_entry_page(title)


@pytest.mark.unit
def test_entry_pages_together_count_as_one():
    count = headlines.count_independent(_block(
        "2383 台光電 - 股市爆料同學會", "2383 台光電 - -202 / -66 / +9 - 股", "外資調升台光電目標價"))
    assert (count.headlines, count.independent, count.entry_pages) == (3, 2, 2)
    assert "3 headlines in the window, 2 independent after de-duplication" in count.describe()


@pytest.mark.unit
def test_a_placeholder_holds_no_headlines():
    assert headlines.count_independent("<FinMind news unavailable for x: down>").headlines == 0


# --- the band, the score and the debate -----------------------------------------------

@pytest.mark.unit
def test_an_unread_sentiment_renders_without_a_score():
    count = headlines.count_independent(_block("外資調升台光電目標價"))
    text = insufficient_report("2383.TW", "2026-10-09", "2026-10-16", count, 5)
    assert text.startswith("**Overall Sentiment:** **Insufficient Data** (Score: n/a)")
    assert "1 headlines in the window, 1 independent" in text


@pytest.mark.unit
def test_the_model_cannot_choose_insufficient_data():
    assert "Insufficient Data" not in {band.value for band in SentimentBand}


@pytest.mark.unit
def test_the_debate_reads_an_unread_sentiment_as_no_finding():
    report = ("**Overall Sentiment:** **Insufficient Data** (Score: n/a)\n**Confidence:** Low\n\n"
              "Stories counted:\n- 外資調升台光電目標價")
    text = sentiment_report_for_debate({"sentiment_report": report})
    assert "Insufficient Data" in text and "do not cite sentiment" in text
    assert "外資調升台光電目標價" not in text
    normal = "**Overall Sentiment:** **Mildly Bullish** (Score: 6.0/10)\nnarrative"
    assert sentiment_report_for_debate({"sentiment_report": normal}) == normal


@pytest.mark.unit
def test_the_bull_researcher_is_told_the_sentiment_was_not_read():
    seen = []

    class LLM:
        def invoke(self, prompt):
            seen.append(prompt)
            return type("R", (), {"content": "arg"})()

    state = {
        "investment_debate_state": {"history": "", "bull_history": "", "bear_history": "",
                                    "current_response": "", "count": 0},
        "market_report": "m", "news_report": "n", "fundamentals_report": "f",
        "sentiment_report": "**Overall Sentiment:** **Insufficient Data** (Score: n/a)\n- 外資調升目標價",
        "company_of_interest": "2383.TW",
    }
    create_bull_researcher(LLM())(state)
    assert "Sentiment: Insufficient Data" in seen[0] and "外資調升目標價" not in seen[0]
