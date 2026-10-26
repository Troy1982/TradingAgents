"""The program's own report text in the run's output language; the labels it reads stay in English."""

import copy
import re

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents import coverage, i18n
from tradingagents.agents.headlines import count_independent
from tradingagents.agents.rating import apply_coverage_cap, parse_confidence, parse_rating
from tradingagents.dataflows.config import set_config
from tradingagents.reporting import coverage_section

CRITICAL = {"score": 0.89, "level": "medium", "capped_by": ["monthly_revenue"],
            "by_analyst": {"market": 1.0, "fundamentals": 0.75},
            "gaps": [{"analyst": "fundamentals", "tool": "monthly_revenue", "status": "unavailable",
                      "detail": "injected: TaiwanStockMonthRevenue unavailable. Report it as unavailable."},
                     {"analyst": "news", "tool": "get_global_news", "status": "unavailable",
                      "detail": "<Yahoo Finance global news unavailable for 2026-10-10..2026-10-17: no historical "
                                "source. Yahoo search serves recent articles only ...>"},
                     {"analyst": "news", "tool": "fred_macro", "status": "not_offered",
                      "detail": "FRED_API_KEY is not set: US macro series are missing; Taiwan indicators (FinMind) "
                                "are offered"},
                     {"analyst": "social", "tool": "news", "status": "ok", "detail": "", "scored": False}]}


@pytest.fixture
def chinese():
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["output_language"] = "Traditional Chinese"
    set_config(config)
    yield
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _english_words(text: str) -> list[str]:
    """Runs of three or more English words: what reads as an English sentence."""
    return re.findall(r"(?:[A-Za-z][A-Za-z'-]+\s+){2,}[A-Za-z][A-Za-z'-]+", text)


@pytest.mark.unit
def test_the_coverage_section_is_written_in_chinese(chinese):
    text = coverage.render_coverage_report(CRITICAL)
    assert text.startswith("**資料覆蓋**：中（分數 0.89）")
    assert "各分析師：市場 1.00、基本面 0.75" in text
    assert "關鍵資料未完整取得，等級上限為中：月營收" in text
    assert "| 基本面 | 月營收 | 無法取得 | 資料來源暫時無法取得 |" in text
    assert "| 新聞 | 全球新聞 | 無法取得 | 歷史日期無來源（不是暫時性失敗，也不代表沒有新聞） |" in text
    assert "| 新聞 | 美國總經（FRED） | 未提供 | 未設定 FRED_API_KEY：缺少美國總經資料；台灣指標（FinMind）已提供 |" in text
    assert "| 情緒 | 新聞標題 | 正常（不計分） |" in text
    assert _english_words(text.replace("FinMind", "").replace("FRED_API_KEY", "")) == []


@pytest.mark.unit
def test_the_prompts_keep_the_english_coverage(chinese):
    text = coverage.render_coverage(CRITICAL)
    assert text.startswith("**Data Coverage**: medium (score 0.89)") and "no historical source" in text


@pytest.mark.unit
@pytest.mark.parametrize("enforce", [True, False])
def test_the_cap_note_is_chinese_and_never_read_as_the_rating(chinese, enforce):
    cap = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": enforce})
    if enforce:
        assert cap["note"] == ("**Coverage Cap**：已套用，評級由 Sell 調為 Underweight，信心由高調為低"
                               "（關鍵資料未完整取得：月營收）")
    else:
        assert cap["note"].startswith("**Coverage Cap**：未套用（影子模式）；若套用，將評級由 Sell 調為 Underweight")
    decision = f"**Rating**: Sell\n\n**Confidence**: high\n\n**Data Limitations**: 缺月營收。\n\n{cap['note']}"
    assert parse_rating(decision) == "Sell" and parse_confidence(decision) == "high"


@pytest.mark.unit
def test_the_report_lines_on_both_managers_are_chinese(chinese):
    research = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": False})
    cap = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": True})
    text = coverage_section({"data_coverage": CRITICAL, "research_coverage_cap": research, "coverage_cap": cap,
                             "final_rating": "Underweight", "final_confidence": "low"})
    assert "本次評級受資料缺口規則約束：關鍵資料未完整取得（月營收）。" in text
    assert "研究經理：建議 Sell，信心高。評級上限不套用於研究計畫；若套用，將評級由 Sell 調為 Underweight" in text
    assert "投資組合經理：模型原始評級為 Sell，信心高；最終評級為 Underweight，信心低。" in text


@pytest.mark.unit
def test_an_insufficient_sentiment_is_chinese_under_english_labels(chinese):
    from tradingagents.agents.analysts.sentiment_analyst import insufficient_report
    counted = count_independent("## 2061.TW News (FinMind), from x:\n\n### 風青9月營收1億2911萬元 (source: S)\n")
    text = insufficient_report("2061.TW", "2026-10-10", "2026-10-17", counted, 5)
    lines = text.splitlines()
    assert lines[0] == "**Overall Sentiment:** **Insufficient Data** (Score: n/a)"
    assert lines[1] == "**Confidence:** Low"
    assert lines[2].startswith("**News Items:** 期間內新聞 1 則，去重後 1 則獨立事件")
    assert "本次不提供情緒判讀：這不是中性訊號" in text
    assert _english_words("\n".join(lines[2:]).replace("StockTwits 與 Reddit", "")) == []


@pytest.mark.unit
def test_english_runs_read_as_before():
    assert coverage.render_coverage_report(CRITICAL).startswith("**Data Coverage**: medium (score 0.89)")
    cap = apply_coverage_cap("Sell", "high", CRITICAL, {"enforce": True})
    assert cap["note"] == ("**Coverage Cap**: applied, moving the call from Sell to Underweight and confidence "
                           "from high to low (critical data not fully served: monthly_revenue)")
    assert i18n.report_language() == i18n.EN


@pytest.mark.unit
@pytest.mark.parametrize("language, code", [
    ("Traditional Chinese", i18n.ZH), ("繁體中文", i18n.ZH), ("Simplified Chinese", i18n.EN), ("Japanese", i18n.EN),
])
def test_which_languages_are_translated(language, code):
    config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    config["output_language"] = language
    set_config(config)
    assert i18n.report_language() == code
