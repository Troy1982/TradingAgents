"""Text the program writes into a report, in the run's output language.

The agents write in ``output_language``; the lines the program writes itself
(the Data Coverage section, the Coverage Cap note, an Insufficient Data
sentiment report) follow it here, so a Traditional Chinese report has no
English paragraphs of the program's own. Labels the program reads back stay in
English in every language: ``**Rating**:``, ``**Confidence**:``,
``**Data Limitations**:``, ``**Coverage Cap**``, ``**Overall Sentiment:**``.

Traditional Chinese is translated; any other language gets the English text,
as before. Prompts keep the English text either way.
"""

from __future__ import annotations

import re

from tradingagents.dataflows.symbols import NOT_FOR_TAIWAN

EN, ZH = "en", "zh-Hant"


def report_language() -> str:
    """``zh-Hant`` for a Traditional Chinese run, else ``en``."""
    from tradingagents.dataflows.config import get_config

    lang = str(get_config().get("output_language") or "English").lower()
    if ("chinese" in lang or "中文" in lang) and "simplified" not in lang and "简" not in lang:
        return ZH
    return EN


_LEVEL = {"high": "高", "medium": "中", "low": "低", "unparsed": "無法判讀"}
_ANALYST = {"market": "市場", "social": "情緒", "news": "新聞", "fundamentals": "基本面", "chips": "籌碼"}
_STATUS = {
    "ok": "正常", "empty": "已確認無資料", "partial": "部分（較舊或缺）", "insufficient": "資料不足",
    "withheld": "歷史日期不提供", "unavailable": "無法取得", "no_data": "無資料", "not_offered": "未提供",
    "error": "錯誤", "missing_report": "缺少報告", "not_applicable": "不適用", "late_publication": "晚公告",
}
_TOOL = {
    "get_stock_data": "股價", "get_indicators": "技術指標", "get_verified_market_snapshot": "行情快照",
    "get_news": "公司新聞", "get_global_news": "全球新聞", "get_macro_indicators": "總經指標",
    "fred_macro": "美國總經（FRED）", "get_prediction_markets": "預測市場（Polymarket）",
    "get_fundamentals": "基本面摘要", "get_income_statement": "損益表", "get_balance_sheet": "資產負債表",
    "get_cashflow": "現金流量表", "get_insider_transactions": "內部人交易", "get_peer_valuation": "同業估值",
    "monthly_revenue": "月營收", "quarterly_statement": "季財報", "freshness": "資料新鮮度",
    "valuation": "估值（基本面摘要）", "prices": "股價（基本面摘要）", "listing": "上市資料",
    "get_chip_summary": "籌碼總覽", "get_institutional_flows": "三大法人", "get_margin_short": "融資融券",
    "get_foreign_holding": "外資持股", "get_securities_lending": "借券",
    "news": "新聞標題", "stocktwits": "StockTwits", "reddit": "Reddit", "sentiment": "情緒判讀",
}


def level(value: str) -> str:
    return _LEVEL.get(value, value) if report_language() == ZH else value


def analyst(key: str) -> str:
    return _ANALYST.get(key, key) if report_language() == ZH else key


def tool(name: str) -> str:
    if report_language() != ZH:
        return name
    timing = re.fullmatch(r"revenue_timing (\d{4}-\d{2})", name)
    return f"月營收公告時間 {timing.group(1)}" if timing else _TOOL.get(name, name)


def status(value: str, scored: bool = True) -> str:
    if report_language() != ZH:
        return value if scored else f"{value} (not scored)"
    label = _STATUS.get(value, value)
    return label if scored else f"{label}（不計分）"


# Program-written check details, read back into Chinese. A detail no rule knows
# is left out rather than shown in English: the status already says what happened.
_DETAIL_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"no historical source"), "歷史日期無來源（不是暫時性失敗，也不代表沒有新聞）"),
    (re.compile(r"only serves recent items"), "此來源只提供近期資料，無法涵蓋該期間"),
    (re.compile(r"FRED_API_KEY is not set: US macro"), "未設定 FRED_API_KEY：缺少美國總經資料；台灣指標（FinMind）已提供"),
    (re.compile(r"FRED_API_KEY"), "未設定 FRED_API_KEY，缺少總經資料"),
    (re.compile(re.escape(NOT_FOR_TAIWAN)), NOT_FOR_TAIWAN),
    (re.compile(r"Insufficient Data: (\d+) headlines in the window, (\d+) independent .*?at least (\d+) needed"),
     lambda m: f"期間內新聞 {m[1]} 則，去重後 {m[2]} 則獨立事件，少於判讀所需的 {m[3]} 則"),
    (re.compile(r"latest monthly revenue is (\S+); (\S+) was due by (\S+)"),
     lambda m: f"最新月營收為 {m[1]}，分析日 {m[3]} 應已有 {m[2]}"),
    (re.compile(r"no monthly revenue served; (\S+) was due by (\S+)"),
     lambda m: f"未取得月營收，分析日 {m[2]} 應已有 {m[1]}"),
    (re.compile(r"get_fundamentals was not called; (\S+) was due by (\S+)"),
     lambda m: f"未呼叫基本面摘要，分析日 {m[2]} 應已有 {m[1]} 月營收"),
    (re.compile(r"latest statement is for (\S+); the quarter ending (\S+) was due by (\S+)"),
     lambda m: f"最新財報為 {m[1]}，分析日 {m[3]} 應已有截至 {m[2]} 的季財報"),
    (re.compile(r"no statement period served; the quarter ending (\S+) was due by (\S+)"),
     lambda m: f"未取得財報期別，分析日 {m[2]} 應已有截至 {m[1]} 的季財報"),
    (re.compile(r"no fundamentals tool was called"), "未呼叫基本面工具，無法檢查新鮮度"),
    (re.compile(r"create_time (\S+) is (\d+) days after its deadline (\S+), past the (\d+) expected"),
     lambda m: f"公告日 {m[1]} 晚於期限 {m[3]} 共 {m[2]} 天（超過預期的 {m[4]} 天），以公告日為準"),
    (re.compile(r"withheld for this date"), "此來源不提供歷史日期的資料，以免前視偏誤"),
    (re.compile(r"the analyst filed no report"), "分析師未產出報告"),
    (re.compile(r"DATA_UNAVAILABLE|SECTION_UNAVAILABLE|unavailable|could not serve|request failed"),
     "資料來源暫時無法取得"),
]


def detail(text: str) -> str:
    """A check's detail in the report language (English as written; Chinese by rule, else nothing)."""
    text = str(text or "")
    if report_language() != ZH:
        return text
    for pattern, out in _DETAIL_RULES:
        found = pattern.search(text)
        if found:
            return out(found) if callable(out) else out
    return ""


def phrase(key: str, **values) -> str:
    """A sentence the program writes into a report."""
    return (_ZH_PHRASES if report_language() == ZH else _EN_PHRASES)[key].format(**values)


_EN_PHRASES = {
    "coverage_title": "**Data Coverage**: {level} (score {score})",
    "per_analyst": "Per analyst: {items}",
    "capped": "Capped at medium: critical data not fully served ({tools})",
    "scoring": ("Scoring: ok, empty 1; partial, insufficient 0.5; withheld, unavailable, no_data, not_offered, "
                "error, missing_report 0; not_applicable, late_publication and checks marked not scored are left out."),
    "table_head": "| Analyst | Tool | Status | Detail |",
    "no_gaps": "No data gaps.",
    "why_low": "data coverage is low (score {score})",
    "why_critical": "critical data not fully served: {tools}",
    "why_medium": "data coverage is medium (score {score})",
    "why_none": "data coverage allows any call",
    "move_call": "the call from {old} to {new}",
    "move_confidence": "confidence from {old} to {new}",
    "and": " and ",
    "cap_none": "**Coverage Cap**: no change ({why}; confidence ceiling {ceiling})",
    "cap_applied": "**Coverage Cap**: applied, moving {change} ({why})",
    "cap_shadow": "**Coverage Cap**: not applied (shadow mode); if applied it would move {change} ({why})",
    "rm_line": ("Research Manager: recommended {rating}, confidence {confidence}. The coverage cap is never "
                "applied to the research plan; {effect}."),
    "rm_would_move": "if applied, it would move {change} ({why})",
    "rm_no_change": "applied, it would change nothing ({why}; confidence ceiling {ceiling})",
    "pm_line": ("Portfolio Manager: the model's call was {rating}, confidence {confidence}. "
                "Final: {final_rating}, confidence {final_confidence}."),
    "rules_applied": "This rating was made under the data-gap rules: critical data was missing ({tools}).",
}

_ZH_PHRASES = {
    "coverage_title": "**資料覆蓋**：{level}（分數 {score}）",
    "per_analyst": "各分析師：{items}",
    "capped": "關鍵資料未完整取得，等級上限為中：{tools}",
    "scoring": ("計分方式：正常、已確認無資料計 1；部分、資料不足計 0.5；歷史日期不提供、無法取得、無資料、未提供、"
                "錯誤、缺少報告計 0；不適用、晚公告與標示不計分的項目不列入計分。"),
    "table_head": "| 分析師 | 資料項目 | 狀態 | 說明 |",
    "no_gaps": "無資料缺口。",
    "why_low": "資料覆蓋等級為低（分數 {score}）",
    "why_critical": "關鍵資料未完整取得：{tools}",
    "why_medium": "資料覆蓋等級為中（分數 {score}）",
    "why_none": "資料覆蓋足以支持任何評級",
    "move_call": "評級由 {old} 調為 {new}",
    "move_confidence": "信心由{old}調為{new}",
    "and": "，",
    "cap_none": "**Coverage Cap**：未調整（{why}；信心上限為{ceiling}）",
    "cap_applied": "**Coverage Cap**：已套用，{change}（{why}）",
    "cap_shadow": "**Coverage Cap**：未套用（影子模式）；若套用，將{change}（{why}）",
    "rm_line": "研究經理：建議 {rating}，信心{confidence}。評級上限不套用於研究計畫；{effect}。",
    "rm_would_move": "若套用，將{change}（{why}）",
    "rm_no_change": "即使套用也不會調整（{why}；信心上限為{ceiling}）",
    "pm_line": "投資組合經理：模型原始評級為 {rating}，信心{confidence}；最終評級為 {final_rating}，信心{final_confidence}。",
    "rules_applied": "本次評級受資料缺口規則約束：關鍵資料未完整取得（{tools}）。",
}
