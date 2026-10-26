"""Hold an agent's output to the run's language: one rewrite, then a note in the report.

Models told to write in Chinese sometimes answer in English (a whole bull
researcher turn on 2881.TW, 2026-08-31). For a Chinese run, an output whose
share of Chinese characters is below ``language_check.min_cjk_share`` (config,
0.3; reports read 0.6-0.9) is rewritten once by the same model. The rewrite must
keep every label line the program reads with the same value; if it does not,
or is still not Chinese, the original stands with a note at its top saying so.
The run is never stopped for it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

logger = logging.getLogger(__name__)

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_LATIN = re.compile(r"[A-Za-z]")
_URL = re.compile(r"https?://\S+")

# Label lines the program reads back; a rewrite must keep each with its value.
_LABELS = re.compile(
    r"^\W*(Rating|Recommendation|Confidence|FINAL TRANSACTION PROPOSAL|Overall Sentiment)\W*[:：]\W*(\w+)",
    re.MULTILINE | re.IGNORECASE)

# Shorter outputs (a one-line note, a refusal) are not judged.
_MIN_LENGTH = 200


def _settings() -> tuple[str, float]:
    from tradingagents.dataflows.config import get_config

    config = get_config()
    return str(config.get("output_language") or "English"), float(
        (config.get("language_check") or {}).get("min_cjk_share", 0.3))


def expects_chinese(language: str) -> bool:
    return "chinese" in language.lower() or "中文" in language


def cjk_share(text: str) -> float:
    """Chinese characters over Chinese characters and Latin letters, links left out."""
    text = _URL.sub("", text or "")
    cjk, latin = len(_CJK.findall(text)), len(_LATIN.findall(text))
    return cjk / (cjk + latin) if cjk + latin else 1.0


def _labels(text: str) -> list[tuple[str, str]]:
    return [(name.lower(), value.lower()) for name, value in _LABELS.findall(text or "")]


def _rewrite_prompt(text: str, language: str) -> str:
    return (
        f"Rewrite the text below entirely in {language}. Keep its meaning, every figure, ticker, date and "
        f"table, and add or drop nothing. Keep these labelled lines exactly as they are, label and value, in "
        f"English: \"**Rating**:\", \"**Recommendation**:\", \"**Confidence**:\", \"**Overall Sentiment:**\", "
        f"\"FINAL TRANSACTION PROPOSAL:\". Reply with the rewritten text only.\n\n{text}"
    )


def in_output_language(text: str, rewrite: Callable[[str], str], agent: str) -> str:
    """``text`` in the run's language: as given, rewritten once by ``rewrite(prompt)``, or flagged."""
    language, minimum = _settings()
    if not expects_chinese(language) or len(text or "") < _MIN_LENGTH or cjk_share(text) >= minimum:
        return text
    logger.warning("%s wrote %.0f%% Chinese against %.0f%% expected; asking for a rewrite",
                   agent, 100 * cjk_share(text), 100 * minimum)
    try:
        rewritten = rewrite(_rewrite_prompt(text, language)) or ""
    except Exception as exc:  # noqa: BLE001 - a failed rewrite leaves the original, flagged
        logger.warning("%s: language rewrite failed: %s", agent, exc)
        rewritten = ""
    # An empty answer has no letters to judge, so it would pass the share: it must say something.
    if rewritten.strip() and cjk_share(rewritten) >= minimum and _labels(rewritten) == _labels(text):
        return rewritten
    logger.warning("%s: output still not in %s after one rewrite; flagged in the report", agent, language)
    return (f"> ⚠️ 語言異常：此段（{agent}）未以設定的語言（{language}）撰寫，重寫一次後仍未改善，以下保留原文。\n\n"
            + text)


def rewriter(llm) -> Callable[[str], str]:
    """A ``rewrite`` for ``in_output_language`` that asks ``llm``."""
    return lambda prompt: llm.invoke(prompt).content
