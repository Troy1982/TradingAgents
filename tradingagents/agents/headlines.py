"""How many independent stories a news block holds, for judging whether a sentiment can be read from it.

A Taiwan listing's sentiment rests on news headlines alone (StockTwits and
Reddit do not cover it), and FinMind's feed repeats itself: one story syndicated
across outlets, the same outlet's article twice, stock-board and quote pages
that are not stories at all. The headlines are counted after de-duplication:

1. Each title is normalized: full-width forms folded (NFKC), a trailing
   " - <source>" or "｜<site>" suffix dropped, punctuation and spaces removed,
   lower-cased.
2. Titles that are equal after normalization, that contain one another, or
   whose similarity (difflib ratio) is at least ``SIMILAR`` are one story:
   the same event syndicated or reposted.
3. Entry pages are not stories: board posts and pages ("股市爆料同學會",
   "討論區", "今日股價與討論") and per-stock quote pages such as
   "2383 台光電 - -202 / -66 / +9". All of them together count as one. A
   newspaper column with "論壇" in its name is a story, not a board.

The block is the news tool's own answer, already trimmed to the analysis window
on the market's clock (``as_of_time``), so only headlines public by then count.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Similarity at or above which two normalized titles are one story.
SIMILAR = 0.8

_HEADLINE = re.compile(r"^### (.+) \(source: (.*)\)$", re.M)
_SUFFIX = re.compile(r"\s*(?:[-|｜–—]\s*[^-|｜–—]{1,30})+$")
_ENTRY_WORDS = ("同學會", "爆料", "討論區", "股價與討論")
_QUOTE_PAGE = re.compile(r"^\d{4,6}[A-Z]?\s*\S+\s*-\s*[-+]?\d")
_STRIP = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True)
class HeadlineCount:
    """Headlines in a news block before and after de-duplication."""

    headlines: int
    independent: int
    entry_pages: int
    stories: tuple[tuple[str, ...], ...]   # each independent story's titles, entry pages last

    def describe(self) -> str:
        return (f"{self.headlines} headlines in the window, {self.independent} independent after "
                f"de-duplication ({self.entry_pages} entry/discussion pages counted as one)")


def headlines(news_block: str) -> list[str]:
    """The article titles in a news tool's answer, in the order given."""
    return [title.strip() for title, _ in _HEADLINE.findall(news_block or "")]


def normalize(title: str) -> str:
    """A title reduced to its words, for comparing reposts of one story."""
    text = unicodedata.normalize("NFKC", title).strip()
    shortened = _SUFFIX.sub("", text)
    # Keep the suffix when dropping it would leave too little to compare.
    text = shortened if len(_STRIP.sub("", shortened)) >= 6 else text
    return _STRIP.sub("", text).lower()


def is_entry_page(title: str) -> bool:
    """Whether a title is a board, community or quote page rather than a story."""
    text = unicodedata.normalize("NFKC", title)
    return any(word in text for word in _ENTRY_WORDS) or bool(_QUOTE_PAGE.match(text))


def _same_story(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) >= 8 and (a in b or b in a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= SIMILAR


def count_independent(news_block: str) -> HeadlineCount:
    """Count the news block's headlines and its independent stories (see the module docstring)."""
    titles = headlines(news_block)
    entries = [t for t in titles if is_entry_page(t)]
    stories: list[tuple[str, list[str]]] = []   # (normalized key, titles)
    for title in titles:
        if title in entries:
            continue
        key = normalize(title)
        for story_key, members in stories:
            if _same_story(key, story_key):
                members.append(title)
                break
        else:
            stories.append((key, [title]))
    grouped = [tuple(members) for _, members in stories] + ([tuple(entries)] if entries else [])
    return HeadlineCount(headlines=len(titles), independent=len(grouped), entry_pages=len(entries),
                         stories=tuple(grouped))
