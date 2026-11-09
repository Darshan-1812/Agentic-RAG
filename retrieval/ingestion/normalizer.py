"""Text normalization helpers shared by parsing and chunking."""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    """Normalize unicode, line endings, control characters, and blank lines.

    Markdown/table layout is preserved: only trailing whitespace and runs of
    blank lines are collapsed. Intra-line spacing is left alone except for
    control characters.
    """
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ")
    value = _ZERO_WIDTH_RE.sub("", value)
    value = _CONTROL_RE.sub("", value)
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    value = _BLANK_LINES_RE.sub("\n\n", value)
    return value.strip()


def normalize_heading(text: str) -> str:
    """Normalize a heading and collapse intra-line whitespace."""
    value = normalize_text(text)
    value = value.replace("\n", " ")
    return _INLINE_WHITESPACE_RE.sub(" ", value).strip()


def normalize_table_markdown(markdown: str) -> str:
    """Normalize exported table markdown without disturbing pipe layout."""
    value = normalize_text(markdown)
    if not value:
        return ""
    lines = [line.rstrip() for line in value.split("\n") if line.strip()]
    return "\n".join(lines)


def split_paragraphs(text: str) -> List[str]:
    """Split normalized text at blank lines, preserving paragraph content."""
    value = normalize_text(text)
    if not value:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", value) if part.strip()]


def is_blank(text: str) -> bool:
    return not normalize_text(text)


def unique_preserving_order(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
