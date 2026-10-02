"""Derived value helpers shared by the in-memory gateway and the response layer.

These mirror `assistant_api._value_part`, `_numeric_value`, `_parse_date_value`, and
`_date_precision`. Postgres owns the production path; this module keeps the
in-process gateway (and its tests) behaving identically without a database.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

_YEAR = re.compile(r"(1[6-9]\d{2}|20\d{2})")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_ISO = re.compile(r"(1[6-9]\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})")
_DAY_FIRST = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](1[6-9]\d{2}|20\d{2})")
_YEAR_MONTH = re.compile(r"(1[6-9]\d{2}|20\d{2})[-/.](\d{1,2})")
_MONTH_NAME_YEAR = re.compile(r"[A-Za-z]{3,}\s+(1[6-9]\d{2}|20\d{2})")
_DAY_MONTH_NAME = re.compile(r"\d{1,2}\s*[A-Za-z]{3,}\s*(1[6-9]\d{2}|20\d{2})")
_ZERO_PART = re.compile(r"[-/.]0{1,2}(?:\D|$)")

_BAND_NUMBER = re.compile(r"\s*#\s*\d+(?:\s*[/-]\s*\d+)*")

VALUE_PARTS = ("raw", "year", "decade", "first_token", "last_token", "number", "base_name")


def value_part(value: Any, part: str | None) -> str | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    if part in (None, "raw"):
        return text
    if part in {"year", "decade"}:
        match = _YEAR.search(text)
        if match is None:
            return None
        if part == "year":
            return match.group(1)
        return f"{(int(match.group(1)) // 10) * 10}s"
    if part in {"first_token", "last_token"}:
        tokens = text.split()
        if not tokens:
            return None
        return tokens[0] if part == "first_token" else tokens[-1]
    if part == "number":
        match = _NUMBER.search(text)
        return match.group(0) if match else None
    if part == "base_name":
        # Community label without its band number, so "Chisasibi #66"
        # and "Chisasibi" group together. Must match the SQL version.
        stripped = _BAND_NUMBER.sub("", text)
        stripped = " ".join(stripped.split()).strip(" /-")
        # A label that was only a band number keeps its original text
        # rather than collapsing every such row into one empty group.
        return stripped or text
    return text


def numeric_value(value: Any, part: str | None = None) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    match = _YEAR.search(text) if part in {"year", "decade"} else _NUMBER.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1) if part in {"year", "decade"} else match.group(0))
    except ValueError:
        return None


def parse_date(value: Any) -> date | None:
    """Best-effort parse of the date shapes this corpus stores."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    match = _ISO.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DAY_FIRST.search(text)
    if match:
        return _safe_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
    match = _YEAR_MONTH.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), 1)
    match = _YEAR.search(text)
    if match:
        return _safe_date(int(match.group(1)), 1, 1)
    return None


def date_precision(value: Any) -> str | None:
    """'day', 'month', or 'year' — how precisely the stored date is written."""
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    if _ISO.search(text) or _DAY_FIRST.search(text):
        return "month" if _ZERO_PART.search(text) else "day"
    if _YEAR_MONTH.search(text) or _MONTH_NAME_YEAR.search(text):
        return "month"
    if _DAY_MONTH_NAME.search(text):
        return "day"
    if _YEAR.search(text):
        return "year"
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    month = min(max(month, 1), 12)
    day = min(max(day, 1), 31)
    for candidate in (day, 1):
        try:
            return date(year, month, candidate)
        except ValueError:
            continue
    return None
