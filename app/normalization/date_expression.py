"""Standardized date expressions. OR is uncertainty; AND records real events.

Filtering reports definite matches: each alternative must establish the condition,
while any recorded event can establish it. Unknown components never become dates.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any

from app.normalization.date_parser import DateParts, parse_date
from app.value_normalization import semantic_scalar


def date_groups(value: Any) -> tuple[tuple[DateParts, ...], ...]:
    text = str(semantic_scalar(value) or "").strip()
    short = re.fullmatch(r"(\d{4}-\d{2}-)(\d{2})-(\d{2})", text)
    if short:
        text = f"{short[1]}{short[2]} | {short[1]}{short[3]}"
    return tuple(
        tuple(_atom(part.strip()) for part in branch.split("&")) for branch in text.split("|")
    )


def _atom(text: str) -> DateParts:
    if ":" in text:
        ends = text.split(":")
        if len(ends) != 2 or any("invalid_date" in _atom(end.strip()).flags for end in ends):
            return DateParts(flags=("invalid_date",))
    standard = re.fullmatch(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", text)
    if standard:
        year, month, day = (int(component or 0) for component in standard.groups())
        if (
            (year and not 1600 <= year <= 2100)
            or month > 12
            or day > 31
            or (not month and day)
            or (month and day > calendar.monthrange(year or 2000, month)[1])
        ):
            return DateParts(flags=("invalid_date",))
    return parse_date(text)


def bounds(parts: DateParts) -> tuple[date, date] | None:
    if parts.year is None or "impossible_day" in parts.flags or "invalid_month" in parts.flags:
        return None
    if "range" in parts.flags and not parts.range_end:
        return None
    month = parts.month
    low = date(parts.year, month or 1, parts.day or 1)
    high = date(
        parts.year, month or 12, parts.day or calendar.monthrange(parts.year, month or 12)[1]
    )
    if parts.range_end:
        end = bounds(parse_date(parts.range_end))
        if end is None or low > end[1]:
            return None
        high = end[1]
    return low, high


def date_matches(value: Any, operator: str, expected: Any) -> bool:
    groups = date_groups(value)
    known = all(
        any(
            bounds(part) is not None
            or (
                part.year is None
                and part.month is not None
                and not part.flags
                and not part.range_end
            )
            for part in branch
        )
        for branch in groups
    )
    if operator == "IS_KNOWN":
        return known
    if operator == "IS_UNKNOWN":
        return not known
    if operator in ("NOT_EQUALS", "NOT_IN"):
        targets = expected if isinstance(expected, list | tuple) else [expected]
        intervals = [bounds(_atom(str(item))) for item in targets]
        if not intervals or any(interval is None for interval in intervals):
            return False
        return all(
            (actual := bounds(part)) is not None
            and all(actual[1] < target[0] or actual[0] > target[1] for target in intervals)
            for branch in groups
            for part in branch
        )
    targets = expected if isinstance(expected, list | tuple) else [expected]

    def matches(part: DateParts) -> bool:
        interval = bounds(part)
        if interval is None:
            return False
        low, high = interval
        if operator == "YEAR_EQUALS":
            target = _atom(str(expected or ""))
            return target.year is not None and low.year == high.year == target.year
        if operator == "DATE_RANGE":
            if len(targets) != 2:
                return False
            start, end = (bounds(_atom(str(item))) for item in targets)
            return bool(start and end and start[0] <= low and high <= end[1])
        if operator in ("EQUALS", "IN"):
            return any(
                (target := bounds(_atom(str(item)))) is not None
                and target[0] <= low
                and high <= target[1]
                for item in targets
            )
        target = bounds(_atom(str(expected or "")))
        if target is None:
            return False
        if operator == "BEFORE":
            return high < target[0]
        if operator == "AFTER":
            return low > target[1]
        return False

    return all(any(matches(part) for part in branch) for branch in groups)


def shared_component(value: Any, component: str) -> int | None:
    def recorded(part: DateParts) -> int | None:
        if part.range_end or "range" in part.flags:
            interval = bounds(part)
            if interval is None:
                return None
            low, high = interval
            if (low.year != high.year or (component != "year" and low.month != high.month)
                    or (component == "day" and low != high)):
                return None
            return getattr(low, component)
        return getattr(part, component)

    values = {recorded(part) for branch in date_groups(value) for part in branch}
    return next(iter(values)) if len(values) == 1 else None


def recorded_event_count(value: Any) -> int:
    """Guaranteed dated events: all AND events, minimum across OR alternatives."""
    return min(
        sum(
            bounds(part) is not None
            or (
                part.year is None
                and part.month is not None
                and not part.flags
                and not part.range_end
            )
            for part in branch
        )
        for branch in date_groups(value)
    )


DATE_OPERATORS = frozenset(
    {
        "IS_KNOWN",
        "IS_UNKNOWN",
        "EQUALS",
        "NOT_EQUALS",
        "IN",
        "NOT_IN",
        "YEAR_EQUALS",
        "BEFORE",
        "AFTER",
        "DATE_RANGE",
    }
)
