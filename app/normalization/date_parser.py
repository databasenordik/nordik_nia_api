"""Split an as-recorded date cell into year, month, day and everything said around them.

The four lists record dates in at least a dozen shapes, and a single cell often carries
more than a date::

    1889-12-24                     a full date
    June 1882                      a month, with no day
    1874                           a year alone
    28 August 1879 2nd time August 1881    two admissions in one cell
    1868-00-00~                    a year, marked approximate
    1902-09-14 (NCTR SOURCE)       a date and its provenance
    Unknown                        no date at all

The existing SQL parser (``assistant_api._parse_date_value``) collapses all of that to a
single DATE, and it does so lossily: "June 1882" matches only its bare-year branch and
becomes 1882-01-01, so the month is discarded and a day that was never recorded is
invented. Anything after the first date is dropped entirely.

This module keeps what the source actually said:

* ``year`` / ``month`` / ``day`` -- only the parts that were recorded, never padded.
* ``precision`` -- day, month or year, so "1882" is never mistaken for the 1st of January.
* ``extra`` -- the further dates in the same cell, which are real events (a second
  admission), not noise.
* ``flags`` -- approximate, uncertain, annotated, range, unparsed.

Only the primary date is treated as canonical; the rest stay beside it, the same way an
alternate surname sits beside the canonical one in ``name_parser``.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_ALT = "|".join(sorted(MONTHS, key=len, reverse=True))

MIN_YEAR, MAX_YEAR = 1600, 2100


@dataclass(frozen=True)
class DateParts:
    """One parsed date cell. Missing components stay missing rather than defaulting."""

    year: int | None = None
    month: int | None = None
    day: int | None = None
    precision: str = ""          # "day" | "month" | "year" | ""
    extra: tuple[str, ...] = ()  # further dates in the same cell, ISO-formatted
    flags: tuple[str, ...] = ()

    @property
    def iso(self) -> str:
        """The date at the precision it was actually recorded to.

        A year-less month (or month-day) uses the ISO 8601 omitted-year form
        ``--MM`` / ``--MM-DD`` so the recorded part is kept without inventing a year.
        """
        if self.year is None:
            if self.month is None:
                return ""
            if self.day is None:
                return f"--{self.month:02d}"
            return f"--{self.month:02d}-{self.day:02d}"
        if self.month is None:
            return f"{self.year:04d}"
        if self.day is None:
            return f"{self.year:04d}-{self.month:02d}"
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    @property
    def sort_key(self) -> str:
        """A padded form for ordering, where an unknown component sorts first."""
        if self.year is None:
            return ""
        return f"{self.year:04d}-{self.month or 0:02d}-{self.day or 0:02d}"


# Provenance and status notes that sit beside a date without being part of it.
_ANNOTATION = re.compile(
    r"\((?:[^)]*)(?:source|nctr|cirnac|registrar|vital|stats?|archive|office)[^)]*\)",
    re.IGNORECASE,
)
# "2nd time", "3rd time", "and", "came again" all introduce a further real date.
_FURTHER = re.compile(
    r"\b(?:\d(?:st|nd|rd|th)\s+time|and\s+second\s+time|and\s+came\s+again|came\s+again|and)\b",
    re.IGNORECASE,
)
_APPROX = re.compile(
    r"~|\best\.?\b|\babout\b|\bcirca\b|\bca\.?\b|\bapprox(?:imately)?\b|\bc\.\s*\d",
    re.IGNORECASE,
)
_UNCERTAIN = re.compile(r"\(\s*\?\s*\)|\?")
_BEFORE = re.compile(r"\bbefore\b|\bprior to\b", re.IGNORECASE)
_AFTER = re.compile(r"\bafter\b|\bpost\b", re.IGNORECASE)
# Year spans only: 1960-61 (61 cannot be a month) or 1960-1961. YYYY-MM is a month.
_YEAR_RANGE = re.compile(
    r"\b(1[6-9]\d{2}|20\d{2})\s*[-–—]\s*((?:1[6-9]\d{2}|20\d{2})|(?:1[3-9]|[2-9]\d))\b"
)
_YEAR_SLASH_RANGE = re.compile(
    r"\b(1[6-9]\d{2}|20\d{2})\s*/\s*(1[6-9]\d{2}|20\d{2})\b"
)
_ALT_MONTHS = re.compile(
    rf"\b(1[6-9]\d{{2}}|20\d{{2}})\s*[-–—]?\s*({_MONTH_ALT})\.?\s+or\s+({_MONTH_ALT})\.?\b",
    re.I,
)
_ALT_MONTHS_TRAILING_YEAR = re.compile(
    rf"\b({_MONTH_ALT})\.?\s+or\s+({_MONTH_ALT})\.?\s+(1[6-9]\d{{2}}|20\d{{2}})\b",
    re.I,
)
_QUARTER = re.compile(
    r"\b(?:first|1st|second|2nd|third|3rd|fourth|4th)\s+quarter\b",
    re.I,
)
_EXCEL_SERIAL = re.compile(r"^-?\d{3,6}$")
# Windows Excel 1900 date system (serial 0 = 1899-12-30).
_EXCEL_EPOCH = datetime(1899, 12, 30)

# Ordered most specific first: an earlier pattern must not steal a longer match.
_ISO = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\b")
# "1943 09 14" -- ISO components separated by spaces rather than dashes.
_YMD_SPACE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\s+(\d{1,2})\s+(\d{1,2})\b")
_ISO_MONTH = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\s*[-/.]\s*(\d{1,2})\b(?!\s*[-/.]\s*\d)")
# "23.09.1940", "00.09.1962" -- day first, and 00 means the day was not recorded.
_DMY = re.compile(r"\b(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(1[6-9]\d{2}|20\d{2})\b")
# Hyphens and dots are allowed: "26-Apr-1946", "9Aug1877".
_D_MONTH_Y = re.compile(
    rf"\b(\d{{1,2}})[-\s.]*({_MONTH_ALT})\.?[-\s.]*((?:1[6-9]\d{{2}}|20\d{{2}}))\b",
    re.I,
)
_MONTH_D_Y = re.compile(rf"\b({_MONTH_ALT})\.?\s*(\d{{1,2}})\s*,?\s*(1[6-9]\d{{2}}|20\d{{2}})\b", re.I)
_MONTH_Y = re.compile(rf"\b({_MONTH_ALT})\.?\s*(1[6-9]\d{{2}}|20\d{{2}})\b", re.I)
_MONTH_DAY = re.compile(rf"\b({_MONTH_ALT})\.?\s+(\d{{1,2}})\b(?!\s*[-/]?\s*(?:1[6-9]\d{{2}}|20\d{{2}}))", re.I)
_MONTH_ONLY = re.compile(rf"\b({_MONTH_ALT})\.?\b", re.I)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def _valid(year: int, month: int | None, day: int | None) -> tuple[int, int | None, int | None, bool]:
    """Keep only components that could be real. A zero means 'not recorded'.

    A day the calendar does not have -- 31 September -- is a transcription error. The
    production SQL parser quietly rewrites it to the 1st, which asserts a day nobody
    recorded; here the day is dropped and the cell falls back to month precision, so the
    part that is trustworthy survives and the error is reported rather than hidden.
    """
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError("year out of range")
    clean_month = month if month and 1 <= month <= 12 else None
    clean_day = day if day and 1 <= day <= 31 else None
    if clean_month is None:
        return year, None, None, False
    impossible = False
    if clean_day is not None:
        last = calendar.monthrange(year, clean_month)[1]
        if clean_day > last:
            clean_day = None
            impossible = True
    return year, clean_month, clean_day, impossible


def _resolve_numeric_dmy(
    first: int, second: int, year: int, matched: str
) -> tuple[int, int | None, int | None, bool, tuple[str, ...]]:
    """Interpret a numeric day/month/year triple.

    Dotted and dashed forms in this corpus are day-first (``23.09.1940``). Slash
    forms are mixed: ``12/14/1890`` can only be December 14, while ``10/9/1951``
    is genuinely ambiguous, so the order is flagged rather than guessed.
    """
    extra_flags: list[str] = []
    slash = "/" in matched
    first_is_month = 1 <= first <= 12
    second_is_month = 1 <= second <= 12
    first_is_day = 1 <= first <= 31
    second_is_day = 1 <= second <= 31

    if slash and second_is_day and not second_is_month and first_is_month:
        month, day = first, second
    elif slash and first_is_month and second_is_month and first != second:
        # Ambiguous MDY/DMY. Default to the list's dominant day-first reading.
        month, day = second, first
        extra_flags.append("ambiguous_order")
    else:
        month, day = second, first
        if not first_is_day:
            day = None
    year, month, day, impossible = _valid(year, month, day)
    return year, month, day, impossible, tuple(extra_flags)


def _match_one(
    text: str,
) -> tuple[int | None, int | None, int | None, bool, int, int, tuple[str, ...]] | None:
    """Find the first date in ``text``; return components plus its span."""
    for pattern, order in (
        (_ISO, "ymd"),
        (_YMD_SPACE, "ymd"),
        (_D_MONTH_Y, "dmy_name"),
        (_MONTH_D_Y, "mdy_name"),
        (_DMY, "dmy"),
        (_ISO_MONTH, "ym"),
        (_MONTH_Y, "my_name"),
        (_YEAR, "y"),
    ):
        match = pattern.search(text)
        if match is None:
            continue
        extra_flags: tuple[str, ...] = ()
        try:
            if order == "ymd":
                year, month, day = int(match[1]), int(match[2]), int(match[3])
                year, month, day, impossible = _valid(year, month, day)
            elif order == "dmy_name":
                day, month, year = int(match[1]), MONTHS[match[2].lower()], int(match[3])
                year, month, day, impossible = _valid(year, month, day)
            elif order == "mdy_name":
                month, day, year = MONTHS[match[1].lower()], int(match[2]), int(match[3])
                year, month, day, impossible = _valid(year, month, day)
            elif order == "dmy":
                year, month, day, impossible, extra_flags = _resolve_numeric_dmy(
                    int(match[1]), int(match[2]), int(match[3]), match.group(0)
                )
            elif order == "ym":
                year, month, day = int(match[1]), int(match[2]), None
                year, month, day, impossible = _valid(year, month, day)
            elif order == "my_name":
                month, year, day = MONTHS[match[1].lower()], int(match[2]), None
                year, month, day, impossible = _valid(year, month, day)
            else:
                year, month, day = int(match[1]), None, None
                year, month, day, impossible = _valid(year, month, day)
        except (ValueError, KeyError):
            continue
        return year, month, day, impossible, match.start(), match.end(), extra_flags
    return None


def _excel_serial_date(text: str) -> datetime | None:
    """Interpret a bare integer as an Excel serial only when it cannot be a year."""
    raw = text.strip()
    if not _EXCEL_SERIAL.fullmatch(raw):
        return None
    serial = int(raw)
    if MIN_YEAR <= serial <= MAX_YEAR:
        return None
    if serial < -100000 or serial > 80000:
        return None
    try:
        converted = _EXCEL_EPOCH + timedelta(days=serial)
    except OverflowError:
        return None
    if not MIN_YEAR <= converted.year <= MAX_YEAR:
        return None
    return converted


def _expand_year_suffix(start: int, suffix: str) -> int:
    if len(suffix) == 4:
        return int(suffix)
    end = (start // 100) * 100 + int(suffix)
    if end < start:
        end += 100
    return end


def _render(year: int | None, month: int | None, day: int | None) -> str:
    return DateParts(year=year, month=month, day=day).iso


def _precision_of(month: int | None, day: int | None) -> str:
    if day is not None:
        return "day"
    if month is not None:
        return "month"
    return "year"


def parse_date(text: str) -> DateParts:
    """Parse one as-recorded date cell."""
    raw = (text or "").strip()
    if not raw:
        return DateParts()

    flags: list[str] = []
    extras: list[str] = []
    working = raw

    annotated = _ANNOTATION.sub(" ", working)
    if annotated != working:
        flags.append("annotated")
        working = annotated

    if _APPROX.search(working):
        flags.append("approximate")
    if _UNCERTAIN.search(working):
        flags.append("uncertain")
    if _BEFORE.search(working):
        flags.append("before")
    if _AFTER.search(working):
        flags.append("after")
    if _QUARTER.search(working):
        flags.append("quarter")

    # "1894- May or July" names two possible months of one year. Keep both; do not
    # pick one as if it were recorded.
    for pattern, year_idx, month_a, month_b in (
        (_ALT_MONTHS, 1, 2, 3),
        (_ALT_MONTHS_TRAILING_YEAR, 3, 1, 2),
    ):
        alt = pattern.search(working)
        if alt is None:
            continue
        year = int(alt[year_idx])
        first_month = MONTHS[alt[month_a].lower()]
        second_month = MONTHS[alt[month_b].lower()]
        flags.append("alternative")
        extras.extend(
            iso
            for iso in (
                _render(year, first_month, None),
                _render(year, second_month, None),
            )
            if iso
        )
        working = working[: alt.start()] + str(year) + working[alt.end() :]
        break

    # "1960-61" / "1960-1961" is a span of years, not a year and a month.
    span = _YEAR_RANGE.search(working)
    if span and not _ISO.search(working):
        start_year = int(span[1])
        end_year = _expand_year_suffix(start_year, span[2])
        flags.append("range")
        extras.append(_render(end_year, None, None))
        working = working[: span.start()] + str(start_year) + working[span.end() :]
    elif (slash_span := _YEAR_SLASH_RANGE.search(working)) and not _ISO.search(working):
        flags.append("range")
        extras.append(_render(int(slash_span[2]), None, None))
        working = working[: slash_span.start()] + slash_span[1] + working[slash_span.end() :]

    serial = _excel_serial_date(working)
    if serial is not None:
        flags.append("excel_serial")
        return DateParts(
            year=serial.year,
            month=serial.month,
            day=serial.day,
            precision="day",
            extra=tuple(dict.fromkeys(extras)),
            flags=tuple(dict.fromkeys(flags)),
        )

    # Each "2nd time"/"and" introduces a further real date -- a second admission is an
    # event in its own right, so the cell is split before anything is parsed.
    segments = [part.strip(" ,;.") for part in _FURTHER.split(working) if part.strip(" ,;.")]
    parsed: list[tuple[int | None, int | None, int | None]] = []
    for segment in segments:
        remaining = segment
        while True:
            found = _match_one(remaining)
            if found is None:
                month_day = _MONTH_DAY.search(remaining)
                month_only = None if month_day else _MONTH_ONLY.search(remaining)
                if month_day:
                    month = MONTHS[month_day[1].lower()]
                    day_n = int(month_day[2])
                    parsed.append((None, month, day_n if 1 <= day_n <= 31 else None))
                    remaining = remaining[: month_day.start()] + " " + remaining[month_day.end() :]
                    continue
                if month_only:
                    parsed.append((None, MONTHS[month_only[1].lower()], None))
                    remaining = remaining[: month_only.start()] + " " + remaining[month_only.end() :]
                    continue
                break
            year, month, day, impossible, start, end, match_flags = found
            if impossible:
                flags.append("impossible_day")
            flags.extend(match_flags)
            parsed.append((year, month, day))
            remaining = remaining[:start] + " " + remaining[end:]
            if not _YEAR.search(remaining) and not _MONTH_ONLY.search(remaining):
                break

    if not parsed:
        month_day = _MONTH_DAY.search(working)
        month_only = _MONTH_ONLY.search(working) if month_day is None else None
        if month_day:
            month = MONTHS[month_day[1].lower()]
            day = int(month_day[2])
            day = day if 1 <= day <= 31 else None
            return DateParts(
                year=None,
                month=month,
                day=day,
                precision=_precision_of(month, day),
                extra=tuple(dict.fromkeys(extras)),
                flags=tuple(dict.fromkeys(flags)),
            )
        if month_only:
            month = MONTHS[month_only[1].lower()]
            return DateParts(
                year=None,
                month=month,
                day=None,
                precision="month",
                extra=tuple(dict.fromkeys(extras)),
                flags=tuple(dict.fromkeys(flags)),
            )
        flags.append("unparsed" if working.strip() else "empty")
        return DateParts(extra=tuple(dict.fromkeys(extras)), flags=tuple(dict.fromkeys(flags)))

    year, month, day = parsed[0]
    for other in parsed[1:]:
        rendered = _render(other[0], other[1], other[2])
        # A quarter note restates the same year; do not treat it as another date.
        if "quarter" in flags and other[1] is None and other[2] is None and other[0] == year:
            continue
        if rendered and rendered not in extras:
            extras.append(rendered)
    if extras:
        flags.append("multiple")
    if re.search(r"\bor\b", raw, re.I) and "alternative" not in flags:
        flags.append("alternative")

    return DateParts(
        year=year,
        month=month,
        day=day,
        precision=_precision_of(month, day),
        extra=tuple(dict.fromkeys(extras)),
        flags=tuple(dict.fromkeys(flags)),
    )
