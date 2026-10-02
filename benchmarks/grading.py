"""Structured comparison of an expected answer against a generated one.

Prose is not compared. Both sides are reduced to the values a reader would check:
the headline figures the expected answer emphasises, the label/count pairs of a
breakdown, and the names of an enumerated set. Those structures are then scored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# A date such as 1889-12-24 is one value, not three: without the lookbehind the
# grader invented headline numbers of -12 and -24 and failed every record summary.
_NUMBER = re.compile(r"(?<![\d.,-])\d[\d,]*(?:\.\d+)?(?![\d,]*[-/.]\d)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_PAIR = re.compile(r"([^;:|\n]{2,80}?)\s*[:=]\s*(\d[\d,]*)(?![\d.%])")
_MARKUP = re.compile("[`*_]|" + chr(92) * 2 + "(?=[a-zA-Z_])")
_SPLIT = re.compile(r"\s*[;|]\s*|\s*\n\s*")
_NOISE_WORDS = frozenset(
    {"", "and", "or", "the", "a", "an", "of", "in", "plus", "including", "e.g", "etc"}
)


def _clean(text: str) -> str:
    return " ".join(_MARKUP.sub("", text or "").split())


def _clean_lines(text: str) -> str:
    """Strip markup but keep line breaks: a numbered list separates items by them."""
    stripped = _MARKUP.sub("", text or "")
    return chr(10).join(" ".join(line.split()) for line in stripped.splitlines())


def _to_number(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass
class AnswerShape:
    """The comparable content of one answer."""

    headline_numbers: list[float] = field(default_factory=list)
    all_numbers: set[float] = field(default_factory=set)
    percents: list[float] = field(default_factory=list)
    pairs: dict[str, float] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)


def shape_of(text: str, *, headline_from_bold: bool = False) -> AnswerShape:
    cleaned_source = text or ""
    shape = AnswerShape()
    if headline_from_bold:
        for bold in _BOLD.findall(cleaned_source):
            for number in _NUMBER.findall(bold):
                shape.headline_numbers.append(_to_number(number))
    cleaned = _clean(cleaned_source)
    shape.all_numbers = {_to_number(item) for item in _NUMBER.findall(cleaned)}
    shape.percents = [float(item) for item in _PERCENT.findall(cleaned)]
    if not shape.headline_numbers:
        shape.headline_numbers = sorted(shape.all_numbers, reverse=True)[:1]
    for label, count in _PAIR.findall(cleaned):
        key = _normalize_label(label)
        if key and key not in _NOISE_WORDS:
            shape.pairs[key] = _to_number(count)
    shape.names = _name_set(_clean_lines(cleaned_source))
    return shape


def _normalize_label(label: str) -> str:
    text = _clean(label).casefold()
    text = re.sub(r"^(?:\d+[.)]\s*|[-\u2013\u2014]\s*)", "", text)
    text = re.sub(r"\s*\(.*?\)\s*$", "", text)
    return text.strip(" .,-\u2013\u2014")


def _name_set(text: str) -> set[str]:
    segments = [segment for segment in _SPLIT.split(text) if segment]
    names: set[str] = set()
    for segment in segments:
        candidate = segment.split(" - ")[0].split(" \u2014 ")[0]
        candidate = re.sub(r"^\d+[.)]\s*", "", candidate)
        if ":" in candidate:
            head, _, tail = candidate.partition(":")
            use_tail = len(tail.split()) >= 2 and not tail.strip()[:1].isdigit()
            candidate = tail if use_tail else head
        candidate = _normalize_label(candidate)
        if len(candidate.split()) >= 2 and not candidate[:1].isdigit():
            names.add(candidate)
    return names


def _overlap(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 1.0
    matched = sum(1 for item in expected if _matches_any(item, actual))
    return matched / len(expected)


def _matches_any(needle: str, haystack: set[str]) -> bool:
    if needle in haystack:
        return True
    return any(needle in item or item in needle for item in haystack)


@dataclass
class Grade:
    verdict: str
    score: float
    reasons: list[str] = field(default_factory=list)
    detail: dict[str, object] = field(default_factory=dict)


def grade(expected_text: str, actual_text: str) -> Grade:
    """Score one answer. Structured values decide; wording never does."""
    expected = shape_of(expected_text, headline_from_bold=True)
    actual = shape_of(actual_text)
    reasons: list[str] = []
    checks: list[float] = []

    headline = [value for value in expected.headline_numbers if value not in (0.0,)]
    if headline:
        hit = [value for value in headline if value in actual.all_numbers]
        ratio = len(hit) / len(headline)
        checks.append(ratio)
        if ratio < 1.0:
            missing = [value for value in headline if value not in actual.all_numbers]
            reasons.append(
                "missing headline value(s): "
                + ", ".join(f"{value:g}" for value in missing[:6])
            )

    if expected.percents:
        hit = [
            value
            for value in expected.percents
            if any(abs(value - other) <= 0.5 for other in actual.percents)
        ]
        ratio = len(hit) / len(expected.percents)
        checks.append(ratio)
        if ratio < 1.0:
            reasons.append("percentage does not match")

    if len(expected.pairs) >= 3:
        # A breakdown is carried by its counts. A bucket counts as matched when
        # the number agrees and either the labels overlap or that number is
        # unambiguous in the expected breakdown, so "Deceased (Yes): 1,066"
        # matches an answer that labels the same bucket "Yes".
        counted = list(expected.pairs.values())
        unique_counts = {value for value in counted if counted.count(value) == 1}
        matched = sum(
            1
            for label, count in expected.pairs.items()
            if any(
                abs(count - other_count) < 0.5
                and (_matches_any(label, {other}) or count in unique_counts)
                for other, other_count in actual.pairs.items()
            )
        )
        ratio = matched / len(expected.pairs)
        checks.append(ratio)
        if ratio < 1.0:
            reasons.append(
                f"breakdown matched {matched}/{len(expected.pairs)} label:count pairs"
            )

    if len(expected.names) >= 3:
        ratio = _overlap(expected.names, actual.names)
        checks.append(ratio)
        if ratio < 1.0:
            reasons.append(
                f"named {int(ratio * len(expected.names))}/{len(expected.names)} expected records"
            )

    if not checks:
        checks.append(1.0 if actual_text.strip() else 0.0)
        if not actual_text.strip():
            reasons.append("empty answer")

    score = sum(checks) / len(checks)
    if score >= 0.95:
        verdict = "pass"
    elif score >= 0.5:
        verdict = "partial"
    else:
        verdict = "fail"
    return Grade(
        verdict=verdict,
        score=round(score, 4),
        reasons=reasons,
        detail={
            "expected_headline": expected.headline_numbers,
            "expected_pairs": len(expected.pairs),
            "expected_names": len(expected.names),
            "actual_pairs": len(actual.pairs),
            "actual_names": len(actual.names),
        },
    )


def same_answer(first: str, second: str) -> bool:
    """Whether two answers carry the same structured content.

    Used for paraphrase self-consistency, where there is no separate golden answer:
    a reworded question must produce the same values as the canonical one.
    """
    left = shape_of(first)
    right = shape_of(second)
    if left.all_numbers != right.all_numbers:
        return False
    if left.pairs != right.pairs:
        return False
    return left.names == right.names
