from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from app.retrieval.normalizer import normalize_search_text
from app.retrieval.types import RetrievalHit

MIN_FUZZY_SCORE = 78.0
CLOSE_SECOND_GAP = 6.0


def score_fuzzy(record: dict[str, Any], query: str) -> RetrievalHit | None:
    needle = normalize_search_text(query)
    if not needle:
        return None
    best_field = None
    best = 0.0
    for field_name, value in (
        ("canonical_name", record.get("canonical_name")),
        ("canonical_community", record.get("canonical_community")),
        ("canonical_school", record.get("canonical_school")),
        ("search_text", record.get("search_text")),
    ):
        if not value:
            continue
        ratio = float(
            max(
                fuzz.token_set_ratio(needle, normalize_search_text(str(value))),
                fuzz.partial_ratio(needle, normalize_search_text(str(value))),
            )
        )
        if ratio > best:
            best = ratio
            best_field = field_name
    if best < MIN_FUZZY_SCORE or best_field is None:
        return None
    return RetrievalHit(
        file_id=int(record["file_id"]),
        version=int(record.get("version") or 1),
        source_row_id=int(record["source_row_id"]),
        canonical_name=record.get("canonical_name"),
        canonical_community=record.get("canonical_community"),
        canonical_school=record.get("canonical_school"),
        method="fuzzy",
        score=round(best / 100.0 * 0.8, 4),
        matched_field=best_field,
        snippet=record.get("canonical_name"),
        record=record,
    )
