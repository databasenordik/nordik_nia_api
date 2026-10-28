from __future__ import annotations

from typing import Any

from app.retrieval.normalizer import normalize_search_text
from app.retrieval.types import RetrievalHit


def score_exact(record: dict[str, Any], query: str) -> RetrievalHit | None:
    needle = normalize_search_text(query)
    if not needle:
        return None
    fields = (
        ("canonical_name", record.get("canonical_name")),
        ("canonical_community", record.get("canonical_community")),
        ("canonical_school", record.get("canonical_school")),
    )
    for field_name, value in fields:
        normalized = normalize_search_text(str(value) if value is not None else "")
        if normalized == needle:
            return _hit(record, "exact", 1.0, field_name)
        if needle and normalized.startswith(needle + " "):
            return _hit(record, "exact", 0.92, field_name)
    return None


def _hit(record: dict[str, Any], method: str, score: float, field_name: str) -> RetrievalHit:
    return RetrievalHit(
        file_id=int(record["file_id"]),
        version=int(record.get("version") or 1),
        source_row_id=int(record["source_row_id"]),
        canonical_name=record.get("canonical_name"),
        canonical_community=record.get("canonical_community"),
        canonical_school=record.get("canonical_school"),
        method=method,  # type: ignore[arg-type]
        score=score,
        matched_field=field_name,
        snippet=record.get("canonical_name"),
        record=record,
    )
