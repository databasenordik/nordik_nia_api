from __future__ import annotations

from typing import Any

from app.retrieval.normalizer import tokens
from app.retrieval.types import RetrievalHit


def score_fts(record: dict[str, Any], query: str) -> RetrievalHit | None:
    query_tokens = tokens(query)
    if not query_tokens:
        return None
    haystack = " ".join(
        [
            str(record.get("search_text") or ""),
            str(record.get("canonical_name") or ""),
            str(record.get("canonical_community") or ""),
            str(record.get("canonical_school") or ""),
        ]
    )
    hay_tokens = set(tokens(haystack))
    if not hay_tokens:
        return None
    overlap = [token for token in query_tokens if token in hay_tokens]
    if not overlap:
        return None
    score = min(0.89, 0.45 + 0.2 * len(overlap) / max(len(query_tokens), 1))
    snippet = _snippet(haystack, overlap[0])
    return RetrievalHit(
        file_id=int(record["file_id"]),
        version=int(record.get("version") or 1),
        source_row_id=int(record["source_row_id"]),
        canonical_name=record.get("canonical_name"),
        canonical_community=record.get("canonical_community"),
        canonical_school=record.get("canonical_school"),
        method="fts",
        score=score,
        matched_field="search_text",
        snippet=snippet,
        record=record,
    )


def _snippet(text: str, token: str, radius: int = 48) -> str:
    lowered = text.casefold()
    index = lowered.find(token)
    if index < 0:
        return text[: radius * 2]
    start = max(0, index - radius)
    end = min(len(text), index + len(token) + radius)
    return text[start:end].strip()
