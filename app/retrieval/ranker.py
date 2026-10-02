from __future__ import annotations

from app.retrieval.types import RetrievalHit

METHOD_WEIGHT = {
    "exact": 1.0,
    "alias": 0.9,
    "fts": 0.75,
    "trgm": 0.7,
    "fuzzy": 0.65,
}


def rank_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    def key(hit: RetrievalHit) -> tuple[float, float, str]:
        weighted = METHOD_WEIGHT.get(hit.method, 0.5) * hit.score
        field_bonus = 0.08 if hit.matched_field and hit.matched_field != "search_text" else 0.0
        return (weighted + field_bonus, hit.score, hit.source_id())

    return sorted(hits, key=key, reverse=True)
