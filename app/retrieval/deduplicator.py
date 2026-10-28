from __future__ import annotations

from app.retrieval.ranker import rank_hits
from app.retrieval.types import RetrievalHit


def dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    best: dict[tuple[int, int], RetrievalHit] = {}
    for hit in rank_hits(hits):
        identity = hit.identity()
        if identity not in best:
            best[identity] = hit
    return rank_hits(list(best.values()))
