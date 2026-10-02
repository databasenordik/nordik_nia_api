from __future__ import annotations

from app.retrieval.types import RetrievalHit


def score_vector(*_args: object, **_kwargs: object) -> list[RetrievalHit]:
    """Disabled until local embeddings are enabled. Do not call a paid API."""
    return []
