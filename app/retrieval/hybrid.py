from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.retrieval.deduplicator import dedupe_hits
from app.retrieval.entity_resolver import ResolvedEntity, resolve_entity
from app.retrieval.exact_search import score_exact
from app.retrieval.fulltext_search import score_fts
from app.retrieval.fuzzy_search import score_fuzzy
from app.retrieval.ranker import rank_hits
from app.retrieval.types import RetrievalHit
from app.security.access_scope import AccessScope


class CandidateSource(Protocol):
    async def retrieve_candidates(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        method: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]: ...


@dataclass
class HybridRetrievalResult:
    hits: list[RetrievalHit]
    methods_run: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    resolved_entity: ResolvedEntity | None = None


class HybridRetriever:
    """Fan-out exact / FTS / fuzzy retrieval with early-cancel on unique exact."""

    def __init__(self, source: CandidateSource) -> None:
        self._source = source

    async def retrieve(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        *,
        limit: int = 12,
        want_exact: bool = True,
        want_fts: bool = True,
        want_fuzzy: bool = True,
    ) -> HybridRetrievalResult:
        resolved = resolve_entity(query)
        search_query = resolved.canonical if resolved and not resolved.ambiguous else query
        methods_run: list[str] = []
        cancelled: list[str] = []

        exact_hits: list[RetrievalHit] = []
        if want_exact:
            exact_hits = await self._branch(scope, file_ids, search_query, "exact", limit)
            methods_run.append("exact")
            if _unique_exact(exact_hits):
                if want_fts:
                    cancelled.append("fts")
                if want_fuzzy:
                    cancelled.append("fuzzy")
                return HybridRetrievalResult(
                    hits=rank_hits(exact_hits)[:limit],
                    methods_run=methods_run,
                    cancelled=cancelled,
                    resolved_entity=resolved,
                )

        async def run(method: str) -> list[RetrievalHit]:
            return await self._branch(scope, file_ids, search_query, method, limit)

        fanout: list[tuple[str, asyncio.Task[list[RetrievalHit]]]] = []
        if want_fts:
            fanout.append(("fts", asyncio.create_task(run("fts"))))
        if want_fuzzy:
            fanout.append(("fuzzy", asyncio.create_task(run("fuzzy"))))
        hits = list(exact_hits)
        if fanout:
            results = await asyncio.gather(*(task for _, task in fanout), return_exceptions=True)
            for (name, _), result in zip(fanout, results, strict=True):
                if isinstance(result, Exception):
                    continue
                methods_run.append(name)
                hits.extend(result)
        fused = dedupe_hits(hits)[:limit]
        return HybridRetrievalResult(
            hits=fused,
            methods_run=methods_run,
            cancelled=cancelled,
            resolved_entity=resolved,
        )

    async def _branch(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        method: str,
        limit: int,
    ) -> list[RetrievalHit]:
        rows = await self._source.retrieve_candidates(scope, file_ids, query, method, limit)
        hits: list[RetrievalHit] = []
        scorer = {"exact": score_exact, "fts": score_fts, "trgm": score_fuzzy, "fuzzy": score_fuzzy}[method]
        for row in rows:
            hit = scorer(row, query)
            if hit is not None:
                hits.append(hit)
        return hits


def _unique_exact(hits: list[RetrievalHit]) -> bool:
    exact = [hit for hit in hits if hit.method == "exact" and hit.score >= 0.99]
    identities = {hit.identity() for hit in exact}
    return len(identities) == 1
