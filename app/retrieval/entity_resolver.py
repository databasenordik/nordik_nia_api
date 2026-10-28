from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from app.retrieval.normalizer import normalize_search_text

MIN_ALIAS_SCORE = 88.0
AMBIGUOUS_GAP = 5.0


@dataclass(frozen=True)
class ReferenceAlias:
    entity_type: str
    canonical: str
    alias: str


@dataclass(frozen=True)
class ResolvedEntity:
    entity_type: str
    canonical: str
    score: float
    ambiguous: bool = False


DEFAULT_REFERENCES = (
    ReferenceAlias("community", "Garden River", "Garden River"),
    ReferenceAlias("community", "Garden River", "Garden River First Nation"),
    ReferenceAlias("community", "Garden River", "Garden Rvr"),
    ReferenceAlias("community", "Sault Ste. Marie", "Sault Ste. Marie"),
    ReferenceAlias("school", "Shingwauk", "Shingwauk"),
    ReferenceAlias("school", "Shingwauk", "Shingwauk Home"),
    ReferenceAlias("school", "Shingwauk", "Shingwauk Residential School"),
)


def resolve_entity(
    text: str,
    references: tuple[ReferenceAlias, ...] = DEFAULT_REFERENCES,
) -> ResolvedEntity | None:
    needle = normalize_search_text(text)
    if not needle:
        return None
    scored: list[tuple[float, ReferenceAlias]] = []
    for item in references:
        score = float(
            max(
                fuzz.ratio(needle, normalize_search_text(item.alias)),
                fuzz.ratio(needle, normalize_search_text(item.canonical)),
            )
        )
        if score >= MIN_ALIAS_SCORE:
            scored.append((score, item))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_score, top = scored[0]
    rivals = [score for score, item in scored[1:] if item.canonical != top.canonical]
    ambiguous = bool(rivals) and (top_score - rivals[0]) < AMBIGUOUS_GAP
    return ResolvedEntity(top.entity_type, top.canonical, top_score, ambiguous=ambiguous)
