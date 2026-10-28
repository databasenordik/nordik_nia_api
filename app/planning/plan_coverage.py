"""Refuse a deterministic plan that quietly dropped part of the question.

The legacy grammars recognise the fields a question names but not always the values
it constrains them to. "Which students died from tuberculosis?" was compiled to an
unfiltered list of every confirmed record, because `cause_of_death` was recognised
as a field to project while "tuberculosis" was never turned into a condition.

The check is deliberately conservative and vocabulary-driven: a question word is
accounted for when it is query language, names a field or dataset in the catalog, or
appears in the plan the fast path produced. Anything left over is a condition the
plan does not express, so the fast path abstains and the compiler plans the turn.
"""

from __future__ import annotations

import re

from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import QueryPlan

_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STEM = 5

# Words that shape a query rather than constrain it.
QUERY_LANGUAGE = frozenset(
    """
    about above after again against all also among another any anyone appear appears
    are around ask asked available back based because been before begin begins
    beginning being below best better between both bottom breakdown but can cannot
    come common compare comparison contain contains containing could count counted
    counts data database different distinct does done down during each earliest
    either else enumerate ever every exact exactly example exist exists fewer fewest
    field fields find first following from full gave give given greatest group
    grouped groups have having here highest how however identify include included
    includes including information into just keep kind know known largest last later
    latest least less like list listed listing lists look looking lowest made make
    many mention mentioned mentions might missing more most much must name named
    names near need needs never next none nonblank nothing number numbers often
    only order other others over overall page part particular people percent
    percentage person persons please present provide provided range rank ranked
    ranking recorded record records report result results same say says search see
    seem seems several show showing shown side since some someone something sort
    sorted specific specify start started starting starts state states still
    student students such summarise summarize summary take tell than that their them
    then there these they this those through time times together told top total
    totals under unknown until upon used using value values various very want was
    were what when where whether which while who whom whose why will with within
    without would year years
    """.split()
)


def plan_covers_question(plan: QueryPlan, question: str, catalog: FieldCatalog) -> bool:
    """True when the plan accounts for every value and every field the question names.

    Two independent checks: no leftover content word (an unexpressed value such as
    "tuberculosis"), and no specifically named field missing from the plan (an
    unexpressed condition such as "...but no discharge date").
    """
    vocabulary = _catalog_vocabulary(plan, catalog) | _plan_vocabulary(plan)
    for token in _WORD.findall(question.lower()):
        if token in QUERY_LANGUAGE:
            continue
        if _covered(token, vocabulary):
            continue
        return False
    named = specific_field_mentions(question, catalog, plan.scope.file_ids)
    return named <= plan_field_references(plan)


def _covered(token: str, vocabulary: set[str]) -> bool:
    head = token[:_STEM]
    return any(word[:_STEM] == head for word in vocabulary)


def _catalog_vocabulary(plan: QueryPlan, catalog: FieldCatalog) -> set[str]:
    words: set[str] = set()
    for dataset in catalog.datasets:
        for label in (dataset.user_facing_label, *dataset.aliases):
            words.update(_WORD.findall(label.lower()))
    for file_id in plan.scope.file_ids:
        for spec in catalog.fields_for(file_id):
            for label in (spec.human_label, spec.semantic_field.replace("_", " "), *spec.aliases):
                words.update(_WORD.findall(label.lower()))
    return words


def _plan_vocabulary(plan: QueryPlan) -> set[str]:
    words: set[str] = set()
    for step in plan.steps:
        for predicate in step.where:
            for leaf in predicate.leaves():
                words.update(_WORD.findall(str(leaf.value).lower()))
                words.update(_WORD.findall(leaf.field.replace("_", " ")))
        if step.query:
            words.update(_WORD.findall(step.query.lower()))
        for name in step.fields:
            words.update(_WORD.findall(name.replace("_", " ")))
    return words


_APOSTROPHE = re.compile(r"(?<=\w)[\u2019'](?=\s)")
_INNER_DASH = re.compile(r"(?<=[a-z])[-\u2010\u2011\u2013\u2014](?=[a-z])")
SPECIFIC_LABEL_LENGTH = 8


def normalize_question(question: str) -> str:
    lowered = " ".join(question.lower().split())
    lowered = _APOSTROPHE.sub("", lowered)
    return _INNER_DASH.sub(" ", lowered)


def _label_pattern(label: str) -> str:
    """Match a field label allowing a plural on its last word.

    Questions say "burial dates" and "communities" where the catalog stores
    "burial date" and "community".
    """
    words = label.split()
    if not words:
        return r"(?!)"
    head = [re.escape(word) for word in words[:-1]]
    tail = re.escape(words[-1])
    if tail.endswith("y"):
        tail = tail[:-1] + "(?:y|ies)"
    else:
        tail = tail + "s?"
    return r"\b" + r"\s+".join([*head, tail]) + r"\b"


def field_mentions(
    question: str,
    catalog: FieldCatalog,
    file_ids: tuple[int, ...] | None = None,
) -> dict[int, set[str]]:
    """Semantic fields the question names, per dataset.

    A match must be a whole word or phrase of at least five characters, and a match
    nested inside a longer field label loses to that longer label, so "community"
    inside "First Nation / Community" does not double-count.
    """
    lowered = normalize_question(question)
    datasets = [
        dataset
        for dataset in catalog.datasets
        if file_ids is None or dataset.file_id in file_ids
    ]
    spans: list[tuple[int, int, int, str]] = []
    for dataset in datasets:
        for spec in catalog.fields_for(dataset.file_id):
            for label in {spec.human_label, spec.semantic_field.replace("_", " "), *spec.aliases}:
                normalized = " ".join(label.lower().split())
                if len(normalized) < 5:
                    continue
                for match in re.finditer(_label_pattern(normalized), lowered):
                    spans.append((match.start(), match.end(), dataset.file_id, spec.semantic_field))
    found: dict[int, set[str]] = {}
    for start, end, file_id, semantic in spans:
        nested = any(
            other_start <= start and end <= other_end and (other_end - other_start) > (end - start)
            for other_start, other_end, _fid, _sem in spans
        )
        if not nested:
            found.setdefault(file_id, set()).add(semantic)
    return found


def specific_field_mentions(
    question: str,
    catalog: FieldCatalog,
    file_ids: tuple[int, ...],
) -> set[str]:
    """Fields named specifically enough that a plan ignoring them dropped a condition."""
    lowered = normalize_question(question)
    named: set[str] = set()
    for file_id in file_ids:
        for spec in catalog.fields_for(file_id):
            for label in {spec.human_label, spec.semantic_field.replace("_", " "), *spec.aliases}:
                normalized = " ".join(label.lower().split())
                if len(normalized) < SPECIFIC_LABEL_LENGTH and " " not in normalized:
                    continue
                if re.search(_label_pattern(normalized), lowered):
                    named.add(spec.semantic_field)
                    break
    return named


def plan_field_references(plan: QueryPlan) -> set[str]:
    referenced: set[str] = set()
    for step in plan.steps:
        referenced.update(step.fields)
        for name in (step.sort_field, step.companion_field, step.interval_start, step.interval_end):
            if name:
                referenced.add(name)
        for predicate in step.where:
            referenced.update(leaf.field for leaf in predicate.leaves())
    return referenced
