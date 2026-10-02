"""Normalize and widen filter values, whichever planner produced them.

Two defects were shared by every planner path:

* a value lifted straight out of the question keeps its article, so
  `location_of_death CONTAINS "the school"` matched nothing;
* a value is matched literally, so "tuberculosis" missed the records recorded as
  consumption, phthisis, or scrofula.

Both are fixed on the finished QueryPlan, so the deterministic grammars, the schema
planner, and the compiler all benefit. The vocabulary comes from the field registry
(`validation_rules.value_synonyms`), never from code.
"""

from __future__ import annotations

import re

from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.plan_schema import FilterOperator, Predicate, QueryPlan

_LEADING_ARTICLE = re.compile(r"^(?:the|a|an|their|his|her|its|any|some)\s+", re.IGNORECASE)
_TRAILING_NOISE = re.compile(r"\s*[.,;]+$")

_WIDENS = {
    FilterOperator.EQUALS: FilterOperator.CONTAINS_ANY,
    FilterOperator.CONTAINS: FilterOperator.CONTAINS_ANY,
    FilterOperator.CONTAINS_ANY: FilterOperator.CONTAINS_ANY,
    FilterOperator.NOT_EQUALS: FilterOperator.NOT_CONTAINS_ANY,
    FilterOperator.NOT_CONTAINS: FilterOperator.NOT_CONTAINS_ANY,
    FilterOperator.NOT_CONTAINS_ANY: FilterOperator.NOT_CONTAINS_ANY,
}

# The same expansion over a closed set of categories, matched exactly.
_EXACT = {
    FilterOperator.EQUALS: FilterOperator.IN,
    FilterOperator.CONTAINS: FilterOperator.IN,
    FilterOperator.CONTAINS_ANY: FilterOperator.IN,
    FilterOperator.NOT_EQUALS: FilterOperator.NOT_IN,
    FilterOperator.NOT_CONTAINS: FilterOperator.NOT_IN,
    FilterOperator.NOT_CONTAINS_ANY: FilterOperator.NOT_IN,
}


def exact_category_terms(recorded: object, terms: object) -> list[str] | None:
    """The categories these terms name exactly, when matching them exactly is right.

    Returns None when the match must stay a substring search.

    Two opposite failures live here, and word boundaries tell them apart.

    Matching too loosely: deceased_status records yes/no/unknown, and "no" is an accidental
    infix of "u-n-k-no-w-n". CONTAINS 'no' returned 1,700 rows where 1,035 are recorded
    "no", so this must become an exact match.

    Matching too tightly: location_of_death records both "School" and "School (NCTR
    SOURCE)". Someone asking about deaths at school means both, so an exact match on
    "School" drops the six rows carrying the provenance note -- 49 returned where 55 died at
    school. That must stay a substring match.

    Structurally the difference is whether the term appears as a whole word inside another
    category. "no" does not occur as a word in "unknown"; "school" does occur as a word in
    "School (NCTR SOURCE)". So a term that is a word of some other category is naming a
    family and keeps substring semantics; otherwise the collision is accidental and the
    match is pinned to the category the user actually named.
    """
    if not isinstance(recorded, list) or not recorded:
        return None
    known = {str(item).strip().casefold(): str(item) for item in recorded}
    candidates = terms if isinstance(terms, list) else [terms]
    if not candidates:
        return None
    matched: list[str] = []
    for item in candidates:
        key = str(item or "").strip().casefold()
        if key not in known:
            return None
        word = re.compile(rf"\b{re.escape(key)}\b")
        if any(other != key and word.search(other) for other in known):
            return None
        matched.append(known[key])
    return matched


def _expanded_operator(
    spec: FieldSpec, operator: FilterOperator, terms: object
) -> FilterOperator:
    """Widen to a substring match only where substrings are safe.

    Expanding a value into its recorded spellings has to match loosely for free text --
    "tuberculosis" should find "pulmonary tuberculosis". Over a closed set of categories it
    depends on the categories themselves; see exact_category_terms.

    Requires the field to allow IN/NOT_IN, since validation is strict about the operator
    list.
    """
    exact = _EXACT.get(operator)
    if exact is None or exact.value not in (spec.allowed_operators or ()):
        return _WIDENS[operator]
    recorded = (spec.validation_rules or {}).get("recorded_values")
    if exact_category_terms(recorded, terms) is None:
        return _WIDENS[operator]
    return exact


def expand_plan_values(
    plan: QueryPlan,
    catalog: FieldCatalog,
    *,
    profile: str = "legacy",
) -> QueryPlan:
    """Return the plan with cleaned and family-expanded filter values."""
    changed = False
    steps = []
    for step in plan.steps:
        file_ids = step.file_ids or plan.scope.file_ids
        rewritten = [_rewrite(item, file_ids, catalog, profile=profile) for item in step.where]
        updates: dict[str, object] = {}
        if rewritten != list(step.where):
            updates["where"] = rewritten
        if profile == "ai" and step.query:
            cleaned = _clean_search_text(step.query)
            if cleaned != step.query:
                updates["query"] = cleaned
        if updates:
            changed = True
            steps.append(step.model_copy(update=updates))
        else:
            steps.append(step)
    return plan.model_copy(update={"steps": steps}) if changed else plan


def _rewrite(
    predicate: Predicate,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
    *,
    profile: str = "legacy",
) -> Predicate:
    if predicate.is_group():
        items = [_rewrite(item, file_ids, catalog, profile=profile) for item in predicate.items]
        if items != list(predicate.items):
            return predicate.model_copy(update={"items": items})
        return predicate
    spec = _spec(predicate.field, file_ids, catalog)
    if predicate.operator is FilterOperator.IS_KNOWN and spec is not None:
        unknown_terms = _unknown_terms(spec)
        if unknown_terms:
            # "a known cause of death" excludes the records whose recorded cause is
            # the literal word Unknown, not just the blank ones.
            return Predicate(
                op="and",
                items=[
                    predicate,
                    Predicate(
                        field=predicate.field,
                        operator=FilterOperator.NOT_CONTAINS_ANY,
                        value=unknown_terms,
                    ),
                ],
            )
    value = predicate.value
    updates: dict[str, object] = {}
    if profile != "ai":
        owner = _field_owning_value(predicate.field, value, file_ids, catalog)
        if owner is not None:
            # The value belongs to another field's recorded vocabulary. "tuberculosis" is
            # a cause of death, not a death factor, and filtering the wrong field matched
            # nothing at all.
            updates["field"] = owner
            spec = _spec(owner, file_ids, catalog)
    if isinstance(value, str):
        cleaned = _TRAILING_NOISE.sub("", value.strip())
        if profile != "ai" or _article_strip_safe(spec):
            cleaned = _LEADING_ARTICLE.sub("", cleaned).strip()
        if cleaned and cleaned != value:
            value = cleaned
            updates["value"] = cleaned
    if spec is None or predicate.operator not in _WIDENS:
        return predicate.model_copy(update=updates) if updates else predicate
    families = (spec.validation_rules or {}).get("value_synonyms")
    if not isinstance(families, dict):
        return predicate.model_copy(update=updates) if updates else predicate
    terms = _family_terms(families, value)
    if terms is not None and terms != value:
        updates["value"] = terms
        updates["operator"] = _expanded_operator(spec, predicate.operator, terms)
    return predicate.model_copy(update=updates) if updates else predicate


def _unknown_terms(spec: FieldSpec) -> list[str]:
    """Spellings this field uses to record "we do not know", from the registry.

    Used to widen IS_KNOWN only, and deliberately not its mirror. IS_UNKNOWN was widened to
    "blank or the literal word" here and reverted: it made "an unknown deceased status"
    right and "missing a cause of death" wrong, because those are different questions that
    compile to the same operator. A cause recorded as Unknown is not a missing cause -- the
    record has a value -- so the distinction lives in the words the researcher used, which
    the compiler no longer has. Encoding it is the planner's job: EQUALS the category for
    "unknown", IS_UNKNOWN for "missing".
    """
    families = (spec.validation_rules or {}).get("value_synonyms")
    if not isinstance(families, dict):
        return []
    for key, members in families.items():
        if key.casefold() == "unknown" and isinstance(members, list):
            return [str(item) for item in members]
    return []


def _field_owning_value(
    field_name: str,
    value: object,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> str | None:
    """The field whose registry vocabulary records this value, when it is not this one."""
    terms = value if isinstance(value, list) else [value]
    needles = {item.strip().casefold() for item in terms if isinstance(item, str) and item.strip()}
    if not needles:
        return None
    current = _spec(field_name, file_ids, catalog)
    if current is not None and _owns(current, needles):
        return None
    owners: set[str] = set()
    for file_id in file_ids:
        for spec in catalog.fields_for(file_id):
            if spec.semantic_field == field_name:
                continue
            if _owns(spec, needles):
                owners.add(spec.semantic_field)
    return next(iter(owners)) if len(owners) == 1 else None


def _owns(spec: FieldSpec, needles: set[str]) -> bool:
    families = (spec.validation_rules or {}).get("value_synonyms")
    if not isinstance(families, dict):
        return False
    for key, members in families.items():
        if key.casefold() in needles:
            return True
        if isinstance(members, list) and any(str(item).casefold() in needles for item in members):
            return True
    return False


def _family_terms(families: dict, value: object) -> list[str] | None:
    candidates = value if isinstance(value, list) else [value]
    widened: list[str] = []
    matched = False
    for item in candidates:
        if not isinstance(item, str):
            return None
        family = _lookup(families, item)
        if family is None:
            widened.append(item)
            continue
        matched = True
        widened.extend(term for term in family if term not in widened)
    if not matched:
        return None
    return widened


def _lookup(families: dict, value: str) -> list[str] | None:
    needle = value.strip().casefold()
    if not needle:
        return None
    for key, members in families.items():
        if isinstance(members, list) and key.casefold() == needle:
            return [str(item) for item in members]
    for members in families.values():
        if isinstance(members, list) and any(str(item).casefold() == needle for item in members):
            return [str(item) for item in members]
    return None


def _article_strip_safe(spec: FieldSpec | None) -> bool:
    if spec is None:
        return False
    rules = spec.validation_rules or {}
    return bool(rules.get("article_strip_safe"))


def _clean_search_text(value: str) -> str:
    return _TRAILING_NOISE.sub("", " ".join(value.split())).strip()


def _spec(name: str, file_ids: tuple[int, ...], catalog: FieldCatalog) -> FieldSpec | None:
    for file_id in file_ids:
        found = catalog.resolve_field(file_id, name)
        if found is not None:
            return found
    return None
