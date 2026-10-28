"""High-confidence, schema-driven planning for common database operations.

This module deliberately reasons over catalog metadata instead of dataset IDs,
column names embedded in code, or benchmark question strings. It only returns a
plan when the operation, field, and value can all be resolved unambiguously;
otherwise the semantic compiler remains the fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.plan_schema import (
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)

_COUNT = re.compile(
    r"\b(?:how\s+many|count|number\s+of|(?:the|a)\s+(?:count|number|total)|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?total|total\s+(?:number\s+of)?)\b",
    re.I,
)
_DISTINCT = re.compile(r"\b(?:distinct|different|unique|various)\b", re.I)
_DISTRIBUTION = re.compile(r"\b(?:distribution|breakdown|frequency|frequencies)\b", re.I)
_MOST_COMMON = re.compile(r"\b(?:most\s+(?:common|frequent)|most\s+often|highest\s+frequency)\b", re.I)
_LEAST_COMMON = re.compile(r"\b(?:least\s+(?:common|frequent)|least\s+often|lowest\s+frequency)\b", re.I)
_MAX = re.compile(r"\b(?:maximum|max|highest|largest|oldest|latest)\b", re.I)
_MIN = re.compile(r"\b(?:minimum|min|lowest|smallest|youngest|earliest)\b", re.I)
_LIST = re.compile(r"\b(?:list|show|find|which|who|give\s+me|look\s+up|lookup)\b", re.I)
_ALL_INFORMATION = re.compile(
    r"\b(?:all|every|complete|full)\s+(?:available\s+)?(?:information|details|fields)\b",
    re.I,
)
_SCHEMA = re.compile(
    r"\b(?:what|which|show|list)\b.{0,40}\b(?:fields|columns|information)\b.{0,30}\b(?:available|exist|included)\b|"
    r"\b(?:what|which)\s+(?:fields|columns)\s+are\s+available\b",
    re.I,
)
_BETWEEN = re.compile(r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\b", re.I)
_NUMBER = re.compile(r"\b(-?\d+(?:\.\d+)?)\b")
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_CONTAINS_AFTER = re.compile(r"\b(?:contain(?:s|ing)?|mention(?:s|ed|ing)?)\s+(.+?)(?:\?|\.|$)", re.I)
_MENTION_AS_FIELD = re.compile(
    r"\bmention(?:s|ed|ing)?\s+(.+?)\s+as\s+(?:the\s+)?(?:cause|reason|place|location|value)\b",
    re.I,
)
_STARTS = re.compile(
    r"\b(?:start|starts|starting|begin|begins|beginning)\s+(?:(?:with|in)\s+)?"
    r"(?:the\s+letter\s+)?([A-Za-z0-9])\b|"
    r"\b(?:with|having)\s+(?:the\s+letter\s+)?([A-Za-z0-9])\s+"
    r"(?:up\s+front|at\s+(?:the\s+)?(?:front|start|beginning))\b|"
    r"\b([A-Za-z0-9])\s+(?:names?|entries|records?)\b",
    re.I,
)
_LIMIT = re.compile(r"\b(?:first|last|latest|earliest|top|bottom)\s+(\d{1,2})\b", re.I)
_PERSON_FOR = re.compile(r"\bfor\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){1,5})\s*[?.!]*$", re.I)
_MISSING = re.compile(
    r"\b(?:missing|without|lack(?:s|ing)?|no\s+recorded|have\s+no|has\s+no|"
    r"do(?:es)?\s+not\s+(?:have|include)|not\s+(?:have|include))\b",
    re.I,
)
_PRESENCE = re.compile(
    r"\b(?:have|has|with|recorded|known|mention(?:s|ed|ing)?|include(?:s|d|ing)?|"
    r"identify|identifies|specify|specifies|contain(?:s|ed|ing)?|say|says|state|states)\b",
    re.I,
)


@dataclass(frozen=True)
class FieldMention:
    spec: FieldSpec
    start: int
    end: int
    phrase: str


def is_schema_priority(text: str) -> bool:
    """Whether typed/catalog planning should precede the legacy grammar."""

    return bool(
        re.search(
            r"\b(?:distribution|breakdown|most\s+(?:common|frequent|often)|"
            r"least\s+(?:common|frequent|often)|maximum|minimum|highest|lowest|"
            r"largest|smallest|oldest|youngest|earliest|latest|contain(?:s|ing)?|"
            r"mention(?:s|ed|ing)?|died\s+at)\b|"
            r"\b(?:which|what)\b[^?]{0,80}\b(?:most|fewest)\b",
            text,
            re.I,
        )
    )


def parse_schema_query(
    text: str,
    *,
    file_id: int,
    catalog: FieldCatalog,
) -> QueryPlan | None:
    """Return a deterministic plan for an unambiguous catalog-backed request."""

    normalized = " ".join(text.strip().split())
    normalized = re.sub(r"(?<=\w)[’'](?=\s)", "", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])[-‐‑–—](?=[A-Za-z])", " ", normalized)
    lowered = normalized.casefold()
    if not normalized or _SCHEMA.search(normalized):
        return None
    if re.search(
        r"\b(?:random|sample|summarize|summary|why|explain|quote|provenance|"
        r"each|organize|compare|versus|except|excluding)\b|"
        r"\bwhat\s+do\s+(?:the\s+)?records?\s+say\b",
        lowered,
    ):
        return None
    if re.search(r"\b(?:first|last|latest|earliest)\s+\d{3,}\b", lowered):
        return None
    specs = catalog.fields_for(file_id)
    if not specs:
        return None
    mentions = _field_mentions(lowered, specs)
    mentioned = _unique_specs(mentions)

    wants_count = bool(_COUNT.search(normalized))
    wants_distribution = bool(_DISTRIBUTION.search(normalized))
    wants_mode = bool(
        _MOST_COMMON.search(normalized)
        or _LEAST_COMMON.search(normalized)
        or re.search(r"\b(?:which|what)\b[^?]{0,80}\b(?:most|fewest)\b", normalized, re.I)
    )
    wants_distinct = bool(_DISTINCT.search(normalized))
    wants_list = bool(_LIST.search(normalized) or _ALL_INFORMATION.search(normalized))
    wants_min = bool(_MIN.search(normalized)) and not wants_mode and not wants_list
    wants_max = bool(_MAX.search(normalized)) and not wants_mode and not wants_list
    if not any((wants_count, wants_distribution, wants_mode, wants_distinct, wants_min, wants_max, wants_list)):
        return None
    if wants_count and wants_list and re.search(
        r"\b(?:count|how\s+many)\b[^.!?;]*\band\b[^.!?;]*\b(?:list|show|find)\b|"
        r"\b(?:list|show|find)\b[^.!?;]*\band\b[^.!?;]*\b(?:count|how\s+many)\b",
        lowered,
    ):
        return None
    person = _person_name(normalized)
    if (
        wants_list
        and not mentioned
        and not person
        and not _ALL_INFORMATION.search(normalized)
        and not re.search(r"\b(?:students?|records?|people|deaths?|master\s+list)\b", lowered)
    ):
        return None

    aggregate_field = _operation_field(
        lowered,
        mentioned,
        specs,
        prefer_date=bool(
            re.search(r"\byear\b|chronological", lowered)
            or (wants_list and re.search(r"\b(?:latest|earliest|newest|oldest)\b", lowered))
        ),
    )
    if (wants_distribution or wants_mode or wants_distinct or wants_min or wants_max) and aggregate_field is None:
        return None

    predicates = (
        []
        if any((wants_distribution, wants_mode, wants_distinct, wants_min, wants_max))
        else _predicates(lowered, mentions, mentioned, allow_exact=not wants_list)
    )
    if person:
        name_spec = _semantic(specs, "student_name")
        if name_spec is None:
            return None
        predicates = [
            predicate
            for predicate in predicates
            if predicate.field != name_spec.semantic_field
        ]
        predicates.append(
            Predicate(
                field=name_spec.semantic_field,
                operator=FilterOperator.CONTAINS,
                value=person,
            )
        )

    if wants_count and not any((wants_distribution, wants_mode, wants_distinct, wants_min, wants_max)):
        # A field-bearing count must preserve the field constraint. Abstain if
        # language suggests one but no compatible predicate could be resolved.
        if mentioned and not predicates and not _plain_dataset_count(lowered):
            return None
        return _count_plan(file_id, predicates)

    if wants_distinct:
        value_part = _value_part(lowered, aggregate_field)
        if wants_count:
            return _aggregate_plan(file_id, aggregate_field, PlanOp.COUNT_DISTINCT, predicates)
        return _group_plan(file_id, aggregate_field, predicates, value_part=value_part, with_count=False)

    if wants_distribution or wants_mode:
        return _group_plan(
            file_id,
            aggregate_field,
            predicates,
            value_part=_value_part(lowered, aggregate_field),
            with_count=True,
        )

    if wants_min or wants_max:
        if not aggregate_field.aggregatable:
            return None
        return _aggregate_plan(
            file_id,
            aggregate_field,
            PlanOp.MIN if wants_min else PlanOp.MAX,
            predicates,
        )

    if wants_list:
        projection = _projection(lowered, mentioned, specs)
        sort_field, sort_direction, window_anchor, window_size, limit = _list_controls(
            lowered,
            aggregate_field,
            specs,
        )
        return _list_plan(
            file_id,
            predicates,
            projection,
            sort_field=sort_field,
            sort_direction=sort_direction,
            limit=limit,
            window_anchor=window_anchor,
            window_size=window_size,
        )
    return None


def _field_mentions(text: str, specs: list[FieldSpec]) -> list[FieldMention]:
    candidates: list[FieldMention] = []
    for spec in specs:
        phrases = {
            spec.semantic_field.replace("_", " "),
            spec.human_label.casefold(),
            *(alias.casefold() for alias in spec.aliases),
            *(str(alias).casefold() for alias in dict(spec.validation_rules.get("value_aliases") or {})),
        }
        for phrase in sorted((item.strip() for item in phrases if item.strip()), key=len, reverse=True):
            variants = _phrase_variants(phrase)
            for variant in variants:
                pattern = r"(?<![a-z0-9])" + re.escape(variant).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
                for match in re.finditer(pattern, text, re.I):
                    candidates.append(FieldMention(spec, match.start(), match.end(), match.group(0)))
    # Prefer longer phrases when two aliases for the same or different fields overlap.
    selected: list[FieldMention] = []
    for item in sorted(candidates, key=lambda value: (-(value.end - value.start), value.start)):
        if any(item.start < other.end and other.start < item.end for other in selected):
            continue
        selected.append(item)
    return sorted(selected, key=lambda item: item.start)


def _phrase_variants(phrase: str) -> set[str]:
    """Return conservative singular/plural surface forms for a field phrase."""

    variants = {phrase}
    words = phrase.split()
    if not words:
        return variants
    for index in {len(words) - 1, 0 if "of" in words else len(words) - 1}:
        for form in _word_number_variants(words[index]):
            changed = list(words)
            changed[index] = form
            variants.add(" ".join(changed))
    return variants


def _word_number_variants(word: str) -> set[str]:
    variants = {word}
    if word.endswith("ies") and len(word) > 3:
        variants.add(word[:-3] + "y")
    elif word.endswith("s") and not word.endswith("ss"):
        variants.add(word[:-1])
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        variants.add(word[:-1] + "ies")
    elif word.endswith(("ch", "sh", "x", "z")):
        variants.add(word + "es")
    else:
        variants.add(word + "s")
    return variants


def _unique_specs(mentions: list[FieldMention]) -> list[FieldSpec]:
    result: list[FieldSpec] = []
    seen: set[str] = set()
    for mention in mentions:
        if mention.spec.semantic_field not in seen:
            result.append(mention.spec)
            seen.add(mention.spec.semantic_field)
    return result


def _operation_field(
    text: str,
    mentioned: list[FieldSpec],
    specs: list[FieldSpec],
    *,
    prefer_date: bool,
) -> FieldSpec | None:
    candidates = [item for item in mentioned if item.aggregatable or item.sortable]
    if prefer_date:
        dated = [item for item in candidates if item.semantic_type == "date"]
        if dated:
            return _best_token_overlap(text, dated)
        all_dates = [item for item in specs if item.semantic_type == "date" and item.aggregatable]
        ranked = _best_token_overlap(text, all_dates)
        if ranked and _token_overlap(text, ranked) > 0:
            return ranked
    return candidates[-1] if candidates else None


def _best_token_overlap(text: str, specs: list[FieldSpec]) -> FieldSpec | None:
    if not specs:
        return None
    return max(specs, key=lambda item: (_token_overlap(text, item), -len(item.semantic_field)))


def _token_overlap(text: str, spec: FieldSpec) -> int:
    text_tokens = {_singular(token) for token in re.findall(r"[a-z0-9]+", text)}
    field_tokens = {
        _singular(token)
        for token in re.findall(
            r"[a-z0-9]+",
            " ".join([spec.semantic_field.replace("_", " "), spec.human_label, *spec.aliases]).casefold(),
        )
        if token not in {"date", "recorded", "information"}
    }
    return len(text_tokens & field_tokens)


def _singular(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _predicates(
    text: str,
    mentions: list[FieldMention],
    specs: list[FieldSpec],
    *,
    allow_exact: bool = True,
) -> list[Predicate]:
    result: list[Predicate] = []
    unknown_match = re.search(r"\bunknown\b", text)
    missing_match = _MISSING.search(text)
    known_match = _PRESENCE.search(text)
    unknown_target = _modifier_target(unknown_match, mentions)
    missing_target = _modifier_target(missing_match, mentions)
    known_target = _modifier_target(known_match, mentions)
    starts = _STARTS.search(text)
    if starts:
        target = _semantic(specs, "student_name")
        if target and "STARTS_WITH" in target.allowed_operators:
            captured = next(value for value in starts.groups() if value is not None)
            value = captured.upper() if len(captured) == 1 else captured
            return [Predicate(field=target.semantic_field, operator=FilterOperator.STARTS_WITH, value=value)]

    for spec in specs:
        operator: FilterOperator | None = None
        value: Any = None
        local = _field_context(text, mentions, spec)
        aliases = {
            str(key).casefold(): raw
            for key, raw in dict(spec.validation_rules.get("value_aliases") or {}).items()
        }
        alias_hit = next((raw for phrase, raw in aliases.items() if _phrase_in(text, phrase)), None)
        if (
            missing_target == spec.semantic_field
            and _MISSING.search(local)
        ):
            operator = FilterOperator.IS_UNKNOWN
        elif (
            unknown_target == spec.semantic_field
            and re.search(r"\bunknown\b", local)
            and "EQUALS" in spec.allowed_operators
        ):
            operator, value = FilterOperator.EQUALS, "unknown"
        elif alias_hit is not None:
            operator, value = FilterOperator.EQUALS, alias_hit
        elif spec.semantic_type == "boolean":
            if re.search(r"\b(?:not|false|no)\b", local):
                operator = FilterOperator.IS_FALSE
            elif re.search(r"\b(?:used|yes|true|deceased|died|dead)\b", local):
                operator = FilterOperator.IS_TRUE

        if operator is None and spec.semantic_type in {"date", "number"}:
            between = _BETWEEN.search(local) or _BETWEEN.search(text)
            if between:
                value = [between.group(1), between.group(2)]
                operator = FilterOperator.DATE_RANGE if spec.semantic_type == "date" else FilterOperator.NUMBER_RANGE
            else:
                numbers = _YEAR.findall(local) if spec.semantic_type == "date" else _NUMBER.findall(local)
                if not numbers:
                    numbers = _YEAR.findall(text) if spec.semantic_type == "date" else []
                if numbers:
                    value = numbers[0]
                    if re.search(r"\bbefore\b", text):
                        operator = FilterOperator.BEFORE
                    elif re.search(r"\bafter\b", text):
                        operator = FilterOperator.AFTER
                    elif re.search(r"\b(?:greater|more|older|above|over)\s+than\b", text):
                        operator = FilterOperator.GREATER_THAN
                    elif re.search(r"\b(?:less|fewer|younger|below|under)\s+than\b", text):
                        operator = FilterOperator.LESS_THAN
                    elif spec.semantic_type == "date":
                        operator = FilterOperator.YEAR_EQUALS
                    else:
                        operator = FilterOperator.EQUALS

        if operator is None:
            contains_value = _contains_value(text, mentions, spec)
            if contains_value and "CONTAINS" in spec.allowed_operators:
                operator, value = FilterOperator.CONTAINS, contains_value

        if operator is None and allow_exact and "EQUALS" in spec.allowed_operators:
            exact_value = _exact_value(text, mentions, spec)
            if exact_value is not None:
                operator, value = FilterOperator.EQUALS, exact_value

        projection_phrase = bool(re.search(r"\bwith\s+(?:their|the)\b", text))
        if (
            operator is None
            and not projection_phrase
            and known_target == spec.semantic_field
            and _PRESENCE.search(local)
        ):
            if "IS_KNOWN" in spec.allowed_operators:
                operator = FilterOperator.IS_KNOWN

        if operator is not None and operator.value in spec.allowed_operators:
            result.append(Predicate(field=spec.semantic_field, operator=operator, value=value))
    return _dedupe_predicates(result)


def _exact_value(text: str, mentions: list[FieldMention], spec: FieldSpec) -> str | None:
    """Extract an explicitly attached scalar value for a registered field."""

    if spec.semantic_type in {"date", "number", "boolean"}:
        return None
    relevant = [item for item in mentions if item.spec.semantic_field == spec.semantic_field]
    for mention in relevant:
        before = text[: mention.start].rstrip()
        preceding = re.search(
            r"\b(?:have|has|give|gives|with|where|whose)\s+"
            r"([^,?;]{1,80}?)\s+as\s+(?:(?:their|the)\s*)?(?:exact\s*)?$",
            before,
            re.I,
        )
        if preceding:
            candidate = _clean_value(preceding.group(1))
            if candidate:
                return candidate

        if any(
            other.end <= mention.start
            and re.search(r"\b(?:contain(?:s|ed|ing)?|mention(?:s|ed|ing)?)\b", text[other.end : mention.start])
            for other in mentions
            if other.spec.semantic_field != spec.semantic_field
        ):
            continue

        after = text[mention.end :].strip(" ,.?;")
        after = re.sub(r"^(?:equals?|equal\s+to|is|of)\s+", "", after, flags=re.I)
        if not after or len(after.split()) > 5:
            continue
        if _MISSING.search(after) or _PRESENCE.search(after) or re.match(
            r"^(?:as|and|or|are|is|was|were|records?|entries|students?)\b",
            after,
        ):
            continue
        if re.fullmatch(r"(?:information|value|values|data|details?|reference|references)", after, re.I):
            continue
        if re.fullmatch(r"(?:someone|somebody|anyone|anybody|anything|something|a\s+person)", after, re.I):
            continue
        if re.search(r"\b(?:before|after|between|greater|less|older|younger|most|least)\b", after):
            continue
        return _clean_value(after)
    return None


def _modifier_target(match: re.Match[str] | None, mentions: list[FieldMention]) -> str | None:
    """Bind a value modifier to the nearest field, preferring a following noun."""

    if match is None or not mentions:
        return None
    following = [item for item in mentions if item.start >= match.end()]
    candidates = following or mentions
    return min(
        candidates,
        key=lambda item: (
            max(item.start - match.end(), 0)
            if item.start >= match.end()
            else match.start() - item.end,
            -(item.end - item.start),
        ),
    ).spec.semantic_field


def _field_context(text: str, mentions: list[FieldMention], spec: FieldSpec) -> str:
    relevant = [item for item in mentions if item.spec.semantic_field == spec.semantic_field]
    if not relevant:
        return text
    start = max(min(item.start for item in relevant) - 45, 0)
    end = min(max(item.end for item in relevant) + 45, len(text))
    return text[start:end]


def _contains_value(text: str, mentions: list[FieldMention], spec: FieldSpec) -> str | None:
    as_match = _MENTION_AS_FIELD.search(text)
    if as_match:
        return _clean_value(as_match.group(1))
    match = _CONTAINS_AFTER.search(text)
    if match:
        candidate = _clean_value(match.group(1))
        if candidate and not any(
            mention.spec.semantic_field == spec.semantic_field
            and mention.phrase.casefold() in candidate.casefold()
            for mention in mentions
        ):
            return candidate
    for mention in mentions:
        if mention.spec.semantic_field != spec.semantic_field:
            continue
        phrase = mention.phrase.casefold().strip()
        embedded = re.match(r"^(?:died\s+)?at\s+(.+)$", phrase)
        if embedded:
            return _clean_value(embedded.group(1))
        if phrase == "died at" or phrase == "at":
            candidate = _clean_value(text[mention.end :])
            if candidate:
                return candidate
    return None


def _clean_value(value: str) -> str:
    cleaned = re.split(r"\s+(?:as|in|from|with|where|who|that)\s+", value.strip(" ,.?"), maxsplit=1)[0]
    return cleaned.strip(" ,.?")


def _dedupe_predicates(items: list[Predicate]) -> list[Predicate]:
    result: list[Predicate] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.field, item.operator.value, repr(item.value))
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _person_name(text: str) -> str | None:
    match = _PERSON_FOR.search(text)
    if not match:
        return None
    candidate = match.group(1).strip()
    if any(word.casefold() in {"records", "students", "deaths", "list", "dataset"} for word in candidate.split()):
        return None
    return candidate.strip(" .?!")


def _projection(text: str, mentioned: list[FieldSpec], specs: list[FieldSpec]) -> list[str]:
    if _ALL_INFORMATION.search(text):
        return [item.semantic_field for item in specs]
    fields = [item.semantic_field for item in mentioned]
    identity = _semantic(specs, "student_name")
    if identity and identity.semantic_field not in fields:
        fields.insert(0, identity.semantic_field)
    return fields or ([identity.semantic_field] if identity else [])


def _list_controls(
    text: str,
    operation_field: FieldSpec | None,
    specs: list[FieldSpec],
) -> tuple[str | None, str, str | None, int | None, int]:
    match = _LIMIT.search(text)
    limit = int(match.group(1)) if match else 25
    identity = _semantic(specs, "student_name")
    directional_date = bool(
        re.search(r"\b(?:latest|earliest|newest|oldest|most\s+recent|chronological(?:ly)?)\b", text)
    )
    explicit_sort = bool(re.search(r"\b(?:sort(?:ed)?|order(?:ed)?|by|top|bottom)\b", text))
    sort_field = (
        operation_field.semantic_field
        if operation_field and operation_field.sortable and (directional_date or explicit_sort)
        else None
    )
    if re.search(r"\balphabetical(?:ly)?\b|\bfull\s+(?:student\s+)?name\b", text):
        sort_field = identity.semantic_field if identity else sort_field
    if sort_field is None and identity:
        sort_field = identity.semantic_field
    if re.search(r"\blatest\b|\bnewest\b|\bmost\s+recent\b", text):
        return sort_field, "DESC", None, None, limit
    if match and match.group(0).casefold().startswith("last"):
        return sort_field, "ASC", "end", limit, limit
    if match and match.group(0).casefold().startswith("first"):
        return sort_field, "ASC", "start", limit, limit
    return sort_field, "ASC", None, None, limit


def _value_part(text: str, spec: FieldSpec) -> str | None:
    return "year" if spec.semantic_type == "date" and re.search(r"\byears?\b", text) else None


def _plain_dataset_count(text: str) -> bool:
    return bool(re.search(r"\b(?:records|students|people)\b", text)) and not bool(
        re.search(r"\b(?:with|have|has|whose|where|missing|recorded|contain|mention)\b", text)
    )


def _semantic(specs: list[FieldSpec], name: str) -> FieldSpec | None:
    return next((item for item in specs if item.semantic_field == name), None)


def _phrase_in(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text, re.I))


def _scope_step(file_id: int) -> PlanStep:
    return PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records", file_ids=(file_id,))


def _filtered_steps(file_id: int, predicates: list[Predicate]) -> tuple[list[PlanStep], str]:
    steps = [_scope_step(file_id)]
    current = "scope_current"
    if predicates:
        steps.append(PlanStep(id="matches", op=PlanOp.FILTER, input=current, where=predicates, file_ids=(file_id,)))
        current = "matches"
    return steps, current


def _count_plan(file_id: int, predicates: list[Predicate]) -> QueryPlan:
    steps, current = _filtered_steps(file_id, predicates)
    steps.append(PlanStep(id="result", op=PlanOp.COUNT, input=current, file_ids=(file_id,), action_id="a0", action_goal="count"))
    return _plan(file_id, ["count"], steps)


def _aggregate_plan(file_id: int, spec: FieldSpec, op: PlanOp, predicates: list[Predicate]) -> QueryPlan:
    steps, current = _filtered_steps(file_id, predicates)
    steps.append(PlanStep(id="result", op=op, input=current, fields=[spec.semantic_field], file_ids=(file_id,), action_id="a0", action_goal="aggregate"))
    return _plan(file_id, ["aggregate", op.value.casefold()], steps)


def _group_plan(
    file_id: int,
    spec: FieldSpec,
    predicates: list[Predicate],
    *,
    value_part: str | None,
    with_count: bool,
) -> QueryPlan:
    steps, current = _filtered_steps(file_id, predicates)
    steps.append(
        PlanStep(
            id="grouped",
            op=PlanOp.GROUP_BY,
            input=current,
            fields=[spec.semantic_field],
            value_part=value_part,
            file_ids=(file_id,),
            action_id="a0",
            action_goal="aggregate" if with_count else "distinct",
        )
    )
    if with_count:
        steps.append(PlanStep(id="result", op=PlanOp.COUNT, input="grouped", file_ids=(file_id,), action_id="a0", action_goal="aggregate"))
    return _plan(file_id, ["aggregate", "group"] if with_count else ["distinct_values"], steps)


def _list_plan(
    file_id: int,
    predicates: list[Predicate],
    projection: list[str],
    *,
    sort_field: str | None,
    sort_direction: str,
    limit: int,
    window_anchor: str | None,
    window_size: int | None,
) -> QueryPlan:
    steps, current = _filtered_steps(file_id, predicates)
    if sort_field:
        steps.append(PlanStep(id="sorted", op=PlanOp.SORT, input=current, sort_field=sort_field, sort_direction=sort_direction, file_ids=(file_id,), action_id="a0", action_goal="list"))
        current = "sorted"
    steps.append(PlanStep(id="total", op=PlanOp.COUNT, input=current, file_ids=(file_id,), action_id="a0", action_goal="list"))
    steps.append(
        PlanStep(
            id="limited",
            op=PlanOp.LIMIT,
            input=current,
            limit=min(max(limit, 1), 50),
            window_anchor=window_anchor,
            window_size=window_size,
            file_ids=(file_id,),
            action_id="a0",
            action_goal="list",
        )
    )
    steps.append(
        PlanStep(
            id="result",
            op=PlanOp.PROJECT,
            input="limited",
            fields=projection,
            limit=min(max(limit, 1), 50),
            window_anchor=window_anchor,
            window_size=window_size,
            file_ids=(file_id,),
            action_id="a0",
            action_goal="list",
        )
    )
    return _plan(file_id, ["list"], steps)


def _plan(file_id: int, goals: list[str], steps: list[PlanStep]) -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=(file_id,), version_mode="current", authorized_only=True),
        goals=goals,
        steps=steps,
        planner_type="deterministic",
    )
