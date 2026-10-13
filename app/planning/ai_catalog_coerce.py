"""Catalog-aware repairs for AI plans. No free-text parsing."""

from __future__ import annotations

import re
from typing import Any

from app.planning.ai_turn_schema import (
    AggregateRequirement,
    FilterLogicRequirement,
    FilterRequirement,
    GoalRequirement,
    GroupingRequirement,
    PlannedQueryAction,
    PlannedTurn,
    ProjectionRequirement,
    SearchTextRequirement,
    SortingRequirement,
    StatedRequirement,
)
from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.turn_schema import AggregateSpec, FilterSpec
from app.planning.value_expansion import exact_category_terms

_TRUE_TOKENS = frozenset({"deceased", "yes", "true", "y", "t", "1"})
_FALSE_TOKENS = frozenset({"no", "false", "n", "f", "0"})
_GENDER_IN = {
    "female": ["F", "female", "f", "woman", "girl"],
    "f": ["F", "female", "f"],
    "male": ["M", "male", "m", "man", "boy"],
    "m": ["M", "male", "m"],
    "woman": ["F", "female", "f", "woman"],
    "man": ["M", "male", "m", "man"],
}


def coerce_stated_requirements(
    requirements: list[StatedRequirement],
    catalog: FieldCatalog,
    file_id: int,
    question: str = "",
) -> list[StatedRequirement]:
    """Apply the same catalog coercions to review reconstruction as to plan actions.

    Review requirements are independently generated. If trusted action coercion turns a
    boolean/value-family/name-token filter into its canonical executable form but the review
    requirement is left in the pre-coercion form, provenance comparison can falsely report
    that the reviewer introduced an unreconstructed filter. Keep both sides of the semantic
    comparison on the same canonical surface.
    """
    coerced: list[StatedRequirement] = []
    for item in requirements:
        coerced.extend(_coerce_requirements(item, catalog, file_id, question))
    coerced = _flatten_compare_requirements(coerced)
    coerced = _rank_requirements(coerced, catalog, file_id)
    coerced = _row_count_requirements(coerced, catalog, file_id)
    return _distinct_count_requirements(coerced) if _asks_distinct_count(question) else coerced


def coerce_planned_turn(
    planned: PlannedTurn,
    catalog: FieldCatalog,
    file_id: int,
    question: str = "",
) -> PlannedTurn:
    """Rewrite catalog-illegal slots so parity and validation stay aligned."""
    actions = []
    changed = False
    flattened: set[int] = set()
    for index, action in enumerate(planned.actions):
        if isinstance(action, PlannedQueryAction):
            if (action.goal in {"aggregate", "count"} and action.aggregate
                    and action.aggregate.function == "count" and not action.group_by
                    and (not action.aggregate.field or catalog.resolve_field(file_id, action.aggregate.field))):
                # COUNT compiles to row count; its unused field is not a measured attribute.
                action = action.model_copy(update={"goal": "count", "aggregate": None})
                changed = True
            dimension = (action.aggregate.field if action.aggregate and action.aggregate.function == "count_distinct"
                         and action.group_by in ([],[action.aggregate.field]) else
                         action.group_by[0] if len(action.group_by) == 1 and action.aggregate is None else None)
            if (_asks_distinct_count(question) and action.goal == "count" and dimension
                    and action.group_value_part in (None, "raw") and action.top_n is None):
                action = action.model_copy(update={"goal":"aggregate", "group_by":[],
                    "aggregate":AggregateSpec(function="count_distinct",field=dimension)})
                changed = True
            single = _flatten_single_compare(action)
            if single is not action:
                flattened.add(index)
            rewritten = _coerce_query(single, catalog, file_id, question)
            changed = changed or rewritten != action
            actions.append(rewritten)
        else:
            actions.append(action)
    requirements: list[Any] = []
    for item in planned.stated_requirements:
        if item.action_index in flattened and getattr(item, "collection", None) == "compare":
            item = item.model_copy(
                update={"collection": "filters", "branch_index": None, "branch_label": None}
            )
        elif item.action_index in flattened and isinstance(item, SearchTextRequirement):
            item = item.model_copy(update={"branch_index": None, "branch_label": None})
        requirements.extend(_coerce_requirements(item, catalog, file_id, question))
    requirements = _flatten_compare_requirements(requirements)
    requirements = _rank_requirements(requirements, catalog, file_id)
    requirements = _row_count_requirements(requirements, catalog, file_id)
    if _asks_distinct_count(question):
        requirements = _distinct_count_requirements(requirements)
    if not changed and requirements == planned.stated_requirements:
        return planned
    return planned.model_copy(update={"actions": actions, "stated_requirements": requirements})


def _asks_distinct_count(question: str) -> bool:
    return bool(re.match(r"^\s*how\s+many\b",question,re.I)
                and re.search(r"\b(?:distinct|unique|different)\b",question,re.I)
                and not re.search(r"\b(?:per|each|by|distribution|breakdown)\b",question,re.I))


def _default_name_order(question: str) -> bool:
    """First/last named records use full-name order unless another order is requested."""
    return bool(re.search(r"\b(?:first|last)\s+\d+\b", question, re.I) and not re.search(
        r"\b(?:sort\w*|order\w*|alphabet\w*|by|earliest|latest|oldest|youngest|highest|lowest|largest|smallest|longest|shortest|reverse|descending)\b",
        question, re.I,
    ))


def _full_name_order(question: str) -> bool:
    return _default_name_order(question) or bool(re.search(
        r"\b(?:sort\w*|order\w*|alphabet\w*)\b[^.!?]*\b(?:full (?:student )?name|student name|display name)\b",
        question, re.I,
    ))


def _rank_requirements(requirements: list[Any], catalog: FieldCatalog, file_id: int) -> list[Any]:
    ranked = {item.action_index for item in requirements
              if isinstance(item, GoalRequirement) and item.goal == "rank"}
    # Count is the rank operation's computed ordering, never a source field.
    return [item for item in requirements if not (
        isinstance(item, SortingRequirement) and item.action_index in ranked
        and item.field.casefold() == "count" and catalog.resolve_field(file_id, item.field) is None
    )]


def _row_count_requirements(requirements: list[Any], catalog: FieldCatalog, file_id: int) -> list[Any]:
    goals = {item.action_index: item.goal for item in requirements if isinstance(item, GoalRequirement)}
    grouped = {item.action_index for item in requirements if isinstance(item, GroupingRequirement)}
    indexes = {item.action_index for item in requirements if isinstance(item, AggregateRequirement)
               and item.function == "count" and item.action_index not in grouped
               and goals.get(item.action_index) in {"aggregate", "count"}
               and (not item.field or catalog.resolve_field(file_id, item.field))}
    return [item.model_copy(update={"goal": "count"})
            if isinstance(item, GoalRequirement) and item.action_index in indexes else item
            for item in requirements if not (isinstance(item, AggregateRequirement) and item.action_index in indexes)]


def _flatten_compare_requirements(requirements: list[Any]) -> list[Any]:
    """Canonicalize one branch's population conditions for a non-comparison goal.

    This mirrors _flatten_single_compare on the independent reconstruction. Nothing
    is compared when there is only one branch. Multiple branches, OR filters, and
    DNF populations keep their coordinates and must pass normal validation.
    """
    goals = {item.action_index: item.goal for item in requirements if isinstance(item, GoalRequirement)}
    branches: dict[int, set[tuple[Any, Any]]] = {}
    blocked = set()
    for item in requirements:
        if (isinstance(item, FilterLogicRequirement) and item.filter_logic != "and") or (
            isinstance(item, FilterRequirement) and item.collection == "filter_groups"
        ):
            blocked.add(item.action_index)
        if (isinstance(item, FilterRequirement) and item.collection == "compare") or (
            isinstance(item, SearchTextRequirement) and (item.branch_index is not None or item.branch_label)
        ):
            branches.setdefault(item.action_index, set()).add((item.branch_index, item.branch_label))
    flatten = {index for index, values in branches.items()
               if len(values) == 1 and index not in blocked and goals.get(index) not in (None, "compare")}
    result = []
    for item in requirements:
        if item.action_index in flatten:
            if isinstance(item, FilterRequirement) and item.collection == "compare":
                item = item.model_copy(update={"collection": "filters", "branch_index": None, "branch_label": None})
            elif isinstance(item, SearchTextRequirement):
                item = item.model_copy(update={"branch_index": None, "branch_label": None})
        result.append(item)
    return result


def _distinct_count_requirements(requirements: list[Any]) -> list[Any]:
    """A count of one selected dimension is count_distinct for this explicit request.

    Canonicalize the same representation on both sides of independent review.
    The AI's chosen field and population are preserved; no field is inferred here.
    """
    grouped = {}
    for item in requirements:
        if isinstance(item, GroupingRequirement):
            grouped.setdefault(item.action_index, []).append(item)
    indexes = {item.action_index for item in requirements
               if isinstance(item,GoalRequirement) and item.goal == "count"
               and ((len(grouped.get(item.action_index, [])) == 1
                     and grouped[item.action_index][0].value_part in (None,"raw"))
                    or (not grouped.get(item.action_index) and any(
                        isinstance(aggregate,AggregateRequirement) and aggregate.action_index == item.action_index
                        and aggregate.function == "count_distinct" and aggregate.field
                        for aggregate in requirements)))}
    result = []
    for item in requirements:
        if item.action_index in indexes:
            if isinstance(item,GoalRequirement):
                item = item.model_copy(update={"goal":"aggregate"})
            elif isinstance(item,GroupingRequirement):
                item = AggregateRequirement(action_index=item.action_index,text=item.text,
                    function="count_distinct",field=item.field)
        result.append(item)
    return result


def _flatten_single_compare(action: PlannedQueryAction) -> PlannedQueryAction:
    """One comparison branch on a query that is not a comparison is just its conditions.

    Branches are compiled only for goal=compare. "Which communities had the largest number of
    students who died from tuberculosis?" came back as a ranking with the tuberculosis
    condition in a lone branch, and the ranking that executed counted every death. With one
    branch there is nothing to compare, so its filters are the population; with several, the
    validator refuses the plan instead of dropping them.
    """
    if action.goal == "compare" or len(action.compare) != 1 or action.filter_groups:
        return action
    branch = action.compare[0]
    if branch.search_text and action.search_text and branch.search_text != action.search_text:
        return action
    if action.filter_logic != "and" and branch.filters:
        return action
    return action.model_copy(
        update={
            "filters": [*action.filters, *branch.filters],
            "search_text": action.search_text or branch.search_text,
            "compare": [],
        }
    )


def _coerce_query(
    action: PlannedQueryAction,
    catalog: FieldCatalog,
    file_id: int,
    question: str = "",
) -> PlannedQueryAction:
    updates: dict[str, Any] = {}
    projection = _coerce_projection(action.requested_fields, catalog, file_id)
    if projection != action.requested_fields:
        updates["requested_fields"] = projection
    filters: list[FilterSpec] = []
    for item in action.filters:
        filters.extend(_coerce_filters(item, catalog, file_id))
    if filters != list(action.filters):
        updates["filters"] = filters

    groups = []
    for group in action.filter_groups:
        rewritten_group: list[FilterSpec] = []
        for item in group.filters:
            # A DNF group is a conjunction, so splitting a token-index lookup into
            # multiple required tokens preserves its meaning inside the group.
            rewritten_group.extend(_coerce_filters(item, catalog, file_id))
        groups.append(group.model_copy(update={"filters": rewritten_group}))
    if groups != list(action.filter_groups):
        updates["filter_groups"] = groups

    denominator: list[FilterSpec] = []
    for item in action.denominator_filters:
        denominator.extend(_coerce_filters(item, catalog, file_id))
    if denominator != list(action.denominator_filters):
        updates["denominator_filters"] = denominator

    compare = []
    for branch in action.compare:
        branch_filters: list[FilterSpec] = []
        for item in branch.filters:
            branch_filters.extend(_coerce_filters(item, catalog, file_id))
        compare.append(branch.model_copy(update={"filters": branch_filters}))
    if compare != list(action.compare):
        updates["compare"] = compare

    group_by = [
        _remap_missing_field(name, action, catalog, file_id) or name for name in action.group_by
    ]
    if group_by != list(action.group_by):
        updates["group_by"] = group_by
    if action.goal == "rank" and action.sort_by and action.sort_by.casefold() == "count" and catalog.resolve_field(file_id, action.sort_by) is None:
        updates["sort_by"] = None
    elif action.goal == "list" and (action.window is not None or action.sort_by) and _full_name_order(question) and catalog.resolve_field(file_id, "student_name"):
        updates["sort_by"] = "student_name"
        if _default_name_order(question):
            updates["sort_direction"] = "asc"
    elif action.sort_by:
        remapped = _remap_missing_field(action.sort_by, action, catalog, file_id)
        sort_spec = catalog.resolve_field(file_id, action.sort_by)
        if sort_spec and sort_spec.validation_rules.get("token_index") and catalog.resolve_field(file_id, "student_name"):
            remapped = "student_name"
        if remapped and remapped != action.sort_by:
            updates["sort_by"] = remapped
    elif action.goal == "list" and action.window is not None:
        if catalog.resolve_field(file_id, "student_name") is not None:
            updates["sort_by"] = "student_name"
            updates["sort_direction"] = action.sort_direction or "asc"
    if action.aggregate is not None and action.aggregate.field:
        remapped = _remap_missing_field(action.aggregate.field, action, catalog, file_id)
        if remapped and remapped != action.aggregate.field:
            updates["aggregate"] = action.aggregate.model_copy(update={"field": remapped})
    return action.model_copy(update=updates) if updates else action


def _coerce_filter(spec: FilterSpec, catalog: FieldCatalog, file_id: int) -> FilterSpec:
    field_name = _remap_missing_field(spec.field, None, catalog, file_id) or spec.field
    field = catalog.resolve_field(file_id, field_name)
    updates: dict[str, Any] = {}
    if field_name != spec.field:
        updates["field"] = field_name
        field = catalog.resolve_field(file_id, field_name)
    if field and field.semantic_type == "date" and spec.operator == "DATE_RANGE" and isinstance(spec.value, str):
        endpoints = re.fullmatch(r"\s*(\d{4}(?:-\d{2}){0,2})\s*:\s*(\d{4}(?:-\d{2}){0,2})\s*", spec.value)
        if endpoints:
            updates["value"] = list(endpoints.groups())
    if field and field.validation_rules.get("token_index") and spec.operator in {"STARTS_WITH", "STARTS_WITH_ANY"}:
        identity = catalog.resolve_field(file_id, "student_name")
        value = spec.value
        operator = spec.operator
        if operator == "STARTS_WITH_ANY" and isinstance(value, list) and len(value) == 1:
            operator, value = "STARTS_WITH", value[0]
        if identity and operator in identity.allowed_operators:
            return spec.model_copy(update={"field":"student_name", "operator":operator, "value":value})
    # Event-year conditions must examine all dates, including multiple admissions.
    # A scalar year is only a projection and cannot represent differing event years.
    year_path = re.fullmatch(r"canonical\.date_parts\.(\w+)\.year", field.canonical_json_path or "") if field else None
    if year_path and catalog.resolve_field(file_id, year_path[1]) is not None:
        converted = _event_year_filter(spec)
        if converted:
            return spec.model_copy(update={"field": year_path[1], **converted})
    if field is not None and (field.semantic_type or "").lower() == "boolean":
        token = _token(spec.value)
        if spec.operator in {"EQUALS", "CONTAINS", "CONTAINS_ANY", "IN"} and token:
            if token in _TRUE_TOKENS:
                return spec.model_copy(
                    update={**updates, "operator": "IS_TRUE", "value": None}
                )
            if token in _FALSE_TOKENS:
                return spec.model_copy(
                    update={**updates, "operator": "IS_FALSE", "value": None}
                )
    exact = _exact_category_match(field, spec)
    if exact is not None:
        return spec.model_copy(update={**updates, **exact})
    if field is not None and field.semantic_field == "gender" and spec.operator in {
        "EQUALS", "CONTAINS", "CONTAINS_ANY", "IN",
        "NOT_EQUALS", "NOT_CONTAINS", "NOT_CONTAINS_ANY", "NOT_IN",
    }:
        widened = _gender_values(spec.value)
        if widened is not None:
            updates["operator"] = "NOT_IN" if spec.operator.startswith("NOT_") else "IN"
            updates["value"] = widened
    return spec.model_copy(update=updates) if updates else spec


def _event_year_filter(spec: FilterSpec) -> dict[str, Any] | None:
    try:
        if spec.operator in {"IN", "NOT_IN"} and isinstance(spec.value, list):
            return {"operator": spec.operator, "value": [f"{int(year):04d}-00-00" for year in spec.value]}
        if spec.operator == "NUMBER_RANGE" and isinstance(spec.value, list) and len(spec.value) == 2:
            return {"operator": "DATE_RANGE", "value": [f"{int(spec.value[0]):04d}-01-01", f"{int(spec.value[1]):04d}-12-31"]}
        if spec.operator in {"EQUALS", "GREATER_THAN", "LESS_THAN", "NOT_EQUALS"}:
            year = int(spec.value)
            operator, value = {"EQUALS": ("YEAR_EQUALS", year),
                "GREATER_THAN": ("AFTER", f"{year:04d}-12-31"),
                "LESS_THAN": ("BEFORE", f"{year:04d}-01-01"),
                "NOT_EQUALS": ("NOT_EQUALS", f"{year:04d}-00-00")}[spec.operator]
            return {"operator": operator, "value": value}
    except (TypeError, ValueError):
        pass
    return None


def _coerce_filters(spec: FilterSpec, catalog: FieldCatalog, file_id: int) -> list[FilterSpec]:
    """One planned filter, as the one or more filters it should compile to.

    Only a token index splits, and sibling filters are ANDed, which is the semantics a
    multi-word name needs: every token must be present.
    """
    field_name = _remap_missing_field(spec.field, None, catalog, file_id) or spec.field
    field = catalog.resolve_field(file_id, field_name)
    tokens = _token_index_tokens(field, spec)
    if tokens is not None:
        return [
            FilterSpec(field=field_name, operator="CONTAINS", value=token) for token in tokens
        ]
    return [_coerce_filter(spec, catalog, file_id)]


def _token_index_tokens(field: FieldSpec | None, spec: FilterSpec) -> list[str] | None:
    """Match a bag of tokens token by token, never as a phrase.

    A field flagged token_index stores one token per recorded form and renders as a joined
    list -- name_search reads "albert, penance, pinnance". Searching that for the phrase
    "Albert Penance" cannot match, so a record that is plainly in the list came back as no
    record at all.

    The words of a name are all required, so this asks for every token. CONTAINS_ANY would
    be an OR and would return every Albert on the list.
    """
    if field is None or spec.operator not in {"CONTAINS", "EQUALS", "CONTAINS_ANY"}:
        return None
    if not (field.validation_rules or {}).get("token_index"):
        return None
    raw = spec.value if isinstance(spec.value, list) else [spec.value]
    tokens: list[str] = []
    for item in raw:
        tokens.extend(part for part in str(item or "").split() if part)
    if len(tokens) < 2:
        return None
    return tokens


def _exact_category_match(field: FieldSpec | None, spec: FilterSpec) -> dict[str, Any] | None:
    """Match an enumerated field by equality where a substring search would be wrong.

    A field that publishes recorded_values holds a closed set of categories, and asking for
    one of them with CONTAINS is a category match written as a text search. Whether that is
    safe depends on the categories -- exact_category_terms decides, and its docstring gives
    both failure modes.

    Only rewrites the operator; the field and value the planner chose are its own.
    """
    if field is None or spec.operator not in {"CONTAINS", "CONTAINS_ANY"}:
        return None
    requested = spec.value if isinstance(spec.value, list) else [spec.value]
    matched = exact_category_terms(
        (field.validation_rules or {}).get("recorded_values"), requested
    )
    if not matched:
        return None
    if len(matched) == 1:
        return {"operator": "EQUALS", "value": matched[0]}
    return {"operator": "IN", "value": matched}


def _coerce_requirements(item: Any, catalog: FieldCatalog, file_id: int, question: str = "") -> list[Any]:
    """Catalog-coerce one requirement, preserving one-to-one parity after splits."""
    if isinstance(item, ProjectionRequirement):
        return [item.model_copy(update={"requested_fields": _coerce_projection(item.requested_fields, catalog, file_id)})]
    if isinstance(item, FilterRequirement):
        spec = FilterSpec(field=item.field, operator=item.operator, value=item.value)
        rewritten = _coerce_filters(spec, catalog, file_id)
        if len(rewritten) == 1 and rewritten[0] == spec:
            return [item]
        return [
            item.model_copy(
                update={
                    "text": f"{part.field} {part.operator} {part.value}",
                    "field": part.field,
                    "operator": part.operator,
                    "value": part.value,
                }
            )
            for part in rewritten
        ]
    if isinstance(item, GroupingRequirement):
        remapped = _remap_missing_field(item.field, None, catalog, file_id)
        if remapped and remapped != item.field:
            return [item.model_copy(update={"field": remapped})]
    if isinstance(item, SortingRequirement):
        if _full_name_order(question) and catalog.resolve_field(file_id, "student_name"):
            updates = {"field": "student_name"}
            if _default_name_order(question):
                updates["direction"] = "asc"
            return [item.model_copy(update=updates)]
        remapped = _remap_missing_field(item.field, None, catalog, file_id)
        if remapped and remapped != item.field:
            return [item.model_copy(update={"field": remapped})]
        field = catalog.resolve_field(file_id, item.field)
        if field and field.validation_rules.get("token_index") and catalog.resolve_field(file_id, "student_name"):
            return [item.model_copy(update={"field":"student_name"})]
    return [item]


def source_projection(catalog: FieldCatalog, file_id: int) -> list[str]:
    """Expand a source-record projection from the authorized registry, once in code."""
    return [field.semantic_field for field in catalog.planner_fields_for(file_id)
            if field.semantic_field == "student_name" or field.raw_json_keys]


def _coerce_projection(fields: list[str], catalog: FieldCatalog, file_id: int) -> list[str]:
    if "*" in fields:
        return source_projection(catalog, file_id)
    result = []
    for name in fields:
        field = catalog.resolve_field(file_id, name)
        if field and field.validation_rules.get("token_index") and catalog.resolve_field(file_id, "student_name"):
            name = "student_name"
        if name not in result:
            result.append(name)
    return result


def _remap_missing_field(
    name: str,
    action: PlannedQueryAction | None,
    catalog: FieldCatalog,
    file_id: int,
) -> str | None:
    if catalog.resolve_field(file_id, name) is not None:
        return None
    if name.strip().casefold() != "count":
        return None
    candidates: list[str] = []
    if action is not None:
        candidates.extend(action.requested_fields)
        if action.companion_field:
            candidates.append(action.companion_field)
        candidates.extend(item for item in action.group_by if item != name)
        if action.sort_by and action.sort_by != name:
            candidates.append(action.sort_by)
    unique = []
    for item in candidates:
        if catalog.resolve_field(file_id, item) is None:
            continue
        if item not in unique:
            unique.append(item)
    return unique[0] if len(unique) == 1 else None


def _token(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            return ""
        value = value[0]
    return str(value or "").strip().casefold()


def _gender_values(value: Any) -> list[str] | None:
    tokens = value if isinstance(value, list) else [value]
    widened: list[str] = []
    matched = False
    for item in tokens:
        family = _GENDER_IN.get(str(item or "").strip().casefold())
        if family is None:
            if item not in (None, "") and str(item) not in widened:
                widened.append(str(item))
            continue
        matched = True
        for term in family:
            if term not in widened:
                widened.append(term)
    return widened if matched else None


def spec_is_boolean(spec: FieldSpec | None) -> bool:
    return spec is not None and (spec.semantic_type or "").lower() == "boolean"
