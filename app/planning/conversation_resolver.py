from __future__ import annotations

import re
from dataclasses import dataclass

from app.planning.catalog import FieldCatalog
from app.planning.input_normalizer import NormalizedTurn
from app.planning.plan_schema import FilterOperator, PlanOp, Predicate, QueryPlan
from app.planning.turn_schema import ActiveQuery
from app.retrieval.entity_resolver import resolve_entity
from app.security.access_scope import AccessScope

_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_BEFORE = re.compile(r"\bbefore\b", re.IGNORECASE)
_AFTER = re.compile(r"\bafter\b", re.IGNORECASE)
_FOLLOWUP_CUE = re.compile(
    r"^(?:and\s+)?(?:what\s+about|how\s+about|make\s+(?:it|that)|instead|rather|"
    r"actually|then|and)?\s*",
    re.IGNORECASE,
)
_LETTER = re.compile(r"\b([A-Za-z])\b")
_COMMUNITY_FOLLOWUP = re.compile(
    r"^(?:and\s+)?(?:what\s+about|how\s+about|make\s+(?:it|that)|instead|rather)?\s*"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,4})\s*[?.!]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationContext:
    """Compact planning state. No source rows are exposed here."""

    active_plan: QueryPlan | None = None
    active_query_text: str = ""
    memory_text: str = ""
    active_query: ActiveQuery | None = None
    topic_frames: tuple[dict, ...] = ()
    last_action_keys: tuple[str, ...] = ()
    user_display_name: str | None = None
    last_assistant_text: str | None = None


@dataclass(frozen=True)
class ConversationResolution:
    plan: QueryPlan | None
    confidence: float = 0.0
    detail: str = ""
    ambiguity: str | None = None


def resolve_conversation_followup(
    turn: NormalizedTurn,
    context: ConversationContext | None,
    scope: AccessScope,
    catalog: FieldCatalog,
) -> ConversationResolution | None:
    """Resolve only high-confidence edits; otherwise let semantic NLU handle it."""

    if context is None or context.active_plan is None:
        return None
    plan = context.active_plan
    if any(file_id not in scope.allowed_file_ids for file_id in plan.scope.file_ids):
        return None

    text = turn.normalized_text.strip()
    if not text:
        return None

    year_match = _YEAR.search(text)
    if year_match and _looks_like_short_followup(text, year_match.group(0)):
        edited = _edit_unique_predicate(
            plan,
            predicate_test=lambda item: _is_date_predicate(item, catalog, plan.scope.file_ids),
            new_value=int(year_match.group(1)),
            new_operator=(
                FilterOperator.BEFORE
                if _BEFORE.search(text)
                else FilterOperator.AFTER
                if _AFTER.search(text)
                else None
            ),
        )
        if edited is not None:
            return ConversationResolution(edited, 0.99, "edited prior date constraint")

    letter_matches = _LETTER.findall(text)
    if len(letter_matches) == 1 and _looks_like_short_followup(text, letter_matches[0]):
        edited = _edit_unique_predicate(
            plan,
            predicate_test=lambda item: item.operator is FilterOperator.STARTS_WITH,
            new_value=letter_matches[0].upper(),
        )
        if edited is not None:
            return ConversationResolution(edited, 0.99, "edited prior prefix constraint")

    community_match = _COMMUNITY_FOLLOWUP.match(text)
    if community_match:
        label = community_match.group(1).strip()
        resolved = resolve_entity(label)
        if resolved and not resolved.ambiguous:
            edited = _edit_unique_predicate(
                plan,
                predicate_test=lambda item: item.field == "community",
                new_value=resolved.canonical,
            )
            if edited is not None:
                return ConversationResolution(edited, 0.98, "edited prior community constraint")

    return None


def _looks_like_short_followup(text: str, value: str) -> bool:
    stripped = _FOLLOWUP_CUE.sub("", text).strip(" .?!,;:")
    stripped = re.sub(r"^(?:before|after)\s+", "", stripped, flags=re.IGNORECASE)
    return stripped.lower() == value.lower() and len(text.split()) <= 7


def _is_date_predicate(
    predicate: Predicate,
    catalog: FieldCatalog,
    file_ids: tuple[int, ...],
) -> bool:
    specs = [catalog.resolve_field(file_id, predicate.field) for file_id in file_ids]
    return bool(specs) and all(spec is not None and spec.semantic_type == "date" for spec in specs)


def _edit_unique_predicate(
    plan: QueryPlan,
    *,
    predicate_test,
    new_value,
    new_operator: FilterOperator | None = None,
) -> QueryPlan | None:
    locations: list[tuple[int, int]] = []
    for step_index, step in enumerate(plan.steps):
        if step.op is not PlanOp.FILTER:
            continue
        for pred_index, predicate in enumerate(step.where):
            if predicate_test(predicate):
                locations.append((step_index, pred_index))
    if len(locations) != 1:
        return None

    step_index, pred_index = locations[0]
    copied = plan.model_copy(deep=True)
    predicate = copied.steps[step_index].where[pred_index]
    copied.steps[step_index].where[pred_index] = Predicate(
        field=predicate.field,
        operator=new_operator or predicate.operator,
        value=new_value,
    )
    copied.goals = list(dict.fromkeys([*copied.goals, "followup_edit"]))
    copied.planner_type = "deterministic"
    return copied
