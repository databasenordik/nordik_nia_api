from __future__ import annotations

import re
from dataclasses import dataclass

from app.memory.types import ConversationRecord, QueryFrame, QueryState, frame_key_for
from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import FilterOperator, Predicate, QueryPlan
from app.planning.query_plan_builder import build_query_plan, query_action_from_state
from app.planning.turn_schema import TurnPlan
from app.retrieval.entity_resolver import resolve_entity
from app.security.access_scope import AccessScope

_NEXT_PAGE = re.compile(
    r"^\s*(?:next(?:\s+\d+)?|keep going|show more|continue)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_WHAT_ABOUT = re.compile(r"^\s*what about\s+(.+?)\??\s*$", re.IGNORECASE)
_ONLY = re.compile(r"^\s*only\b(.+)$", re.IGNORECASE)
_EXCLUDE = re.compile(r"\bexclude\b.+\b(unknown|missing)\b.+\b(discharge|discharged)\b", re.IGNORECASE)
_GOING_BACK = re.compile(r"going back to\s+(.+)", re.IGNORECASE)
_ORDINAL = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b",
    re.IGNORECASE,
)
_THOSE_DECEASED = re.compile(r"\b(those|them|these)\b.*\bdeceased\b|\bdeceased\b.*\b(those|them|these)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_COMMUNITY = re.compile(r"\b(?:from|in|community)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)")

_ORDINAL_INDEX = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
    "last": -1,
}


@dataclass
class FollowUpResolution:
    kind: str
    plan: QueryPlan | None = None
    source_ids: list[str] | None = None
    frame_key: str | None = None
    query_state: QueryState | None = None
    detail: str = ""


def is_followup_utterance(text: str) -> bool:
    lowered = text.strip()
    return bool(
        _NEXT_PAGE.match(lowered)
        or _WHAT_ABOUT.match(lowered)
        or _ONLY.match(lowered)
        or _EXCLUDE.search(lowered)
        or _GOING_BACK.search(lowered)
        or _ORDINAL.search(lowered)
        or _THOSE_DECEASED.search(lowered)
    )


def _plan_from_state(state: QueryState, scope: AccessScope, catalog: FieldCatalog) -> QueryPlan:
    action = query_action_from_state(state)
    turn = TurnPlan(
        normalized_request="repeat stored action",
        actions=[action],
        final_response="deterministic",
        confidence=1.0,
    )
    return build_query_plan(turn, scope=scope, catalog=catalog)


def resolve_listed_ordinal(
    conversation: ConversationRecord,
    *,
    selector: str | None = None,
    index: int | None = None,
    target: str | None = None,
) -> FollowUpResolution:
    if target == "action":
        return _resolve_action_frame(conversation, selector=selector, index=index)
    frame = conversation.active_frame()
    if frame is None:
        return FollowUpResolution(kind="unresolved", detail="no listed result set to index")
    result = conversation.result_sets.get(frame.result_set_id or "")
    if not result or not result.ordered_source_ids:
        return FollowUpResolution(kind="unresolved", detail="no listed result set to index")
    if selector:
        position = _ORDINAL_INDEX.get(selector.lower())
        if position is None:
            return FollowUpResolution(kind="unresolved", detail="unknown result position")
        index = position
    if index is None:
        return FollowUpResolution(kind="unresolved", detail="no listed result set to index")
    if index < 0:
        index = len(result.ordered_source_ids) - 1
    if index >= len(result.ordered_source_ids):
        return FollowUpResolution(kind="unresolved", detail="that position is outside the last result set")
    return FollowUpResolution(
        kind="ordinal",
        source_ids=[result.ordered_source_ids[index]],
        frame_key=frame.frame_key,
        query_state=frame.query,
    )


def _resolve_action_frame(
    conversation: ConversationRecord,
    *,
    selector: str | None,
    index: int | None,
) -> FollowUpResolution:
    keys = list(conversation.last_action_keys)
    if not keys:
        return FollowUpResolution(kind="unresolved", detail="no prior actions to index")
    if selector:
        position = _ORDINAL_INDEX.get(selector.lower())
        if position is None:
            return FollowUpResolution(kind="unresolved", detail="unknown action position")
        index = position
    if index is None:
        return FollowUpResolution(kind="unresolved", detail="no prior actions to index")
    if index < 0:
        index = len(keys) - 1
    if index >= len(keys):
        return FollowUpResolution(kind="unresolved", detail="that position is outside the last action list")
    key = keys[index]
    frame = conversation.frames.get(key)
    if frame is None:
        return FollowUpResolution(kind="unresolved", detail="no matching action frame")
    return FollowUpResolution(
        kind="action_frame",
        frame_key=frame.frame_key,
        query_state=frame.query,
    )


def resolve_followup(
    question: str,
    scope: AccessScope,
    catalog: FieldCatalog,
    conversation: ConversationRecord,
) -> FollowUpResolution | None:
    text = " ".join(question.strip().split())
    if not text:
        return None

    going = _GOING_BACK.search(text)
    if going:
        frame = _find_frame(conversation, going.group(1), catalog)
        if frame is None:
            return FollowUpResolution(kind="unresolved", detail="no matching topic frame")
        plan = _plan_from_state(frame.query, scope, catalog)
        return FollowUpResolution(kind="restore_frame", plan=plan, frame_key=frame.frame_key, query_state=frame.query)

    ordinal = _ORDINAL.search(text)
    if ordinal and conversation.active_frame():
        frame = conversation.active_frame()
        assert frame is not None
        result = conversation.result_sets.get(frame.result_set_id or "")
        if not result or not result.ordered_source_ids:
            return FollowUpResolution(kind="unresolved", detail="no listed result set to index")
        index = _ORDINAL_INDEX[ordinal.group(1).lower()]
        if index < 0:
            index = len(result.ordered_source_ids) - 1
        if index >= len(result.ordered_source_ids):
            return FollowUpResolution(kind="unresolved", detail="that position is outside the last result set")
        return FollowUpResolution(
            kind="ordinal",
            source_ids=[result.ordered_source_ids[index]],
            frame_key=frame.frame_key,
            query_state=frame.query,
        )

    frame = conversation.active_frame()
    if frame is None:
        return None
    if not (
        _WHAT_ABOUT.match(text)
        or _ONLY.match(text)
        or _EXCLUDE.search(text)
        or _THOSE_DECEASED.search(text)
    ):
        return None

    merged = frame.query.model_copy(deep=True)
    if _EXCLUDE.search(text):
        merged.filters = _upsert(merged.filters, Predicate(field="discharged_date", operator=FilterOperator.IS_KNOWN))
    if _THOSE_DECEASED.search(text):
        merged.filters = _upsert(
            merged.filters, Predicate(field="deceased_status", operator=FilterOperator.IS_TRUE)
        )
        merged.goal = "exact_count"
    year = _YEAR.search(text)
    if year:
        merged.filters = _upsert(
            merged.filters,
            Predicate(field="admitted_date", operator=FilterOperator.YEAR_EQUALS, value=int(year.group(1))),
        )
    community = _COMMUNITY.search(text)
    if community:
        resolved = resolve_entity(community.group(1))
        label = resolved.canonical if resolved and not resolved.ambiguous else community.group(1)
        merged.filters = _upsert(
            merged.filters, Predicate(field="community", operator=FilterOperator.EQUALS, value=label)
        )
    named = catalog.resolve_dataset_ids(text)
    named = [file_id for file_id in named if file_id in scope.allowed_file_ids]
    if named:
        merged.file_ids = tuple(named)

    plan = _plan_from_state(merged, scope, catalog)
    return FollowUpResolution(
        kind="edit_query",
        plan=plan,
        frame_key=frame.frame_key,
        query_state=merged,
    )


def _upsert(filters: list[Predicate], incoming: Predicate) -> list[Predicate]:
    kept = [item for item in filters if item.field != incoming.field]
    kept.append(incoming)
    return kept


def _find_frame(conversation: ConversationRecord, phrase: str, catalog: FieldCatalog) -> QueryFrame | None:
    lowered = phrase.lower()
    for key, frame in conversation.frames.items():
        if key in lowered or frame.topic.lower() in lowered:
            return frame
        dataset = catalog.dataset(frame.query.file_ids[0]) if frame.query.file_ids else None
        if dataset and any(alias.lower() in lowered for alias in (dataset.user_facing_label, *dataset.aliases)):
            return frame
    named = catalog.resolve_dataset_ids(phrase)
    if named:
        key = frame_key_for(tuple(named))
        if key in conversation.frames:
            return conversation.frames[key]
    return None
