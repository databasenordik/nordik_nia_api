from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from app.execution.executor import ActionResult, ExecutionResult
from app.memory.types import (
    ConversationRecord,
    QueryFrame,
    QueryState,
    ResultSet,
    frame_key_for,
    make_source_id,
)
from app.planning.plan_schema import PlanOp, Predicate, QueryPlan


def update_memory_from_turn(
    conversation: ConversationRecord,
    *,
    question: str,
    answer: str,
    plan: QueryPlan | None,
    execution: ExecutionResult | None,
    query_state: QueryState | None = None,
    frame_key: str | None = None,
    mode: str = "text",
    interrupted: bool = False,
    citations: list[str] | None = None,
    source_actions: dict[str, object] | None = None,
) -> ConversationRecord:
    from app.planning.user_name import extract_introduced_name

    introduced = extract_introduced_name(question)
    if introduced:
        conversation.user_display_name = introduced
    _append_turn(conversation, role="user", text=question, mode=mode)
    _append_turn(
        conversation,
        role="assistant",
        text=answer,
        mode=mode,
        interrupted=interrupted,
        citations=list(citations or []),
    )
    conversation.recent_turns = conversation.recent_turns[-12:]
    if conversation.title is None:
        conversation.title = question[:80]

    if execution is not None and execution.action_results and plan is not None:
        keys = _persist_action_frames(
            conversation,
            plan,
            execution,
            source_actions=source_actions,
            question=_frame_question(question),
        )
        if keys:
            conversation.last_action_keys = keys
            conversation.rolling_summary = _summary(conversation)
            return conversation

    state = query_state or (extract_query_state(plan) if plan else None)
    if state is None:
        conversation.rolling_summary = _summary(conversation)
        return conversation

    key = frame_key or frame_key_for(state.file_ids)
    _write_frame(
        conversation,
        key=key,
        state=state,
        rows=_rows(execution),
        question=_frame_question(question),
    )
    conversation.active_frame_key = key
    conversation.last_action_keys = [key]
    conversation.rolling_summary = _summary(conversation)
    return conversation


def _append_turn(
    conversation: ConversationRecord,
    *,
    role: str,
    text: str,
    mode: str,
    interrupted: bool = False,
    citations: list[str] | None = None,
) -> None:
    sequence = conversation.message_sequence
    conversation.message_sequence += 1
    conversation.recent_turns.append(
        {
            "message_id": str(uuid4()),
            "turn_id": f"turn-{sequence}",
            "created_at": datetime.now(UTC).isoformat(),
            "role": role,
            "text": text,
            "mode": mode,
            "interrupted": interrupted,
            "citations": list(citations or []),
        }
    )


def extract_query_state(plan: QueryPlan) -> QueryState:
    filters: list[Predicate] = []
    retrieval_query = None
    limit = None
    sample = False
    for step in plan.steps:
        if step.op is PlanOp.FILTER:
            filters.extend(step.where)
        if step.query:
            retrieval_query = step.query
        if step.limit:
            limit = step.limit
        if step.op is PlanOp.SAMPLE:
            sample = True
    goal = "list" if "list" in plan.goals else "exact_count"
    return QueryState(
        file_ids=plan.scope.file_ids,
        version_mode=plan.scope.version_mode,
        filters=filters,
        goal=goal,
        retrieval_query=retrieval_query,
        sample=sample,
        limit=limit,
    )


def extract_action_query_state(
    plan: QueryPlan,
    action: ActionResult,
    *,
    source_action=None,
) -> QueryState:
    retrieval_query = None
    limit = None
    file_ids = plan.scope.file_ids
    for step in plan.steps:
        if step.action_id and step.action_id != action.action_id:
            continue
        if step.file_ids:
            file_ids = step.file_ids
        if step.query:
            retrieval_query = step.query
        if step.limit:
            limit = step.limit
    goal = "list" if action.goal == "list" else action.goal
    if goal == "count":
        goal = "exact_count"
    return QueryState(
        file_ids=file_ids,
        version_mode=plan.scope.version_mode,
        filters=list(action.predicates),
        goal=goal,
        retrieval_query=retrieval_query,
        limit=limit,
        offset=getattr(source_action, "offset", None),
        exhaustive=bool(getattr(source_action, "exhaustive", False)),
        presentation=getattr(source_action, "presentation", None),
        sort_by=getattr(source_action, "sort_by", None),
        sort_direction=getattr(source_action, "sort_direction", None),
        sample=bool(getattr(source_action, "sample", False)),
        requested_fields=list(getattr(source_action, "requested_fields", None) or []),
        window=getattr(source_action, "window", None),
        total_count=action.total_count,
        returned_count=len(action.rows),
        has_more=action.has_more,
        action_id=action.action_id,
        source_action=source_action,
    )


def _persist_action_frames(
    conversation: ConversationRecord,
    plan: QueryPlan,
    execution: ExecutionResult,
    *,
    source_actions: dict[str, object] | None = None,
    question: str | None = None,
) -> list[str]:
    from app.planning.query_plan_builder import reconstruct_query_action
    from app.planning.turn_schema import QueryAction

    keys: list[str] = []
    last_key = None
    last_frame: QueryFrame | None = None
    multi = len(execution.action_results) > 1
    sources = source_actions or {}
    for action in execution.action_results:
        raw = sources.get(action.action_id)
        source = None
        if isinstance(raw, QueryAction):
            source = raw
        elif raw is not None:
            source = QueryAction.model_validate(raw)
        if source is None:
            source = reconstruct_query_action(
                plan,
                action_id=action.action_id,
                goal=action.goal,
                predicates=action.predicates,
            )
        state = extract_action_query_state(plan, action, source_action=source)
        base = frame_key_for(state.file_ids)
        key = f"{base}-{action.action_id}" if multi else base
        frame = _write_frame(
            conversation,
            key=key,
            state=state,
            rows=action.rows,
            result=action,
            question=question,
        )
        keys.append(key)
        last_key = base
        last_frame = frame
    if multi and last_frame is not None and last_key is not None:
        conversation.frames[last_key] = last_frame
        conversation.active_frame_key = last_key
    elif keys:
        conversation.active_frame_key = keys[0]
    return keys


def _write_frame(
    conversation: ConversationRecord,
    *,
    key: str,
    state: QueryState,
    rows: list[dict],
    result=None,
    question: str | None = None,
) -> QueryFrame:
    previous = conversation.frames.get(key)
    source_ids = _source_ids_from_rows(rows)
    result_set_id = None
    if source_ids or (result is not None and getattr(result, "total_count", None) is not None):
        result_set_id = str(uuid4())
        total = getattr(result, "total_count", None) if result is not None else None
        offset = getattr(result, "offset", 0) if result is not None else 0
        page_size = getattr(result, "page_size", None) if result is not None else state.limit
        has_more = bool(getattr(result, "has_more", False)) if result is not None else False
        conversation.result_sets[result_set_id] = ResultSet(
            id=result_set_id,
            file_ids=state.file_ids,
            ordered_source_ids=source_ids,
            source_turn_id=str(uuid4()),
            action_id=state.action_id,
            total_count=total,
            returned_count=len(source_ids),
            offset=offset or 0,
            limit=page_size,
            has_more=has_more,
            sort_by=state.sort_by,
            sort_direction=state.sort_direction,
            sample=state.sample,
            requested_fields=list(state.requested_fields),
            window=state.window,
        )
    frame = QueryFrame(
        frame_key=key,
        topic=_topic(state),
        query=state,
        result_set_id=result_set_id,
        last_source_ids=source_ids,
        action_id=state.action_id,
        # A turn with no question of its own -- a pasted correction re-asking an earlier
        # one -- keeps the question the frame already answers.
        question=question if question is not None else (previous.question if previous else None),
    )
    conversation.frames[key] = frame
    return frame


def _frame_question(question: str) -> str | None:
    from app.planning.corrections import is_pasted_correction

    text = " ".join((question or "").split())
    if not text or is_pasted_correction(question):
        return None
    return text[:4000]


def _rows(execution: ExecutionResult | None) -> list[dict]:
    if execution is None:
        return []
    return list(execution.rows)


def _source_ids(execution: ExecutionResult | None) -> list[str]:
    return _source_ids_from_rows(_rows(execution))


def _source_ids_from_rows(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        file_id = row.get("file_id")
        source_row_id = row.get("source_row_id")
        if file_id is None or source_row_id is None:
            continue
        ids.append(make_source_id(int(file_id), int(row.get("version") or 1), int(source_row_id)))
    return ids


def _topic(state: QueryState) -> str:
    if state.file_ids == (91,):
        return "confirmed-deaths records"
    if state.file_ids == (49,):
        return "master-list students"
    return "research records"


def _summary(conversation: ConversationRecord) -> str:
    bits = [turn["text"] for turn in conversation.recent_turns[-4:]]
    text = " ".join(bits)
    return text[:400]


def record_candidate_list(
    conversation: ConversationRecord,
    *,
    file_ids: tuple[int, ...],
    records: list[tuple[int, int]],
    key: str,
) -> None:
    """Make an offered list of records indexable by a follow-up.

    A turn that offers candidates -- "did you mean one of these" -- is answered naturally by
    pointing at one: "the second one". That only works if the offer is registered as a
    result set, which normally happens as a side effect of running a QueryPlan. This path
    has no plan and no execution, so it registers the list directly.
    """
    if not records:
        return
    _write_frame(
        conversation,
        key=key,
        state=QueryState(
            file_ids=tuple(file_ids),
            goal="lookup",
            limit=len(records),
            returned_count=len(records),
            total_count=len(records),
        ),
        rows=[
            {"file_id": file_id, "source_row_id": source_row_id}
            for file_id, source_row_id in records
        ],
    )
    # The offer is what the next turn is about, so it becomes the frame an ordinal indexes.
    conversation.active_frame_key = key
