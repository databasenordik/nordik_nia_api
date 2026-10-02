from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.planning.analytic_guards import is_analytic_question
from app.planning.catalog import FieldCatalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.deterministic_parser import parse_deterministic
from app.planning.input_normalizer import NormalizedTurn, normalize_user_turn
from app.planning.plan_coverage import plan_covers_question
from app.planning.plan_schema import QueryPlan
from app.planning.plan_validator import PlanValidationError, validate_query_plan
from app.planning.query_plan_builder import (
    QueryPlanBuildError,
    UnsupportedFieldsError,
    build_query_plan,
)
from app.planning.response_policy import (
    acknowledgement,
    canned_response,
    decide_final_response,
    only_clarification,
    only_direct_response,
    only_reference,
)
from app.planning.schema_query import is_schema_priority, parse_schema_query
from app.planning.semantic_compiler import (
    CompilerResult,
    SemanticCompiler,
    UnavailableSemanticCompiler,
)
from app.planning.timing import PlanningTimings, TimingRecorder
from app.planning.trusted_overrides import apply_trusted_overrides, trusted_fast_turn
from app.planning.turn_schema import TurnPlan
from app.planning.turn_validator import validate_turn_plan
from app.planning.value_expansion import expand_plan_values
from app.security.access_scope import AccessScope


@dataclass(frozen=True)
class PlanRouteResult:
    plan: QueryPlan | None
    status: str
    source: str
    normalized_turn: NormalizedTurn
    detail: str = ""
    response_text: str | None = None
    clarification_question: str | None = None
    needs_response_model: bool = False
    reasoning_calls: int = 0
    turn_plan: TurnPlan | None = None
    compiler: CompilerResult | None = None
    final_response: str | None = None
    timings: PlanningTimings = PlanningTimings()
    planner_attempts: int = 0
    planner_retry_count: int = 0
    review_attempts: int = 0
    review_status: str | None = None
    planner_total_model_calls: int = 0
    planner_failure_stage: str | None = None
    suggested_file_id: int | None = None
    planner_provider: str = ""
    planner_model: str = ""
    planner_usage: dict | None = None
    original_planned: object | None = None
    # Legacy fields kept so older traces/tests can still inspect skipped layers.
    deterministic: None = None
    canonical_query: None = None
    interpreter: CompilerResult | None = None


async def plan_user_turn(
    question: str,
    scope: AccessScope,
    catalog: FieldCatalog,
    *,
    semantic_compiler: SemanticCompiler | UnavailableSemanticCompiler | None = None,
    semantic_interpreter: SemanticCompiler | UnavailableSemanticCompiler | None = None,
    constrained_planner=None,
    conversation: ConversationContext | None = None,
    memory_text: str = "",
    input_mode: Literal["text", "voice"] = "text",
    canonical_min_confidence: float = 0.80,
    selected_file_id: int | None = None,
    reasoner=None,
    authorized_catalog: FieldCatalog | None = None,
) -> PlanRouteResult:
    """Plan one FINAL user turn: compiler → TurnPlan → trusted QueryPlan.

    Partial STT hypotheses must never call this function.
    ``canonical_min_confidence`` is accepted for call-site compatibility and ignored.
    """

    del constrained_planner, canonical_min_confidence
    from app.config import get_settings

    if get_settings().planner_mode == "ai":
        from app.planning.ai_planner import plan_user_turn_ai

        if selected_file_id is None:
            normalized = normalize_user_turn(question)
            return PlanRouteResult(
                plan=None,
                status="dataset_selection_required",
                source="ai_planner",
                normalized_turn=normalized,
                detail="Select a list before asking.",
                response_text="Select a list before asking.",
                review_status="skipped",
            )
        return await plan_user_turn_ai(
            question,
            scope,
            catalog,
            selected_file_id=selected_file_id,
            semantic_compiler=semantic_compiler,
            conversation=conversation,
            memory_text=memory_text,
            input_mode=input_mode,
            reasoner=reasoner,
            authorized_catalog=authorized_catalog,
        )

    timing = TimingRecorder()
    with timing.measure("normalize_ms"):
        turn = normalize_user_turn(question)
        scoped_catalog = catalog.for_scope(scope)

    if not (turn.normalized_text or question).strip():
        return PlanRouteResult(
            plan=None,
            status="empty",
            source="direct",
            normalized_turn=turn,
            detail="empty transcript",
            response_text="Sorry, I didn't catch that.",
            timings=timing.finish(),
        )

    if input_mode == "voice" and _looks_like_incomplete_utterance(question):
        message = "I heard an incomplete phrase. Please finish the request."
        return PlanRouteResult(
            plan=None,
            status="clarification",
            source="direct",
            normalized_turn=turn,
            detail="incomplete voice utterance",
            response_text=message,
            clarification_question=message,
            timings=timing.finish(),
        )

    context = conversation or ConversationContext(memory_text=memory_text)
    if memory_text and not context.memory_text:
        context = ConversationContext(
            active_plan=context.active_plan,
            active_query_text=context.active_query_text,
            memory_text=memory_text,
            active_query=context.active_query,
            topic_frames=context.topic_frames,
            last_action_keys=context.last_action_keys,
            user_display_name=context.user_display_name,
            last_assistant_text=context.last_assistant_text,
        )

    has_active_query = context.active_query is not None or bool(context.topic_frames)
    with timing.measure("trusted_fast_path_ms"):
        fast_turn = trusted_fast_turn(
            question,
            has_active_query=has_active_query,
            catalog=scoped_catalog,
            last_assistant_text=context.last_assistant_text,
        )
    if fast_turn is not None:
        with timing.measure("turn_validation_ms"):
            validated_fast_turn = validate_turn_plan(
                fast_turn,
                scope=scope,
                catalog=scoped_catalog,
                has_active_query=has_active_query,
            )
        return _route_validated_turn(
            validated_fast_turn,
            source="trusted_fast_path",
            normalized_turn=turn,
            scope=scope,
            catalog=scoped_catalog,
            context=context,
            input_mode=input_mode,
            timing=timing,
        )

    # Ranking, percentage, statistics, duplicate, distinct, and per-group questions
    # go straight to the compiler. The legacy grammars predate those computations and
    # would answer them with a bare count or an arbitrary first page.
    analytic = is_analytic_question(question)

    if not analytic and is_schema_priority(question):
        with timing.measure("schema_parse_ms"):
            selected_dataset = (
                scoped_catalog.datasets[0] if len(scoped_catalog.datasets) == 1 else None
            )
            priority_plan = (
                parse_schema_query(
                    question,
                    file_id=selected_dataset.file_id,
                    catalog=scoped_catalog,
                )
                if selected_dataset is not None
                else None
            )
        if priority_plan is not None and not plan_covers_question(
            priority_plan, question, scoped_catalog
        ):
            priority_plan = None
        if priority_plan is not None:
            with timing.measure("plan_validation_ms"):
                priority_plan = validate_query_plan(expand_plan_values(priority_plan, scoped_catalog), scope, scoped_catalog)
            return PlanRouteResult(
                plan=priority_plan,
                status="planned",
                source="schema_deterministic",
                normalized_turn=turn,
                detail="schema-driven deterministic plan",
                reasoning_calls=0,
                final_response="deterministic",
                timings=timing.finish(),
            )

    with timing.measure("deterministic_parse_ms"):
        deterministic = _no_deterministic_plan() if analytic else parse_deterministic(
            question,
            scope,
            scoped_catalog,
            conversation=context,
            normalized_turn=turn,
        )
    if deterministic.plan is not None and not plan_covers_question(
        deterministic.plan, question, scoped_catalog
    ):
        deterministic = _no_deterministic_plan()
    if deterministic.plan is not None:
        with timing.measure("plan_validation_ms"):
            deterministic_plan = validate_query_plan(
                expand_plan_values(deterministic.plan, scoped_catalog),
                scope,
                scoped_catalog,
            )
        return PlanRouteResult(
            plan=deterministic_plan,
            status="planned",
            source="deterministic_fast_path",
            normalized_turn=turn,
            detail=deterministic.detail,
            reasoning_calls=0,
            final_response="deterministic",
            timings=timing.finish(),
        )
    if deterministic.status == "access_restricted":
        return PlanRouteResult(
            plan=None,
            status="access_restricted",
            source="deterministic_fast_path",
            normalized_turn=turn,
            detail=deterministic.detail,
            response_text=deterministic.detail,
            reasoning_calls=0,
            final_response="deterministic",
            timings=timing.finish(),
        )

    # Extend the legacy deterministic grammar only after it has had the first
    # chance to resolve its mature fast paths and conversation follow-ups.
    if not analytic and deterministic.status == "needs_planning":
        explicit_followup = bool(
            re.search(
                r"\b(?:next|previous|more|them|those|these|previous|same|also|add|beside|each\s+one)\b",
                turn.normalized_text,
                re.IGNORECASE,
            )
        )
        if not (has_active_query and explicit_followup):
            with timing.measure("schema_parse_ms"):
                selected_dataset = (
                    scoped_catalog.datasets[0] if len(scoped_catalog.datasets) == 1 else None
                )
                schema_plan = (
                    parse_schema_query(
                        question,
                        file_id=selected_dataset.file_id,
                        catalog=scoped_catalog,
                    )
                    if selected_dataset is not None
                    else None
                )
            if schema_plan is not None and not plan_covers_question(
                schema_plan, question, scoped_catalog
            ):
                schema_plan = None
            if schema_plan is not None:
                with timing.measure("plan_validation_ms"):
                    schema_plan = validate_query_plan(
                        expand_plan_values(schema_plan, scoped_catalog), scope, scoped_catalog
                    )
                return PlanRouteResult(
                    plan=schema_plan,
                    status="planned",
                    source="schema_deterministic",
                    normalized_turn=turn,
                    detail="schema-driven deterministic plan",
                    reasoning_calls=0,
                    final_response="deterministic",
                    timings=timing.finish(),
                )

    compiler = semantic_compiler or semantic_interpreter or UnavailableSemanticCompiler(
        "semantic compiler was not injected"
    )
    with timing.measure("semantic_compiler_ms"):
        compiled = await compiler.compile(
            text=question,
            context=context,
            catalog=scoped_catalog,
            mode=input_mode,
            scope=scope,
        )

    if compiled.turn is None:
        return PlanRouteResult(
            plan=None,
            status="needs_reasoning",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=compiled.error or "semantic compiler could not produce a TurnPlan",
            response_text=_compiler_unavailable_text(input_mode, compiled),
            clarification_question=_compiler_unavailable_text(input_mode, compiled),
            reasoning_calls=compiled.reasoning_calls,
            compiler=compiled,
            interpreter=compiled,
            timings=timing.finish(),
        )

    compiled_turn = apply_trusted_overrides(
        compiled.turn,
        question,
        has_active_query=has_active_query,
        catalog=scoped_catalog,
        last_assistant_text=context.last_assistant_text,
    )
    with timing.measure("turn_validation_ms"):
        validated = validate_turn_plan(
            compiled_turn,
            scope=scope,
            catalog=scoped_catalog,
            has_active_query=has_active_query,
        )

    policy = decide_final_response(validated)
    frame_map = _frame_map(context)

    if only_reference(validated):
        return PlanRouteResult(
            plan=None,
            status="reference",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=validated.normalized_request or "reference previous result",
            reasoning_calls=compiled.reasoning_calls,
            turn_plan=validated,
            compiler=compiled,
            interpreter=compiled,
            final_response="deterministic",
            timings=timing.finish(),
        )

    if only_clarification(validated):
        question_text = validated.clarify_actions()[0].question
        return PlanRouteResult(
            plan=None,
            status="clarification",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=validated.normalized_request or "semantic ambiguity remains",
            response_text=question_text,
            clarification_question=question_text,
            reasoning_calls=compiled.reasoning_calls,
            turn_plan=validated,
            compiler=compiled,
            interpreter=compiled,
            final_response="compiler_response",
            timings=timing.finish(),
        )

    if only_direct_response(validated) or policy == "compiler_response":
        response = _compiler_response_text(
            validated,
            scoped_catalog,
            input_mode,
            user_display_name=context.user_display_name,
            last_assistant_text=context.last_assistant_text,
        )
        return PlanRouteResult(
            plan=None,
            status="conversational",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=validated.normalized_request or "conversational turn",
            response_text=response,
            reasoning_calls=compiled.reasoning_calls,
            turn_plan=validated,
            compiler=compiled,
            interpreter=compiled,
            final_response="compiler_response",
            timings=timing.finish(),
        )

    try:
        with timing.measure("query_plan_build_ms"):
            plan = build_query_plan(
                validated,
                scope=scope,
                catalog=scoped_catalog,
                active_query=context.active_query,
                frames=frame_map,
            )
        with timing.measure("plan_validation_ms"):
            plan = validate_query_plan(expand_plan_values(plan, scoped_catalog), scope, scoped_catalog)
    except PlanValidationError:
        raise
    except UnsupportedFieldsError as exc:
        response = _unsupported_fields_text(exc)
        return PlanRouteResult(
            plan=None,
            status="unsupported_fields",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=str(exc),
            response_text=response,
            reasoning_calls=compiled.reasoning_calls,
            turn_plan=validated,
            compiler=compiled,
            interpreter=compiled,
            final_response="deterministic",
            timings=timing.finish(),
        )
    except QueryPlanBuildError as exc:
        return PlanRouteResult(
            plan=None,
            status="needs_reasoning",
            source="semantic_compiler",
            normalized_turn=turn,
            detail=str(exc),
            response_text=_unplanned_clarification(input_mode),
            clarification_question=_unplanned_clarification(input_mode),
            reasoning_calls=compiled.reasoning_calls,
            turn_plan=validated,
            compiler=compiled,
            interpreter=compiled,
            final_response=policy,
            timings=timing.finish(),
        )

    prefix = acknowledgement(validated)
    detail = validated.normalized_request or "compiled query planned"
    if prefix:
        detail = f"{prefix} {detail}".strip()

    return PlanRouteResult(
        plan=plan,
        status="planned",
        source="semantic_compiler",
        normalized_turn=turn,
        detail=detail,
        reasoning_calls=compiled.reasoning_calls,
        turn_plan=validated,
        compiler=compiled,
        interpreter=compiled,
        final_response=policy,
        timings=timing.finish(),
    )


def _no_deterministic_plan():
    """Abstain from the legacy grammar without pretending it ran."""
    from app.planning.deterministic_parser import ParseResult

    return ParseResult(plan=None, status="needs_planning", detail="analytic question")


def _route_validated_turn(
    validated: TurnPlan,
    *,
    source: str,
    normalized_turn: NormalizedTurn,
    scope: AccessScope,
    catalog: FieldCatalog,
    context: ConversationContext,
    input_mode: Literal["text", "voice"],
    timing: TimingRecorder,
) -> PlanRouteResult:
    """Route an already validated zero-model TurnPlan."""

    policy = decide_final_response(validated)
    if only_reference(validated):
        return PlanRouteResult(
            plan=None,
            status="reference",
            source=source,
            normalized_turn=normalized_turn,
            detail=validated.normalized_request or "reference previous result",
            turn_plan=validated,
            final_response="deterministic",
            timings=timing.finish(),
        )

    if only_clarification(validated):
        question_text = validated.clarify_actions()[0].question
        return PlanRouteResult(
            plan=None,
            status="clarification",
            source=source,
            normalized_turn=normalized_turn,
            detail=validated.normalized_request or "semantic ambiguity remains",
            response_text=question_text,
            clarification_question=question_text,
            turn_plan=validated,
            final_response="compiler_response",
            timings=timing.finish(),
        )

    if only_direct_response(validated) or policy == "compiler_response":
        response = _compiler_response_text(
            validated,
            catalog,
            input_mode,
            user_display_name=context.user_display_name,
            last_assistant_text=context.last_assistant_text,
        )
        return PlanRouteResult(
            plan=None,
            status="conversational",
            source=source,
            normalized_turn=normalized_turn,
            detail=validated.normalized_request or "conversational turn",
            response_text=response,
            turn_plan=validated,
            final_response="compiler_response",
            timings=timing.finish(),
        )

    try:
        with timing.measure("query_plan_build_ms"):
            plan = build_query_plan(
                validated,
                scope=scope,
                catalog=catalog,
                active_query=context.active_query,
                frames=_frame_map(context),
            )
        with timing.measure("plan_validation_ms"):
            plan = validate_query_plan(expand_plan_values(plan, catalog), scope, catalog)
    except PlanValidationError:
        raise
    except UnsupportedFieldsError as exc:
        response = _unsupported_fields_text(exc)
        return PlanRouteResult(
            plan=None,
            status="unsupported_fields",
            source=source,
            normalized_turn=normalized_turn,
            detail=str(exc),
            response_text=response,
            turn_plan=validated,
            final_response="deterministic",
            timings=timing.finish(),
        )
    except QueryPlanBuildError as exc:
        clarification = _unplanned_clarification(input_mode)
        return PlanRouteResult(
            plan=None,
            status="needs_reasoning",
            source=source,
            normalized_turn=normalized_turn,
            detail=str(exc),
            response_text=clarification,
            clarification_question=clarification,
            turn_plan=validated,
            final_response=policy,
            timings=timing.finish(),
        )

    detail = validated.normalized_request or "deterministic query planned"
    prefix = acknowledgement(validated)
    if prefix:
        detail = f"{prefix} {detail}".strip()
    return PlanRouteResult(
        plan=plan,
        status="planned",
        source=source,
        normalized_turn=normalized_turn,
        detail=detail,
        turn_plan=validated,
        final_response=policy,
        timings=timing.finish(),
    )


def _compiler_response_text(
    turn: TurnPlan,
    catalog: FieldCatalog,
    input_mode: Literal["text", "voice"],
    user_display_name: str | None = None,
    last_assistant_text: str | None = None,
) -> str:
    catalog_text = describe_available_data(catalog)
    canned = canned_response(
        turn,
        mode=input_mode,
        catalog_text=catalog_text,
        user_display_name=user_display_name,
        last_assistant_text=last_assistant_text,
    )
    if canned:
        return canned
    if turn.clarify_actions():
        return turn.clarify_actions()[0].question
    ack = acknowledgement(turn)
    if ack:
        return ack
    return (
        "I'm Nia." if input_mode == "voice"
        else "I'm Nia, a research assistant. Ask me about the authorized records."
    )


def describe_available_data(catalog: FieldCatalog, file_ids: tuple[int, ...] = ()) -> str:
    datasets = list(catalog.datasets)
    if file_ids:
        wanted = set(file_ids)
        scoped = [item for item in datasets if item.file_id in wanted]
        if scoped:
            datasets = scoped
    if not datasets:
        return "There are no authorized research records in this session."
    chunks: list[str] = []
    for dataset in datasets:
        # Every registered field, internal matching aids included. Filtering to
        # planner_fields_for here looked tidier -- a researcher is told the list covers
        # "Name cell residue" -- and was wrong twice over: exposure_policy exists to keep
        # the planner prompt small, not to decide what a person may see, and hiding only
        # the internal subset left the answer incoherent, still listing the derived
        # *_spellings and *_month columns while dropping *_flags and *_iso. Which derived
        # columns belong in a schema answer is a product decision, not a filter to slip in.
        labels = [field.human_label for field in catalog.fields_for(dataset.file_id)]
        if labels:
            chunks.append(f"{dataset.user_facing_label}, covering {', '.join(labels)}")
        else:
            chunks.append(dataset.user_facing_label)
    if len(chunks) == 1:
        return f"The authorized research records are the {chunks[0]}."
    return "The authorized research records are: " + "; ".join(chunks) + "."


def _unsupported_fields_text(exc: UnsupportedFieldsError) -> str:
    requested = ", ".join(_humanize_field(item) for item in exc.requested)
    datasets = ", ".join(exc.dataset_labels) or "the selected records"
    available = ", ".join(exc.available_fields) or "no displayable fields"
    verb = "is" if len(exc.requested) == 1 else "are"
    return (
        f"I understood the request, but the requested field"
        f"{'s' if len(exc.requested) != 1 else ''} ({requested}) {verb} not available in {datasets}. "
        f"Available fields are: {available}."
    )


def _humanize_field(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


def _unplanned_clarification(input_mode: Literal["text", "voice"]) -> str:
    if input_mode == "voice":
        return (
            "Could you say that another way? Ask about a name, a community, "
            "a year, or the research records."
        )
    return (
        "I understood you, but I need a more specific question about the research records — "
        "for example a count, a name, a community, or a year."
    )


def _looks_like_incomplete_utterance(text: str) -> bool:
    stripped = text.strip()
    return bool(re.search(r"(?:[A-Za-z0-9][\-–—]|\.{3}|…)$", stripped))


def _compiler_unavailable_text(
    input_mode: Literal["text", "voice"],
    compiled: CompilerResult,
) -> str:
    del input_mode, compiled
    return _unplanned_clarification("text")


def _frame_map(context: ConversationContext) -> dict:
    from app.planning.turn_schema import ActiveQuery

    frames: dict[str, ActiveQuery] = {}
    if context.active_query is not None:
        key = context.active_query.frame_key or "active"
        frames[key] = context.active_query
    for item in context.topic_frames:
        if not isinstance(item, dict):
            continue
        try:
            query = ActiveQuery.model_validate(
                {
                    "datasets": item.get("datasets") or item.get("file_ids") or [],
                    "goal": item.get("goal") or "count",
                    "filters": item.get("filters") or [],
                    "group_by": item.get("group_by") or [],
                    "sort_by": item.get("sort_by"),
                    "result_set_id": item.get("result_set_id"),
                    "search_text": item.get("search_text"),
                    "limit": item.get("limit"),
                    "offset": item.get("offset"),
                    "sort_direction": item.get("sort_direction"),
                    "requested_fields": item.get("requested_fields") or [],
                    "window": item.get("window"),
                    "frame_key": item.get("frame_key") or item.get("key"),
                    "topic": item.get("topic"),
                    "action_id": item.get("action_id"),
                }
            )
        except Exception:
            continue
        frames[query.frame_key or query.topic or str(len(frames))] = query
    return frames
