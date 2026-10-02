from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from app.concurrency.cancellation import CancellationToken
from app.config import get_settings
from app.data_gateway.protocol import DataGateway
from app.execution.executor import ExecutionResult, QueryExecutor
from app.execution.formatter import (
    format_access_denied,
    format_deterministic_answer,
    format_spoken_answer,
    format_unplanned,
    is_plain_count,
)
from app.execution.match_review import read_matches, unchecked_words
from app.execution.name_correction import (
    ask_text,
    corrected_prefix,
    from_rows,
    is_name_field,
    rewrite_for,
    searched_name,
)
from app.execution.roster import Roster, fetch_rows, roster_from_rows
from app.execution.row_projection import field_descriptors, public_projected_row
from app.execution.text_search import TextSearchRefinement, refine_text_search
from app.llm.base import ReasoningProvider
from app.llm.citations import validate_synthesis
from app.llm.factory import reasoning_unavailable_reason
from app.llm.prompts import (
    SYNTHESIS_SYSTEM,
    WHOLE_LIST_SYSTEM,
    synthesis_user_prompt,
    whole_list_user_prompt,
)
from app.llm.schemas import SynthesisAnswer
from app.memory.followups import resolve_listed_ordinal
from app.memory.store import InMemoryMemoryStore, MemoryStore
from app.memory.types import ConversationRecord, QueryState, parse_source_id
from app.memory.updates import record_candidate_list, update_memory_from_turn
from app.observability.audit import default_auditor
from app.observability.tracing import get_tracer
from app.planning.catalog import FieldCatalog, static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.llm_planner import ConstrainedPlanner
from app.planning.plan_coverage import _label_pattern, field_mentions
from app.planning.plan_schema import QueryPlan
from app.planning.plan_validator import PlanValidationError, validate_query_plan
from app.planning.router import PlanRouteResult, plan_user_turn
from app.planning.semantic_compiler import (
    SemanticCompiler,
    UnavailableSemanticCompiler,
)
from app.planning.turn_schema import ActiveQuery, FilterSpec
from app.retrieval.context_builder import (
    build_evidence_packet,
    build_whole_list_packet,
    whole_list_facts,
)
from app.retrieval.types import EvidencePacket
from app.security.access_scope import AccessScope
from app.tts.humanize import humanize_for_speech
from app.tts.phrases import PhraseBuffer, split_phrases
from app.value_normalization import json_object

SpokenPhraseFn = Callable[[str], Awaitable[None]]
ProgressFn = Callable[[str, dict[str, Any]], Awaitable[None]]
logger = logging.getLogger("nia.execution")


@dataclass
class TurnResult:
    status: str
    answer: str
    plan: QueryPlan | None = None
    facts: list[dict[str, Any]] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    list_results: list[dict[str, Any]] = field(default_factory=list)
    reasoning_calls: int = 0
    planner_type: str | None = None
    detail: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    methods_run: list[str] = field(default_factory=list)
    parallel_step_peak: int = 0
    critical_path_ms: float = 0.0
    conversation_id: str | None = None
    selected_file_id: int | None = None
    suggested_file_id: int | None = None
    error_code: str | None = None
    citations: list[str] = field(default_factory=list)
    inference: bool = False
    spoken_text: str | None = None
    generation_id: str | None = None
    interrupted: bool = False
    delivered_text: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    planner_attempts: int = 0
    planner_retry_count: int = 0
    review_attempts: int = 0
    review_status: str | None = None
    planner_total_model_calls: int = 0
    planner_failure_stage: str | None = None
    planner_provider: str | None = None
    planner_model: str | None = None
    planner_usage: dict[str, Any] = field(default_factory=dict)
    original_planned: object | None = None
    latency: dict[str, Any] = field(default_factory=dict)


class AssistantTurnService:
    """Shared Assistant Core entry point for text and, later, voice."""

    def __init__(
        self,
        gateway: DataGateway,
        catalog: FieldCatalog | None = None,
        store: MemoryStore | None = None,
        reasoner: ReasoningProvider | None = None,
        auditor: Any | None = None,
    ) -> None:
        self._gateway = gateway
        self._catalog = catalog or static_catalog()
        self._store = store if store is not None else InMemoryMemoryStore()
        self._reasoner = reasoner
        # API and voice services are constructed after the database pool starts,
        # so production turns use PostgresAuditor. Unit tests naturally retain
        # the process-local auditor when no pool exists.
        self._auditor = auditor if auditor is not None else default_auditor()
        self._semantic_compiler_cache: SemanticCompiler | UnavailableSemanticCompiler | None = None
        self._latency_bag: dict[str, Any] | None = None
        # HTTP builds a service per request; the voice agent reuses one. Set and
        # cleared per turn, same as _latency_bag, so a GUI model pick cannot leak.
        self._turn_reasoner: ReasoningProvider | None = None
        # Set when a pasted correction re-asks an earlier question: the conversation still
        # records what the researcher typed, and the answer says what was checked.
        self._turn_user_text: str | None = None
        self._turn_preface: str | None = None

    async def answer(
        self,
        scope: AccessScope,
        question: str,
        *,
        mode: str = "text",
        conversation_id: str | None = None,
        selected_file_id: int | None = None,
        model: str | None = None,
        cancellation: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        on_progress: ProgressFn | None = None,
    ) -> TurnResult:
        bag: dict[str, Any] = {}
        self._latency_bag = bag
        self._turn_reasoner = self._reasoner_for(model)
        started = time.perf_counter()
        first_token_at: list[float] = []

        async def timed_progress(event: str, data: dict[str, Any]) -> None:
            if event == "answer_delta" and not first_token_at:
                first_token_at.append(time.perf_counter())
            if on_progress is not None:
                await on_progress(event, data)

        try:
            with get_tracer().span("turn", mode=mode, principal_id=scope.principal_id):
                result = await self._answer(
                    scope,
                    question,
                    mode=mode,
                    conversation_id=conversation_id,
                    selected_file_id=selected_file_id,
                    model=model,
                    cancellation=cancellation,
                    on_spoken_phrase=on_spoken_phrase,
                    on_progress=timed_progress,
                )
        finally:
            self._latency_bag = None
            self._turn_reasoner = None
            self._turn_user_text = None
            self._turn_preface = None
        bag["e2e_ms"] = (time.perf_counter() - started) * 1000
        bag["ttft_ms"] = (
            (first_token_at[0] - started) * 1000 if first_token_at else None
        )
        bag.update(_planner_latency_from_trace(result.trace))
        bag["planner_attempts"] = result.planner_attempts
        bag["review_attempts"] = result.review_attempts
        bag["cached_tokens"] = (result.planner_usage or {}).get("cached_tokens")
        bag["planner_unavailable"] = result.status == "planner_unavailable"
        result.latency = bag
        await self._auditor.record(
            event_type="query_completed" if result.status == "answered" else result.status,
            principal_id=scope.principal_id,
            authorization_outcome=(
                "denied"
                if result.status in {"access_restricted", "dataset_unavailable"}
                else "allowed"
            ),
            conversation_id=result.conversation_id,
            dataset_ids=(result.selected_file_id,) if result.selected_file_id else scope.allowed_file_ids,
            provider_metadata={
                "planner_type": result.planner_type,
                "reasoning_calls": result.reasoning_calls,
                "mode": mode,
                "planner_attempts": result.planner_attempts,
                "review_status": result.review_status,
                "planner_failure_stage": result.planner_failure_stage,
                "planner_provider": result.planner_provider,
                "planner_model": result.planner_model,
                "planner_usage": result.planner_usage,
            },
        )
        return result

    async def _answer(
        self,
        scope: AccessScope,
        question: str,
        *,
        mode: str,
        conversation_id: str | None,
        selected_file_id: int | None,
        model: str | None = None,
        cancellation: CancellationToken | None,
        on_spoken_phrase: SpokenPhraseFn | None,
        on_progress: ProgressFn | None,
    ) -> TurnResult:
        cancel = cancellation or CancellationToken()
        if get_settings().planner_mode == "ai":
            return await self._answer_ai(
                scope,
                question,
                mode=mode,
                conversation_id=conversation_id,
                selected_file_id=selected_file_id,
                model=model,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                on_progress=on_progress,
            )
        full_catalog, conversation = await asyncio.gather(
            self._planning_catalog(scope),
            self._store.get_or_create(scope, conversation_id),
        )
        # An explicit UI selection wins; then the list this conversation is already
        # tied to; only an unpinned, unselected turn lets the question choose.
        pinned = conversation.selected_file_id
        selected, selection_error = _resolve_selected_dataset(
            scope,
            full_catalog,
            selected_file_id or pinned,
            question,
        )
        if selection_error is not None:
            return TurnResult(
                status="dataset_unavailable",
                error_code="dataset_unavailable",
                answer=selection_error,
                conversation_id=conversation.id,
                selected_file_id=selected_file_id,
            )
        assert selected is not None

        # A question naming more than one list is answered once per list and merged,
        # instead of being refused. Each branch still reads exactly one dataset.
        fan_out = _fan_out_datasets(question, full_catalog, scope)
        if len(fan_out) > 1:
            return await self._answer_fan_out(
                scope,
                question,
                full_catalog,
                fan_out,
                conversation,
                mode=mode,
                cancel=cancel,
                on_progress=on_progress,
            )

        if conversation.selected_file_id is None:
            if conversation.recent_turns:
                return TurnResult(
                    status="legacy_unscoped_conversation",
                    error_code="legacy_unscoped_conversation",
                    answer=(
                        "This older chat was not tied to one list. Start a new chat to continue "
                        "with the selected list; this history remains available to read."
                    ),
                    conversation_id=conversation.id,
                    selected_file_id=selected,
                )
            conversation.selected_file_id = selected
            await self._store.save(conversation)
        elif conversation.selected_file_id != selected:
            return TurnResult(
                status="dataset_scope_conflict",
                error_code="dataset_scope_conflict",
                answer=(
                    "This conversation is already tied to another list. "
                    "Start a new chat for the selected list."
                ),
                conversation_id=conversation.id,
                selected_file_id=conversation.selected_file_id,
                suggested_file_id=selected,
            )

        # A pinned conversation stays on its list. Naming another list, or naming a
        # field only that list carries, is answered with a switch suggestion rather
        # than by silently answering from the wrong records.
        elsewhere = _other_list_for_question(question, full_catalog, scope, selected)
        if elsewhere is not None:
            other = full_catalog.dataset(elsewhere)
            label = other.user_facing_label if other is not None else "the other list"
            result = TurnResult(
                status="dataset_mismatch",
                error_code="dataset_mismatch",
                answer=(
                    f"That question is about {label}, but this chat is using a "
                    f"different list. Switch to {label} to continue."
                ),
                conversation_id=conversation.id,
                selected_file_id=selected,
                suggested_file_id=elsewhere,
            )
            await self._persist(
                conversation, question, result.answer, None, None, None, None, mode
            )
            return result


        scope = scope.narrow((selected,))
        catalog = full_catalog.for_scope(scope)
        if conversation.user_display_name is None:
            if scope.display_name:
                conversation.user_display_name = scope.display_name
            elif scope.principal_id == get_settings().standalone_demo_principal_id:
                conversation.user_display_name = get_settings().standalone_demo_username
        try:
            try:
                route = await plan_user_turn(
                    question,
                    scope,
                    catalog,
                    semantic_compiler=self._semantic_compiler(),
                    conversation=_conversation_context(conversation),
                    memory_text=_memory_text(conversation),
                    input_mode="voice" if mode == "voice" else "text",
                )
            except PlanValidationError as exc:
                # A planner (including the LLM) tried to reach outside the
                # AccessScope or emit a disallowed plan. Fail closed.
                status = exc.code if exc.code == "access_restricted" else "invalid_plan"
                return TurnResult(
                    status=status,
                    answer=format_access_denied() if status == "access_restricted" else format_unplanned(str(exc)),
                    reasoning_calls=1 if self._reasoner is not None else 0,
                    detail=str(exc),
                    conversation_id=conversation.id,
                    selected_file_id=selected,
                )

            if route.status == "reference" and route.turn_plan is not None:
                result = await self._answer_reference(
                    scope,
                    question,
                    conversation,
                    route,
                    mode,
                    cancel=cancel,
                    on_spoken_phrase=on_spoken_phrase,
                    on_progress=on_progress,
                )
                result.selected_file_id = selected
                return result

            if route.plan is not None:
                # A list small enough to read in full is answered from its rows. The
                # planner has already done the language work -- this only replaces how the
                # data request is served, and only when the whole list actually fits.
                whole = await self._answer_whole_list(
                    scope,
                    question,
                    selected,
                    conversation,
                    mode=mode,
                    cancel=cancel,
                    on_spoken_phrase=on_spoken_phrase,
                    catalog=catalog,
                    route=route,
                    on_progress=on_progress,
                )
                if whole is not None:
                    whole.selected_file_id = selected
                    return whole
                result = await self._run_plan(
                    scope,
                    question,
                    route.plan,
                    conversation,
                    mode=mode,
                    cancel=cancel,
                    on_spoken_phrase=on_spoken_phrase,
                    catalog=catalog,
                    route=route,
                    on_progress=on_progress,
                )
                result.selected_file_id = selected
                return result

            if route.status == "access_restricted":
                return TurnResult(
                    status="access_restricted",
                    answer=format_access_denied(),
                    reasoning_calls=route.reasoning_calls,
                    detail=route.detail,
                    planner_type=route.source,
                    conversation_id=conversation.id,
                    selected_file_id=selected,
                )

            if route.status == "unsupported_fields":
                # Name the list that does carry the requested field instead of
                # leaving the researcher to guess which chat to open.
                other = _field_dataset_suggestion(question, full_catalog, selected)
                hint = full_catalog.dataset(other) if other is not None else None
                if hint is not None:
                    route = replace(
                        route,
                        response_text=(
                            f"{route.response_text} "
                            f"{hint.user_facing_label} does record it — open a chat on "
                            f"that list to ask there."
                        ),
                    )

            # Conversational / clarification / still-unplannable turns all answer
            # in words. No QueryPlan is built and no database work is done.
            result = await self._direct_reply(
                route,
                question,
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                on_progress=on_progress,
            )
            result.selected_file_id = selected
            return result
        except asyncio.CancelledError:
            result = TurnResult(
                status="interrupted",
                answer="",
                conversation_id=conversation.id,
                selected_file_id=selected,
                interrupted=True,
                spoken_text="",
            )
            await self._persist(
                conversation,
                question,
                result.answer,
                None,
                None,
                None,
                None,
                mode,
                interrupted=True,
            )
            return result

    def _reasoner_for(self, model: str | None):
        """The reasoner that *answers* this turn, honouring a model the researcher picked.

        QueryRequest.model is answer-only. Planning and review always use the service
        default (PLANNER_MODEL). This helper is for synthesis, whole-list narrative,
        and general conversation.

        The service is built once per request by a FastAPI dependency, before the body is
        available, so the model cannot be chosen at construction time. The voice agent
        reuses one instance across turns, so the caller assigns the result to
        ``_turn_reasoner`` at the start of ``answer`` and clears it in the same
        ``finally`` as ``_latency_bag``. Providers are cached per model in the factory,
        so selecting one costs a dict lookup.

        Falls back to the configured reasoner whenever no model was named, the name is not
        on the allowlist, or the factory cannot build one -- a bad model name must not turn
        into a failed turn.
        """
        if not model:
            return self._reasoner
        try:
            from app.llm.factory import get_reasoning_provider, resolve_model

            if resolve_model(model) == get_settings().reasoning_model:
                return self._reasoner
            return get_reasoning_provider(model) or self._reasoner
        except Exception:
            logger.warning("could not build a provider for model %s", model, exc_info=True)
            return self._reasoner

    def _answer_reasoner(self) -> ReasoningProvider | None:
        """Provider that writes the answer. Planning uses ``self._reasoner`` instead."""
        if self._turn_reasoner is not None:
            return self._turn_reasoner
        return self._reasoner

    async def _answer_ai(
        self,
        scope: AccessScope,
        question: str,
        *,
        mode: str,
        conversation_id: str | None,
        selected_file_id: int | None,
        cancel: CancellationToken,
        on_spoken_phrase: SpokenPhraseFn | None,
        on_progress: ProgressFn | None,
        model: str | None = None,
    ) -> TurnResult:
        from app.planning.ai_planner import PLANNER_INVALID_TEXT, PLANNER_UNAVAILABLE_TEXT

        full_catalog, conversation = await asyncio.gather(
            self._planning_catalog(scope),
            self._store.get_or_create(scope, conversation_id),
        )
        if not question.strip():
            return TurnResult(
                status="unintelligible",
                answer="Sorry, I didn't catch that.",
                conversation_id=conversation.id,
                selected_file_id=conversation.selected_file_id,
                spoken_text="Sorry, I didn't catch that." if mode == "voice" else None,
                review_status="skipped",
            )
        selected = selected_file_id if selected_file_id is not None else conversation.selected_file_id
        if selected is None:
            answer = "Select a list before asking."
            return TurnResult(
                status="dataset_selection_required",
                error_code="dataset_selection_required",
                answer=answer,
                conversation_id=conversation.id,
                spoken_text=answer if mode == "voice" else None,
                review_status="skipped",
            )
        dataset = full_catalog.dataset(selected)
        if dataset is None or not scope.allows_file(selected, private=dataset.private):
            return TurnResult(
                status="dataset_unavailable",
                error_code="dataset_unavailable",
                answer="The selected list is unavailable or unauthorized.",
                conversation_id=conversation.id,
                selected_file_id=selected,
                review_status="skipped",
            )
        if conversation.selected_file_id is not None and selected_file_id is not None:
            if conversation.selected_file_id != selected_file_id:
                return TurnResult(
                    status="dataset_scope_conflict",
                    error_code="dataset_scope_conflict",
                    answer=(
                        "This conversation is already tied to another list. "
                        "Start a new chat for the selected list."
                    ),
                    conversation_id=conversation.id,
                    selected_file_id=conversation.selected_file_id,
                    suggested_file_id=selected_file_id,
                    review_status="skipped",
                )
        if conversation.selected_file_id is None:
            conversation.selected_file_id = selected
            await self._store.save(conversation)
        elif conversation.selected_file_id != selected:
            return TurnResult(
                status="dataset_scope_conflict",
                error_code="dataset_scope_conflict",
                answer=(
                    "This conversation is already tied to another list. "
                    "Start a new chat for the selected list."
                ),
                conversation_id=conversation.id,
                selected_file_id=conversation.selected_file_id,
                suggested_file_id=selected,
                review_status="skipped",
            )

        earlier = _question_being_corrected(conversation, selected, question)
        if earlier is not None:
            # The paste is a correction of an earlier answer, not a question: ask that one
            # again, so NIA's figures can be set beside the researcher's.
            self._turn_user_text = question
            self._turn_preface = (
                "Thank you for checking. I asked your earlier question again against the "
                f"list (“{_shorten(earlier, 200)}”), so the two can be compared:"
            )
            question = earlier

        authorized_catalog = full_catalog.for_scope(scope)
        scope = scope.narrow((selected,))
        catalog = full_catalog.for_scope(scope)
        if conversation.user_display_name is None:
            if scope.display_name:
                conversation.user_display_name = scope.display_name
            elif scope.principal_id == get_settings().standalone_demo_principal_id:
                conversation.user_display_name = get_settings().standalone_demo_username
        try:
            route = await plan_user_turn(
                question,
                scope,
                catalog,
                semantic_compiler=self._semantic_compiler(),
                conversation=_conversation_context_for_dataset(conversation, selected),
                memory_text=_memory_text(conversation),
                input_mode="voice" if mode == "voice" else "text",
                selected_file_id=selected,
                reasoner=self._reasoner,
                authorized_catalog=authorized_catalog,
            )
        except PlanValidationError as exc:
            status = exc.code if exc.code == "access_restricted" else "invalid_plan"
            return TurnResult(
                status=status,
                answer=format_access_denied() if status == "access_restricted" else format_unplanned(str(exc)),
                conversation_id=conversation.id,
                selected_file_id=selected,
                error_code=status,
                review_status="skipped",
            )

        if route.status in {"planner_unavailable", "planner_invalid_output"}:
            table = PLANNER_UNAVAILABLE_TEXT if route.status == "planner_unavailable" else PLANNER_INVALID_TEXT
            answer = table["voice" if mode == "voice" else "text"]
            result = TurnResult(
                status=route.status,
                error_code=route.status,
                answer=answer,
                spoken_text=table["voice"] if mode == "voice" else None,
                conversation_id=conversation.id,
                selected_file_id=selected,
                planner_type="ai_planner",
                reasoning_calls=route.reasoning_calls,
                detail=route.detail,
                trace=_route_trace(route),
            )
            _copy_planner_meta(result, route)
            await self._auditor.record(
                event_type=route.status,
                principal_id=scope.principal_id,
                authorization_outcome="allowed",
                conversation_id=conversation.id,
                dataset_ids=(selected,),
                provider_metadata={
                    "planner_failure_stage": route.planner_failure_stage,
                    "planner_attempts": route.planner_attempts,
                },
            )
            await self._persist(conversation, question, result.answer, None, None, None, None, mode)
            return result

        if route.status == "reference" and route.turn_plan is not None:
            result = await self._answer_reference(
                scope,
                question,
                conversation,
                route,
                mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                on_progress=on_progress,
            )
            result.selected_file_id = selected
            _copy_planner_meta(result, route)
            return result

        if route.plan is not None:
            # A list small enough to read in full is answered from its rows. The planner
            # has already read the question; this only changes how the data request is
            # served, and only when the whole list actually fits.
            whole = await self._answer_whole_list(
                scope,
                question,
                selected,
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                catalog=catalog,
                route=route,
                on_progress=on_progress,
            )
            if whole is not None:
                whole.selected_file_id = selected
                _copy_planner_meta(whole, route)
                whole.planner_type = "whole_list"
                return whole
            result = await self._run_plan(
                scope,
                question,
                route.plan,
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                catalog=catalog,
                route=route,
                on_progress=on_progress,
            )
            corrected = await self._correct_missed_name(
                scope,
                result,
                route.plan,
                catalog,
                question,
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                route=route,
                on_progress=on_progress,
            )
            if corrected is not None:
                result = corrected
            result.selected_file_id = selected
            _copy_planner_meta(result, route)
            return result

        if route.status == "access_restricted":
            result = TurnResult(
                status="access_restricted",
                error_code="access_restricted",
                answer=format_access_denied(),
                reasoning_calls=route.reasoning_calls,
                detail=route.detail,
                planner_type=route.source,
                conversation_id=conversation.id,
                selected_file_id=selected,
                trace=_route_trace(route),
            )
            _copy_planner_meta(result, route)
            return result

        result = await self._direct_reply(
            route,
            question,
            conversation,
            mode=mode,
            cancel=cancel,
            on_spoken_phrase=on_spoken_phrase,
            on_progress=on_progress,
        )
        result.selected_file_id = selected
        result.suggested_file_id = route.suggested_file_id
        if route.status == "dataset_not_selected":
            result.status = "dataset_not_selected"
            result.error_code = "dataset_not_selected"
        _copy_planner_meta(result, route)
        return result

    async def answer_from_planned(
        self,
        scope: AccessScope,
        planned,
        *,
        selected_file_id: int,
        question: str = "",
    ) -> TurnResult:
        """Execute an already-produced PlannedTurn with no additional planner call."""
        from app.planning.ai_planner import bind_scope
        from app.planning.predicate_contract import validate_predicate_contract
        from app.planning.query_plan_builder import build_query_plan
        from app.planning.turn_validator import validate_turn_plan
        from app.planning.value_expansion import expand_plan_values

        catalog = (await self._planning_catalog(scope)).for_scope(scope.narrow((selected_file_id,)))
        bound = bind_scope(planned, selected_file_id)
        validated = validate_turn_plan(
            bound,
            scope=scope.narrow((selected_file_id,)),
            catalog=catalog,
            strict=True,
        )
        plan = build_query_plan(validated, scope=scope.narrow((selected_file_id,)), catalog=catalog)
        validate_predicate_contract(plan)
        plan = validate_query_plan(expand_plan_values(plan, catalog, profile="ai"), scope.narrow((selected_file_id,)), catalog)
        conversation = await self._store.get_or_create(scope, None)
        conversation.selected_file_id = selected_file_id
        return await self._run_plan(
            scope.narrow((selected_file_id,)),
            question,
            plan,
            conversation,
            mode="text",
            catalog=catalog,
        )

    async def _answer_fan_out(
        self,
        scope: AccessScope,
        question: str,
        full_catalog: FieldCatalog,
        file_ids: list[int],
        conversation: ConversationRecord,
        *,
        mode: str,
        cancel: CancellationToken,
        on_progress: ProgressFn | None,
    ) -> TurnResult:
        """Answer a question that names several lists, one list at a time.

        Nia reads a single dataset per execution, so a cross-list question is run
        independently against each named list and the labelled answers are merged.
        No branch ever sees another list's rows, and no cross-dataset join happens.
        """
        parts: list[str] = []
        rows: list[dict[str, Any]] = []
        list_results: list[dict[str, Any]] = []
        facts: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        reasoning_calls = 0
        planner_type: str | None = None
        for file_id in file_ids:
            branch_scope = scope.narrow((file_id,))
            branch_catalog = full_catalog.for_scope(branch_scope)
            dataset = full_catalog.dataset(file_id)
            label = dataset.user_facing_label if dataset is not None else f"dataset {file_id}"
            branch_conversation = ConversationRecord(
                id=f"{conversation.id}:{file_id}",
                principal_id=conversation.principal_id,
                selected_file_id=file_id,
            )
            branch_question = _question_for_dataset(question, full_catalog, file_id)
            try:
                route = await plan_user_turn(
                    branch_question,
                    branch_scope,
                    branch_catalog,
                    semantic_compiler=self._semantic_compiler(),
                    conversation=_conversation_context(branch_conversation),
                    memory_text="",
                    input_mode="voice" if mode == "voice" else "text",
                )
                reasoning_calls += route.reasoning_calls
                planner_type = planner_type or route.source
                if route.plan is not None:
                    branch = await self._run_plan(
                        branch_scope,
                        question,
                        route.plan,
                        branch_conversation,
                        mode="text",
                        cancel=cancel,
                        catalog=branch_catalog,
                        route=route,
                    )
                else:
                    branch = TurnResult(
                        status=route.status,
                        answer=(route.response_text or "").strip()
                        or "I could not answer that for this list.",
                        reasoning_calls=route.reasoning_calls,
                        planner_type=route.source,
                    )
            except PlanValidationError as exc:
                branch = TurnResult(
                    status="invalid_plan", answer=format_unplanned(str(exc))
                )
            parts.append(f"{label}: {branch.answer.strip()}")
            rows.extend(branch.rows)
            list_results.extend(branch.list_results)
            facts.extend(branch.facts)
            evidence.extend(branch.evidence)
        answer = "\n\n".join(parts)
        await _emit_progress(on_progress, "answer_delta", {"text": answer})
        result = TurnResult(
            status="answered",
            answer=answer,
            facts=facts,
            rows=rows,
            list_results=list_results,
            evidence=evidence,
            reasoning_calls=reasoning_calls,
            planner_type=planner_type or "semantic_compiler",
            detail=f"answered separately for datasets {file_ids}",
            conversation_id=conversation.id,
            selected_file_id=conversation.selected_file_id,
            spoken_text=humanize_for_speech(answer) if mode == "voice" else None,
        )
        await self._persist(
            conversation, question, result.answer, None, None, None, None, mode
        )
        return result

    async def _whole_list_fields(self, catalog: FieldCatalog, file_id: int) -> list[str]:
        """The as-recorded fields of one list, excluding anything this pipeline derived.

        Derived name and date columns exist so a query planner can filter on them. When the
        model reads the rows itself they add nothing and cost payload, so the list it sees
        is the list a person actually recorded.
        """
        return [
            spec.semantic_field
            for spec in catalog.fields_for(file_id)
            if not (spec.canonical_json_path or "").startswith(
                ("canonical.name_parts.", "canonical.date_parts.")
            )
        ]

    async def _answer_whole_list(
        self,
        scope: AccessScope,
        question: str,
        file_id: int,
        conversation: ConversationRecord,
        *,
        mode: str,
        cancel: CancellationToken,
        on_spoken_phrase: SpokenPhraseFn | None,
        catalog: FieldCatalog,
        route: PlanRouteResult,
        on_progress: ProgressFn | None,
    ) -> TurnResult | None:
        """Answer from every row of a small list, or return None to use the query path.

        Returns None -- rather than raising or guessing -- whenever the list is too large,
        the reasoner is unavailable, or the rendered payload exceeds its budget. The caller
        then runs the compiled QueryPlan exactly as before, so this is additive.
        """
        settings = get_settings()
        reasoner = self._answer_reasoner()
        if not settings.whole_list_answer or reasoner is None:
            return None

        total = await self._gateway.count_records(scope, (file_id,), [])
        if total <= 0 or total > settings.whole_list_max_rows:
            return None

        records = await self._gateway.list_records(
            scope, (file_id,), [], limit=total, offset=0
        )
        if not records:
            return None

        packet = build_whole_list_packet(
            question,
            records,
            catalog=catalog,
            fields=await self._whole_list_fields(catalog, file_id),
            max_chars=settings.whole_list_max_chars,
        )
        if packet is None:
            # Too big to read after all; the query path is the correct answer here.
            return None
        # Exact tallies computed from the same rows. Counting is the one thing the model
        # measurably gets wrong on a list this size, so it is never asked to.
        packet = packet.model_copy(
            update={"facts": tuple(whole_list_facts(packet, total_rows=total))}
        )

        dataset = catalog.dataset(file_id)
        label = dataset.user_facing_label if dataset is not None else f"list {file_id}"
        prompt = whole_list_user_prompt(
            question,
            packet,
            _memory_text(conversation),
            mode=mode,
            dataset_label=label,
        )
        with get_tracer().span("llm", planner_type="whole_list"):
            synthesis = await reasoner.answer_structured(
                system=WHOLE_LIST_SYSTEM, user=prompt
            )
        # Unchanged from every other synthesized path: a citation outside the packet is
        # stripped, which here means outside the selected list.
        synthesis = validate_synthesis(synthesis, packet, scope)

        spoken_text = None
        if mode == "voice":
            spoken_text = humanize_for_speech(synthesis.answer)
            if on_spoken_phrase is not None:
                await _emit_phrases(spoken_text, cancel, on_spoken_phrase)
        await _emit_progress(on_progress, "answer_delta", {"text": synthesis.answer})

        result = TurnResult(
            status="answered",
            answer=synthesis.answer,
            reasoning_calls=route.reasoning_calls + 1,
            planner_type="whole_list",
            evidence=[item.model_dump(mode="json") for item in packet.items],
            conversation_id=conversation.id,
            citations=synthesis.citations,
            inference=synthesis.inference,
            spoken_text=spoken_text,
            interrupted=cancel.cancelled,
            trace=_route_trace(route),
            detail=f"answered from all {len(packet.items)} records of {label}",
        )
        # No QueryPlan and no QueryState: this turn had no compiled query to remember.
        # Follow-ups still work through conversation memory, and the whole list is resent
        # on the next turn anyway, so there is nothing a query frame would add.
        await self._persist(
            conversation,
            question,
            result.answer,
            None,
            None,
            None,
            None,
            mode,
            citations=synthesis.citations,
        )
        return result

    async def _correct_missed_name(
        self,
        scope: AccessScope,
        result: TurnResult,
        plan: QueryPlan,
        catalog: FieldCatalog,
        question: str,
        conversation: ConversationRecord,
        *,
        mode: str,
        cancel: CancellationToken | None,
        on_spoken_phrase: SpokenPhraseFn | None,
        route: PlanRouteResult,
        on_progress: ProgressFn | None,
    ) -> TurnResult | None:
        """Answer a misspelt name instead of reporting nothing found.

        Returns None whenever the turn was not a name lookup that found nothing, or no
        record is close enough -- "no matching records" is then the true answer and stands.
        """
        settings = get_settings()
        if not settings.name_correction or result.status != "answered":
            return None
        if _found_any(result):
            return None
        searched_for = searched_name(plan, catalog, question)
        if not searched_for:
            return None
        suggest = getattr(self._gateway, "suggest_name_matches", None)
        if not callable(suggest):
            return None
        rows = await suggest(
            scope,
            tuple(plan.scope.file_ids),
            searched_for,
            max_distance=settings.name_correction_max_distance,
            limit=settings.name_correction_limit,
        )
        correction = from_rows(searched_for, rows)
        if correction is None:
            return None

        certain = correction.certain
        if certain is not None:
            rerun = await self._run_plan(
                scope,
                question,
                rewrite_for(plan, catalog, certain.display_name, searched_for=searched_for),
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                catalog=catalog,
                route=route,
                on_progress=on_progress,
            )
            if rerun.status == "answered" and _found_any(rerun):
                prefix = corrected_prefix(searched_for, certain.display_name)
                rerun.answer = f"{prefix} {rerun.answer}".strip()
                rerun.detail = "name_correction_applied"
                return rerun
            # The corrected search still found nothing -- the suggestion stands, so offer it
            # rather than falling back to "no matching records" for a person who is there.
        # Two or three letters out, several people equally close, or a one-letter guess the
        # corrected search could not confirm: the researcher knows which one they meant, so
        # ask rather than pick.
        answer = ask_text(correction)
        await _emit_progress(on_progress, "answer_delta", {"text": answer})
        # Persisted like any other turn: the question and the names offered belong in
        # the conversation, so the researcher can scroll back to them and so a reply
        # naming one of them has something to resolve against.
        await self._persist(
            conversation, question, answer, None, None, None, None, mode
        )
        # Register the offer so "the second one" resolves against it. Normally a result
        # set falls out of running a QueryPlan; this turn has no plan to produce one.
        record_candidate_list(
            conversation,
            file_ids=tuple(plan.scope.file_ids),
            records=[
                (item.file_id, item.source_row_id) for item in correction.suggestions
            ],
            key="name_correction",
        )
        await self._store.save(conversation)
        return TurnResult(
            status="answered",
            answer=answer,
            reasoning_calls=result.reasoning_calls,
            planner_type=result.planner_type,
            conversation_id=conversation.id,
            detail="name_correction_ask",
            trace=result.trace,
            list_results=[
                {
                    "title": f'Records close to "{correction.searched_for}"',
                    "fields": ["student_name"],
                    "rows": [
                        {
                            "source_row_id": item.source_row_id,
                            "file_id": item.file_id,
                            "values": {"student_name": item.display_name},
                        }
                        for item in correction.suggestions
                    ],
                }
            ],
        )


    async def _run_plan(
        self,
        scope: AccessScope,
        question: str,
        raw_plan: QueryPlan,
        conversation: ConversationRecord,
        *,
        mode: str,
        query_state: QueryState | None = None,
        frame_key: str | None = None,
        cancel: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        catalog: FieldCatalog | None = None,
        route: PlanRouteResult | None = None,
        on_progress: ProgressFn | None = None,
    ) -> TurnResult:
        cancel = cancel or CancellationToken()
        catalog = catalog or self._catalog
        try:
            plan = validate_query_plan(raw_plan, scope, catalog)
        except PlanValidationError as exc:
            status = exc.code if exc.code == "access_restricted" else "invalid_plan"
            answer = format_access_denied() if status == "access_restricted" else format_unplanned(str(exc))
            return TurnResult(
                status=status,
                answer=answer,
                reasoning_calls=0,
                detail=str(exc),
                conversation_id=conversation.id,
            )

        refinement = await self._refine_text_search(scope, plan, catalog, question)
        if refinement is not None:
            plan = refinement.plan

        await _emit_progress(
            on_progress,
            "plan_validated",
            {
                "ops": plan.op_names(),
                "file_ids": list(plan.scope.file_ids),
                "reasoning_calls": route.reasoning_calls if route is not None else 0,
            },
        )

        retrieval_started = time.perf_counter()
        with get_tracer().span("retrieval", file_ids=list(plan.scope.file_ids)):
            execution = await QueryExecutor(self._gateway, catalog).execute(
                plan, scope, cancellation=cancel
            )
        self._note_latency("retrieval_ms", (time.perf_counter() - retrieval_started) * 1000)
        persist_started = time.perf_counter()
        await self._record_query_run(plan, execution, status="cancelled" if cancel.cancelled else "completed")
        self._note_latency("persist_ms", (time.perf_counter() - persist_started) * 1000, add=True)
        await _emit_progress(
            on_progress,
            "retrieval_completed",
            {
                "methods_run": execution.methods_run,
                "evidence_count": len(_public_evidence(execution)),
            },
        )
        for fact in execution.facts:
            await _emit_progress(on_progress, "computed_fact", fact.__dict__)
        # The router already counted every paid call it made getting to this plan.
        if route is not None:
            reasoning_calls = route.reasoning_calls
            planner_type = route.source
        else:
            reasoning_calls = 1 if plan.planner_type == "constrained_llm" else 0
            planner_type = plan.planner_type
        ack = None
        if route is not None and route.turn_plan is not None:
            from app.planning.response_policy import acknowledgement

            ack = acknowledgement(route.turn_plan)
        read_applied = (
            await self._read_matches(question, plan, refinement, catalog, mode)
            if refinement is not None
            else False
        )
        if read_applied:
            reasoning_calls += 1
        roster = await self._roster_for(scope, plan, execution, catalog)
        answer = format_deterministic_answer(question, plan, execution, catalog, roster=roster)
        if refinement is not None:
            answer = refinement.compose(answer)
        if self._turn_preface:
            answer = f"{self._turn_preface}\n\n{answer}"
        if route is not None and route.turn_plan is not None and route.turn_plan.verify_actions():
            answer = _verify_preface(conversation, execution) + answer
        if ack:
            answer = f"{ack} {answer}".strip()
        packet = _packet_for_synthesis(question, execution, catalog)
        citations: list[str] = list(dict.fromkeys(item.source_id for item in packet.items))
        if packet.items:
            execution.evidence = packet
        inference = read_applied
        spoken_text = None
        streamed = False
        if self._should_synthesize(question, plan, route) and self._answer_reasoner() is not None:
            synth_started = time.perf_counter()
            with get_tracer().span("llm", planner_type=planner_type):
                synthesis, packet = await self._synthesize(
                    question,
                    execution,
                    conversation,
                    mode,
                    scope,
                    cancel=cancel,
                    on_spoken_phrase=on_spoken_phrase,
                    catalog=catalog,
                )
            self._note_latency("synthesis_ms", (time.perf_counter() - synth_started) * 1000)
            answer = synthesis.answer
            citations = synthesis.citations
            inference = synthesis.inference
            reasoning_calls += 1
            execution.evidence = packet
            streamed = mode == "voice"
        if mode == "voice":
            # Deterministic results get a terse spoken form; synthesized prose is
            # already conversational and only needs citation stripping/bounding.
            concise = None if streamed else format_spoken_answer(plan, execution, catalog)
            spoken_text = humanize_for_speech(concise or answer)
            if on_spoken_phrase is not None and not streamed:
                await _emit_phrases(spoken_text, cancel, on_spoken_phrase)
        await _emit_progress(on_progress, "answer_delta", {"text": answer})
        result = TurnResult(
            status="answered",
            answer=answer,
            plan=plan,
            facts=[fact.__dict__ for fact in execution.facts],
            rows=_public_rows(execution, catalog),
            list_results=_public_list_results(execution, catalog, plan),
            reasoning_calls=reasoning_calls,
            planner_type=planner_type,
            evidence=_public_evidence(execution),
            methods_run=execution.methods_run,
            parallel_step_peak=execution.run_trace.parallel_step_peak if execution.run_trace else 0,
            critical_path_ms=execution.run_trace.critical_path_ms if execution.run_trace else 0.0,
            conversation_id=conversation.id,
            citations=citations,
            inference=inference,
            spoken_text=spoken_text,
            interrupted=cancel.cancelled,
            trace=_route_trace(route) if route is not None else {},
        )
        await self._persist(
            conversation,
            question,
            result.answer,
            plan,
            execution,
            query_state,
            frame_key,
            mode,
            interrupted=result.interrupted,
            citations=result.citations,
            source_actions=self._source_actions(route, scope, catalog, conversation),
        )
        return result

    async def _refine_text_search(
        self,
        scope: AccessScope,
        plan: QueryPlan,
        catalog: FieldCatalog,
        question: str,
    ) -> TextSearchRefinement | None:
        """Check a plan's narrative word matches against the records; None leaves it as asked.

        Additive in the same way as name correction: any failure here -- a gateway error, a
        rewritten plan the validator refuses -- runs the planner's plan unchanged rather than
        failing a turn that would otherwise have been answered.
        """
        if not get_settings().text_search_refinement:
            return None
        try:
            refinement = await refine_text_search(self._gateway, scope, plan, catalog, question)
            if refinement is None:
                return None
            refinement.plan = validate_query_plan(refinement.plan, scope, catalog)
            return refinement
        except Exception:
            logger.warning("text search refinement skipped; running the plan as asked", exc_info=True)
            return None

    async def _read_matches(
        self,
        question: str,
        plan: QueryPlan,
        refinement: TextSearchRefinement,
        catalog: FieldCatalog,
        mode: str,
    ) -> bool:
        """Read the listed records against what the word search could not check.

        Only for a short, listed text-match answer whose question has words beyond the search,
        and never in voice, where a spoken answer cannot carry a table of reasons. Any failure,
        a timeout included, leaves the exact count to stand alone.
        """
        settings = get_settings()
        reasoner = self._answer_reasoner()
        if not settings.text_match_review or reasoner is None or mode == "voice":
            return False
        if not refinement.excerpts:
            return False
        labels = [
            *refinement.field_labels,
            *(
                dataset.user_facing_label
                for file_id in plan.scope.file_ids
                if (dataset := catalog.dataset(file_id)) is not None
            ),
        ]
        unchecked = unchecked_words(question, refinement.terms, labels)
        if not unchecked:
            return False
        try:
            async with asyncio.timeout(settings.text_match_review_timeout_s):
                readings = await read_matches(reasoner, question, unchecked, refinement.records())
        except TimeoutError:
            logger.warning(
                "match reading took longer than %ss; the exact count stands alone",
                settings.text_match_review_timeout_s,
            )
            return False
        except Exception:
            logger.warning("match reading skipped; the exact count stands alone", exc_info=True)
            return False
        if not readings:
            return False
        refinement.unchecked = unchecked
        refinement.readings = readings
        return True

    async def _roster_for(
        self,
        scope: AccessScope,
        plan: QueryPlan,
        execution: ExecutionResult,
        catalog: FieldCatalog,
    ) -> Roster | None:
        """The people behind a count on a short list, so rows and people can be told apart.

        Only for a plain count -- alone, or beside a breakdown of the same records -- that is
        not about a named person, and only when the list is short enough to read. Anything
        else, or any failure reading the rows, leaves the count exactly as it was.
        """
        limit = get_settings().roster_max_rows
        actions = execution.action_results
        if not limit or not actions or len(actions) > 2:
            return None
        counts = [item for item in actions if is_plain_count(item)]
        if len(counts) != 1:
            return None
        count = counts[0]
        file_ids = tuple(count.file_ids or plan.scope.file_ids)
        total = int(next((fact.value for fact in count.facts if fact.name == "count"), 0) or 0)
        if total <= 0 or total > limit:
            return None
        if not any(catalog.resolve_field(file_id, "student_name") for file_id in file_ids):
            return None
        if any(
            is_name_field(leaf.field, file_ids, catalog)
            for predicate in count.predicates
            for leaf in predicate.leaves()
        ):
            return None
        try:
            rows = await fetch_rows(self._gateway, scope, file_ids, list(count.predicates), total)
        except Exception:
            logger.warning("roster skipped; answering with the row count alone", exc_info=True)
            return None
        return roster_from_rows(rows, catalog)

    def _semantic_compiler(self) -> SemanticCompiler | UnavailableSemanticCompiler:
        """Always return a compiler layer. Missing xAI is an explicit unavailable result."""
        if self._semantic_compiler_cache is not None:
            return self._semantic_compiler_cache
        structured = getattr(self._reasoner, "structured_output", None) if self._reasoner else None
        if self._reasoner is None or not callable(structured):
            reason = reasoning_unavailable_reason() or (
                "reasoning provider has no structured_output method"
                if self._reasoner is not None
                else "XAI_API_KEY is not configured on this process; the semantic compiler cannot run"
            )
            self._semantic_compiler_cache = UnavailableSemanticCompiler(reason)
            return self._semantic_compiler_cache
        self._semantic_compiler_cache = SemanticCompiler(
            structured,
            provider_name=getattr(self._reasoner, "provider_name", "xai"),
            model_name=getattr(self._reasoner, "model_name", "") or "",
        )
        return self._semantic_compiler_cache

    def _source_actions(
        self,
        route: PlanRouteResult | None,
        scope: AccessScope,
        catalog: FieldCatalog,
        conversation: ConversationRecord,
    ) -> dict[str, object]:
        if route is None or route.turn_plan is None:
            return {}
        from app.planning.query_plan_builder import compiled_query_actions

        frames = {}
        selected = conversation.selected_file_id
        for frame in conversation.frames.values():
            if (
                get_settings().planner_mode == "ai"
                and selected is not None
                and list(getattr(frame.query, "file_ids", ()) or ()) != [selected]
            ):
                continue
            active = _active_query_from_frame(frame)
            if active is not None:
                frames[frame.frame_key] = active
        active_frame = conversation.active_frame()
        if (
            get_settings().planner_mode == "ai"
            and selected is not None
            and active_frame is not None
            and list(getattr(active_frame.query, "file_ids", ()) or ()) != [selected]
        ):
            active_frame = None
        queries = compiled_query_actions(
            route.turn_plan,
            scope=scope,
            catalog=catalog,
            active_query=_active_query_from_frame(active_frame),
            frames=frames,
        )
        return {f"a{index}": query for index, query in enumerate(queries)}

    async def _direct_reply(
        self,
        route: PlanRouteResult,
        question: str,
        conversation: ConversationRecord,
        *,
        mode: str,
        cancel: CancellationToken,
        on_spoken_phrase: SpokenPhraseFn | None,
        on_progress: ProgressFn | None,
    ) -> TurnResult:
        """Answer in words with no QueryPlan and no database work."""
        from app.planning.response_policy import only_general_conversation

        reasoning_calls = route.reasoning_calls
        general_conversation = only_general_conversation(route.turn_plan)
        explicit_general_response = bool(
            route.turn_plan
            and any(item.response.strip() for item in route.turn_plan.respond_actions())
        )
        if _is_unclear_single_word(question) and (
            route.status == "clarification" or general_conversation
        ):
            answer = "I may have misheard that. Could you say it again?"
            spoken = answer if mode == "voice" else None
            if spoken and on_spoken_phrase is not None:
                await _emit_phrases(spoken, cancel, on_spoken_phrase)
            await _emit_progress(on_progress, "answer_delta", {"text": answer})
        elif general_conversation and self._answer_reasoner() is not None and not explicit_general_response:
            answer = await self._answer_general(
                question,
                conversation,
                mode=mode,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                on_progress=on_progress,
            )
            reasoning_calls += 1
            spoken = humanize_for_speech(answer) if mode == "voice" else None
        else:
            answer = (route.response_text or "").strip() or _fallback_reply(route, mode)
            spoken = humanize_for_speech(answer) if mode == "voice" else None
            if spoken and on_spoken_phrase is not None:
                await _emit_phrases(spoken, cancel, on_spoken_phrase)
            await _emit_progress(on_progress, "answer_delta", {"text": answer})
        result = TurnResult(
            status=_direct_status(route),
            answer=answer,
            reasoning_calls=reasoning_calls,
            planner_type=route.source,
            detail=route.detail,
            conversation_id=conversation.id,
            spoken_text=spoken,
            trace=_route_trace(route),
        )
        await self._persist(
            conversation, question, result.answer, None, None, None, None, mode
        )
        return result

    async def _answer_general(
        self,
        question: str,
        conversation: ConversationRecord,
        *,
        mode: str,
        cancel: CancellationToken,
        on_spoken_phrase: SpokenPhraseFn | None,
        on_progress: ProgressFn | None,
    ) -> str:
        from app.llm.prompts import GENERAL_ANSWER_SYSTEM

        if _is_unclear_single_word(question):
            answer = "I may have misheard that. Could you say it again?"
            await _emit_progress(on_progress, "answer_delta", {"text": answer})
            if mode == "voice" and on_spoken_phrase is not None:
                await _emit_phrases(answer, cancel, on_spoken_phrase)
            return answer

        memory = _memory_text_for_general(conversation)
        user = question if not memory else f"Recent conversation:\n{memory}\n\nUser request:\n{question}"
        parts: list[str] = []
        reasoner = self._answer_reasoner()
        assert reasoner is not None
        async for delta in reasoner.stream_answer(system=GENERAL_ANSWER_SYSTEM, user=user):
            cancel.raise_if_cancelled()
            if delta:
                parts.append(delta)
                await _emit_progress(on_progress, "answer_delta", {"text": delta})
        answer = "".join(parts).strip() or "I can help with that, but I need a bit more detail."
        if not parts:
            await _emit_progress(on_progress, "answer_delta", {"text": answer})
        if mode == "voice" and on_spoken_phrase is not None:
            await _emit_phrases(humanize_for_speech(answer), cancel, on_spoken_phrase)
        return answer

    async def _reasoned_plan_and_answer(
        self,
        scope: AccessScope,
        question: str,
        conversation: ConversationRecord,
        mode: str,
        *,
        cancel: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        catalog: FieldCatalog | None = None,
        on_progress: ProgressFn | None = None,
    ) -> TurnResult:
        assert self._reasoner is not None
        catalog = catalog or self._catalog
        memory = _memory_text(conversation)
        try:
            raw_plan = await ConstrainedPlanner(self._reasoner, catalog).plan(
                question, scope, memory
            )
        except PlanValidationError as exc:
            status = exc.code if exc.code == "access_restricted" else "invalid_plan"
            return TurnResult(
                status=status,
                answer=format_access_denied() if status == "access_restricted" else format_unplanned(str(exc)),
                reasoning_calls=1,
                detail=str(exc),
                conversation_id=conversation.id,
                planner_type="constrained_llm",
            )
        except Exception as exc:
            return TurnResult(
                status="planner_failed",
                answer=format_unplanned("the constrained planner could not produce a valid plan"),
                reasoning_calls=1,
                detail=str(exc),
                conversation_id=conversation.id,
                planner_type="constrained_llm",
            )
        return await self._run_plan(
            scope,
            question,
            raw_plan,
            conversation,
            mode=mode,
            cancel=cancel,
            on_spoken_phrase=on_spoken_phrase,
            catalog=catalog,
            on_progress=on_progress,
        )

    def _note_latency(self, name: str, elapsed_ms: float, *, add: bool = False) -> None:
        bag = self._latency_bag
        if bag is None:
            return
        if add:
            bag[name] = float(bag.get(name) or 0.0) + float(elapsed_ms)
        else:
            bag[name] = float(elapsed_ms)

    async def _planning_catalog(self, scope: AccessScope) -> FieldCatalog:
        started = time.perf_counter()
        try:
            loader = getattr(self._gateway, "load_catalog", None)
            if callable(loader):
                try:
                    loaded = await loader(scope)
                    return loaded.for_scope(scope)
                except Exception:
                    if get_settings().app_env == "production":
                        raise
            return (self._catalog or static_catalog()).for_scope(scope)
        finally:
            self._note_latency("catalog_ms", (time.perf_counter() - started) * 1000)

    def _should_synthesize(
        self,
        question: str,
        plan: QueryPlan,
        route: PlanRouteResult | None = None,
    ) -> bool:
        del question
        if route is not None and route.final_response == "deterministic":
            return False
        if route is not None and route.final_response == "llm_synthesis":
            return True
        if route is not None and route.turn_plan is not None:
            turn = route.turn_plan
            if turn.needs_inference:
                return True
        return any(goal in {"synthesis", "cause_summary"} for goal in plan.goals)

    async def _synthesize(
        self,
        question: str,
        execution: ExecutionResult,
        conversation: ConversationRecord,
        mode: str,
        scope: AccessScope,
        *,
        cancel: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        catalog: FieldCatalog | None = None,
    ) -> tuple[SynthesisAnswer, EvidencePacket]:
        reasoner = self._answer_reasoner()
        assert reasoner is not None
        packet = _packet_for_synthesis(question, execution, catalog)
        prompt = synthesis_user_prompt(question, packet, _memory_text(conversation), mode=mode)
        if mode == "voice":
            spoken_parts: list[str] = []
            buffer = PhraseBuffer()
            buffer.start("voice-synth")
            async for delta in reasoner.stream_answer(system=SYNTHESIS_SYSTEM, user=prompt):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                for phrase in buffer.push(delta):
                    spoken = humanize_for_speech(phrase.text)
                    if spoken:
                        spoken_parts.append(spoken)
                        if on_spoken_phrase is not None:
                            await on_spoken_phrase(spoken)
            for phrase in buffer.flush():
                spoken = humanize_for_speech(phrase.text)
                if spoken:
                    spoken_parts.append(spoken)
                    if on_spoken_phrase is not None:
                        await on_spoken_phrase(spoken)
            raw = SynthesisAnswer(
                answer=" ".join(spoken_parts).strip(),
                citations=[item.source_id for item in packet.items],
                inference=False,
            )
            return validate_synthesis(raw, packet, scope), packet
        raw = await reasoner.answer_structured(system=SYNTHESIS_SYSTEM, user=prompt)
        return validate_synthesis(raw, packet, scope), packet

    async def _answer_reference(
        self,
        scope: AccessScope,
        question: str,
        conversation: ConversationRecord,
        route: PlanRouteResult,
        mode: str,
        *,
        cancel: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        on_progress: ProgressFn | None = None,
    ) -> TurnResult:
        action = route.turn_plan.reference_actions()[0] if route.turn_plan else None
        resolved = resolve_listed_ordinal(
            conversation,
            selector=action.selector if action else None,
            index=action.index if action else None,
            target=action.target if action else None,
        )
        if resolved.kind == "action_frame" and resolved.query_state is not None:
            from app.planning.query_plan_builder import build_query_plan, query_action_from_state
            from app.planning.turn_schema import TurnPlan

            catalog = await self._planning_catalog(scope)
            replayed = query_action_from_state(resolved.query_state)
            replay_turn = TurnPlan(
                normalized_request=question,
                actions=[replayed],
                final_response=(
                    route.turn_plan.final_response
                    if route.turn_plan is not None
                    else "deterministic"
                ),
                needs_evidence=replayed.goal in {"quote", "search", "provenance"}
                or (route.turn_plan.needs_evidence if route.turn_plan else False),
                needs_explanation=route.turn_plan.needs_explanation if route.turn_plan else False,
                confidence=1.0,
            )
            plan = build_query_plan(replay_turn, scope=scope, catalog=catalog)
            return await self._run_plan(
                scope,
                question,
                plan,
                conversation,
                mode=mode,
                query_state=resolved.query_state,
                frame_key=resolved.frame_key,
                cancel=cancel,
                on_spoken_phrase=on_spoken_phrase,
                catalog=catalog,
                route=route,
                on_progress=on_progress,
            )
        if resolved.kind != "ordinal" or not resolved.source_ids:
            result = TurnResult(
                status="unresolved_followup",
                answer=resolved.detail or "I could not resolve that follow-up against the current conversation.",
                reasoning_calls=route.reasoning_calls,
                planner_type=route.source,
                conversation_id=conversation.id,
                trace=_route_trace(route),
            )
            await self._persist(conversation, question, result.answer, None, None, None, None, mode)
            await _emit_progress(on_progress, "answer_delta", {"text": result.answer})
            return result
        return await self._answer_ordinal(
            scope,
            question,
            conversation,
            resolved.source_ids,
            mode,
            cancel=cancel,
            on_spoken_phrase=on_spoken_phrase,
            reasoning_calls=route.reasoning_calls,
            planner_type=route.source,
            route=route,
            on_progress=on_progress,
        )

    async def _answer_ordinal(
        self,
        scope: AccessScope,
        question: str,
        conversation: ConversationRecord,
        source_ids: list[str],
        mode: str,
        *,
        cancel: CancellationToken | None = None,
        on_spoken_phrase: SpokenPhraseFn | None = None,
        reasoning_calls: int = 0,
        planner_type: str | None = "semantic_compiler",
        route: PlanRouteResult | None = None,
        on_progress: ProgressFn | None = None,
    ) -> TurnResult:
        row_ids = []
        for source_id in source_ids:
            parsed = parse_source_id(source_id)
            if parsed:
                row_ids.append(parsed[1])
        records = await self._gateway.get_records_by_source_ids(scope, row_ids)
        rows = _public_row_dicts(records)
        names = [row.get("canonical_name") or "unnamed record" for row in rows]
        answer = (
            f"That result is {names[0]}."
            if names
            else "I could not load that listed record."
        )
        if len(records) == 1:
            # One record pointed at -- "yes" to "did you mean Alexander KNAGGS?", "the second
            # one" -- is a request to see that person, not to be told their name back.
            details = _recorded_details(records[0], await self._planning_catalog(scope))
            if details:
                answer = f"{answer}\n\n{details}"
        execution = ExecutionResult(rows=records)
        await _emit_progress(
            on_progress,
            "retrieval_completed",
            {"methods_run": [], "evidence_count": 0},
        )
        spoken_text = humanize_for_speech(answer) if mode == "voice" else None
        if spoken_text and on_spoken_phrase is not None:
            await _emit_phrases(spoken_text, cancel or CancellationToken(), on_spoken_phrase)
        await _emit_progress(on_progress, "answer_delta", {"text": answer})
        result = TurnResult(
            status="answered",
            answer=answer,
            rows=rows,
            reasoning_calls=reasoning_calls,
            planner_type=planner_type,
            conversation_id=conversation.id,
            spoken_text=spoken_text,
            trace=_route_trace(route) if route is not None else {},
        )
        await self._persist(
            conversation,
            question,
            result.answer,
            None,
            execution,
            None,
            None,
            mode,
            citations=result.citations,
        )
        return result

    async def _persist(
        self,
        conversation: ConversationRecord,
        question: str,
        answer: str,
        plan: QueryPlan | None,
        execution: ExecutionResult | None,
        query_state: QueryState | None,
        frame_key: str | None,
        mode: str,
        *,
        interrupted: bool = False,
        citations: list[str] | None = None,
        source_actions: dict[str, object] | None = None,
    ) -> None:
        if self._turn_user_text is not None:
            question = self._turn_user_text
        update_memory_from_turn(
            conversation,
            question=question,
            answer=answer,
            plan=plan,
            execution=execution,
            query_state=query_state,
            frame_key=frame_key,
            mode=mode,
            interrupted=interrupted,
            citations=citations,
            source_actions=source_actions,
        )
        started = time.perf_counter()
        await self._store.save(conversation)
        self._note_latency("persist_ms", (time.perf_counter() - started) * 1000, add=True)

    async def _record_query_run(
        self,
        plan: QueryPlan | None,
        execution: ExecutionResult,
        *,
        status: str,
    ) -> None:
        try:
            await self._gateway.save_query_run(
                turn_id=uuid4().hex,
                plan=plan,
                execution=execution,
                status=status,
            )
        except Exception:
            logger.warning("failed to persist query run trace", exc_info=True)


async def _emit_phrases(text: str, cancel: CancellationToken, on_spoken_phrase: SpokenPhraseFn) -> None:
    for phrase in split_phrases(text):
        cancel.raise_if_cancelled()
        await on_spoken_phrase(phrase)


async def _emit_progress(
    callback: ProgressFn | None,
    event: str,
    data: dict[str, Any],
) -> None:
    if callback is not None:
        await callback(event, data)


def _public_rows(execution: ExecutionResult, catalog: FieldCatalog) -> list[dict[str, Any]]:
    fields: list[str] = []
    for action in execution.action_results:
        for field_name in action.projected_fields:
            if field_name not in fields:
                fields.append(field_name)
    if not fields:
        fields = ["student_name", "community"]
    return [public_projected_row(row, fields, catalog) for row in execution.rows]


def _recorded_details(row: dict[str, Any], catalog: FieldCatalog) -> str:
    """Every field this record has a value for, labelled, in the list's own field order."""
    from app.execution.row_projection import display_value, semantic_value

    file_id = int(row.get("file_id") or 0)
    parts: list[str] = []
    for spec in catalog.planner_fields_for(file_id):
        if spec.semantic_field == "student_name" or spec.canonical_json_path == "names":
            continue
        value = semantic_value(row, spec.semantic_field, catalog)
        if value in (None, "", [], ()):
            continue
        parts.append(f"{spec.human_label}: {display_value(value)}")
    return "; ".join(parts)


def _public_row_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for row in rows:
        public.append(
            {
                "file_id": row.get("file_id"),
                "source_row_id": row.get("source_row_id"),
                "canonical_name": _display_name(row),
                "canonical_community": _display_community(row),
            }
        )
    return public


def _display_name(row: dict[str, Any]) -> Any:
    """The stored display name, not the folded lookup column."""
    canonical = json_object(row.get("row_data_normalized")).get("canonical")
    if isinstance(canonical, dict):
        value = canonical.get("display_name") or canonical.get("name")
        if value:
            return value
    return row.get("canonical_name")


def _display_community(row: dict[str, Any]) -> Any:
    canonical = json_object(row.get("row_data_normalized")).get("canonical")
    if isinstance(canonical, dict) and canonical.get("community"):
        return canonical["community"]
    return row.get("canonical_community")


def _public_list_results(
    execution: ExecutionResult,
    catalog: FieldCatalog,
    plan: QueryPlan,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for action in execution.action_results:
        if action.goal != "list" or action.total_count is None:
            continue
        fields = list(action.projected_fields) or ["student_name"]
        file_ids = action.file_ids or plan.scope.file_ids
        dataset_labels = [
            catalog.dataset(file_id).user_facing_label
            for file_id in file_ids
            if catalog.dataset(file_id) is not None
        ]
        results.append(
            {
                "action_id": action.action_id,
                "dataset_label": ", ".join(dataset_labels) or "Authorized records",
                "fields": field_descriptors(fields, file_ids, catalog),
                "rows": [public_projected_row(row, fields, catalog) for row in action.rows],
                "total_count": action.total_count,
                "offset": action.offset,
                "page_size": action.page_size,
                "has_more": action.has_more,
                "window": (
                    {
                        "anchor": action.window_anchor,
                        "size": action.window_size,
                        "count": action.window_count,
                        "cursor": action.window_cursor,
                    }
                    if action.window_anchor and action.window_size
                    else None
                ),
            }
        )
    return results


def _public_evidence(execution: ExecutionResult) -> list[dict[str, Any]]:
    if not execution.evidence:
        return []
    return [
        {
            "source_id": item.source_id,
            "file_id": item.file_id,
            "fields": item.fields,
        }
        for item in execution.evidence.items
    ]


def _route_trace(route: PlanRouteResult) -> list[dict[str, Any]]:
    """Ordered operator-facing layers. Never spoken. No source rows or table names."""
    turn = route.normalized_turn
    timings = route.timings.as_dict() if route.timings is not None else {}
    if route.source == "ai_planner":
        return [
            {
                "layer": "normalize",
                "duration_ms": timings.get("normalize_ms", 0.0),
                "normalized_text": getattr(turn, "normalized_text", ""),
                "changed": bool(getattr(turn, "changed", False)),
            },
            {
                "layer": "ai_planner",
                "status": route.status,
                "attempts": route.planner_attempts,
                "retry_count": route.planner_retry_count,
                "duration_ms": timings.get("ai_planner_ms", 0.0),
            },
            {
                "layer": "turn_validator",
                "status": "valid" if route.turn_plan is not None else "skipped",
                "duration_ms": timings.get("turn_validation_ms", 0.0),
            },
            {
                "layer": "query_plan_builder",
                "status": "built" if route.plan is not None else "SKIPPED",
                "duration_ms": timings.get("query_plan_build_ms", 0.0),
            },
            {
                "layer": "predicate_contract",
                "status": "valid" if route.plan is not None else "skipped",
                "duration_ms": timings.get("predicate_contract_ms", 0.0),
            },
            {
                "layer": "ai_self_review",
                "status": route.review_status or "skipped",
                "attempts": route.review_attempts,
                "duration_ms": timings.get("ai_self_review_ms", 0.0),
                "provider": route.planner_provider,
                "model": route.planner_model,
                "usage": route.planner_usage,
            },
            {
                "layer": "value_expansion",
                "status": "applied" if route.plan is not None else "skipped",
                "duration_ms": timings.get("value_expansion_ms", 0.0),
            },
            {
                "layer": "plan_validator",
                "status": "valid" if route.plan is not None else "skipped",
                "duration_ms": timings.get("plan_validation_ms", 0.0),
            },
            {
                "layer": "route",
                "status": route.status,
                "source": route.source,
                "detail": route.detail,
                "reasoning_calls": route.reasoning_calls,
                "final_response": route.final_response,
                "planner_failure_stage": route.planner_failure_stage,
                "timings_ms": timings,
            },
        ]
    layers: list[dict[str, Any]] = [
        {
            "layer": "normalize",
            "duration_ms": timings.get("normalize_ms", 0.0),
            "normalized_text": getattr(turn, "normalized_text", ""),
            "changed": bool(getattr(turn, "changed", False)),
        }
    ]
    layers.append(
        {
            "layer": "deterministic_fast_path",
            "status": "matched" if route.source in {"trusted_fast_path", "deterministic_fast_path"} else "abstained",
            "duration_ms": round(
                timings.get("trusted_fast_path_ms", 0.0)
                + timings.get("deterministic_parse_ms", 0.0),
                3,
            ),
        }
    )
    if route.compiler is not None:
        layers.append(route.compiler.as_layer())
    elif route.interpreter is not None:
        layers.append(route.interpreter.as_layer())
    else:
        layers.append(
            {
                "layer": "semantic_compiler",
                "status": "skipped",
                "reasoning_calls": 0,
            }
        )
    layers.append(
        {
            "layer": "turn_validator",
            "status": "valid" if route.turn_plan is not None else "skipped",
            "duration_ms": timings.get("turn_validation_ms", 0.0),
        }
    )
    if route.plan is None:
        layers.append(
            {
                "layer": "query_plan",
                "status": "SKIPPED",
                "duration_ms": timings.get("query_plan_build_ms", 0.0),
            }
        )
    else:
        layers.append(
            {
                "layer": "query_plan_builder",
                "status": "built",
                "duration_ms": timings.get("query_plan_build_ms", 0.0),
                "ops": route.plan.op_names(),
            }
        )
        layers.append(
            {
                "layer": "plan_validator",
                "status": "valid",
                "duration_ms": timings.get("plan_validation_ms", 0.0),
            }
        )
    layers.append(
        {
            "layer": "route",
            "status": route.status,
            "source": route.source,
            "detail": route.detail,
            "reasoning_calls": route.reasoning_calls,
            "final_response": route.final_response,
            "timings_ms": timings,
        }
    )
    return layers


def _direct_status(route: PlanRouteResult) -> str:
    """Report what actually happened.

    A conversational turn really was answered. Empty audio is unintelligible.
    A valid transcript we could not execute keeps its diagnostic status.
    """
    if route.status in {"conversational", "greeting"}:
        return "answered"
    if route.status == "clarification":
        return "clarification"
    if route.status == "empty":
        return "unintelligible"
    if route.compiler is not None or route.interpreter is not None:
        return route.status
    return route.status


def _fallback_reply(route: PlanRouteResult, mode: str = "text") -> str:
    """Last-resort wording when no layer produced a response.

    "Sorry, I didn't catch that." is reserved for empty / unintelligible STT.
    A valid transcript that we could not plan gets a real clarification.
    """
    if route.clarification_question:
        return route.clarification_question
    if _is_unintelligible(route):
        return "Sorry, I didn't catch that."
    if mode == "voice":
        return (
            "Could you say that another way? Ask about a name, a community, "
            "a year, or the research records."
        )
    return (
        "I understood you, but I need a more specific question about the research records — "
        "for example a count, a name, a community, or a year."
    )


def _is_unintelligible(route: PlanRouteResult) -> bool:
    return route.status == "empty"


def _resolve_selected_dataset(
    scope: AccessScope,
    catalog: FieldCatalog,
    requested_file_id: int | None,
    question: str = "",
) -> tuple[int | None, str | None]:
    """Resolve one assistant-enabled dataset without ever broadening access.

    An explicit selection always wins. Without one, the question itself chooses the
    list, so a researcher asking about causes of death does not have to know which
    file carries that field.
    """

    available = {item.file_id for item in catalog.datasets}
    if not available:
        return None, "No assistant-enabled datasets are available to this account."
    selected = requested_file_id or select_dataset_for_question(
        question, catalog, scope, catalog.default_people_file_id
    )
    if selected not in available or not scope.allows_file(selected):
        return None, "The selected list is unavailable or is not authorized for this account."
    return selected, None


def _other_list_for_question(
    question: str,
    catalog: FieldCatalog,
    scope: AccessScope,
    selected_file_id: int,
) -> int | None:
    """The single other authorized list this question is really about, if any."""
    named = [
        file_id
        for file_id in dict.fromkeys(catalog.resolve_dataset_ids(question))
        if file_id != selected_file_id
        and (dataset := catalog.dataset(file_id)) is not None
        and scope.allows_file(file_id, private=dataset.private)
    ]
    if len(named) == 1 and selected_file_id not in catalog.resolve_dataset_ids(question):
        return named[0]
    return None


def _specific_label(
    catalog: FieldCatalog,
    file_id: int,
    semantic: str,
    question: str,
) -> bool:
    lowered = " ".join(question.lower().split())
    spec = catalog.resolve_field(file_id, semantic)
    if spec is None:
        return False
    for label in {spec.human_label, semantic.replace("_", " "), *spec.aliases}:
        normalized = " ".join(label.lower().split())
        if len(normalized) < 8 and " " not in normalized:
            continue
        if re.search(_label_pattern(normalized), lowered):
            return True
    return False


def select_dataset_for_question(
    question: str,
    catalog: FieldCatalog,
    scope: AccessScope,
    default_file_id: int,
) -> int:
    """Pick the one authorized list a question is about.

    An explicit list name wins. Otherwise the dataset whose exclusive fields the
    question names wins. With no signal either way the default people scope is used.
    Selection never widens access: only datasets already in the AccessScope compete.
    """

    authorized = [
        item.file_id
        for item in catalog.datasets
        if scope.allows_file(item.file_id, private=item.private)
    ]
    if not authorized:
        return default_file_id
    named = [item for item in catalog.resolve_dataset_ids(question) if item in authorized]
    if named:
        return named[0]
    mentions = field_mentions(question, catalog)
    exclusive: dict[int, int] = {}
    for file_id in authorized:
        owned = mentions.get(file_id, set())
        elsewhere = {
            semantic
            for other, semantics in mentions.items()
            if other != file_id
            for semantic in semantics
        }
        # A bare common noun such as "school" is too weak to choose a list on its
        # own; an exclusive signal has to be a multi-word or long field label.
        unique = {
            semantic
            for semantic in owned
            if all(
                catalog.resolve_field(other, semantic) is None
                for other in authorized
                if other != file_id
            )
            and _specific_label(catalog, file_id, semantic, question)
        }
        del elsewhere
        if unique:
            exclusive[file_id] = len(unique)
    if not exclusive:
        # No specific label pointed anywhere. Fall back to distinguishing fields: a
        # field only one authorized list has. Fields every list shares (a student's
        # name) say nothing about which list the question is about.
        distinguishing = {
            file_id: {
                semantic
                for semantic in mentions.get(file_id, set())
                if all(
                    catalog.resolve_field(other, semantic) is None
                    for other in authorized
                    if other != file_id
                )
            }
            for file_id in authorized
        }
        signalled = [file_id for file_id, owned in distinguishing.items() if owned]
        if len(signalled) == 1 and signalled[0] != default_file_id:
            return signalled[0]
        return default_file_id if default_file_id in authorized else authorized[0]
    best = max(exclusive.values())
    winners = sorted(file_id for file_id, score in exclusive.items() if score == best)
    if len(winners) != 1:
        return default_file_id if default_file_id in authorized else authorized[0]
    return winners[0]


def _wrong_list_for_question(
    question: str,
    catalog: FieldCatalog,
    scope: AccessScope,
    selected_file_id: int,
) -> int | None:
    """The list to switch to, when the selected one cannot serve the question at all.

    Redirect only when the question names a field that exists solely on another
    authorized list *and* names no field of the selected list. A question that
    mentions both is answerable here and must not be bounced.
    """

    mentions = field_mentions(question, catalog)
    if mentions.get(selected_file_id):
        return None
    authorized = [
        item.file_id
        for item in catalog.datasets
        if item.file_id != selected_file_id
        and scope.allows_file(item.file_id, private=item.private)
    ]
    candidates = [
        file_id
        for file_id in authorized
        if any(
            catalog.resolve_field(selected_file_id, semantic) is None
            for semantic in mentions.get(file_id, set())
        )
    ]
    return candidates[0] if len(candidates) == 1 else None

# "What does the database tell us about <person>?" is answered from every list the
# researcher is authorized to read, one list at a time, and the summaries merged.
# A named person is required: "what do the records say" on its own is a synthesis
# request about the current result, not a person summary.
_SUMMARY_LEAD = re.compile(
    r"\bwhat (?:does|do) the (?:database|records?|data)\s+(?:tell|say)\b"
    r"|\btell me (?:everything|all|what you know)\b",
    re.IGNORECASE,
)
_ABOUT_PERSON = re.compile(
    r"\babout\s+(?:the\s+life\s+of\s+)?(?:his|her|their)?\s*"
    r"[A-Z][A-Za-z'\u2019-]+\s+[A-Z]"
)
_RELATED_PAIR = re.compile(
    r"\b[Ww]ere\s+[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*\s+and\s+"
    r"[A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)*\s+related\b",
)


def is_person_summary(question: str) -> bool:
    if _RELATED_PAIR.search(question):
        return True
    return bool(_SUMMARY_LEAD.search(question) and _ABOUT_PERSON.search(question))


def _question_for_dataset(question: str, catalog: FieldCatalog, file_id: int) -> str:
    """The question with the other lists' names removed.

    Each fan-out branch sees only its own list. Leaving "and the Confirmed
    Shingwauk list" in the text makes the compiler emit a second query action for a
    dataset that is not in the branch scope, and the whole branch then fails to plan.
    """
    from app.planning.catalog import dataset_match_labels

    text = question
    for dataset in catalog.datasets:
        if dataset.file_id == file_id:
            continue
        labels = sorted(
            (label for label in dataset_match_labels(dataset) if len(label) >= 5),
            key=len,
            reverse=True,
        )
        for label in labels:
            name = r"(?:the\s+)?" + re.escape(label) + r"(?:\s+list)?"
            # Remove the whole conjunct, not just the name, so no dangling "and".
            text = re.sub(rf"\s*(?:,\s*)?(?:and|or)\s+{name}", " ", text, flags=re.I)
            text = re.sub(rf"{name}\s+(?:and|or)\s+", " ", text, flags=re.I)
            text = re.sub(rf"\s*{name}", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*(?:,|and|or)\s*(?=[?.!]|$)", "", text).strip()
    return text or question


def _fan_out_datasets(
    question: str,
    catalog: FieldCatalog,
    scope: AccessScope,
) -> list[int]:
    """Authorized datasets this question explicitly asks about.

    Returns more than one id only when the question names several lists, or asks
    about "both"/"all" of them. The caller then runs one single-dataset turn per id.
    """

    authorized = [
        item.file_id
        for item in catalog.datasets
        if scope.allows_file(item.file_id, private=item.private)
    ]
    if is_person_summary(question):
        return authorized
    if re.search(
        r"\b(?:both|all|each)\s+(?:authorized\s+)?(?:datasets?|lists?|sources?)\b"
        r"|\bacross\s+(?:the\s+)?(?:datasets?|lists?)\b",
        question,
        re.IGNORECASE,
    ):
        return authorized
    mentioned = [
        file_id
        for file_id in dict.fromkeys(catalog.resolve_dataset_ids(question))
        if file_id in authorized
    ]
    return mentioned


def _field_dataset_suggestion(
    question: str,
    catalog: FieldCatalog,
    selected_file_id: int,
) -> int | None:
    """Find an explicitly named field that exists only on another list."""

    lowered = " ".join(question.lower().split())
    lowered = re.sub(r"(?<=\w)[’'](?=\s)", "", lowered)
    lowered = re.sub(r"(?<=[a-z])[-‐‑–—](?=[a-z])", " ", lowered)
    selected_semantics = {
        item.semantic_field for item in catalog.fields_for(selected_file_id)
    }
    selected_labels = {
        " ".join(label.lower().split())
        for item in catalog.fields_for(selected_file_id)
        for label in {item.human_label, item.semantic_field.replace("_", " "), *item.aliases}
        if label.strip()
    }
    selected_spans = [
        match.span()
        for label in selected_labels
        if len(label) >= 3
        for match in re.finditer(rf"\b{re.escape(label)}\b", lowered)
    ]
    matches: set[int] = set()
    for dataset in catalog.datasets:
        if dataset.file_id == selected_file_id:
            continue
        for catalog_field in catalog.fields_for(dataset.file_id):
            if catalog_field.semantic_field in selected_semantics:
                continue
            labels = {
                catalog_field.human_label,
                catalog_field.semantic_field.replace("_", " "),
                *catalog_field.aliases,
            }
            normalized_labels = {
                " ".join(label.lower().split()) for label in labels if label.strip()
            } - selected_labels
            for label in normalized_labels:
                if len(label) < 5:
                    continue
                for match in re.finditer(rf"\b{re.escape(label)}\b", lowered):
                    if any(
                        selected_start <= match.start()
                        and match.end() <= selected_end
                        for selected_start, selected_end in selected_spans
                    ):
                        continue
                    matches.add(dataset.file_id)
    return next(iter(matches)) if len(matches) == 1 else None


def _planner_latency_from_trace(trace: object) -> dict[str, Any]:
    if not isinstance(trace, list):
        return {}
    timings: dict[str, Any] = {}
    final_response = None
    for layer in trace:
        if not isinstance(layer, dict):
            continue
        if layer.get("layer") == "route":
            timings = dict(layer.get("timings_ms") or {})
            final_response = layer.get("final_response")
    return {
        "plan_ms": timings.get("ai_planner_ms"),
        "review_ms": timings.get("ai_self_review_ms"),
        "final_response": final_response,
    }


def _copy_planner_meta(result: TurnResult, route: PlanRouteResult) -> None:
    result.planner_attempts = route.planner_attempts
    result.planner_retry_count = route.planner_retry_count
    result.review_attempts = route.review_attempts
    result.review_status = route.review_status
    result.planner_total_model_calls = route.planner_total_model_calls
    result.planner_failure_stage = route.planner_failure_stage
    result.planner_provider = route.planner_provider or None
    result.planner_model = route.planner_model or None
    result.planner_usage = dict(route.planner_usage or {})
    if route.suggested_file_id is not None:
        result.suggested_file_id = route.suggested_file_id
    result.original_planned = route.original_planned


def _conversation_context_for_dataset(
    conversation: ConversationRecord, selected_file_id: int
) -> ConversationContext:
    from app.planning.ai_planner import filter_conversation_for_dataset

    return filter_conversation_for_dataset(_conversation_context(conversation), selected_file_id)


def _conversation_context(conversation: ConversationRecord) -> ConversationContext:
    """Compact planning view of the conversation. Never exposes source rows."""
    frame = conversation.active_frame()
    return ConversationContext(
        active_plan=None,
        active_query_text=(frame.topic if frame else ""),
        memory_text=_memory_text(conversation),
        active_query=_active_query_from_frame(frame),
        topic_frames=tuple(_frame_payload(item) for item in conversation.frames.values()),
        last_action_keys=tuple(conversation.last_action_keys),
        user_display_name=conversation.user_display_name,
        last_assistant_text=next(
            (
                str(item.get("text") or "")
                for item in reversed(conversation.recent_turns)
                if item.get("role") == "assistant"
            ),
            None,
        ),
    )


def _verify_preface(conversation: ConversationRecord, execution: ExecutionResult) -> str:
    previous = None
    frame = conversation.active_frame()
    if frame and frame.result_set_id:
        previous = conversation.result_sets.get(frame.result_set_id)
    shown = 0
    if previous is not None:
        shown = previous.returned_count or len(previous.ordered_source_ids)
    action = execution.action_results[0] if execution.action_results else None
    total = action.total_count if action is not None else None
    if total is None:
        fact = next((item for item in execution.facts if item.name == "count"), None)
        if fact is not None:
            total = int(fact.value)
    if total is None:
        return "I checked the previous result against the database. "
    if shown and shown < total:
        return (
            f"You're right. There are {total} matching records; I only displayed {shown}. "
            "That was a page, not the full list. I'll show them in pages.\n\n"
        )
    return f"I checked again. There are {total} matching records.\n\n"


def _filter_specs(predicates) -> list[FilterSpec]:
    """Flatten stored predicates into follow-up filter specs.

    A boolean group carries no single field name, so it is expanded to its leaves for
    the conversation frame; the executed plan keeps the full tree.
    """
    specs: list[FilterSpec] = []
    for item in predicates:
        leaves = item.leaves() if hasattr(item, "leaves") else [item]
        for leaf in leaves:
            if not leaf.field or leaf.operator is None:
                continue
            specs.append(
                FilterSpec(field=leaf.field, operator=str(leaf.operator), value=leaf.value)
            )
    return specs


def _active_query_from_frame(frame) -> ActiveQuery | None:
    if frame is None:
        return None
    state = frame.query
    goal = "count" if state.goal in {"exact_count", "count"} else state.goal
    # QueryState does not persist the grouping shape, but it keeps the original
    # QueryAction. Without carrying group_by forward, any follow-up to a
    # grouped/distinct answer loses its grouping and fails validation with
    # "distinct requires a group_by field".
    origin = getattr(state, "source_action", None)

    def _from_origin(name: str, default=None):
        return getattr(origin, name, default) if origin is not None else default

    return ActiveQuery(
        group_by=list(_from_origin("group_by", []) or []),
        group_value_part=_from_origin("group_value_part"),
        secondary_group_value_part=_from_origin("secondary_group_value_part"),
        having_min_count=_from_origin("having_min_count"),
        top_n=_from_origin("top_n"),
        per_group_top_n=_from_origin("per_group_top_n"),
        include_missing=bool(_from_origin("include_missing", False)),
        companion_field=_from_origin("companion_field"),
        stats_value_part=_from_origin("stats_value_part"),
        interval_start=_from_origin("interval_start"),
        interval_end=_from_origin("interval_end"),
        interval_min_days=_from_origin("interval_min_days"),
        interval_max_days=_from_origin("interval_max_days"),
        datasets=list(state.file_ids),
        goal=goal,
        filters=list(_from_origin("filters", _filter_specs(state.filters)) or []),
        filter_groups=[
            group.model_copy(deep=True)
            for group in (_from_origin("filter_groups", []) or [])
        ],
        denominator_filters=list(_from_origin("denominator_filters", []) or []),
        filter_logic=_from_origin("filter_logic", "and") or "and",
        search_text=_from_origin("search_text", state.retrieval_query),
        sort_by=getattr(state, "sort_by", None),
        sort_direction=(
            direction
            if (direction := (getattr(state, "sort_direction", None) or "").lower())
            in {"asc", "desc"}
            else None
        ),
        requested_fields=list(
            _from_origin("requested_fields", getattr(state, "requested_fields", None)) or []
        ),
        window=_from_origin("window", getattr(state, "window", None)),
        sample=bool(_from_origin("sample", getattr(state, "sample", False))),
        limit=_from_origin("limit", state.limit),
        offset=_from_origin("offset", getattr(state, "offset", None)),
        exhaustive=bool(
            _from_origin("exhaustive", getattr(state, "exhaustive", False))
        ),
        presentation=_from_origin("presentation", getattr(state, "presentation", None)),
        aggregate=(
            _from_origin("aggregate").model_copy(deep=True)
            if _from_origin("aggregate") is not None
            else None
        ),
        compare=[
            branch.model_copy(deep=True)
            for branch in (_from_origin("compare", []) or [])
        ],
        total_count=getattr(state, "total_count", None),
        returned_count=getattr(state, "returned_count", None),
        has_more=bool(getattr(state, "has_more", False)),
        result_set_id=frame.result_set_id,
        frame_key=frame.frame_key,
        topic=frame.topic,
        action_id=getattr(frame, "action_id", None),
    )


def _frame_payload(frame) -> dict[str, Any]:
    state = frame.query
    goal = "count" if state.goal in {"exact_count", "count"} else state.goal
    return {
        "frame_key": frame.frame_key,
        "key": frame.frame_key,
        "topic": frame.topic,
        "datasets": list(state.file_ids),
        "file_ids": list(state.file_ids),
        "goal": goal,
        "filters": [
            item.model_dump(mode="json")
            for item in (
                list(getattr(getattr(state, "source_action", None), "filters", []) or [])
                or _filter_specs(state.filters)
            )
        ],
        "filter_groups": [
            group.model_dump(mode="json")
            for group in (getattr(getattr(state, "source_action", None), "filter_groups", []) or [])
        ],
        "filter_logic": getattr(getattr(state, "source_action", None), "filter_logic", "and"),
        "search_text": state.retrieval_query,
        "limit": state.limit,
        "offset": state.offset,
        "sort_by": state.sort_by,
        "sort_direction": state.sort_direction,
        "requested_fields": list(getattr(state, "requested_fields", None) or []),
        "window": state.window.model_dump(mode="json") if state.window else None,
        "result_set_id": frame.result_set_id,
        "action_id": getattr(frame, "action_id", None),
    }


def _memory_text(conversation: ConversationRecord) -> str:
    if conversation.rolling_summary:
        recent = " | ".join(f"{item['role']}: {item['text']}" for item in conversation.recent_turns[-4:])
        return f"{conversation.rolling_summary}\n{recent}".strip()
    return " | ".join(f"{item['role']}: {item['text']}" for item in conversation.recent_turns[-6:])


def _memory_text_for_general(conversation: ConversationRecord) -> str:
    parts: list[str] = []
    for item in conversation.recent_turns[-6:]:
        text = str(item.get("text") or "")
        if _looks_like_list_page(text):
            text = "(listed a page of research records)"
        text = _SOURCE_ID.sub("", text)
        parts.append(f"{item.get('role')}: {text[:160]}")
    return " | ".join(parts)[:400]


def _is_unclear_single_word(question: str) -> bool:
    return len(re.findall(r"[A-Za-z0-9']+", question)) == 1


_SOURCE_ID = re.compile(r"file:\d+:v\d+:row:\d+")


def _looks_like_list_page(text: str) -> bool:
    return "Showing " in text and " of " in text


def _describe_predicate(predicate) -> str:
    """Readable filter summary that also covers boolean groups."""
    if predicate.is_group():
        joined = f" {predicate.op.upper()} ".join(
            _describe_predicate(item) for item in predicate.items
        )
        return f"({joined})"
    operator = predicate.operator.value if predicate.operator is not None else "?"
    return f"{predicate.field}:{operator}={predicate.value}"


def _packet_for_synthesis(
    question: str, execution: ExecutionResult, catalog: FieldCatalog | None = None
) -> EvidencePacket:
    from app.execution.executor import hit_from_row
    from app.retrieval.types import EvidenceBranch

    if execution.action_results:
        branches: list[EvidenceBranch] = []
        facts: list[str] = []
        items = []
        truncated = False
        tokens = 0
        for action in execution.action_results:
            action_facts = [f"{action.action_id}.{item.name}={item.value}" for item in action.facts]
            if action.predicates:
                action_facts.append(
                    f"{action.action_id}.filters="
                    + ",".join(_describe_predicate(pred) for pred in action.predicates)
                )
            if action.evidence and action.evidence.items:
                action_items = action.evidence.items
                tokens += action.evidence.token_estimate
                truncated = truncated or action.evidence.truncated
            else:
                hits = list(action.hits) or [
                    hit_from_row(row) for row in action.rows if row.get("source_row_id")
                ]
                built = build_evidence_packet(question, hits, facts=action_facts, catalog=catalog)
                action_items = built.items
                tokens += built.token_estimate
                truncated = truncated or built.truncated
            branches.append(
                EvidenceBranch(
                    action_id=action.action_id,
                    goal=action.goal,
                    facts=tuple(action_facts),
                    items=action_items,
                )
            )
            facts.extend(action_facts)
            items.extend(action_items)
        return EvidencePacket(
            question=question,
            facts=tuple(facts),
            items=tuple(items),
            token_estimate=tokens,
            truncated=truncated,
            branches=tuple(branches),
        )

    facts = tuple(f"{item.name}={item.value}" for item in execution.facts)
    if execution.evidence and execution.evidence.items:
        return EvidencePacket(
            question=question,
            facts=facts or execution.evidence.facts,
            items=execution.evidence.items,
            token_estimate=execution.evidence.token_estimate,
            truncated=execution.evidence.truncated,
        )
    hits = list(execution.hits) or [hit_from_row(row) for row in execution.rows if row.get("source_row_id")]
    return build_evidence_packet(question, hits, facts=list(facts), catalog=catalog)


def _question_being_corrected(
    conversation: ConversationRecord, selected_file_id: int | None, text: str
) -> str | None:
    """The earlier question a pasted correction is about, or None when it is not one."""
    from app.planning.corrections import is_pasted_correction

    if not is_pasted_correction(text):
        return None
    frame = conversation.active_frame()
    if frame is not None and frame.question:
        if selected_file_id is None or tuple(frame.query.file_ids) == (selected_file_id,):
            return frame.question
    # Conversations saved before frames kept their question: the last thing the researcher
    # asked that was not itself a pasted correction.
    for item in reversed(conversation.recent_turns):
        if item.get("role") != "user":
            continue
        candidate = " ".join(str(item.get("text") or "").split())
        if candidate and not is_pasted_correction(str(item.get("text") or "")):
            return candidate
    return None


def _shorten(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _found_any(result: TurnResult) -> bool:
    """Whether the turn actually located records, however it reported them."""
    if result.rows or result.evidence:
        return True
    for item in result.list_results or []:
        if item.get("rows"):
            return True
    for fact in result.facts or []:
        if fact.get("name") in {"count", "count_distinct"} and int(fact.get("value") or 0) > 0:
            return True
    return False
