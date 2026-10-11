import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.ai_catalog_coerce import coerce_planned_turn
from app.planning.ai_planner import (
    AIPlanner,
    _review_eligible,
    authorized_dataset_options,
    bind_scope,
    planner_visible_labels,
    should_honor_dataset_not_selected,
)
from app.planning.ai_turn_schema import (
    PlannedConversationAction,
    PlannedQueryAction,
    PlannedTurn,
    PlanReviewResponse,
    ReviewComplete,
    ReviewCorrected,
    planned_turn_contains_dataset_id,
)
from app.planning.catalog import DatasetSpec, FieldCatalog, FieldSpec, static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.plan_schema import FilterOperator, PlanScope, PlanStep, Predicate, QueryPlan
from app.planning.plan_validator import PlanValidationError
from app.planning.predicate_contract import validate_predicate_contract
from app.planning.query_plan_builder import build_query_plan
from app.planning.requirement_parity import check_requirement_parity
from app.planning.response_policy import resolve_final_response
from app.planning.turn_schema import (
    AggregateSpec,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    TurnPlan,
    VerifyPreviousAction,
)
from app.planning.turn_validator import validate_turn_plan
from app.planning.value_expansion import expand_plan_values
from app.security.access_scope import AccessScope


def _ai_mode(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


def _count_plan(goal="count", **kwargs) -> PlannedTurn:
    filters = kwargs.pop("filters", [])
    requirements = [
        {
            "kind": "goal",
            "action_index": 0,
            "text": f"goal {goal}",
            "goal": goal,
        }
    ]
    for spec in filters:
        requirements.append(
            {
                "kind": "filter",
                "action_index": 0,
                "text": f"{spec['field']} {spec['operator']} {spec.get('value')}",
                "collection": "filters",
                "field": spec["field"],
                "operator": spec["operator"],
                "value": spec.get("value"),
            }
        )
    return PlannedTurn.model_validate(
        {
            "normalized_request": kwargs.pop("normalized_request", "count records"),
            "actions": [
                {
                    "type": "query",
                    "goal": goal,
                    "filters": filters,
                    **kwargs,
                }
            ],
            "stated_requirements": requirements,
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )


def _complete_review(planned: PlannedTurn) -> ReviewComplete:
    return ReviewComplete(reconstructed_requirements=list(planned.stated_requirements))


@pytest.fixture
def ai_settings(monkeypatch):
    _ai_mode(monkeypatch)
    yield
    monkeypatch.delenv("PLANNER_MODE", raising=False)
    get_settings.cache_clear()


def test_planned_turn_rejects_datasets_field():
    with pytest.raises(ValidationError):
        PlannedQueryAction.model_validate({"goal": "count", "datasets": [49]})


def test_provider_json_schema_requires_union_discriminators():
    planned_schema = PlannedTurn.model_json_schema()
    action_schemas = [
        item
        for item in planned_schema["$defs"].values()
        if "type" in item.get("properties", {})
    ]
    assert action_schemas
    assert all("type" in item.get("required", []) for item in action_schemas)
    assert "kind" in planned_schema["$defs"]["GoalRequirement"]["required"]
    review_schema = PlanReviewResponse.model_json_schema()
    assert "verdict" in review_schema["$defs"]["ReviewComplete"]["required"]


def test_planned_turn_preserves_query_semantics():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "stats",
                    "search_text": "tuberculosis",
                    "secondary_group_value_part": "year",
                    "having_min_count": 2,
                    "stats_value_part": "year",
                    "verify_previous": True,
                    "filters": [{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "stats", "goal": "stats"}
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    action = planned.query_actions()[0]
    assert action.search_text == "tuberculosis"
    assert action.secondary_group_value_part == "year"
    assert action.having_min_count == 2
    assert action.stats_value_part == "year"
    assert action.verify_previous is True
    bound = bind_scope(planned, 49)
    assert bound.query_actions()[0].datasets == [49]
    assert planned_turn_contains_dataset_id(planned.model_dump()) is False


def test_extra_fields_rejected_on_envelopes():
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "clarify", "question": "Which?"}],
                "final_response": "compiler_response",
                "file_id": 49,
            }
        )
    with pytest.raises(ValidationError):
        ReviewComplete.model_validate({"verdict": "complete", "extra": True})


def test_illegal_review_states_rejected():
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate({"verdict": "corrected"})
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate({"verdict": "needs_clarification"})
    parsed = PlanReviewResponse.model_validate(
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"}
            ],
        }
    )
    assert isinstance(parsed.root, ReviewComplete)


def test_review_complete_rejects_contradictory_correction_fields():
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate(
            {
                "verdict": "complete",
                "reconstructed_requirements": [
                    {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"}
                ],
                "missing": ["use cause_of_death for confirmed deaths"],
                "plan": {
                    "actions": [{"type": "query", "goal": "count"}],
                    "final_response": "deterministic",
                    "confidence": 0.9,
                },
            }
        )


def test_review_complete_strips_only_empty_compatibility_extras():
    parsed = PlanReviewResponse.model_validate(
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"}
            ],
            "missing": [],
            "plan": None,
        }
    )
    assert isinstance(parsed.root, ReviewComplete)


def test_conversational_plan_drops_query_requirements():
    planned = PlannedTurn.model_validate(
        {
            "actions": [{"type": "respond", "intent": "describe_available_data", "response": ""}],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "fields", "goal": "distinct"}
            ],
            "final_response": "compiler_response",
            "confidence": 0.9,
        }
    )
    assert planned.stated_requirements == []
    assert planned.respond_actions()[0].intent == "describe_available_data"


def test_general_conversation_requires_response():
    with pytest.raises(ValidationError):
        PlannedConversationAction.model_validate(
            {"intent": "general_conversation", "response": ""}
        )
    PlannedConversationAction.model_validate({"intent": "dataset_not_selected", "response": ""})


def test_requirement_index_and_compare_branch_validation():
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "query", "goal": "count"}],
                "stated_requirements": [
                    {"kind": "goal", "action_index": 3, "text": "count", "goal": "count"}
                ],
                "final_response": "deterministic",
            }
        )
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "query", "goal": "compare", "compare": []}],
                "stated_requirements": [
                    {
                        "kind": "filter",
                        "action_index": 0,
                        "text": "community",
                        "collection": "compare",
                        "field": "community",
                        "operator": "EQUALS",
                        "value": "Garden River",
                    }
                ],
                "final_response": "deterministic",
            }
        )


def test_requirement_parity_one_to_one():
    planned = _count_plan(
        filters=[
            {"field": "community", "operator": "EQUALS", "value": "Garden River"},
            {"field": "admitted_date", "operator": "AFTER", "value": 1905},
        ]
    )
    bound = bind_scope(planned, 49)
    check_requirement_parity(planned, bound)
    planned.stated_requirements.append(
        planned.stated_requirements[1].model_copy(update={"text": "duplicate garden river"})
    )
    with pytest.raises(PlanValidationError) as exc:
        check_requirement_parity(planned, bound)
    assert exc.value.code == "dropped_requirement"


def test_bind_scope_never_reads_model_file_ids():
    planned = _count_plan()
    bound = bind_scope(planned, 91)
    assert bound.query_actions()[0].datasets == [91]


@pytest.mark.asyncio
async def test_primary_retries_validation_error_only():
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        {"actions": [{"type": "query"}], "final_response": "deterministic"},
        _count_plan(),
        {"verdict": "complete"},
    ]
    planner = AIPlanner(provider)
    result = await planner.plan(
        "How many students?",
        catalog=static_catalog().for_scope(_scope()),
        selected_file_id=49,
        authorized_options=authorized_dataset_options(static_catalog().for_scope(_scope())),
        context=ConversationContext(),
    )
    assert result.status == "planned"
    assert result.planner_attempts == 2
    assert result.planner_retry_count == 1
    assert result.planner_total_model_calls == 2
    assert result.review_status == "skipped"
    assert provider.structured_log[1]["user"].startswith("SCHEMA-VALIDATION RETRY")


@pytest.mark.asyncio
async def test_no_retry_on_provider_failure():
    provider = FakeReasoningProvider()
    provider.structured_queue = [TimeoutError("xai timeout")]
    planner = AIPlanner(provider)
    result = await planner.plan(
        "How many students?",
        catalog=static_catalog().for_scope(_scope()),
        selected_file_id=49,
        authorized_options=authorized_dataset_options(static_catalog().for_scope(_scope())),
        context=ConversationContext(),
    )
    assert result.status == "planner_unavailable"
    assert result.planner_failure_stage == "primary"
    assert result.planner_attempts == 1
    assert result.planner_total_model_calls == 1


@pytest.mark.asyncio
async def test_malformed_primary_twice():
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        {"actions": [{"type": "query"}], "final_response": "deterministic"},
        {"actions": [{"type": "query"}], "final_response": "deterministic"},
    ]
    planner = AIPlanner(provider)
    result = await planner.plan(
        "How many students?",
        catalog=static_catalog().for_scope(_scope()),
        selected_file_id=49,
        authorized_options=authorized_dataset_options(static_catalog().for_scope(_scope())),
        context=ConversationContext(),
    )
    assert result.status == "planner_invalid_output"
    assert result.planner_failure_stage == "primary"
    assert result.planner_attempts == 2
    assert result.review_attempts == 0


@pytest.mark.asyncio
async def test_greeting_skips_review(ai_settings):
    provider = FakeReasoningProvider()
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(_scope(), "hello", selected_file_id=49)
    assert result.status == "answered"
    # Slice 5: greetings are exact trusted conversation shapes, so the AI path no longer
    # pays for a primary model call just to return the same bounded canned response.
    assert result.planner_total_model_calls == 0
    assert result.review_status == "skipped"
    assert result.reasoning_calls == 0


@pytest.mark.asyncio
async def test_data_request_reviews_even_with_zero_filters(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _count_plan(goal="list", normalized_request="show students admitted after 1905"),
        _complete_review(_count_plan(goal="list")),
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "Show students admitted after 1905",
        selected_file_id=49,
    )
    assert result.review_attempts == 1
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_review_correction_not_reviewed_again(ai_settings):
    corrected = _count_plan(
        filters=[{"field": "community", "operator": "EQUALS", "value": "Garden River"}]
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _count_plan(),
        ReviewCorrected(
            missing=["community Garden River"],
            reconstructed_requirements=list(corrected.stated_requirements),
            plan=corrected,
        ),
        _complete_review(corrected),
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(_scope(), "How many from Garden River?", selected_file_id=49)
    assert result.review_status == "corrected"
    assert result.planner_total_model_calls == 2
    assert provider.structured_calls == 2


@pytest.mark.asyncio
async def test_dataset_selection_required(ai_settings):
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = await service.answer(_scope(), "How many students are there?")
    assert result.status == "dataset_selection_required"
    assert result.planner_total_model_calls == 0


@pytest.mark.asyncio
async def test_pin_conflict_before_planner(ai_settings):
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    first = await service.answer(_scope(), "How many students?", selected_file_id=49)
    second = await service.answer(
        _scope(),
        "How many records?",
        conversation_id=first.conversation_id,
        selected_file_id=91,
    )
    assert second.status == "dataset_scope_conflict"
    assert second.suggested_file_id == 91


@pytest.mark.asyncio
async def test_authorized_other_list_suggestion(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        PlannedTurn.model_validate(
            {
                "actions": [
                    {
                        "type": "respond",
                        "intent": "dataset_not_selected",
                        "response": "Confirmed deaths",
                    }
                ],
                "stated_requirements": [],
                "final_response": "compiler_response",
                "confidence": 0.95,
            }
        )
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "How many records are in confirmed deaths?",
        selected_file_id=49,
    )
    assert result.status == "dataset_not_selected"
    assert result.suggested_file_id == 91
    assert "Confirmed deaths" in result.answer
    assert result.plan is None


@pytest.mark.asyncio
async def test_unauthorized_list_is_not_disclosed(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        PlannedTurn.model_validate(
            {
                "actions": [
                    {"type": "respond", "intent": "dataset_not_selected", "response": "Potential records"}
                ],
                "final_response": "compiler_response",
                "confidence": 0.9,
            }
        )
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(_scope(), "Search the potential list", selected_file_id=49)
    assert result.suggested_file_id is None
    assert "Potential" not in result.answer
    assert result.plan is None


@pytest.mark.asyncio
async def test_no_fan_out_in_ai_mode(ai_settings):
    provider = FakeReasoningProvider()
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "Compare the master list and confirmed deaths.",
        selected_file_id=49,
    )
    assert result.selected_file_id == 49
    assert result.status != "dataset_mismatch"


def test_strict_validation_no_orphan_promotion():
    turn = TurnPlan(
        actions=[ModifyPreviousAction(changes=[FilterSpec(field="community", operator="EQUALS", value="Garden River")])],
        final_response="deterministic",
        confidence=0.9,
    )
    validated = validate_turn_plan(
        turn,
        scope=_scope(),
        catalog=static_catalog().for_scope(_scope()),
        has_active_query=False,
        strict=True,
    )
    assert validated.clarify_actions()
    assert not validated.query_actions()


def test_predicate_and_or_contract():
    and_plan = QueryPlan(
        planner_type="ai_planner",
        goals=["count"],
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="f",
                op="FILTER",
                input="current_records",
                where=[
                    Predicate(
                        op="and",
                        items=[
                            Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"),
                            Predicate(field="community", operator=FilterOperator.EQUALS, value="Sault Ste. Marie"),
                        ],
                    )
                ],
            )
        ],
    )
    with pytest.raises(PlanValidationError):
        validate_predicate_contract(and_plan)
    or_plan = QueryPlan(
        planner_type="ai_planner",
        goals=["count"],
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="f",
                op="FILTER",
                input="current_records",
                where=[
                    Predicate(
                        op="or",
                        items=[
                            Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"),
                            Predicate(field="community", operator=FilterOperator.EQUALS, value="Sault Ste. Marie"),
                        ],
                    )
                ],
            )
        ],
    )
    validate_predicate_contract(or_plan)


def test_ai_canonicalization_does_not_reassign_fields():
    catalog = static_catalog().for_scope(_scope())
    plan = QueryPlan(
        planner_type="ai_planner",
        goals=["count"],
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="f",
                op="FILTER",
                input="current_records",
                where=[Predicate(field="notes", operator=FilterOperator.CONTAINS, value="the school.")],
            )
        ],
    )
    expanded = expand_plan_values(plan, catalog, profile="ai")
    leaf = expanded.steps[0].where[0]
    assert leaf.field == "notes"
    assert leaf.value == "school"


def test_sibling_predicates_are_anded_for_contradictions():
    and_plan = QueryPlan(
        planner_type="ai_planner",
        goals=["count"],
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="f",
                op="FILTER",
                input="current_records",
                where=[
                    Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"),
                    Predicate(field="community", operator=FilterOperator.EQUALS, value="Sault Ste. Marie"),
                ],
            )
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_predicate_contract(and_plan)
    assert exc.value.code == "op_contract"


def test_transport_normalize_keeps_fillers_and_polite_wrappers():
    from app.planning.input_normalizer import normalize_transport, normalize_user_turn

    text = "um please could you tell me how many students, sorry, records"
    transport = normalize_transport(text)
    legacy = normalize_user_turn(text)
    assert "um" in transport.normalized_text.lower()
    assert "please" in transport.normalized_text.lower()
    assert "um" not in legacy.normalized_text.lower()


def test_modify_parity_includes_limit_and_presentation():
    from app.planning.requirement_parity import extract_expressed
    from app.planning.turn_schema import ModifyPreviousAction, ResultWindow

    turn = TurnPlan(
        actions=[
            ModifyPreviousAction(
                goal="list",
                limit=10,
                offset=5,
                sort_by="student_name",
                sort_direction="desc",
                sample=False,
                requested_fields=["student_name"],
                window=ResultWindow(anchor="start", size=10, cursor=0),
                exhaustive=True,
                presentation="table",
            )
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    kinds = {item.kind: item for item in extract_expressed(turn) if item.kind == "modify_edit"}
    edits = {item.edit for item in extract_expressed(turn) if item.kind == "modify_edit"}
    assert edits >= {
        "goal",
        "limit",
        "offset",
        "sort_by",
        "sort_direction",
        "sample",
        "requested_fields",
        "window",
        "exhaustive",
        "presentation",
    }
    del kinds


def test_missing_requirement_kind_is_rejected():
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "query", "goal": "count"}],
                "stated_requirements": [
                    {"action_index": 0, "text": "count", "goal": "count"}
                ],
                "final_response": "deterministic",
                "confidence": 0.9,
            }
        )


def test_out_of_range_requirement_is_rejected():
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "query", "goal": "count"}],
                "stated_requirements": [
                    {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
                    {"kind": "goal", "action_index": 4, "text": "ghost", "goal": "list"},
                ],
                "final_response": "deterministic",
                "confidence": 0.9,
            }
        )


def test_mode_aggregate_compiles_to_stats_instead_of_crashing():
    turn = TurnPlan(
        actions=[
            QueryAction(
                datasets=[49],
                goal="aggregate",
                aggregate=AggregateSpec(function="mode", field="age"),
            )
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    plan = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    assert plan.op_names() == ["USE_CURRENT_VERSION", "STATS"]


def test_ai_does_not_strip_articles_on_community():
    catalog = static_catalog().for_scope(_scope())
    plan = QueryPlan(
        planner_type="ai_planner",
        goals=["count"],
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="f",
                op="FILTER",
                input="current_records",
                where=[Predicate(field="community", operator=FilterOperator.EQUALS, value="The Pas")],
            )
        ],
    )
    expanded = expand_plan_values(plan, catalog, profile="ai")
    assert expanded.steps[0].where[0].value == "The Pas"


def test_planner_visible_labels_exclude_file_ids():
    dataset = static_catalog().dataset(49)
    labels = planner_visible_labels(dataset)
    assert all("file 49" not in label.lower() for label in labels)
    assert "Student master list" in labels


def test_response_contract_rejects_suppressed_query():
    turn = TurnPlan(
        actions=[QueryAction(datasets=[49], goal="count")],
        final_response="compiler_response",
        confidence=0.9,
    )
    with pytest.raises(PlanValidationError) as exc:
        resolve_final_response(turn)
    assert exc.value.code == "response_contract"


@pytest.mark.asyncio
async def test_source_actions_drops_stale_active_frame(ai_settings):
    from app.execution.turn import AssistantTurnService
    from app.memory.types import ConversationRecord, QueryFrame, QueryState
    from app.planning.input_normalizer import NormalizedTurn
    from app.planning.plan_validator import PlanValidationError
    from app.planning.router import PlanRouteResult
    from app.planning.turn_schema import ModifyPreviousAction, TurnPlan

    conversation = ConversationRecord(
        id="00000000-0000-0000-0000-000000000099",
        principal_id="researcher",
        selected_file_id=49,
        frames={
            "old": QueryFrame(
                frame_key="old",
                topic="old",
                query=QueryState(file_ids=(91,), goal="count"),
            )
        },
        active_frame_key="old",
    )
    route = PlanRouteResult(
        plan=None,
        status="planned",
        source="ai_planner",
        normalized_turn=NormalizedTurn(raw_text="next", normalized_text="next"),
        turn_plan=TurnPlan(
            actions=[ModifyPreviousAction(page="next")],
            final_response="deterministic",
            confidence=0.9,
        ),
    )
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore())
    try:
        compiled = service._source_actions(
            route, _scope(), static_catalog().for_scope(_scope()), conversation
        )
    except PlanValidationError:
        compiled = {}
    assert all(getattr(query, "datasets", [49]) == [49] for query in compiled.values())


@pytest.mark.asyncio
async def test_schema_question_uses_selected_catalog_text(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        {
            "actions": [{"type": "respond", "intent": "describe_available_data", "response": ""}],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "fields", "goal": "distinct"}
            ],
            "final_response": "compiler_response",
            "confidence": 0.9,
        }
    ]
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(
        _scope(),
        "What fields are available in the master list?",
        selected_file_id=49,
    )
    assert result.status == "answered"
    assert result.plan is None
    assert "student name" in result.answer.casefold()
    assert result.review_status == "skipped"


@pytest.mark.asyncio
async def test_ungrounded_dataset_not_selected_retries(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        {
            "actions": [
                {"type": "respond", "intent": "dataset_not_selected", "response": ""}
            ],
            "final_response": "compiler_response",
            "confidence": 0.9,
        },
        _count_plan(),
        _complete_review(_count_plan()),
    ]
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(
        _scope(),
        "How many students were admitted between 1940 and 1945?",
        selected_file_id=49,
    )
    assert result.status == "answered"
    assert result.planner_attempts == 2
    assert result.planner_retry_count == 1
    assert provider.structured_log[0]["user"].find("SCHEMA-VALIDATION RETRY") == -1
    assert provider.structured_log[1]["user"].startswith("SCHEMA-VALIDATION RETRY")


@pytest.mark.asyncio
async def test_review_prompt_lists_only_selected_fields(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _count_plan(),
        _complete_review(_count_plan()),
    ]
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    await service.answer(_scope(), "How many students?", selected_file_id=49)
    review_prompt = provider.structured_log[1]["user"]
    assert "SELECTED FIELDS:" in review_prompt
    assert "cause_of_death" not in review_prompt


def test_dataset_not_selected_requires_named_other_source():
    action = PlannedConversationAction(intent="dataset_not_selected", response="")
    options = authorized_dataset_options(static_catalog().for_scope(_scope()))
    assert (
        should_honor_dataset_not_selected(
            action,
            question="How many students were admitted between 1940 and 1945?",
            selected_file_id=49,
            options=options,
        )
        is False
    )
    assert (
        should_honor_dataset_not_selected(
            action,
            question="How many records are in confirmed deaths?",
            selected_file_id=49,
            options=options,
        )
        is True
    )


def test_a_question_naming_the_selected_list_is_never_sent_elsewhere():
    """P09: a reply that named no list was believed over a question that named this one."""
    scope = AccessScope(
        principal_id="researcher", allowed_file_ids=(49, 91, 94), can_use_private_files=True
    )
    options = authorized_dataset_options(static_catalog().for_scope(scope))
    action = PlannedConversationAction(
        intent="dataset_not_selected",
        response="Garden River is not something this list records.",
    )
    assert (
        should_honor_dataset_not_selected(
            action,
            question="How many potential records are from Garden River?",
            selected_file_id=94,
            options=options,
        )
        is False
    )
    # Naming a different list is still what the redirect is for.
    assert (
        should_honor_dataset_not_selected(
            action,
            question="How many confirmed deaths are from Garden River?",
            selected_file_id=94,
            options=options,
        )
        is True
    )


@pytest.mark.asyncio
async def test_malformed_review_has_no_retry(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _count_plan(),
        {"verdict": "corrected"},
    ]
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(_scope(), "How many students?", selected_file_id=49)
    assert result.status == "planner_invalid_output"
    assert result.review_status == "invalid_output"
    assert result.review_attempts == 1
    assert result.plan is None
    assert result.planner_failure_stage == "review"
    assert provider.structured_calls == 2


def test_boolean_equals_is_coerced_to_is_true():
    planned = _count_plan(
        filters=[{"field": "deceased_status", "operator": "EQUALS", "value": "deceased"}]
    )
    coerced = coerce_planned_turn(planned, static_catalog().for_scope(_scope()), 49)
    spec = coerced.query_actions()[0].filters[0]
    assert spec.operator == "IS_TRUE"
    assert spec.value is None
    assert coerced.stated_requirements[1].operator == "IS_TRUE"


def test_projection_requirement_accepts_long_field_list():
    fields = [
        "student_name",
        "cause_of_death",
        "death_factor",
        "birth_date",
        "death_date",
        "burial_date",
        "first_admitted_date",
        "discharged_date",
        "death_registration_date",
        "age_at_death",
        "gender",
        "nation",
        "community",
        "school",
        "place_of_death",
        "location_of_death",
        "place_of_burial",
        "student_number",
        "indigenous_name",
        "parents_names",
        "ancestry_information",
        "admission_method",
        "reason_for_discharge",
        "documentation",
        "investigative_notes",
        "specific_file_notes",
        "information_shared_with",
        "death_registration_number",
        "other_schools",
        "other_links",
        "other_information",
        "census_documents_used",
        "census_year",
        "nia_comments",
    ]
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "list",
                    "requested_fields": fields,
                    "filters": [
                        {"field": "student_name", "operator": "CONTAINS", "value": "Albert"}
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "list", "goal": "list"},
                {
                    "kind": "projection",
                    "action_index": 0,
                    "text": "project " + ", ".join(fields),
                    "requested_fields": fields,
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    bound = bind_scope(planned, 91)
    check_requirement_parity(planned, bound)


def test_review_corrected_uses_verdict_discriminator():
    corrected = _count_plan()
    parsed = PlanReviewResponse.model_validate(
        {
            "verdict": "corrected",
            "missing": ["use cause_of_death"],
            "reconstructed_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"}
            ],
            "plan": corrected.model_dump(mode="json"),
        }
    )
    assert parsed.root.verdict == "corrected"


@pytest.mark.asyncio
async def test_unknown_field_review_can_correct(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _count_plan(filters=[{"field": "deceased_status", "operator": "IS_UNKNOWN"}]),
        ReviewCorrected(
            missing=["deceased_status is not on this list"],
            reconstructed_requirements=list(
                _count_plan(filters=[{"field": "gender", "operator": "IS_UNKNOWN"}]).stated_requirements
            ),
            plan=_count_plan(filters=[{"field": "gender", "operator": "IS_UNKNOWN"}]),
        ),
    ]
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(
        AccessScope(principal_id="researcher", allowed_file_ids=(91,)),
        "How many confirmed death records have no recorded gender?",
        selected_file_id=91,
    )
    assert result.review_status == "corrected"
    assert result.status in {"answered", "clarification"}
    if result.plan is not None:
        fields = [leaf.field for step in result.plan.steps for leaf in step.where if leaf.field]
        assert "deceased_status" not in fields


def test_application_default_is_ai():
    from app.config import Settings

    assert Settings.model_fields["planner_mode"].default == "ai"


def test_review_mode_accepts_only_implemented_values():
    """shadow and conditional used to be accepted and silently ran full review."""
    from app.config import Settings

    assert Settings.model_fields["planner_review_mode"].default == "always"
    Settings(planner_review_mode="off")
    Settings(planner_review_mode="always")
    for mode in ("shadow", "conditional"):
        with pytest.raises(ValidationError):
            Settings(planner_review_mode=mode)


@pytest.mark.asyncio
async def test_legacy_rollback_still_plans(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "legacy")
    get_settings.cache_clear()
    assert get_settings().planner_mode == "legacy"
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = await service.answer(_scope(), "How many students are there?", selected_file_id=49)
    assert result.status == "answered"
    assert result.plan is not None
    assert result.planner_type != "ai_planner"


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_ui_model_does_not_select_the_planner(ai_settings):
    """The researcher-picked model is for synthesis, not planning."""
    from app.llm.factory import resolve_model

    assert resolve_model("grok-4.6") == "grok-4.6"
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = await service.answer(
        _scope(),
        "How many students are there?",
        selected_file_id=49,
        model="grok-4.6",
    )
    assert result.status == "answered"
    assert result.planner_type == "ai_planner"
    assert result.planner_model in {None, "", "fake", "grok-4.20-0309-non-reasoning"}


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_ui_model_selects_the_synthesis_provider(ai_settings, monkeypatch):
    """The researcher-picked model answers; planning stays on the service default.

    The voice agent reuses one AssistantTurnService, so the pick is also cleared
    after the turn — a later turn without a model must not keep the last pick.
    """
    from app.llm.factory import allowed_models

    planner = FakeReasoningProvider()
    planner.model_name = "fake-planner"
    selected = FakeReasoningProvider()
    selected.model_name = "picked"
    picked = next(
        name for name in allowed_models() if name != get_settings().reasoning_model
    )

    monkeypatch.setattr(
        "app.llm.factory.get_reasoning_provider",
        lambda model_override=None: selected if model_override == picked else planner,
    )
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=planner
    )
    result = await service.answer(
        _scope(),
        "Summarize the causes of death in the authorized records.",
        selected_file_id=91,
        model=picked,
    )
    assert result.status == "answered"
    assert selected.synthesis_calls >= 1
    assert planner.synthesis_calls == 0
    assert planner.structured_calls >= 1
    assert selected.structured_calls == 0
    assert service._turn_reasoner is None

    later = await service.answer(
        _scope(),
        "Summarize the causes of death in the authorized records.",
        selected_file_id=91,
    )
    assert later.status == "answered"
    assert planner.synthesis_calls >= 1
    assert selected.synthesis_calls == 1


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_ai_default_path_plans_with_fake_provider(ai_settings):
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = await service.answer(_scope(), "How many students are there?", selected_file_id=49)
    assert result.status == "answered"
    assert result.planner_type == "ai_planner"


@pytest.mark.asyncio
async def test_three_condition_parity_reaches_plan(ai_settings):
    planned = _count_plan(
        filters=[
            {"field": "community", "operator": "EQUALS", "value": "Garden River"},
            {"field": "admitted_date", "operator": "AFTER", "value": 1905},
            {"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"},
        ]
    )
    # cause_of_death is not on file 49; use 91 for this semantic check via bind+parity only
    planned = _count_plan(
        filters=[
            {"field": "community", "operator": "EQUALS", "value": "Garden River"},
            {"field": "admitted_date", "operator": "AFTER", "value": 1905},
            {"field": "deceased_status", "operator": "IS_TRUE"},
        ]
    )
    bound = bind_scope(planned, 49)
    check_requirement_parity(planned, bound)
    fields = {leaf.field for spec in bound.query_actions()[0].filters for leaf in [spec]}
    assert fields == {"community", "admitted_date", "deceased_status"}


def test_a_follow_up_keeps_its_turn_when_it_states_the_filter_it_carries():
    """A modify_previous-only turn used to die on the requirement it honestly stated.

    Live, on the York thread: "I mean the 4 children, what are their names" planned as a
    modify_previous -- correct, the filter comes from the active query -- and attached the
    filter it was carrying as a stated requirement. Nothing could target it, so the whole
    turn was rejected and the researcher was told the planning service was unavailable.

    The requirement is dropped, not the turn. There is no query action here for parity to
    check it against; the filter itself still reaches the plan through the active query.
    """
    turn = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "modify_previous",
                    "changes": [
                        {"field": "admitted_date", "operator": "YEAR_EQUALS", "value": 1910}
                    ],
                }
            ],
            "stated_requirements": [
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "the 4 children",
                    "field": "age",
                    "operator": "EQUALS",
                    "value": 4,
                },
                {"kind": "goal", "action_index": 0, "text": "names", "goal": "list"},
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert [action.type for action in turn.actions] == ["modify_previous"]
    assert turn.stated_requirements == []


def test_a_requirement_naming_a_query_action_is_still_kept_and_checked():
    """The drop is by target, so a real query keeps every requirement parity relies on."""
    turn = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "modify_previous",
                    "changes": [{"field": "age", "operator": "EQUALS", "value": 4}],
                },
                {"type": "query", "goal": "list"},
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 1, "text": "names", "goal": "list"},
                {"kind": "goal", "action_index": 0, "text": "stray", "goal": "list"},
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    kept = turn.stated_requirements
    assert [item.action_index for item in kept] == [1]
    assert kept[0].text == "names"


@pytest.mark.asyncio
async def test_a_timed_out_planner_call_is_asked_again_before_giving_up():
    """The York thread lost a turn to one ReadTimeout.

    The retry budget was already there -- two attempts -- but only a schema disagreement
    could spend it, so a transport hiccup returned "unavailable" on the first try. Nothing
    about the question caused the timeout, so the second attempt sends it unchanged: no
    SCHEMA-VALIDATION RETRY preamble, which would be answering a complaint nobody made.
    """
    from app.llm.base import StructuredOutputResult
    from app.llm.http_reasoner import TransientProviderError
    from app.planning.ai_planner import AIPlanner

    seen: list[str] = []

    class _FlakyHost:
        provider_name = "http"
        model_name = "grok-test"

        async def structured_output_with_metadata(self, *, system, user, response_model):
            seen.append(user)
            if len(seen) == 1:
                raise TransientProviderError("ReadTimeout from reasoning provider")
            return StructuredOutputResult(
                value=response_model.model_validate(
                    {
                        "actions": [{"type": "query", "goal": "list"}],
                        "final_response": "deterministic",
                        "confidence": 0.9,
                    }
                ),
                provider="http",
                model="grok-test",
            )

    stage, planned, retry_count = await AIPlanner(_FlakyHost())._primary(
        "USER PROMPT", question="q", selected_file_id=49, authorized_options=[]
    )

    assert stage.status == "ok"
    assert planned is not None
    assert stage.attempts == 2
    assert len(seen) == 2 and seen[1] == "USER PROMPT"
    # A timeout is not the planner getting the schema wrong, so it is not counted as one.
    assert retry_count == 0


@pytest.mark.asyncio
async def test_a_host_that_stays_down_is_reported_unavailable_not_retried_forever():
    from app.llm.http_reasoner import TransientProviderError
    from app.planning.ai_planner import AIPlanner

    calls: list[int] = []

    class _DeadHost:
        provider_name = "http"
        model_name = "grok-test"

        async def structured_output_with_metadata(self, *, system, user, response_model):
            calls.append(1)
            raise TransientProviderError("ReadTimeout from reasoning provider")

    stage, planned, _ = await AIPlanner(_DeadHost())._primary(
        "USER PROMPT", question="q", selected_file_id=49, authorized_options=[]
    )
    assert stage.status == "unavailable"
    assert planned is None
    assert len(calls) == 2


def test_a_blank_conversational_flourish_does_not_take_the_query_with_it():
    """Seen repeatedly in the matrix: a good query plus an empty general_conversation.

    The action is unusable and the query beside it is fine, so the action goes and the turn
    stays. Previously the whole turn was rejected and the planner spent a retry call being
    told about a field it had left blank on purpose.
    """
    turn = PlannedTurn.model_validate(
        {
            "actions": [
                {"type": "query", "goal": "count"},
                {"type": "respond", "intent": "general_conversation", "response": "  "},
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"}
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert [action.type for action in turn.actions] == ["query"]
    assert [item.action_index for item in turn.stated_requirements] == [0]


def test_requirement_indexes_follow_the_action_that_moved():
    """Dropping action 0 renumbers action 1, and a requirement naming it has to move too."""
    turn = PlannedTurn.model_validate(
        {
            "actions": [
                {"type": "respond", "intent": "general_conversation", "response": ""},
                {"type": "query", "goal": "list"},
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 1, "text": "names", "goal": "list"}
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert [action.type for action in turn.actions] == ["query"]
    assert [item.action_index for item in turn.stated_requirements] == [0]


def test_small_talk_that_is_the_whole_turn_still_has_to_say_something():
    """The drop is about a flourish beside real work, not a licence to plan an empty turn."""
    with pytest.raises(ValidationError):
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "respond", "intent": "general_conversation", "response": ""}],
                "final_response": "deterministic",
                "confidence": 0.9,
            }
        )


def test_a_flourish_that_actually_says_something_is_kept():
    turn = PlannedTurn.model_validate(
        {
            "actions": [
                {"type": "query", "goal": "count"},
                {"type": "respond", "intent": "general_conversation", "response": "Happy to help."},
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert [action.type for action in turn.actions] == ["query", "respond"]


def test_strict_low_confidence_executable_becomes_clarification():
    """Characterization: AI/strict validation clarifies before review when confidence is below 0.75.

    So every plan that reaches review already has confidence >= the threshold.
    A later 'conditional review if confidence < 0.75' gate would never fire
    on surviving single-action queries.
    """
    turn = TurnPlan(
        actions=[QueryAction(datasets=[49], goal="count")],
        final_response="deterministic",
        confidence=0.2,
    )
    validated = validate_turn_plan(
        turn,
        scope=_scope(),
        catalog=static_catalog().for_scope(_scope()),
        has_active_query=False,
        strict=True,
    )
    assert validated.clarify_actions()
    assert not validated.query_actions()
    assert "low_confidence" in validated.unresolved


def test_verify_previous_action_is_review_eligible():
    """Characterization: a verify_previous action is review-eligible."""
    planned = PlannedTurn.model_validate(
        {
            "actions": [{"type": "verify_previous"}],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert _review_eligible(planned) is True
    assert planned.verify_actions()
    assert isinstance(planned.verify_actions()[0], VerifyPreviousAction)


def test_navigation_only_modify_skips_review():
    """Characterization: page/restore-only modify is not review-eligible."""
    planned = PlannedTurn.model_validate(
        {
            "actions": [{"type": "modify_previous", "page": "next"}],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert _review_eligible(planned) is False


def test_evidence_disallowed_fields_are_not_packed():
    """Characterization: needs_evidence with no allowed fields is sensitivity."""
    catalog = FieldCatalog(
        datasets=(DatasetSpec(file_id=49, user_facing_label="Master"),),
        fields=(
            FieldSpec(
                file_id=49,
                semantic_field="notes",
                human_label="Notes",
                allowed_operators=("CONTAINS", "IS_KNOWN"),
                evidence_allowed=False,
            ),
        ),
        default_people_file_id=49,
    )
    turn = TurnPlan(
        actions=[QueryAction(datasets=[49], goal="count")],
        final_response="deterministic",
        confidence=0.9,
        needs_evidence=True,
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_turn_plan(
            turn,
            scope=AccessScope(principal_id="researcher", allowed_file_ids=(49,)),
            catalog=catalog,
            strict=True,
        )
    assert exc.value.code == "sensitivity"


@pytest.mark.asyncio
async def test_empty_registry_fails_closed_in_production(monkeypatch):
    """Characterization: production load_catalog with no fields is fail-closed."""
    from app.data_gateway.repository import AssistantDataGateway

    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    gateway = AssistantDataGateway.__new__(AssistantDataGateway)

    async def _datasets(_scope):
        return [{"file_id": 49}]

    async def _fields(_scope, _file_id):
        return []

    gateway.list_datasets = _datasets  # type: ignore[method-assign]
    gateway.get_dataset_fields = _fields  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="field_registry"):
        await gateway.load_catalog(_scope())
    monkeypatch.delenv("APP_ENV", raising=False)
    get_settings.cache_clear()


def test_a_question_naming_the_selected_list_is_recognized_as_such():
    """Registry labels decide this, so it carries no dataset vocabulary of its own."""
    from app.planning.ai_planner import (
        AuthorizedDatasetOption,
        question_names_only_the_selected_list,
    )

    options = [
        AuthorizedDatasetOption(file_id=49, label="Student master list", aliases=("master",)),
        AuthorizedDatasetOption(
            file_id=93, label="Additional Deaths", aliases=("additional deaths",)
        ),
    ]
    names_own = question_names_only_the_selected_list
    assert names_own("How many records are in additional deaths?", 93, options) is True
    # Naming another authorized list is the case dataset_not_selected exists for.
    assert names_own("How many are in the master list?", 93, options) is False
    # Naming neither is an ordinary question about whatever is selected.
    assert names_own("How many records have a community?", 93, options) is False
    # Naming both is ambiguous, so the reviewer keeps its say.
    assert names_own("compare additional deaths with the master list", 93, options) is False
    # A file the principal cannot see is not an option, so nothing is claimed about it.
    assert names_own("How many records are in additional deaths?", 93, []) is False


def _grouped_plan(**kwargs) -> PlannedTurn:
    return _count_plan(
        goal="duplicates",
        group_by=["parents_names"],
        normalized_request="families with several children at the same time",
        **kwargs,
    )


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_a_same_period_question_gets_one_retry_to_compare_periods(ai_settings):
    """The planner wrote "during the same period" down and still grouped by names alone."""
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _grouped_plan(),
        _grouped_plan(interval_start="admitted_date", interval_end="discharged_date"),
        _complete_review(_grouped_plan(interval_start="admitted_date", interval_end="discharged_date")),
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "Are there families with several children attending during the same period?",
        selected_file_id=49,
    )
    assert result.status == "answered", result.answer
    assert result.planner_retry_count == 1
    [step] = [item for item in result.plan.steps if item.op.value == "DUPLICATES"]
    assert (step.interval_start, step.interval_end) == ("admitted_date", "discharged_date")


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_a_second_plan_without_periods_is_still_answered(ai_settings):
    """The retry is feedback, not a gate: failing it twice must not cost the researcher an answer."""
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        _grouped_plan(),
        _grouped_plan(),
        _complete_review(_grouped_plan()),
    ]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "Are there families with several children attending during the same period?",
        selected_file_id=49,
    )
    assert result.status == "answered", result.answer
    assert result.planner_retry_count == 1


def test_a_review_correction_keeps_the_periods_a_same_time_question_compares():
    from app.planning.ai_planner import _keep_period_comparison

    original = _grouped_plan(interval_start="admitted_date", interval_end="discharged_date")
    corrected = _count_plan(goal="rank", group_by=["parents_names"])
    question = "Are there families with several children attending during the same period?"
    kept = _keep_period_comparison(corrected, original, question)
    [action] = kept.query_actions()
    assert (action.goal, action.interval_start, action.interval_end) == (
        "duplicates",
        "admitted_date",
        "discharged_date",
    )
    assert [item.goal for item in kept.stated_requirements if item.kind == "goal"] == ["duplicates"]
    # A question about no period leaves the correction exactly as the reviewer wrote it.
    assert _keep_period_comparison(corrected, original, "Which parents names repeat?") is corrected


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_a_grouping_question_about_no_period_is_not_retried(ai_settings):
    provider = FakeReasoningProvider()
    provider.structured_queue = [_grouped_plan(), _complete_review(_grouped_plan())]
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(), "Which parents names appear on more than one record?", selected_file_id=49
    )
    assert result.status == "answered", result.answer
    assert result.planner_retry_count == 0


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_review_may_not_ask_which_list_when_the_question_named_it(ai_settings, monkeypatch):
    """A01 lived here: a validated count was replaced by a clarification, every run.

    The reviewer answered "the selected list does not contain records about additional
    deaths" while the selected list *was* Additional Deaths. The primary path already
    refuses that premise; review had no equivalent, so it overruled a plan that had passed
    bind, validate, build and predicate.
    """
    from app.llm.base import StructuredOutputResult
    from app.planning.ai_planner import PlanReviewResponse

    provider = FakeReasoningProvider()
    real = provider.structured_output_with_metadata

    async def always_wants_clarification(*, system, user, response_model):
        if response_model is PlanReviewResponse:
            return StructuredOutputResult(
                value=PlanReviewResponse.model_validate(
                    {
                        "verdict": "needs_clarification",
                        "missing": ["which list"],
                        "reason": "dataset_identity",
                        "question": "The selected list does not contain records about "
                        "additional deaths. Did you mean the master list?",
                    }
                ),
                provider="fake",
                model="fake",
            )
        return await real(system=system, user=user, response_model=response_model)

    monkeypatch.setattr(
        provider, "structured_output_with_metadata", always_wants_clarification
    )
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(
        _scope(), "How many records are in confirmed deaths?", selected_file_id=91
    )
    assert result.status == "answered", result.answer
    assert result.review_status == "clarification_overruled"


@pytest.mark.planner_ai
@pytest.mark.asyncio
async def test_review_still_clarifies_when_the_question_names_no_list(ai_settings, monkeypatch):
    """The overrule is narrow: it only answers the premise it can check."""
    from app.llm.base import StructuredOutputResult
    from app.planning.ai_planner import PlanReviewResponse

    provider = FakeReasoningProvider()
    real = provider.structured_output_with_metadata

    async def always_wants_clarification(*, system, user, response_model):
        if response_model is PlanReviewResponse:
            return StructuredOutputResult(
                value=PlanReviewResponse.model_validate(
                    {
                        "verdict": "needs_clarification",
                        "missing": ["community"],
                        "reason": "ambiguous_value",
                        "question": "Which community?",
                    }
                ),
                provider="fake",
                model="fake",
            )
        return await real(system=system, user=user, response_model=response_model)

    monkeypatch.setattr(
        provider, "structured_output_with_metadata", always_wants_clarification
    )
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    result = await service.answer(
        _scope(), "How many records have a recorded community?", selected_file_id=91
    )
    assert result.status == "clarification"


def test_a_plan_that_fails_validation_never_shows_its_reason_verbatim():
    """These messages are written for whoever fixes the plan, not whoever asked.

    "operator STARTS_WITH is not allowed on name_search for file 49" reached a researcher
    as the answer. The forbidden_table / forbidden_sql codes would be worse: internal
    schema names in a chat transcript.
    """
    from app.planning.ai_planner import _safe_clarification
    from app.planning.plan_validator import PlanValidationError

    leaky = [
        PlanValidationError("operator STARTS_WITH is not allowed on name_search for file 49", code="operator"),
        PlanValidationError("table assistant.query_runs is not readable here", code="forbidden_table"),
        PlanValidationError("select * from pg_class is refused", code="forbidden_sql"),
        PlanValidationError("step fan-out exceeds the dag budget of 8", code="dag"),
        PlanValidationError("something nobody enumerated", code=""),
    ]
    import re

    # Whole words only: "selected list" is legitimate user-facing wording and must not be
    # mistaken for the SQL verb.
    internals = ("STARTS_WITH", "name_search", "pg_class", "dag", "select", "table", "operator")
    for exc in leaky:
        text = _safe_clarification(exc)
        assert text != str(exc)
        assert "file 49" not in text and "assistant." not in text
        for internal in internals:
            assert not re.search(rf"{re.escape(internal)}", text, re.IGNORECASE), (
                exc.code,
                internal,
                text,
            )
        assert text.strip().endswith(("?", ".")), text


def test_an_operator_the_field_forbids_is_worth_a_second_look():
    """unknown_field and operator are the same mistake: naming what the registry forbids.

    Only the first was salvageable, so a question whose operator did not suit the field the
    planner chose died there, with a valid alternative field on the same list.
    """
    from app.planning.ai_planner import _REVIEWABLE_VALIDATION_CODES

    assert "operator" in _REVIEWABLE_VALIDATION_CODES
    assert "unknown_field" in _REVIEWABLE_VALIDATION_CODES
    # access_restricted must never be "salvaged" -- it is re-raised before this is consulted.
    assert "access_restricted" not in _REVIEWABLE_VALIDATION_CODES


def test_a_condition_no_requirement_accounts_for_is_recorded(caplog):
    """The dominant remaining failure mode, made visible rather than guessed at.

    "How many confirmed death records are missing a cause of death?" came back filtered to
    those with a known death date as well -- 2 instead of 6 -- and "missing a community"
    picked up a known cause of death, giving 0 instead of 1. Both are conditions no stated
    requirement accounts for, and both undercount while sounding certain.

    Not an error: the reverse direction (a stated requirement with nothing to match) is one,
    because dropping a condition answers a question nobody asked, and dropping an
    unaccounted condition here would risk exactly that when the planner merely under-declared
    a filter the researcher did ask for. Logged so the rate can be measured before anyone
    decides.
    """
    import logging

    planned = _count_plan(
        filters=[
            {"field": "community", "operator": "EQUALS", "value": "Garden River"},
            {"field": "admitted_date", "operator": "AFTER", "value": 1905},
        ]
    )
    bound = bind_scope(planned, 49)
    # The planner stated one of the two conditions it planned.
    planned.stated_requirements.pop()
    with caplog.at_level(logging.INFO, logger="nia.planning.parity"):
        check_requirement_parity(planned, bound)
    messages = [record.getMessage() for record in caplog.records]
    assert any("no stated requirement accounts for" in text for text in messages), messages
    assert any("admitted_date" in text for text in messages), messages


def test_a_plan_whose_conditions_are_all_accounted_for_logs_nothing(caplog):
    import logging

    planned = _count_plan(
        filters=[{"field": "community", "operator": "EQUALS", "value": "Garden River"}]
    )
    bound = bind_scope(planned, 49)
    with caplog.at_level(logging.INFO, logger="nia.planning.parity"):
        check_requirement_parity(planned, bound)
    assert not [r for r in caplog.records if "no stated requirement" in r.getMessage()]


def test_a_list_is_named_by_words_not_by_substring():
    """"List the first 5 additional death names" is about Additional Deaths.

    Substring matching could not see it -- the string "additional deaths" is not in that
    sentence -- so a question about the selected list looked like a question about no list,
    and the reviewer's "this list holds no such thing" stood unchallenged.
    """
    from app.planning.ai_planner import (
        AuthorizedDatasetOption,
        question_names_only_the_selected_list,
    )

    options = [
        AuthorizedDatasetOption(file_id=49, label="Student master list", aliases=("master",)),
        AuthorizedDatasetOption(file_id=91, label="Confirmed deaths", aliases=("confirmed",)),
        AuthorizedDatasetOption(
            file_id=93, label="Additional Deaths", aliases=("additional deaths",)
        ),
    ]
    names_own = question_names_only_the_selected_list

    # Plural label, singular mention, and extra words after it.
    assert names_own("List the first 5 additional death names.", 93, options) is True
    assert names_own("How many records are in additional deaths?", 93, options) is True
    # Still only fires for the list the question actually names.
    assert names_own("List the first 5 additional death names.", 49, options) is False
    assert names_own("how many confirmed deaths are there?", 93, options) is False


def test_word_matching_does_not_fire_inside_a_longer_word():
    """Stricter than the substring test it replaced: "master" is not "postmaster"."""
    from app.planning.ai_planner import AuthorizedDatasetOption, _text_names_option

    master = AuthorizedDatasetOption(file_id=49, label="Student master list", aliases=("master",))
    assert _text_names_option("who was the postmaster?", master) is False
    assert _text_names_option("how many are in the master list?", master) is True


def test_plural_folding_is_not_applied_to_short_or_double_s_words():
    """Folding every trailing s would turn "class" into "clas" and "is" into "i"."""
    from app.planning.ai_planner import _phrase_tokens

    assert _phrase_tokens("deaths names lists") == ["death", "name", "list"]
    assert _phrase_tokens("class is") == ["class", "is"]
    assert _phrase_tokens("Additional Deaths") == ["additional", "death"]
    # Punctuation and case never reach the comparison.
    assert _phrase_tokens("Additional-Deaths, ") == ["additional", "death"]


def test_conditions_the_plan_never_claimed_are_reported_not_removed():
    """The signal behind A03/D04, where an invented filter undercounted with confidence.

    "missing a cause of death" was also filtered to a known death date (2 instead of 6) and
    "missing a community" picked up a known cause of death (0 instead of 1). Nothing is
    deleted on this signal: an invented filter and one the planner simply forgot to declare
    are identical here, and dropping the second would be the condition loss parity exists to
    prevent. The reviewer sees the question and decides.
    """
    from app.planning.requirement_parity import unaccounted_conditions

    planned = _count_plan(
        filters=[
            {"field": "community", "operator": "EQUALS", "value": "Garden River"},
            {"field": "admitted_date", "operator": "AFTER", "value": 1905},
        ]
    )
    bound = bind_scope(planned, 49)
    assert unaccounted_conditions(planned, bound) == []

    planned.stated_requirements.pop()
    reported = unaccounted_conditions(planned, bound)
    assert len(reported) == 1 and "admitted_date" in reported[0]
    # Reporting is not removal: the plan still carries both conditions.
    assert len(bound.query_actions()[0].filters) == 2


def test_the_review_prompt_is_unchanged_when_every_condition_is_accounted_for():
    """The extra section must not perturb the ordinary turn, or the cacheable prefix."""
    from app.planning.ai_planner import _review_user_prompt
    from app.planning.catalog import static_catalog

    catalog = static_catalog()
    planned = _count_plan()
    without = _review_user_prompt("how many students?", planned, catalog, 49)
    empty = _review_user_prompt("how many students?", planned, catalog, 49, unaccounted=())
    assert without == empty
    assert "CONDITIONS NOT STATED AS REQUIREMENTS" not in without


def test_the_review_is_told_which_conditions_were_never_claimed():
    from app.planning.ai_planner import _review_user_prompt
    from app.planning.catalog import static_catalog

    catalog = static_catalog()
    planned = _count_plan()
    text = _review_user_prompt(
        "how many records are missing a cause of death?",
        planned,
        catalog,
        49,
        unaccounted=("death_date IS_KNOWN",),
    )
    assert "CONDITIONS NOT STATED AS REQUIREMENTS" in text
    assert "- death_date IS_KNOWN" in text
    # The instruction has to protect a condition the question really did ask for.
    assert "Never drop a condition" in text


def _two_list_catalog():
    """One list records a student number; the other only a student name."""
    from app.planning.catalog import DatasetSpec, FieldCatalog, FieldSpec

    def field(file_id, name, label, aliases=()):
        return FieldSpec(
            file_id=file_id,
            semantic_field=name,
            human_label=label,
            aliases=tuple(aliases),
            allowed_operators=("EQUALS", "IS_KNOWN", "IS_UNKNOWN"),
        )

    return FieldCatalog(
        datasets=(
            DatasetSpec(file_id=49, user_facing_label="Student master list"),
            DatasetSpec(file_id=94, user_facing_label="Potential"),
        ),
        fields=(
            field(49, "student_number", "Student number"),
            field(49, "student_name", "Student name", ("student", "full name")),
            field(94, "student_name", "Student name", ("student", "full name")),
            field(94, "nation", "First Nation", ("band",)),
        ),
        default_people_file_id=49,
    )


def test_a_broad_alias_cannot_swallow_a_more_specific_request():
    """P10 lived here: "student number" answered from student_name, alias "student".

    The list records no number, so the count of records that have a *name* was returned as
    if it answered the question. Only multi-word names count as naming a field -- one broad
    word is how the wrong field gets picked in the first place.
    """
    from app.planning.ai_planner import absent_fields_named

    catalog = _two_list_catalog()
    # Named on the master list, absent from Potential.
    assert absent_fields_named("how many potential records have a student number?", 94, catalog) == [
        "Student number"
    ]
    # The list that does record it is not accused of lacking it.
    assert absent_fields_named("how many records have a student number?", 49, catalog) == []


def test_a_single_broad_word_is_not_treated_as_naming_a_field():
    """"student" alone must not make every list that lacks some student_* field refuse."""
    from app.planning.ai_planner import absent_fields_named

    catalog = _two_list_catalog()
    assert absent_fields_named("how many students are there?", 94, catalog) == []
    assert absent_fields_named("list the students", 94, catalog) == []


def test_a_field_both_lists_share_is_never_reported_absent():
    from app.planning.ai_planner import absent_fields_named

    catalog = _two_list_catalog()
    assert absent_fields_named("show the student name for each record", 94, catalog) == []


def test_the_overrule_stands_down_when_a_named_field_is_missing_here():
    """Otherwise the A01 fix would force a wrong answer instead of a correct refusal.

    "How many potential records have a student number" names the selected list too, so the
    overrule would have fired and turned a correct clarification into a count of 53.
    """
    from app.planning.ai_planner import (
        AuthorizedDatasetOption,
        question_names_field_absent_from_selected,
        question_names_only_the_selected_list,
    )

    catalog = _two_list_catalog()
    options = [
        AuthorizedDatasetOption(file_id=49, label="Student master list", aliases=("master",)),
        AuthorizedDatasetOption(file_id=94, label="Potential", aliases=("potential records",)),
    ]
    question = "How many potential records have a student number?"
    assert question_names_only_the_selected_list(question, 94, options) is True
    assert question_names_field_absent_from_selected(question, 94, catalog) is True


def test_the_review_prompt_names_the_missing_field_only_when_there_is_one():
    from app.planning.ai_planner import _review_user_prompt
    from app.planning.catalog import static_catalog

    catalog = static_catalog()
    planned = _count_plan()
    plain = _review_user_prompt("how many students?", planned, catalog, 49)
    assert "NAMED BUT NOT ON THIS LIST" not in plain
    assert plain == _review_user_prompt(
        "how many students?", planned, catalog, 49, absent_fields=()
    )

    flagged = _review_user_prompt(
        "how many have a student number?",
        planned,
        catalog,
        49,
        absent_fields=("Student number",),
    )
    assert "NAMED BUT NOT ON THIS LIST" in flagged
    assert "- Student number" in flagged
    assert "name merely overlaps" in flagged
