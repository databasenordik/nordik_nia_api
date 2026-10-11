"""Slice 2 regressions for semantic completeness and review containment.

These tests intentionally focus on failures that can return a plausible but wrong answer:
unexplained narrowing, reviewer anchoring/coverage, correction drift, clarification overrides,
and missing follow-up/scope context.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.llm.fake import FakeReasoningProvider
from app.planning.ai_planner import (
    AI_PLANNER_SYSTEM,
    AI_REVIEW_SYSTEM,
    _planner_user_prompt,
    _review_user_prompt,
    absent_fields_named,
    authorized_dataset_options,
    plan_user_turn_ai,
)
from app.planning.ai_turn_schema import PlanReviewResponse, PlannedTurn, bind_scope
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.predicate_contract import validate_predicate_contract
from app.planning.query_plan_builder import build_query_plan
from app.planning.requirement_parity import unaccounted_narrowing_conditions
from app.planning.turn_schema import ActiveQuery
from app.planning.turn_validator import validate_turn_plan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


@pytest.fixture
def ai_settings(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    monkeypatch.setenv("PLANNER_REVIEW_MODE", "always")
    monkeypatch.setenv("PLANNER_SELF_REVIEW", "true")
    monkeypatch.setenv("PLANNER_REQUIREMENT_CHECK", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _goal_requirement(goal: str = "count") -> dict:
    return {
        "kind": "goal",
        "action_index": 0,
        "text": f"goal {goal}",
        "goal": goal,
    }


def _filter_requirement(field: str, operator: str, value=None) -> dict:
    return {
        "kind": "filter",
        "action_index": 0,
        "text": f"{field} {operator} {value}",
        "collection": "filters",
        "field": field,
        "operator": operator,
        "value": value,
    }


def _count_plan(*, filters=(), stated_filters=(), filter_logic="and") -> PlannedTurn:
    requirements = [_goal_requirement()]
    if filter_logic == "or":
        requirements.append(
            {
                "kind": "filter_logic",
                "action_index": 0,
                "text": "alternatives",
                "filter_logic": "or",
            }
        )
    for spec in stated_filters:
        requirements.append(
            _filter_requirement(spec["field"], spec["operator"], spec.get("value"))
        )
    return PlannedTurn.model_validate(
        {
            "normalized_request": "count records",
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filter_logic": filter_logic,
                    "filters": list(filters),
                }
            ],
            "stated_requirements": requirements,
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )


async def _plan(question: str, provider: FakeReasoningProvider, *, context=None):
    catalog = static_catalog()
    return await plan_user_turn_ai(
        question,
        _scope(),
        catalog,
        selected_file_id=49,
        conversation=context or ConversationContext(),
        reasoner=provider,
        authorized_catalog=catalog,
    )


def _complete(requirements) -> dict:
    return {"verdict": "complete", "reconstructed_requirements": requirements}


@pytest.mark.asyncio
async def test_unstated_narrowing_fails_closed_when_review_output_is_invalid(ai_settings):
    extra = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _count_plan(filters=[extra], stated_filters=[])
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, {"verdict": "complete"}]  # missing reconstruction

    result = await _plan("How many students are there?", provider)

    assert result.status == "planner_invalid_output"
    assert result.plan is None
    assert result.review_status == "invalid_output"
    assert result.planner_failure_stage == "review"
    assert result.planner_total_model_calls == 2
    assert "unverified" not in (result.response_text or "").casefold()  # raw validator text is hidden


@pytest.mark.asyncio
async def test_unstated_narrowing_fails_closed_when_review_is_unavailable(ai_settings):
    extra = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _count_plan(filters=[extra], stated_filters=[])
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, RuntimeError("review host down")]

    result = await _plan("How many students are there?", provider)

    assert result.status == "planner_unavailable"
    assert result.plan is None
    assert result.review_status == "unavailable"
    assert result.planner_failure_stage == "review"
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_complete_review_cannot_omit_a_candidate_filter(ai_settings):
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _count_plan(filters=[filt], stated_filters=[filt])
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, _complete([_goal_requirement()])]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.status == "clarification"
    assert result.plan is None
    assert result.review_status == "inconsistent"


@pytest.mark.asyncio
async def test_complete_review_cannot_reconstruct_a_filter_the_candidate_dropped(ai_settings):
    primary = _count_plan()
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        _complete(
            [
                _goal_requirement(),
                _filter_requirement("community", "EQUALS", "Garden River"),
            ]
        ),
    ]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.status == "clarification"
    assert result.plan is None
    assert result.review_status == "inconsistent"


@pytest.mark.asyncio
async def test_review_correction_cannot_introduce_unreconstructed_narrowing(ai_settings):
    primary = _count_plan()
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    corrected = _count_plan(filters=[filt], stated_filters=[filt])
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community"],
            "reconstructed_requirements": [_goal_requirement()],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.status == "clarification"
    assert result.plan is None
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_matching_review_correction_succeeds_without_a_third_llm_call(ai_settings):
    primary = _count_plan()
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    corrected = _count_plan(filters=[filt], stated_filters=[filt])
    reconstruction = [
        _goal_requirement(),
        _filter_requirement("community", "EQUALS", "Garden River"),
    ]
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community"],
            "reconstructed_requirements": reconstruction,
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"
    assert result.planner_total_model_calls == 2
    assert provider.structured_calls == 2


@pytest.mark.asyncio
async def test_review_may_remove_an_unstated_suspect_filter(ai_settings):
    extra = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _count_plan(filters=[extra], stated_filters=[])
    corrected = _count_plan()
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["unrequested community filter"],
            "reconstructed_requirements": [_goal_requirement()],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan("How many students are there?", provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_named_selected_list_does_not_override_non_dataset_ambiguity(ai_settings):
    primary = _count_plan()
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "needs_clarification",
            "missing": ["range around 1900"],
            "reason": "ambiguous_range",
            "reconstructed_requirements": [_goal_requirement()],
            "question": "What year range should 'around 1900' mean?",
        },
    ]

    result = await _plan(
        "In the Student master list, how many were admitted around 1900?", provider
    )

    assert result.status == "clarification"
    assert result.review_status == "needs_clarification"
    assert "year range" in (result.clarification_question or "").casefold()


def test_review_prompt_has_scope_followup_context_but_not_internal_ids():
    catalog = static_catalog()
    options = authorized_dataset_options(catalog)
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filters=[{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
        requested_fields=["student_name"],
        result_set_id="rs-secret",
        frame_key="frame-secret",
        action_id="action-secret",
        total_count=9,
    )
    context = ConversationContext(
        active_query=active,
        memory_text="The previous question asked about Garden River.",
    )
    planned = _count_plan()

    text = _review_user_prompt(
        "Which of those were admitted later?",
        planned,
        catalog,
        49,
        authorized_options=options,
        context=context,
    )

    assert "SELECTED LIST: Student master list" in text
    assert "Confirmed deaths" in text
    assert "The previous question asked about Garden River." in text
    active_block = text.split("ACTIVE QUERY:\n", 1)[1].split("\nMEMORY:", 1)[0]
    assert '"goal":"count"' in active_block
    assert "Garden River" in active_block
    assert "datasets" not in active_block
    assert "result_set_id" not in active_block
    assert "frame_key" not in active_block
    assert "action_id" not in active_block
    assert "rs-secret" not in active_block


def test_primary_prompt_also_sanitizes_active_query_ids():
    catalog = static_catalog()
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        result_set_id="rs-secret",
        frame_key="frame-secret",
        action_id="action-secret",
    )
    context = ConversationContext(active_query=active)
    text = _planner_user_prompt(
        "Show the next page",
        catalog=catalog,
        selected_file_id=49,
        authorized_options=authorized_dataset_options(catalog),
        context=context,
        input_mode="text",
        min_confidence=0.75,
    )
    active_block = text.split("ACTIVE QUERY:\n", 1)[1].split("\nMEMORY:", 1)[0]
    assert '"goal":"count"' in active_block
    assert "datasets" not in active_block
    assert "rs-secret" not in active_block
    assert "frame-secret" not in active_block
    assert "action-secret" not in active_block


def test_complete_review_requires_independent_reconstruction():
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate({"verdict": "complete"})


def test_clarification_reason_is_required_and_typed():
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate(
            {
                "verdict": "needs_clarification",
                "missing": ["range"],
                "question": "Which range?",
            }
        )
    with pytest.raises(ValidationError):
        PlanReviewResponse.model_validate(
            {
                "verdict": "needs_clarification",
                "missing": ["range"],
                "reason": "anything_goes",
                "question": "Which range?",
            }
        )


def test_punctuation_only_absent_field_scan_is_always_a_list():
    result = absent_fields_named("?! ...", 49, static_catalog())
    assert result == []
    assert isinstance(result, list)


def test_primary_prompt_supports_flat_or_and_slice3_bounded_nesting():
    assert "Flat sibling OR is also valid" in AI_PLANNER_SYSTEM
    assert "bounded DNF" in AI_PLANNER_SYSTEM
    assert "(A AND B) OR C" in AI_PLANNER_SYSTEM
    assert "clarify" in AI_PLANNER_SYSTEM and "never approximate" in AI_PLANNER_SYSTEM
    assert "Preserve every stated condition as its own filter under filter_logic=and" not in AI_PLANNER_SYSTEM


def test_primary_and_review_have_the_same_missing_bucket_policy():
    primary = AI_PLANNER_SYSTEM.casefold()
    review = AI_REVIEW_SYSTEM.casefold()
    for text in (primary, review):
        assert "distribution" in text or "breakdown" in text
        assert "include_missing=true" in text
        assert "include_missing=false" in text
        assert "superlative" in text


def test_field_coverage_cannot_change_semantic_identity():
    text = " ".join(AI_PLANNER_SYSTEM.casefold().split())
    assert "never semantic evidence" in text
    assert "do not choose another field because it is more populated" in text
    assert "prefer one that is populated" not in text


def test_flat_cross_field_or_reaches_boolean_query_plan():
    filters = [
        {"field": "community", "operator": "EQUALS", "value": "Garden River"},
        {"field": "deceased_status", "operator": "IS_TRUE"},
    ]
    planned = _count_plan(filters=filters, stated_filters=filters, filter_logic="or")
    scope = AccessScope(principal_id="researcher", allowed_file_ids=(49,))
    catalog = static_catalog().for_scope(scope)
    bound = validate_turn_plan(
        bind_scope(planned, 49),
        scope=scope,
        catalog=catalog,
        has_active_query=False,
        strict=True,
    )
    query_plan = build_query_plan(bound, scope=scope, catalog=catalog)
    validate_predicate_contract(query_plan)
    filter_step = next(step for step in query_plan.steps if step.op == "FILTER")
    assert len(filter_step.where) == 1
    assert filter_step.where[0].op == "or"
    assert {item.field for item in filter_step.where[0].items} == {
        "community",
        "deceased_status",
    }


def test_narrowing_provenance_gate_does_not_promote_presentation_differences():
    planned = PlannedTurn.model_validate(
        {
            "normalized_request": "break down by community",
            "actions": [{"type": "query", "goal": "rank", "group_by": ["community"]}],
            "stated_requirements": [_goal_requirement("rank")],
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )
    scope = AccessScope(principal_id="researcher", allowed_file_ids=(49,))
    catalog = static_catalog().for_scope(scope)
    bound = validate_turn_plan(
        bind_scope(planned, 49),
        scope=scope,
        catalog=catalog,
        has_active_query=False,
        strict=True,
    )
    assert unaccounted_narrowing_conditions(planned, bound) == []

    narrowed = _count_plan(
        filters=[{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
        stated_filters=[],
    )
    narrowed_bound = validate_turn_plan(
        bind_scope(narrowed, 49),
        scope=scope,
        catalog=catalog,
        has_active_query=False,
        strict=True,
    )
    assert unaccounted_narrowing_conditions(narrowed, narrowed_bound)
