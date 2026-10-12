"""Slice 5 latency/cost regressions for the AI planner path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.llm.fake import FakeReasoningProvider
from app.llm.http_reasoner import _compact_response_schema
from app.planning.ai_planner import (
    _compact_planned_turn_json,
    _safe_active_query_json,
    plan_user_turn_ai,
)
from app.planning.ai_turn_schema import (
    GoalRequirement,
    PlannedQueryAction,
    PlannedTurn,
    PlanReviewResponse,
)
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.plan_schema import PlanOp
from app.planning.turn_schema import ActiveQuery, FilterSpec
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


@pytest.fixture
def ai_settings(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    monkeypatch.setenv("PLANNER_REVIEW_MODE", "always")
    monkeypatch.setenv("PLANNER_SELF_REVIEW", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


@pytest.mark.asyncio
async def test_greeting_uses_zero_model_calls(ai_settings):
    provider = FakeReasoningProvider()
    result = await _plan("Hello", provider)
    assert result.status == "conversational"
    assert result.planner_total_model_calls == 0
    assert result.review_status == "skipped"
    assert provider.structured_calls == 0


@pytest.mark.asyncio
async def test_describe_available_fields_uses_zero_model_calls(ai_settings):
    provider = FakeReasoningProvider()
    result = await _plan("What fields are available?", provider)
    assert result.status == "conversational"
    assert result.planner_total_model_calls == 0
    assert provider.structured_calls == 0


@pytest.mark.asyncio
async def test_next_page_uses_zero_model_calls_and_keeps_active_query(ai_settings):
    provider = FakeReasoningProvider()
    context = ConversationContext(
        active_query=ActiveQuery(
            datasets=[49],
            goal="list",
            filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
            requested_fields=["student_name", "community"],
            limit=25,
            offset=0,
        )
    )
    result = await _plan("next page", provider, context=context)
    assert result.status == "planned"
    assert result.planner_total_model_calls == 0
    assert provider.structured_calls == 0
    assert result.plan is not None
    filter_step = next(step for step in result.plan.steps if step.op == PlanOp.FILTER)
    assert any(getattr(item, "field", None) == "community" for item in filter_step.where)


@pytest.mark.asyncio
async def test_verify_previous_uses_zero_model_calls_and_preserves_count(ai_settings):
    provider = FakeReasoningProvider()
    context = ConversationContext(
        active_query=ActiveQuery(
            datasets=[49],
            goal="count",
            filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
        )
    )
    result = await _plan("Are you sure?", provider, context=context)
    assert result.status == "planned"
    assert result.planner_total_model_calls == 0
    assert provider.structured_calls == 0
    assert result.plan is not None
    assert any(step.op == PlanOp.COUNT for step in result.plan.steps)


@pytest.mark.asyncio
async def test_next_page_on_a_non_list_is_not_taken_by_the_zero_call_fast_path(ai_settings):
    provider = FakeReasoningProvider()
    context = ConversationContext(active_query=ActiveQuery(datasets=[49], goal="count"))
    result = await _plan("next page", provider, context=context)
    assert result.planner_total_model_calls >= 1
    assert provider.structured_calls >= 1


@pytest.mark.asyncio
async def test_new_research_query_still_uses_primary_and_review(ai_settings):
    provider = FakeReasoningProvider()
    result = await _plan("How many students were admitted in 1905?", provider)
    assert result.status in {"planned", "clarification"}
    assert result.planner_total_model_calls == 2
    assert provider.structured_calls == 2


def test_active_query_prompt_drops_inert_defaults_but_keeps_semantics():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
        result_set_id="secret",
    )
    text = _safe_active_query_json(ConversationContext(active_query=active))
    assert '"goal":"count"' in text
    assert "Garden River" in text
    assert "result_set_id" not in text
    assert '"filter_logic":"and"' not in text
    assert '"requested_fields":[]' not in text


def test_review_candidate_serialization_is_compact_and_keeps_discriminators():
    planned = PlannedTurn(
        actions=[PlannedQueryAction(goal="count")],
        stated_requirements=[GoalRequirement(action_index=0, text="count records", goal="count")],
        final_response="deterministic",
        confidence=0.95,
    )
    compact = _compact_planned_turn_json(planned)
    full = planned.model_dump_json()
    parsed = json.loads(compact)
    assert parsed["actions"][0]["type"] == "query"
    assert parsed["stated_requirements"][0]["kind"] == "goal"
    assert len(compact) < len(full) * 0.4


def test_http_structured_schema_removes_only_annotations_and_is_materially_smaller():
    for model in (PlannedTurn, PlanReviewResponse):
        raw = model.model_json_schema()
        compact = _compact_response_schema(model)
        raw_text = json.dumps(raw, separators=(",", ":"))
        compact_text = json.dumps(compact, separators=(",", ":"))
        assert len(compact_text) < len(raw_text) * 0.82
        assert compact.get("required") == raw.get("required")
        assert compact.get("additionalProperties") == raw.get("additionalProperties")
        assert "$defs" in compact
        assert "title" not in compact_text
        # Structural discriminators and constraints survive the annotation strip.
        if model is PlannedTurn:
            assert compact["$defs"]["GoalRequirement"]["properties"]["kind"]["const"] == "goal"
            assert "kind" in compact["$defs"]["GoalRequirement"]["required"]
