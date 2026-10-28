"""Regressions derived from the 2026-09-15 live qualification review failures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.llm.fake import FakeReasoningProvider
from app.planning.ai_planner import plan_user_turn_ai
from app.planning.ai_turn_schema import PlannedTurn
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
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


def _goal(goal: str) -> dict:
    return {
        "kind": "goal",
        "action_index": 0,
        "text": f"goal {goal}",
        "goal": goal,
    }


def _filter(field: str, operator: str, value=None) -> dict:
    return {
        "kind": "filter",
        "action_index": 0,
        "text": f"{field} {operator} {value}",
        "collection": "filters",
        "field": field,
        "operator": operator,
        "value": value,
    }


def _plan(*, limit=None, presentation=None, with_filter=False, state_filter=True) -> PlannedTurn:
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    action = {"type": "query", "goal": "list"}
    if limit is not None:
        action["limit"] = limit
    if presentation is not None:
        action["presentation"] = presentation
    if with_filter:
        action["filters"] = [filt]
    requirements = [_goal("list")]
    if with_filter and state_filter:
        requirements.append(_filter("community", "EQUALS", "Garden River"))
    return PlannedTurn.model_validate(
        {
            "normalized_request": "list students",
            "actions": [action],
            "stated_requirements": requirements,
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )


async def _run(provider: FakeReasoningProvider):
    catalog = static_catalog()
    return await plan_user_turn_ai(
        "List 5 students from Garden River",
        _scope(),
        catalog,
        selected_file_id=49,
        conversation=ConversationContext(),
        reasoner=provider,
        authorized_catalog=catalog,
    )


@pytest.mark.asyncio
async def test_review_correction_preserves_unrestated_unchanged_limit(ai_settings):
    """A full replacement must not re-prove unchanged primary semantics.

    This reproduces the live pattern behind errors such as
    ``correction contains unstated semantics: limit 5``.
    """
    primary = _plan(limit=5, with_filter=False)
    corrected = _plan(limit=5, with_filter=True)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community Garden River"],
            "reconstructed_requirements": [
                _goal("list"),
                _filter("community", "EQUALS", "Garden River"),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _run(provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_review_correction_does_not_execute_new_unreconstructed_limit(ai_settings):
    """A reviewer still cannot silently introduce a result-changing limit.

    The verified part of the correction -- the reconstructed community filter -- is kept; the
    limit it added without reconstructing is put back to the primary plan's value rather than
    refusing the whole turn.
    """
    primary = _plan(with_filter=False)
    corrected = _plan(limit=5, with_filter=True)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community Garden River"],
            "reconstructed_requirements": [
                _goal("list"),
                _filter("community", "EQUALS", "Garden River"),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _run(provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert all(step.limit != 5 for step in result.plan.steps)
    filters = [
        predicate
        for step in result.plan.steps
        for predicate in step.where
    ]
    assert any(item.field == "community" for item in filters)


@pytest.mark.asyncio
async def test_review_correction_allows_presentation_only_materialization(ai_settings):
    """Formatting chosen while materializing a full corrected object is not query meaning."""
    primary = _plan(limit=5, with_filter=False)
    corrected = _plan(limit=5, presentation="numbered_list", with_filter=True)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community Garden River"],
            "reconstructed_requirements": [
                _goal("list"),
                _filter("community", "EQUALS", "Garden River"),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _run(provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"


@pytest.mark.asyncio
async def test_reconstruction_can_supply_narrowing_provenance_for_correction(ai_settings):
    """The reviewer reconstruction, not duplicated stated text, is correction provenance."""
    primary = _plan(limit=5, with_filter=False)
    corrected = _plan(limit=5, with_filter=True, state_filter=False)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["community Garden River"],
            "reconstructed_requirements": [
                _goal("list"),
                _filter("community", "EQUALS", "Garden River"),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _run(provider)

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"

@pytest.mark.asyncio
async def test_review_reconstruction_is_catalog_coerced_before_narrowing_comparison(ai_settings):
    """Trusted filter coercion must not create a false provenance mismatch.

    The action/replacement side canonicalizes deceased_status EQUALS "deceased" to IS_TRUE.
    The independently reconstructed requirement starts in the model's pre-coercion form and
    must receive the same canonicalization before comparison.
    """
    primary = PlannedTurn.model_validate(
        {
            "normalized_request": "count deceased students",
            "actions": [{"type": "query", "goal": "count"}],
            "stated_requirements": [_goal("count")],
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )
    raw_filter = _filter("deceased_status", "EQUALS", "deceased")
    corrected = PlannedTurn.model_validate(
        {
            "normalized_request": "count deceased students",
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [
                        {
                            "field": "deceased_status",
                            "operator": "EQUALS",
                            "value": "deceased",
                        }
                    ],
                }
            ],
            "stated_requirements": [_goal("count"), raw_filter],
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["deceased"],
            "reconstructed_requirements": [_goal("count"), raw_filter],
            "plan": corrected.model_dump(mode="json"),
        },
    ]
    catalog = static_catalog()
    result = await plan_user_turn_ai(
        "How many deceased students are there?",
        _scope(),
        catalog,
        selected_file_id=49,
        conversation=ConversationContext(),
        reasoner=provider,
        authorized_catalog=catalog,
    )

    assert result.status == "planned"
    assert result.plan is not None
    assert result.review_status == "corrected"
