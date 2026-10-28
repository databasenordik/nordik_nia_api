"""Reviewer reconstruction compared by meaning, not by representation.

Derived from the 2026-09-16 live qualification, where most refusals were two model outputs
saying the same thing in different shapes: "first 5" as a window against a limit, an explicit
include_missing=false or filter_logic=and against the implicit default, a correction that
dropped `confidence` and was clarified as unsure. The narrowing guarantees are unchanged and
are exercised here next to each relaxation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings
from app.execution.formatter import _constraint_text
from app.llm.fake import FakeReasoningProvider
from app.planning.ai_planner import plan_user_turn_ai
from app.planning.ai_turn_schema import PlannedTurn, bind_scope
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.plan_schema import FilterOperator, Predicate
from app.planning.requirement_parity import reconcile_reconstruction
from app.security.access_scope import AccessScope


@pytest.fixture
def ai_settings(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    monkeypatch.setenv("PLANNER_REVIEW_MODE", "always")
    monkeypatch.setenv("PLANNER_SELF_REVIEW", "true")
    monkeypatch.setenv("PLANNER_REQUIREMENT_CHECK", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _req(kind: str, **fields) -> dict:
    return {"kind": kind, "action_index": 0, "text": fields.pop("text", kind), **fields}


def _turn(action: dict, requirements: list[dict], **extra) -> PlannedTurn:
    payload = {
        "normalized_request": "request",
        "actions": [{"type": "query", **action}],
        "stated_requirements": requirements,
        "final_response": "deterministic",
        "confidence": 0.95,
    }
    payload.update(extra)
    return PlannedTurn.model_validate(payload)


async def _plan(question: str, provider: FakeReasoningProvider):
    catalog = static_catalog()
    return await plan_user_turn_ai(
        question,
        AccessScope(principal_id="researcher", allowed_file_ids=(49, 91)),
        catalog,
        selected_file_id=49,
        conversation=ConversationContext(),
        reasoner=provider,
        authorized_catalog=catalog,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_complete_cannot_approve_a_trusted_goal_contradiction(ai_settings, repair):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    approval = {"verdict": "complete", "reconstructed_requirements": [_req("goal", goal="count")]}
    requirements = [_req("goal", goal="aggregate"),
                    _req("aggregate", function="count_distinct", field="community")]
    corrected = _turn({"goal": "aggregate", "aggregate": {"function": "count_distinct", "field": "community"}}, requirements)
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, approval,
        {"verdict": "corrected", "missing": ["distinct count"], "plan": corrected.model_dump(),
         "reconstructed_requirements": requirements} if repair else approval]
    result = await _plan("How many distinct communities are recorded?", provider)
    assert result.planner_total_model_calls == 3
    if repair:
        assert result.status == "planned"
        assert any(step.op.value == "COUNT_DISTINCT" for step in result.plan.steps)
    else:
        assert result.plan is None
        assert result.status == "clarification"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_correction_that_keeps_a_goal_contradiction_has_one_bounded_repair(ai_settings, repair):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    wrong = {"verdict": "corrected", "missing": ["distinct count"], "plan": primary.model_dump(),
             "reconstructed_requirements": [_req("goal", goal="count")]}
    requirements = [_req("goal", goal="aggregate"),
                    _req("aggregate", function="count_distinct", field="community")]
    corrected = _turn({"goal": "aggregate", "aggregate": {"function": "count_distinct", "field": "community"}}, requirements)
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, wrong,
        {"verdict": "corrected", "missing": ["distinct count"], "plan": corrected.model_dump(),
         "reconstructed_requirements": requirements} if repair else wrong]
    result = await _plan("How many distinct communities are recorded?", provider)
    assert result.planner_total_model_calls == 3
    assert result.status == ("planned" if repair else "clarification")


def _check(action: dict, reconstructed: list[dict]):
    turn = _turn(action, [_req("goal", goal=action["goal"])])
    parsed = PlannedTurn.model_validate(
        {
            "actions": [{"type": "query", "goal": action["goal"]}],
            "stated_requirements": reconstructed,
            "final_response": "deterministic",
        }
    ).stated_requirements
    return reconcile_reconstruction(list(parsed), bind_scope(turn, 49))


def test_start_window_limit_and_stated_defaults_are_the_same_request():
    check = _check(
        {"goal": "list", "sort_by": "student_name", "limit": 5, "offset": 0},
        [
            _req("goal", goal="list"),
            _req("sorting", field="student_name", direction="asc"),
            _req("window", anchor="start", size=5, cursor=0),
            _req("option", option="include_missing", value=False),
            _req("option", option="exhaustive", value=False),
            _req("filter_logic", filter_logic="and"),
        ],
    )
    assert check.consistent
    assert check.fills == []


def test_end_window_is_not_a_start_limit():
    check = _check(
        {"goal": "list", "sort_by": "student_name", "limit": 5},
        [_req("goal", goal="list"), _req("window", anchor="end", size=5, cursor=0)],
    )
    assert not check.consistent
    assert check.conflicts


def test_narrowing_still_has_to_agree_both_ways():
    missing = _check(
        {"goal": "count"},
        [
            _req("goal", goal="count"),
            _req("filter", field="community", operator="EQUALS", value="Garden River"),
        ],
    )
    assert missing.missing_narrowing
    invented = _check(
        {
            "goal": "count",
            "filters": [{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
        },
        [_req("goal", goal="count")],
    )
    assert invented.extra_narrowing


def test_a_different_goal_or_sort_is_a_conflict_not_a_fill():
    assert _check({"goal": "count"}, [_req("goal", goal="list")]).conflicts
    assert _check(
        {"goal": "list", "sort_by": "student_name", "sort_direction": "asc"},
        [_req("goal", goal="list"), _req("sorting", field="student_name", direction="desc")],
    ).conflicts


@pytest.mark.asyncio
async def test_complete_review_fills_a_projection_the_candidate_left_out(ai_settings):
    primary = _turn(
        {"goal": "list", "sort_by": "student_name", "limit": 5},
        [_req("goal", goal="list"), _req("sorting", field="student_name", direction="asc")],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                _req("goal", goal="list"),
                _req("sorting", field="student_name", direction="asc"),
                _req("window", anchor="start", size=5, cursor=0),
                _req("projection", requested_fields=["student_name", "community"]),
            ],
        },
    ]

    result = await _plan("List the first 5 students with their communities", provider)

    assert result.status == "planned"
    assert result.review_status == "completed_from_reconstruction"
    projected = {
        field for step in result.plan.steps if step.op.value == "PROJECT" for field in step.fields
    }
    assert "community" in projected
    assert result.planner_total_model_calls == 2


@pytest.mark.asyncio
async def test_complete_review_is_provenance_for_a_filter_the_planner_did_not_state(ai_settings):
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _turn({"goal": "count", "filters": [filt]}, [_req("goal", goal="count")])
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                _req("goal", goal="count"),
                _req("filter", collection="filters", **filt),
            ],
        },
    ]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.status == "planned"
    assert result.plan is not None


@pytest.mark.asyncio
async def test_a_correction_without_a_confidence_keeps_the_primary_confidence(ai_settings):
    after = {"field": "admitted_date", "operator": "AFTER", "value": "1965-12-31"}
    primary = _turn(
        {"goal": "count", "filters": [after]},
        [_req("goal", goal="count"), _req("filter", collection="filters", **after)],
    )
    corrected = primary.model_dump(mode="json")
    corrected.pop("confidence")
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["nothing but a restated plan"],
            "reconstructed_requirements": [
                _req("goal", goal="count"),
                _req("filter", collection="filters", **after),
            ],
            "plan": corrected,
        },
    ]

    result = await _plan("How many students were admitted after 1965?", provider)

    assert result.status == "planned"
    assert result.plan is not None


@pytest.mark.asyncio
async def test_a_correction_that_fills_confidence_with_zero_is_not_a_doubt(ai_settings):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    corrected = primary.model_dump(mode="json")
    corrected["confidence"] = 0.0
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["unsure"],
            "reconstructed_requirements": [_req("goal", goal="count")],
            "plan": corrected,
        },
    ]

    result = await _plan("How many students are there?", provider)

    assert result.status == "planned"
    assert result.plan is not None


@pytest.mark.asyncio
async def test_unreconstructed_top_n_on_a_ranking_is_restored_not_refused(ai_settings):
    primary = _turn(
        {"goal": "rank", "group_by": ["community"]},
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)],
    )
    corrected = _turn(
        {"goal": "rank", "group_by": ["community"], "top_n": 1, "limit": 1},
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["top_n"],
            "reconstructed_requirements": [
                _req("goal", goal="rank"),
                _req("grouping", field="community", position=0),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan("Which community appears most often?", provider)

    assert result.status == "planned"
    assert all(step.top_n != 1 for step in result.plan.steps)


@pytest.mark.asyncio
async def test_unreconstructed_grouping_in_a_correction_is_still_refused(ai_settings):
    primary = _turn({"goal": "rank", "group_by": ["community"]}, [_req("goal", goal="rank")])
    corrected = _turn(
        {"goal": "rank", "group_by": ["community", "gender"]}, [_req("goal", goal="rank")]
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["second grouping"],
            "reconstructed_requirements": [_req("goal", goal="rank")],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan("Which community appears most often?", provider)

    assert result.status == "clarification"
    assert "introduced semantics" in result.detail


def test_an_answer_names_the_value_condition_it_was_counted_under():
    text = _constraint_text(
        [
            Predicate(field="deceased_status", operator=FilterOperator.EQUALS, value="yes"),
            Predicate(field="community", operator=FilterOperator.IS_UNKNOWN),
        ],
        static_catalog(),
        (49,),
    )
    assert 'deceased status is "yes"' in text
    assert "no recorded" in text


async def _plan_on(question: str, provider: FakeReasoningProvider, file_id: int):
    catalog = static_catalog()
    return await plan_user_turn_ai(
        question,
        AccessScope(
            principal_id="researcher",
            allowed_file_ids=(49, 91, 93, 94),
            can_use_private_files=True,
        ),
        catalog,
        selected_file_id=file_id,
        conversation=ConversationContext(),
        reasoner=provider,
        authorized_catalog=catalog,
    )


@pytest.mark.asyncio
async def test_a_search_for_the_selected_lists_own_name_is_not_a_condition(ai_settings):
    primary = _turn(
        {"goal": "list", "search_text": "additional deaths", "limit": 5},
        [
            _req("goal", goal="list"),
            _req("search_text", search_text="additional deaths"),
            _req("numeric_slot", slot="limit", value=5),
        ],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                _req("goal", goal="list"),
                _req("search_text", search_text="additional deaths"),
                _req("numeric_slot", slot="limit", value=5),
            ],
        },
    ]

    result = await _plan_on("List the first 5 additional death names", provider, 93)

    assert result.status == "planned"
    assert not [step for step in result.plan.steps if step.query or step.where]


@pytest.mark.asyncio
async def test_a_correction_cannot_add_a_presence_condition_the_question_never_names(ai_settings):
    primary = _turn(
        {"goal": "list", "sort_by": "student_name", "limit": 5},
        [_req("goal", goal="list"), _req("numeric_slot", slot="limit", value=5)],
    )
    known = {"field": "death_date", "operator": "IS_KNOWN"}
    corrected = _turn(
        {"goal": "list", "sort_by": "student_name", "limit": 5, "filters": [known]},
        [_req("goal", goal="list"), _req("filter", collection="filters", **known)],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["only records that are deaths"],
            "reconstructed_requirements": [
                _req("goal", goal="list"),
                _req("numeric_slot", slot="limit", value=5),
                _req("filter", collection="filters", **known),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan_on("List the first 5 additional death names", provider, 93)

    assert result.status == "planned"
    assert not [step for step in result.plan.steps if step.where]


@pytest.mark.asyncio
async def test_a_presence_condition_the_question_asks_about_is_kept(ai_settings):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    unknown = {"field": "cause_of_death", "operator": "IS_UNKNOWN"}
    corrected = _turn(
        {"goal": "count", "filters": [unknown]},
        [_req("goal", goal="count"), _req("filter", collection="filters", **unknown)],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["no known cause of death"],
            "reconstructed_requirements": [
                _req("goal", goal="count"),
                _req("filter", collection="filters", **unknown),
            ],
            "plan": corrected.model_dump(mode="json"),
        },
    ]

    result = await _plan_on(
        "How many confirmed deaths have no known cause of death?", provider, 91
    )

    assert result.status == "planned"
    assert any(step.where for step in result.plan.steps)


@pytest.mark.asyncio
async def test_a_correction_cannot_answer_an_absent_field_from_a_different_one(ai_settings):
    primary = _turn(
        {"goal": "count", "filters": [{"field": "student_number", "operator": "IS_KNOWN"}]},
        [
            _req("goal", goal="count"),
            _req("filter", collection="filters", field="student_number", operator="IS_KNOWN"),
        ],
    )
    substitute = {"field": "student_name", "operator": "IS_KNOWN"}
    corrected = _turn(
        {"goal": "count", "filters": [substitute]},
        [_req("goal", goal="count"), _req("filter", collection="filters", **substitute)],
    )
    review = {
        "verdict": "corrected",
        "missing": ["student_number is not a field in this list"],
        "reconstructed_requirements": [
            _req("goal", goal="count"),
            _req("filter", collection="filters", **substitute),
        ],
        "plan": corrected.model_dump(mode="json"),
    }
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, review]

    result = await _plan_on("How many potential records have a student number?", provider, 94)

    assert result.status in {"unsupported_fields", "clarification"}
    assert result.plan is None


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_independent_reconstruction_retry_is_bounded(ai_settings, repair):
    filt = {"field": "community", "operator": "EQUALS", "value": "Garden River"}
    primary = _turn({"goal": "count", "filters": [filt]},
        [_req("goal", goal="count"), _req("filter", collection="filters", **filt)])
    bad = {"verdict": "complete", "reconstructed_requirements": [_req("goal", goal="count")]}
    good = {"verdict": "complete", "reconstructed_requirements": [
        _req("goal", goal="count"), _req("filter", collection="filters", **filt)]}
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, bad, good if repair else bad]
    result = await _plan("How many students are from Garden River?", provider)
    assert result.status == ("planned" if repair else "clarification")
    assert result.review_attempts == 2
    assert result.planner_total_model_calls == 3
    assert result.review_status == ("repaired_reconstruction" if repair else "inconsistent")


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", [True, False])
async def test_review_action_index_retry_preserves_requirements(ai_settings, repair):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    malformed = primary.model_dump(mode="json")
    malformed["stated_requirements"][0]["action_index"] = 1
    bad_review = {
        "verdict": "corrected",
        "missing": ["correct action index"],
        "reconstructed_requirements": [_req("goal", goal="count")],
        "plan": malformed,
    }
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, bad_review,
        {"verdict": "complete", "reconstructed_requirements": [_req("goal", goal="count")]}
        if repair else bad_review]
    result = await _plan("How many students are there?", provider)
    assert result.status == ("planned" if repair else "planner_invalid_output")
    assert result.review_attempts == 2
    assert result.planner_total_model_calls == 3


@pytest.mark.asyncio
async def test_one_transient_review_failure_is_retried(ai_settings):
    from app.llm.base import TransientReasoningError

    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        TransientReasoningError("503 from reasoning provider: server is shutting down"),
        {"verdict": "complete", "reconstructed_requirements": [_req("goal", goal="count")]},
    ]

    result = await _plan("How many students are there?", provider)

    assert result.status == "planned"
    assert result.review_attempts == 2
    assert result.planner_total_model_calls == 3


@pytest.mark.asyncio
async def test_review_that_stays_unavailable_still_fails_closed(ai_settings):
    from app.llm.base import TransientReasoningError

    primary = _turn({"goal": "count"}, [_req("goal", goal="count")])
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        TransientReasoningError("503"),
        TransientReasoningError("503"),
    ]

    result = await _plan("How many students are there?", provider)

    assert result.status == "planner_unavailable"
    assert result.plan is None


@pytest.mark.asyncio
async def test_a_stated_window_and_projection_the_action_omits_are_written_in(ai_settings):
    primary = _turn(
        {"goal": "list", "sort_by": "student_name"},
        [
            _req("goal", goal="list"),
            _req("sorting", field="student_name", direction="asc"),
            _req("window", anchor="start", size=5, cursor=0),
            _req("projection", requested_fields=["student_name", "community"]),
        ],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "complete",
            "reconstructed_requirements": [
                _req("goal", goal="list"),
                _req("sorting", field="student_name", direction="asc"),
                _req("numeric_slot", slot="limit", value=5),
                _req("projection", requested_fields=["student_name", "community"]),
            ],
        },
    ]

    result = await _plan("List the first 5 students with their names and communities", provider)

    assert result.status == "planned"
    assert any(step.window_size == 5 or step.limit == 5 for step in result.plan.steps)
    projected = {
        field for step in result.plan.steps if step.op.value == "PROJECT" for field in step.fields
    }
    assert "community" in projected


@pytest.mark.asyncio
async def test_a_stated_filter_the_action_omits_is_still_a_dropped_condition(ai_settings):
    primary = _turn(
        {"goal": "count"},
        [
            _req("goal", goal="count"),
            _req("filter", collection="filters", field="community", operator="EQUALS", value="Garden River"),
        ],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "needs_clarification",
            "missing": ["community"],
            "reason": "other",
            "question": "Which community?",
        },
    ]

    result = await _plan("How many students are from Garden River?", provider)

    assert result.plan is None


@pytest.mark.asyncio
async def test_a_primary_below_the_confidence_threshold_is_still_clarified(ai_settings):
    primary = _turn({"goal": "count"}, [_req("goal", goal="count")], confidence=0.2)
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary]

    result = await _plan("How many students are there?", provider)

    assert result.status == "clarification"
    assert result.plan is None


def test_distinct_without_a_grouping_is_sent_back_for_review_not_guessed():
    """Validation refuses it with a reviewable code; the builder never picks a field for it."""
    from app.planning.plan_validator import PlanValidationError
    from app.planning.turn_validator import validate_turn_plan

    turn = _turn(
        {
            "goal": "distinct",
            "filters": [{"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}],
            "requested_fields": ["cause_of_death"],
        },
        [_req("goal", goal="distinct")],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_turn_plan(
            bind_scope(turn, 91),
            scope=AccessScope(principal_id="researcher", allowed_file_ids=(49, 91)),
            catalog=static_catalog(),
            has_active_query=False,
            strict=True,
        )
    assert exc.value.code == "op_contract"


def test_a_value_list_compares_as_a_set():
    check = _check(
        {
            "goal": "count",
            "filters": [{"field": "gender", "operator": "IN", "value": ["F", "female"]}],
        },
        [
            _req("goal", goal="count"),
            _req("filter", field="gender", operator="IN", value=["female", "F", "female"]),
        ],
    )
    assert check.consistent


def test_an_overlong_review_note_is_shortened_not_fatal():
    from app.planning.ai_turn_schema import PlanReviewResponse

    parsed = PlanReviewResponse.model_validate(
        {
            "verdict": "needs_clarification",
            "missing": ["x. " * 300],
            "reason": "other",
            "reconstructed_requirements": [_req("goal", goal="count", text="y " * 300)],
        }
    )
    assert len(parsed.root.missing[0]) <= 400
    assert len(parsed.root.reconstructed_requirements[0].text) <= 400


def _compile(action: dict, file_id: int = 91):
    from app.planning.ai_catalog_coerce import coerce_planned_turn
    from app.planning.query_plan_builder import build_query_plan
    from app.planning.turn_validator import validate_turn_plan

    catalog = static_catalog()
    scope = AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))
    turn = coerce_planned_turn(_turn(action, [_req("goal", goal=action["goal"])]), catalog, file_id)
    validated = validate_turn_plan(
        bind_scope(turn, file_id), scope=scope, catalog=catalog, has_active_query=False, strict=True
    )
    return build_query_plan(validated, scope=scope, catalog=catalog)


def test_a_lone_comparison_branch_on_a_ranking_still_filters_the_ranking():
    tb = [{"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}]
    plan = _compile(
        {
            "goal": "rank",
            "group_by": ["community"],
            "compare": [{"label": "tuberculosis", "filters": tb}],
        }
    )
    assert any(
        predicate.field == "cause_of_death" for step in plan.steps for predicate in step.where
    )


def test_several_comparison_branches_on_a_ranking_are_refused_not_ignored():
    from app.planning.plan_validator import PlanValidationError

    with pytest.raises(PlanValidationError) as exc:
        _compile(
            {
                "goal": "rank",
                "group_by": ["community"],
                "compare": [
                    {"label": "a", "filters": [{"field": "gender", "operator": "EQUALS", "value": "M"}]},
                    {"label": "b", "filters": [{"field": "gender", "operator": "EQUALS", "value": "F"}]},
                ],
            }
        )
    assert exc.value.code == "op_contract"


def test_yes_or_the_same_name_again_accepts_a_single_name_offer_only():
    from app.planning.trusted_overrides import accepts_offered_name

    offer = 'No record matches "Alexander knegs". Did you mean Alexander KNAGGS (Manitoba Vital Stats)?'
    several = 'No record matches "Ann". Did you mean one of these: Anne A; Anna B?'
    assert accepts_offered_name("yes", offer)
    assert accepts_offered_name("Give me information about Alexander knegs?", offer)
    assert not accepts_offered_name("How many deaths in 1900?", offer)
    assert not accepts_offered_name("yes", several)
    assert not accepts_offered_name("yes", "There are 82 records in the Confirmed deaths.")


def test_a_top_one_ranking_does_not_claim_its_group_is_the_whole_list():
    from app.data_gateway.protocol import GroupRow
    from app.execution.executor import ActionResult
    from app.execution.formatter import _format_ranked

    action = ActionResult(
        action_id="a0",
        goal="rank",
        file_ids=(91,),
        ranked_groups=[GroupRow(file_id=91, label="1882", secondary_label=None, count=5)],
        group_fields=("death_date_year", None),
        requested_top_n=1,
    )
    text = _format_ranked(action, "Confirmed deaths")
    assert "5 of 5" not in text
    assert "5 records" in text


# --------------------------------------------------------------------- 2026-09-25 run
# Every gate refusal in the 222-observation qualification of 2026-09-25 was one thing: a
# presence condition written by one model call and not the other, on a question where the
# two encodings select identical records. These lock the equivalence and the guarantees
# that must survive it.


def _grouped(include_missing: bool, with_presence: bool, field: str = "community") -> PlannedTurn:
    action: dict = {"goal": "rank", "group_by": [field], "top_n": 1}
    requirements = [_req("goal", goal="rank"), _req("grouping", field=field, position=0)]
    if include_missing:
        action["include_missing"] = True
        requirements.append(_req("option", option="include_missing", value=True))
    if with_presence:
        action["filters"] = [{"field": field, "operator": "IS_KNOWN"}]
        requirements.append(_req("filter", collection="filters", field=field, operator="IS_KNOWN"))
    return _turn(action, requirements)


def _reconstruction(items: list[dict]):
    return list(
        PlannedTurn.model_validate(
            {
                "actions": [{"type": "query", "goal": "rank"}],
                "stated_requirements": items,
                "final_response": "deterministic",
            }
        ).stated_requirements
    )


def test_presence_on_a_grouped_field_is_the_same_request_as_dropping_missing_values():
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    # The plan states it as a filter, the reviewer states nothing: the same records.
    assert reconcile_reconstruction(recon, bind_scope(_grouped(False, True), 49)).consistent
    # And the other way round: the reviewer states the filter the plan never wrote.
    with_filter = _reconstruction(
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("filter", collection="filters", field="community", operator="IS_KNOWN"),
        ]
    )
    assert reconcile_reconstruction(with_filter, bind_scope(_grouped(False, False), 49)).consistent


def test_presence_still_narrows_when_the_grouping_keeps_missing_values():
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    check = reconcile_reconstruction(recon, bind_scope(_grouped(True, True), 49))
    assert not check.consistent
    assert "IS_KNOWN" in check.detail()


def test_presence_on_a_field_the_query_does_not_group_by_still_narrows():
    plan = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "filters": [{"field": "birth_date", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("filter", collection="filters", field="birth_date", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    check = reconcile_reconstruction(recon, bind_scope(plan, 49))
    assert not check.consistent
    assert "birth_date" in check.detail()


def test_a_correction_may_spell_the_superlative_rule_as_a_presence_filter():
    from app.planning.requirement_parity import unverified_correction_changes

    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    assert not unverified_correction_changes(
        recon, bind_scope(_grouped(False, False), 49), bind_scope(_grouped(False, True), 49)
    )


def test_a_correction_adding_an_unrelated_condition_is_still_unverified():
    from app.planning.requirement_parity import unverified_correction_changes

    corrected = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "top_n": 1,
            "filters": [{"field": "deceased_status", "operator": "IS_TRUE"}],
        },
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("filter", collection="filters", field="deceased_status", operator="IS_TRUE"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    unverified = unverified_correction_changes(
        recon, bind_scope(_grouped(False, False), 49), bind_scope(corrected, 49)
    )
    assert [item.field for item in unverified] == ["deceased_status"]


def test_a_correction_that_rewrites_no_action_is_agreement():
    from app.planning.ai_planner import _correction_changes_nothing

    candidate = _turn({"goal": "count"}, [_req("goal", goal="count")])
    restated = _turn(
        # Spelling out a default the candidate left unset changes no record.
        {"goal": "count", "include_missing": False, "exhaustive": False},
        [_req("goal", goal="count"), _req("option", option="include_missing", value=False)],
    )
    changed = _turn(
        {"goal": "count", "filters": [{"field": "community", "operator": "IS_KNOWN"}]},
        [_req("goal", goal="count")],
    )
    assert _correction_changes_nothing(restated, candidate)
    assert not _correction_changes_nothing(changed, candidate)
    assert not _correction_changes_nothing(None, candidate)


def _two_counts() -> PlannedTurn:
    return PlannedTurn.model_validate(
        {
            "normalized_request": "How many students were discharged after 1965?",
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [
                        {"field": "discharged_date", "operator": "AFTER", "value": "1965-12-31"}
                    ],
                },
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [{"field": "discharged_date", "operator": "IS_KNOWN"}],
                },
            ],
            "stated_requirements": [
                _req("goal", goal="count"),
                _req(
                    "filter",
                    collection="filters",
                    field="discharged_date",
                    operator="AFTER",
                    value="1965-12-31",
                ),
                {"kind": "goal", "action_index": 1, "goal": "count", "text": "recorded discharges"},
                {
                    "kind": "filter",
                    "action_index": 1,
                    "collection": "filters",
                    "field": "discharged_date",
                    "operator": "IS_KNOWN",
                    "text": "discharge date is recorded",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )


def _asked_for_the_count():
    return _reconstruction(
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="discharged_date",
                operator="AFTER",
                value="1965-12-31",
            ),
        ]
    )


def test_an_unreconstructed_action_restating_another_condition_is_dropped():
    from app.planning.ai_planner import _without_unrequested_context_actions

    recon = _asked_for_the_count()
    trimmed = _without_unrequested_context_actions(_two_counts(), recon)
    assert trimmed is not None
    planned, reconstructed_out = trimmed
    assert len(planned.actions) == 1
    assert {item.action_index for item in planned.stated_requirements} == {0}
    assert reconcile_reconstruction(reconstructed_out, bind_scope(planned, 49)).consistent


def test_a_goal_on_the_extra_count_does_not_keep_a_restated_presence_filter():
    from app.planning.ai_planner import _without_unrequested_context_actions
    from app.planning.ai_turn_schema import GoalRequirement

    recon = _asked_for_the_count() + [
        GoalRequirement(action_index=1, text="recorded discharges", goal="count")
    ]
    trimmed = _without_unrequested_context_actions(_two_counts(), recon)
    assert trimmed is not None
    planned, reconstructed_out = trimmed
    assert len(planned.actions) == 1
    assert reconcile_reconstruction(reconstructed_out, bind_scope(planned, 49)).consistent


def test_grouped_aggregate_with_missing_values_dropped_matches_a_presence_filter():
    plan = _turn(
        {
            "goal": "aggregate",
            "group_by": ["mapping_location"],
            "aggregate": {"function": "count"},
            "sort_direction": "desc",
            "top_n": 1,
            "filters": [{"field": "mapping_location", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="aggregate"),
            _req("grouping", field="mapping_location", position=0),
            _req("filter", collection="filters", field="mapping_location", operator="IS_KNOWN"),
        ],
    )
    without = _turn(
        {
            "goal": "aggregate",
            "group_by": ["mapping_location"],
            "aggregate": {"function": "count"},
            "sort_direction": "desc",
            "top_n": 1,
        },
        [
            _req("goal", goal="aggregate"),
            _req("grouping", field="mapping_location", position=0),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="aggregate"),
            _req("grouping", field="mapping_location", position=0),
        ]
    )
    assert reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent
    from app.planning.requirement_parity import unverified_correction_changes

    assert not unverified_correction_changes(
        recon, bind_scope(without, 49), bind_scope(plan, 49)
    )


def test_an_action_the_reviewer_reconstructed_is_never_dropped():
    from app.planning.ai_planner import _without_unrequested_context_actions

    both = PlannedTurn.model_validate(
        {
            "actions": [
                {"type": "query", "goal": "count"},
                {"type": "query", "goal": "count"},
            ],
            "stated_requirements": [
                _req("goal", goal="count"),
                {
                    "kind": "filter",
                    "action_index": 1,
                    "collection": "filters",
                    "field": "discharged_date",
                    "operator": "IS_KNOWN",
                    "text": "and how many have one recorded",
                },
            ],
            "final_response": "deterministic",
        }
    ).stated_requirements
    assert _without_unrequested_context_actions(_two_counts(), list(both)) is None


def test_an_action_with_a_condition_of_its_own_is_never_dropped():
    from app.planning.ai_planner import _without_unrequested_context_actions

    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [
                        {"field": "discharged_date", "operator": "AFTER", "value": "1965-12-31"}
                    ],
                },
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [{"field": "deceased_status", "operator": "IS_KNOWN"}],
                },
            ],
            "stated_requirements": [_req("goal", goal="count")],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    )
    assert _without_unrequested_context_actions(planned, _asked_for_the_count()) is None


def test_a_presence_condition_on_the_denominator_is_not_the_grouping():
    """A share is two populations; the one under the line is not the grouped one."""
    plan = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "denominator_filters": [{"field": "community", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req(
                "filter",
                collection="denominator_filters",
                field="community",
                operator="IS_KNOWN",
            ),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_presence_on_the_field_an_aggregate_measures_is_the_same_request():
    """The oldest recorded age is the same number with or without the empty records."""
    plan = _turn(
        {
            "goal": "aggregate",
            "aggregate": {"function": "max", "field": "age"},
            "filters": [{"field": "age", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="aggregate"),
            _req("aggregate", function="max", field="age"),
            _req("filter", collection="filters", field="age", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="aggregate"), _req("aggregate", function="max", field="age")]
    )
    assert reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_presence_still_narrows_a_count_of_records():
    """How many records there are is a question about the population itself."""
    plan = _turn(
        {
            "goal": "aggregate",
            "aggregate": {"function": "count"},
            "filters": [{"field": "age", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="aggregate"),
            _req("aggregate", function="count"),
            _req("filter", collection="filters", field="age", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="aggregate"), _req("aggregate", function="count")]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_presence_on_another_field_than_the_one_measured_still_narrows():
    plan = _turn(
        {
            "goal": "aggregate",
            "aggregate": {"function": "max", "field": "age"},
            "filters": [{"field": "birth_date", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="aggregate"),
            _req("aggregate", function="max", field="age"),
            _req("filter", collection="filters", field="birth_date", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="aggregate"), _req("aggregate", function="max", field="age")]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def _count_with(filters: list[dict], requirements: list[dict], **extra) -> PlannedTurn:
    return _turn({"goal": "count", "filters": filters, **extra}, requirements)


def test_a_comparison_already_requires_the_value_to_be_recorded():
    plan = _count_with(
        [
            {"field": "admitted_date", "operator": "BEFORE", "value": "1900-01-01"},
            {"field": "admitted_date", "operator": "IS_KNOWN"},
        ],
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
            _req("filter", collection="filters", field="admitted_date", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
        ]
    )
    assert reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_presence_beside_an_alternative_is_not_implied_by_it():
    """Under OR the presence condition widens the population instead of restating it."""
    plan = _count_with(
        [
            {"field": "admitted_date", "operator": "BEFORE", "value": "1900-01-01"},
            {"field": "admitted_date", "operator": "IS_KNOWN"},
        ],
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
            _req("filter", collection="filters", field="admitted_date", operator="IS_KNOWN"),
            _req("filter_logic", filter_logic="or"),
        ],
        filter_logic="or",
    )
    recon = _reconstruction(
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
            _req("filter_logic", filter_logic="or"),
        ]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_a_negative_condition_does_not_imply_the_value_is_recorded():
    """"Not tuberculosis" is true of a record that recorded no cause at all."""
    plan = _count_with(
        [
            {"field": "community", "operator": "NOT_EQUALS", "value": "Bkejwanong"},
            {"field": "community", "operator": "IS_KNOWN"},
        ],
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="community",
                operator="NOT_EQUALS",
                value="Bkejwanong",
            ),
            _req("filter", collection="filters", field="community", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="community",
                operator="NOT_EQUALS",
                value="Bkejwanong",
            ),
        ]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_presence_on_a_field_no_condition_mentions_still_narrows():
    plan = _count_with(
        [
            {"field": "admitted_date", "operator": "BEFORE", "value": "1900-01-01"},
            {"field": "community", "operator": "IS_KNOWN"},
        ],
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
            _req("filter", collection="filters", field="community", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="count"),
            _req(
                "filter",
                collection="filters",
                field="admitted_date",
                operator="BEFORE",
                value="1900-01-01",
            ),
        ]
    )
    check = reconcile_reconstruction(recon, bind_scope(plan, 49))
    assert not check.consistent
    assert "community" in check.detail()


def test_a_ranking_that_also_states_a_count_is_still_a_grouping():
    """A rank compiles to a grouping and never reads its aggregate spec."""
    plan = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "aggregate": {"function": "count"},
            "filters": [{"field": "community", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("aggregate", function="count"),
            _req("filter", collection="filters", field="community", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("aggregate", function="count"),
        ]
    )
    assert reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_a_grouped_aggregate_over_another_field_still_narrows():
    """When the aggregate is the answer, the filter changes the records it reads."""
    plan = _turn(
        {
            "goal": "aggregate",
            "group_by": ["community"],
            "aggregate": {"function": "avg", "field": "age"},
            "filters": [{"field": "community", "operator": "IS_KNOWN"}],
        },
        [
            _req("goal", goal="aggregate"),
            _req("grouping", field="community", position=0),
            _req("aggregate", function="avg", field="age"),
            _req("filter", collection="filters", field="community", operator="IS_KNOWN"),
        ],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="aggregate"),
            _req("grouping", field="community", position=0),
            _req("aggregate", function="avg", field="age"),
        ]
    )
    assert not reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_a_group_threshold_of_one_keeps_every_group_and_so_states_nothing():
    plan = _turn(
        {"goal": "rank", "group_by": ["community"], "having_min_count": 1},
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("having_min_count", having_min_count=1, text="groups of at least one"),
        ],
    )
    recon = _reconstruction(
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    )
    assert reconcile_reconstruction(recon, bind_scope(plan, 49)).consistent


def test_a_real_group_threshold_is_still_carried_into_the_plan():
    """Only a threshold of one is a default. A real one the reviewer read in the question
    is filled into the plan rather than dropped with it."""
    plan = _turn(
        {"goal": "rank", "group_by": ["community"]},
        [_req("goal", goal="rank"), _req("grouping", field="community", position=0)],
    )
    recon = _reconstruction(
        [
            _req("goal", goal="rank"),
            _req("grouping", field="community", position=0),
            _req("having_min_count", having_min_count=2, text="more than one student"),
        ]
    )
    check = reconcile_reconstruction(recon, bind_scope(plan, 49))
    assert [item.text for item in check.fills] == ["more than one student"]


@pytest.mark.asyncio
async def test_correction_adding_an_unreconstructed_condition_executes_without_it(ai_settings):
    """The reviewer's invention is dropped, not the turn.

    Live, "the most rare causes of death" came back as a valid ranking plus a correction
    that attached a condition excluding "unknown" which the same review never reconstructed.
    Refusing told the user only that a correction could not be verified; executing it would
    have answered a narrower question. The ranking the reconstruction does state survives.
    """
    requirements = [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    primary = _turn({"goal": "rank", "group_by": ["community"]}, requirements)
    invented = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "filters": [{"field": "community", "operator": "EQUALS", "value": "Lake Town"}],
        },
        requirements,
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["one community"],
            "plan": invented.model_dump(),
            "reconstructed_requirements": requirements,
        },
    ]
    result = await _plan("Which communities appear least often?", provider)
    assert result.status == "planned"
    assert not any(predicate.field == "community" for step in result.plan.steps for predicate in step.where)


@pytest.mark.asyncio
async def test_a_correction_whose_own_requirement_states_the_condition_keeps_it(ai_settings):
    """The same shape with provenance still executes the condition."""
    stated = _req("filter", collection="filters", field="community",
                  operator="EQUALS", value="Lake Town")
    requirements = [_req("goal", goal="rank"), _req("grouping", field="community", position=0)]
    primary = _turn({"goal": "rank", "group_by": ["community"]}, requirements)
    corrected = _turn(
        {
            "goal": "rank",
            "group_by": ["community"],
            "filters": [{"field": "community", "operator": "EQUALS", "value": "Lake Town"}],
        },
        [*requirements, stated],
    )
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["one community"],
            "plan": corrected.model_dump(),
            "reconstructed_requirements": [*requirements, stated],
        },
    ]
    result = await _plan("Which recorded communities appear least often?", provider)
    assert result.status == "planned"
    assert any(predicate.field == "community" for step in result.plan.steps for predicate in step.where)


@pytest.mark.asyncio
async def test_an_unreconstructed_free_text_search_is_dropped_too(ai_settings):
    """A text search narrows exactly as a filter does, so it goes the same way."""
    requirements = [_req("goal", goal="count")]
    primary = _turn({"goal": "count"}, requirements)
    corrected = _turn({"goal": "count", "search_text": "hospital"}, requirements)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["hospital"],
            "plan": corrected.model_dump(),
            "reconstructed_requirements": requirements,
        },
    ]
    result = await _plan("How many students are recorded?", provider)
    assert result.status == "planned"
    assert all(not step.query for step in result.plan.steps)


@pytest.mark.asyncio
async def test_an_unreconstructed_statistic_still_fails_closed(ai_settings):
    """Dropping only ever removes conditions; a measurement nobody reconstructed refuses."""
    requirements = [_req("goal", goal="list")]
    primary = _turn({"goal": "list"}, requirements)
    corrected = _turn(
        {"goal": "aggregate", "aggregate": {"function": "avg", "field": "age"}}, requirements
    )
    unverified = {
        "verdict": "corrected",
        "missing": ["average age"],
        "plan": corrected.model_dump(),
        "reconstructed_requirements": requirements,
    }
    provider = FakeReasoningProvider()
    provider.structured_queue = [primary, unverified, unverified]
    result = await _plan("List the students.", provider)
    assert result.plan is None
    assert result.status == "clarification"


@pytest.mark.asyncio
async def test_a_review_operator_outside_the_schema_gets_one_repair_not_a_dead_turn(ai_settings):
    """The auditor was right about the records and wrong about the token.

    "Students explicitly linked to parents Sabine Williams and Emily Watson" was planned with
    CONTAINS_ANY; the review said both names must match and spelled it CONTAINS_ALL, which the
    schema does not define. The whole turn failed on the token. One precise retry keeps the
    independent check and the stricter reading it asked for.
    """
    both = [
        {"field": "parents_names", "operator": "CONTAINS", "value": "Sabine Williams"},
        {"field": "parents_names", "operator": "CONTAINS", "value": "Emily Watson"},
    ]
    requirements = [
        _req("goal", goal="list"),
        *[_req("filter", collection="filters", **item) for item in both],
    ]
    primary = _turn(
        {
            "goal": "list",
            "filters": [{"field": "parents_names", "operator": "CONTAINS_ANY",
                         "value": ["Sabine Williams", "Emily Watson"]}],
        },
        [_req("goal", goal="list"),
         _req("filter", collection="filters", field="parents_names", operator="CONTAINS_ANY",
              value=["Sabine Williams", "Emily Watson"])],
    )
    corrected = _turn({"goal": "list", "filters": both, "filter_logic": "and"}, requirements)
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        primary,
        {
            "verdict": "corrected",
            "missing": ["both parents, not either"],
            "plan": corrected.model_dump(),
            "reconstructed_requirements": [
                _req("goal", goal="list"),
                _req("filter", collection="filters", field="parents_names",
                     operator="CONTAINS_ALL", value=["Sabine Williams", "Emily Watson"]),
            ],
        },
        {
            "verdict": "corrected",
            "missing": ["both parents, not either"],
            "plan": corrected.model_dump(),
            "reconstructed_requirements": requirements,
        },
    ]
    result = await _plan(
        "Which students are explicitly linked to parents Sabine Williams and Emily Watson?",
        provider,
    )
    assert result.status == "planned"
    # One extra review call, not a failed turn, and the strict reading it asked for.
    assert result.review_attempts == 2
    values = {
        str(predicate.value)
        for step in result.plan.steps
        for predicate in step.where
        if predicate.field == "parents_names"
    }
    assert values == {"Sabine Williams", "Emily Watson"}
