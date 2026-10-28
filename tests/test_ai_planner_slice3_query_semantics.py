"""Slice 3 regressions for Boolean query semantics and follow-up fidelity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.execution.turn import _active_query_from_frame
from app.memory.types import QueryFrame, QueryState
from app.planning.ai_planner import AI_PLANNER_SYSTEM, AI_REVIEW_SYSTEM
from app.planning.ai_turn_schema import FilterRequirement, PlannedTurn, bind_scope
from app.planning.catalog import static_catalog
from app.planning.plan_schema import FilterOperator, Predicate
from app.planning.plan_validator import PlanValidationError
from app.planning.predicate_contract import validate_predicate_contract
from app.planning.query_plan_builder import build_query_plan, compiled_query_actions
from app.planning.requirement_parity import (
    check_requirement_parity,
    requirements_not_reconstructed,
)
from app.planning.trusted_overrides import _apply_random_sample, _bare_research_dump
from app.planning.turn_schema import (
    ActiveQuery,
    AggregateSpec,
    FilterGroupSpec,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    ResultWindow,
    TurnPlan,
    VerifyPreviousAction,
)
from app.planning.turn_validator import validate_turn_plan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49,))


def _catalog():
    return static_catalog().for_scope(_scope())


def _f(field: str, operator: str, value=None) -> FilterSpec:
    return FilterSpec(field=field, operator=operator, value=value)


def _dnf_action(*, goal: str = "count") -> QueryAction:
    # (community=Garden River OR community=Batchewana) AND deceased
    return QueryAction(
        datasets=[49],
        goal=goal,
        filter_groups=[
            FilterGroupSpec(
                filters=[
                    _f("community", "EQUALS", "Garden River"),
                    _f("deceased_status", "IS_TRUE"),
                ]
            ),
            FilterGroupSpec(
                filters=[
                    _f("community", "EQUALS", "Batchewana"),
                    _f("deceased_status", "IS_TRUE"),
                ]
            ),
        ],
    )


def test_bounded_dnf_compiles_to_nested_predicate_tree():
    turn = TurnPlan(actions=[_dnf_action()], final_response="deterministic", confidence=0.95)
    validated = validate_turn_plan(
        turn, scope=_scope(), catalog=_catalog(), strict=True, has_active_query=False
    )
    plan = build_query_plan(validated, scope=_scope(), catalog=_catalog())
    validate_predicate_contract(plan)

    step = next(item for item in plan.steps if item.op == "FILTER")
    assert len(step.where) == 1
    root = step.where[0]
    assert root.op == "or"
    assert len(root.items) == 2
    assert all(item.op == "and" for item in root.items)
    assert [
        {(leaf.field, str(leaf.operator), leaf.value) for leaf in branch.items}
        for branch in root.items
    ] == [
        {
            ("community", "EQUALS", "Garden River"),
            ("deceased_status", "IS_TRUE", None),
        },
        {
            ("community", "EQUALS", "Batchewana"),
            ("deceased_status", "IS_TRUE", None),
        },
    ]


def test_contradiction_is_rejected_inside_one_dnf_branch_but_not_across_alternatives():
    contradictory = QueryAction(
        datasets=[49],
        goal="count",
        filter_groups=[
            FilterGroupSpec(
                filters=[
                    _f("community", "EQUALS", "Garden River"),
                    _f("community", "EQUALS", "Batchewana"),
                ]
            ),
            FilterGroupSpec(filters=[_f("deceased_status", "IS_TRUE")]),
        ],
    )
    turn = TurnPlan(actions=[contradictory], final_response="deterministic", confidence=0.95)
    plan = build_query_plan(turn, scope=_scope(), catalog=_catalog())
    with pytest.raises(PlanValidationError, match="contradictory"):
        validate_predicate_contract(plan)

    alternatives = QueryAction(
        datasets=[49],
        goal="count",
        filter_groups=[
            FilterGroupSpec(filters=[_f("community", "EQUALS", "Garden River")]),
            FilterGroupSpec(filters=[_f("community", "EQUALS", "Batchewana")]),
        ],
    )
    plan = build_query_plan(
        TurnPlan(actions=[alternatives], final_response="deterministic", confidence=0.95),
        scope=_scope(),
        catalog=_catalog(),
    )
    validate_predicate_contract(plan)


def test_query_rejects_mixed_flat_and_dnf_surfaces():
    with pytest.raises(ValidationError, match="cannot mix"):
        QueryAction(
            datasets=[49],
            goal="count",
            filters=[_f("community", "EQUALS", "Garden River")],
            filter_groups=[FilterGroupSpec(filters=[_f("deceased_status", "IS_TRUE")])],
        )


def test_dnf_is_bounded_for_structured_output_safety():
    groups = [
        {"filters": [{"field": "community", "operator": "EQUALS", "value": f"Place {i}"}]}
        for i in range(9)
    ]
    with pytest.raises(ValidationError):
        QueryAction(datasets=[49], goal="count", filter_groups=groups)


def test_refining_flat_or_distributes_new_condition_instead_of_widening():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filters=[
            _f("community", "EQUALS", "Garden River"),
            _f("community", "EQUALS", "Batchewana"),
        ],
        filter_logic="or",
    )
    turn = TurnPlan(
        actions=[
            ModifyPreviousAction(
                changes=[_f("deceased_status", "IS_TRUE")], change_mode="refine"
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    query = compiled_query_actions(
        turn, scope=_scope(), catalog=_catalog(), active_query=active
    )[0]
    assert query.filters == []
    assert query.filter_logic == "and"
    assert len(query.filter_groups) == 2
    assert all(
        {item.field for item in group.filters} == {"community", "deceased_status"}
        for group in query.filter_groups
    )

    plan = build_query_plan(
        turn, scope=_scope(), catalog=_catalog(), active_query=active
    )
    root = next(item for item in plan.steps if item.op == "FILTER").where[0]
    assert root.op == "or"
    assert all(branch.op == "and" for branch in root.items)


def test_refining_existing_dnf_adds_condition_to_every_alternative():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filter_groups=[group.model_copy(deep=True) for group in _dnf_action().filter_groups],
    )
    turn = TurnPlan(
        actions=[
            ModifyPreviousAction(
                changes=[_f("admitted_date", "YEAR_EQUALS", 1905)], change_mode="refine"
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    query = compiled_query_actions(
        turn, scope=_scope(), catalog=_catalog(), active_query=active
    )[0]
    assert all(
        any(item.field == "admitted_date" and item.value == 1905 for item in group.filters)
        for group in query.filter_groups
    )


def test_explicit_replace_keeps_historical_same_field_edit_semantics():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filters=[_f("admitted_date", "YEAR_EQUALS", 1905)],
    )
    turn = TurnPlan(
        actions=[
            ModifyPreviousAction(
                changes=[_f("admitted_date", "YEAR_EQUALS", 1910)], change_mode="replace"
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    query = compiled_query_actions(
        turn, scope=_scope(), catalog=_catalog(), active_query=active
    )[0]
    assert [(item.field, item.value) for item in query.filters] == [("admitted_date", 1910)]


def test_replace_on_flat_or_removes_stale_same_field_alternatives():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filters=[
            _f("community", "EQUALS", "Garden River"),
            _f("community", "EQUALS", "Batchewana"),
        ],
        filter_logic="or",
    )
    query = compiled_query_actions(
        TurnPlan(
            actions=[
                ModifyPreviousAction(
                    changes=[_f("community", "EQUALS", "Sault Ste. Marie")],
                    change_mode="replace",
                )
            ],
            final_response="deterministic",
            confidence=0.95,
        ),
        scope=_scope(),
        catalog=_catalog(),
        active_query=active,
    )[0]
    assert [(item.field, item.value) for item in query.filters] == [
        ("community", "Sault Ste. Marie")
    ]


def test_replace_in_dnf_changes_only_branches_that_contain_the_field():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filter_groups=[
            FilterGroupSpec(
                filters=[
                    _f("community", "EQUALS", "Garden River"),
                    _f("deceased_status", "IS_TRUE"),
                ]
            ),
            FilterGroupSpec(filters=[_f("cause_of_death", "CONTAINS", "tuberculosis")]),
        ],
    )
    query = compiled_query_actions(
        TurnPlan(
            actions=[
                ModifyPreviousAction(
                    changes=[_f("community", "EQUALS", "Batchewana")],
                    change_mode="replace",
                )
            ],
            final_response="deterministic",
            confidence=0.95,
        ),
        scope=_scope(),
        catalog=_catalog(),
        active_query=active,
    )[0]
    assert {(item.field, item.value) for item in query.filter_groups[0].filters} == {
        ("community", "Batchewana"),
        ("deceased_status", None),
    }
    assert [(item.field, item.value) for item in query.filter_groups[1].filters] == [
        ("cause_of_death", "tuberculosis")
    ]


def test_replace_absent_field_in_dnf_fails_closed_instead_of_narrowing_every_branch():
    active = ActiveQuery(
        datasets=[49],
        goal="count",
        filter_groups=[group.model_copy(deep=True) for group in _dnf_action().filter_groups],
    )
    turn = TurnPlan(
        actions=[
            ModifyPreviousAction(
                changes=[_f("cause_of_death", "CONTAINS", "tuberculosis")],
                change_mode="replace",
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    with pytest.raises(PlanValidationError, match="cannot replace absent"):
        compiled_query_actions(turn, scope=_scope(), catalog=_catalog(), active_query=active)


@pytest.mark.parametrize(
    "goal,extra",
    [
        ("count", {}),
        ("rank", {"group_by": ["community"], "top_n": 5}),
        (
            "aggregate",
            {"aggregate": AggregateSpec(function="count_distinct", field="community")},
        ),
        ("duplicates", {"group_by": ["community"], "having_min_count": 2}),
        (
            "interval",
            {
                "interval_start": "admitted_date",
                "interval_end": "discharged_date",
                "interval_min_days": 30,
            },
        ),
        ("completeness", {}),
    ],
)
def test_verify_previous_replays_same_goal_and_semantics(goal, extra):
    active = ActiveQuery(
        datasets=[49],
        goal=goal,
        filters=[_f("community", "EQUALS", "Garden River")],
        requested_fields=["student_name", "community"],
        **extra,
    )
    turn = TurnPlan(
        actions=[VerifyPreviousAction()], final_response="deterministic", confidence=0.95
    )
    query = compiled_query_actions(
        turn, scope=_scope(), catalog=_catalog(), active_query=active
    )[0]
    assert query.goal == goal
    assert query.verify_previous is True
    assert query.filters == active.filters
    assert query.group_by == active.group_by
    assert query.top_n == active.top_n
    assert query.aggregate == active.aggregate
    assert query.interval_start == active.interval_start
    assert query.interval_end == active.interval_end


def test_verify_previous_preserves_window_and_page_for_list():
    active = ActiveQuery(
        datasets=[49],
        goal="list",
        filters=[_f("community", "EQUALS", "Garden River")],
        requested_fields=["student_name", "community"],
        sort_by="student_name",
        sort_direction="asc",
        window=ResultWindow(anchor="end", size=100, cursor=25),
        limit=25,
        offset=25,
        exhaustive=True,
    )
    query = compiled_query_actions(
        TurnPlan(
            actions=[VerifyPreviousAction()], final_response="deterministic", confidence=0.95
        ),
        scope=_scope(),
        catalog=_catalog(),
        active_query=active,
    )[0]
    assert query.goal == "list"
    assert query.window == active.window
    assert query.offset == 25
    assert query.limit == 25


def test_explicit_stale_action_target_does_not_fall_back_to_unrelated_active_query():
    active = ActiveQuery(
        datasets=[49], goal="count", filters=[_f("community", "EQUALS", "Garden River")]
    )
    turn = TurnPlan(
        actions=[VerifyPreviousAction(target_action_id="a-stale")],
        final_response="deterministic",
        confidence=0.95,
    )
    with pytest.raises(PlanValidationError, match="stale"):
        compiled_query_actions(
            turn, scope=_scope(), catalog=_catalog(), active_query=active, frames={}
        )


def test_explicit_unknown_restore_frame_does_not_fall_back_to_current_topic():
    active = ActiveQuery(
        datasets=[49], goal="count", filters=[_f("community", "EQUALS", "Garden River")]
    )
    turn = TurnPlan(
        actions=[ModifyPreviousAction(restore_frame="missing topic")],
        final_response="deterministic",
        confidence=0.95,
    )
    with pytest.raises(PlanValidationError, match="stale"):
        compiled_query_actions(
            turn,
            scope=_scope(),
            catalog=_catalog(),
            active_query=active,
            frames={"master": active},
        )


def test_active_query_frame_preserves_source_boolean_and_analytic_semantics():
    source = QueryAction(
        datasets=[49],
        goal="aggregate",
        filter_groups=[group.model_copy(deep=True) for group in _dnf_action().filter_groups],
        aggregate=AggregateSpec(function="count_distinct", field="community"),
        denominator_filters=[_f("admitted_date", "IS_KNOWN")],
        requested_fields=["student_name", "community"],
    )
    state = QueryState(
        file_ids=(49,),
        filters=[
            Predicate(
                op="or",
                items=[
                    Predicate(
                        op="and",
                        items=[
                            Predicate(
                                field="community",
                                operator=FilterOperator.EQUALS,
                                value="Garden River",
                            ),
                            Predicate(
                                field="deceased_status",
                                operator=FilterOperator.IS_TRUE,
                            ),
                        ],
                    )
                ],
            )
        ],
        goal="aggregate",
        source_action=source,
    )
    frame = QueryFrame(frame_key="master", topic="test", query=state)
    active = _active_query_from_frame(frame)
    assert active is not None
    assert active.filter_groups == source.filter_groups
    assert active.aggregate == source.aggregate
    assert active.denominator_filters == source.denominator_filters
    assert active.requested_fields == source.requested_fields


def test_dnf_requirements_are_checked_with_group_coordinates():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filter_groups": [
                        {
                            "filters": [
                                {"field": "community", "operator": "EQUALS", "value": "Garden River"},
                                {"field": "deceased_status", "operator": "IS_TRUE"},
                            ]
                        },
                        {
                            "filters": [
                                {"field": "community", "operator": "EQUALS", "value": "Batchewana"},
                                {"field": "deceased_status", "operator": "IS_TRUE"},
                            ]
                        },
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "Garden River",
                    "collection": "filter_groups",
                    "group_index": 0,
                    "field": "community",
                    "operator": "EQUALS",
                    "value": "Garden River",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "deceased branch one",
                    "collection": "filter_groups",
                    "group_index": 0,
                    "field": "deceased_status",
                    "operator": "IS_TRUE",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "Batchewana",
                    "collection": "filter_groups",
                    "group_index": 1,
                    "field": "community",
                    "operator": "EQUALS",
                    "value": "Batchewana",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "deceased branch two",
                    "collection": "filter_groups",
                    "group_index": 1,
                    "field": "deceased_status",
                    "operator": "IS_TRUE",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.95,
        }
    )
    bound = validate_turn_plan(
        bind_scope(planned, 49),
        scope=_scope(),
        catalog=_catalog(),
        has_active_query=False,
        strict=True,
    )
    check_requirement_parity(planned, bound)


def test_slice3_prompts_define_dnf_and_refine_vs_replace_contract():
    primary = " ".join(AI_PLANNER_SYSTEM.split())
    review = " ".join(AI_REVIEW_SYSTEM.split())
    assert "bounded DNF" in primary
    assert "filter_groups" in primary
    assert "(A OR B) AND C" in primary
    assert "change_mode=refine" in primary
    assert "change_mode=replace" in primary
    assert "filter_groups" in review
    assert "refine narrows" in review


def test_dnf_review_requirement_matching_is_group_order_insensitive():
    candidate = [
        FilterRequirement(
            action_index=0, text="Garden River", collection="filter_groups", group_index=0,
            field="community", operator="EQUALS", value="Garden River"
        ),
        FilterRequirement(
            action_index=0, text="deceased", collection="filter_groups", group_index=0,
            field="deceased_status", operator="IS_TRUE"
        ),
        FilterRequirement(
            action_index=0, text="Batchewana", collection="filter_groups", group_index=1,
            field="community", operator="EQUALS", value="Batchewana"
        ),
    ]
    # Same Boolean expression, but the reviewer emits the OR branches in reverse order
    # and reverses predicate order inside the first conjunction.
    reconstructed = [
        candidate[2].model_copy(update={"group_index": 0}),
        candidate[1].model_copy(update={"group_index": 1}),
        candidate[0].model_copy(update={"group_index": 1}),
    ]
    assert requirements_not_reconstructed(candidate, reconstructed) == []


def test_random_sample_override_preserves_dnf_population():
    base = TurnPlan(
        actions=[_dnf_action(goal="list")], final_response="deterministic", confidence=0.95
    )
    sampled = _apply_random_sample(
        base, "show a random sample of 5", catalog=_catalog()
    )
    assert sampled is not None
    query = sampled.query_actions()[0]
    assert query.filter_groups == base.query_actions()[0].filter_groups
    assert query.filters == []
    assert query.sample is True
    assert query.limit == 5


def test_dnf_query_is_not_misclassified_as_bare_research_dump():
    turn = TurnPlan(
        actions=[_dnf_action(goal="list")], final_response="deterministic", confidence=0.95
    )
    assert _bare_research_dump(turn) is False
