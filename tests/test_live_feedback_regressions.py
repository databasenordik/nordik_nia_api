"""Concrete plan variants observed in the October live qualification."""

import json

import pytest

from app.normalization.date_expression import date_matches
from app.planning.ai_catalog_coerce import coerce_planned_turn, coerce_stated_requirements
from app.planning.ai_planner import _compact_planned_turn_json
from app.planning.ai_turn_schema import PlannedTurn
from app.planning.catalog import static_catalog
from app.planning.plan_validator import (
    PlanValidationError,
    validate_filter_value,
    validate_query_plan,
)
from app.planning.query_plan_builder import build_query_plan
from app.planning.turn_schema import QueryAction, TurnPlan
from app.planning.turn_validator import validate_turn_plan
from app.security.access_scope import AccessScope


def test_colon_range_operand_is_canonical_on_both_review_sides():
    requirements = [{"kind": "filter", "action_index": 0, "text": "admitted 1940 through 1945",
                     "field": "admitted_date", "operator": "DATE_RANGE", "value": "1940-01-01:1945-12-31"}]
    planned = PlannedTurn(actions=[{"type": "query", "goal": "count", "filters": [
        {"field": "admitted_date", "operator": "DATE_RANGE", "value": "1940-01-01:1945-12-31"}
    ]}], stated_requirements=requirements, final_response="deterministic", confidence=1)
    result = coerce_planned_turn(planned, static_catalog(), 49)
    condition = result.actions[0].filters[0]
    assert condition.value == ["1940-01-01", "1945-12-31"]
    assert date_matches("1942-00-00", condition.operator, condition.value)
    assert not date_matches("1946-01-01", condition.operator, condition.value)
    assert coerce_stated_requirements(planned.stated_requirements, static_catalog(), 49) == result.stated_requirements
    assert coerce_planned_turn(result, static_catalog(), 49) == result


@pytest.mark.parametrize("value", ["1940:1945", [], ["1940"], ["1940", "1941", "1945"],
                                  ["1945-01-01", "1940-01-01"], ["0000-00-00", "1945-01-01"],
                                  ["1940-09-31", "1945-01-01"], ["1940 | 1941", "1945"]])
def test_invalid_date_range_never_executes_as_a_zero_count(value):
    scope = AccessScope(principal_id="test", allowed_file_ids=(49,))
    action = QueryAction(datasets=[49], goal="count", filters=[
        {"field": "admitted_date", "operator": "DATE_RANGE", "value": value}
    ])
    turn = TurnPlan(actions=[action], final_response="deterministic", confidence=1)
    with pytest.raises(PlanValidationError):
        validate_turn_plan(turn, scope=scope, catalog=static_catalog())
    # The lower-level query API must enforce the same operand contract.
    plan = build_query_plan(turn, scope=scope, catalog=static_catalog())
    with pytest.raises(PlanValidationError):
        validate_query_plan(plan, scope, static_catalog())


@pytest.mark.parametrize("value", [["1940", "1945"], ["1940-00-00", "1945-00-00"],
                                  ["1940-01-01", "1945-12-31"]])
def test_valid_date_range_preserves_recorded_precision(value):
    validate_filter_value("DATE_RANGE", value)


@pytest.mark.parametrize("explicit", [None, "last name", "full student name"])
def test_first_page_default_order_does_not_override_explicit_surname_order(explicit):
    question = "List the first 5 students with their names and communities."
    if explicit:
        question += f" Sort by {explicit}."
    planned = PlannedTurn(actions=[{"type": "query", "goal": "list", "sort_by": "last_name",
        "sort_direction": "asc", "window": {"anchor": "start", "size": 5},
        "requested_fields": ["student_name", "first_name", "last_name", "community"]}],
        stated_requirements=[{"kind": "sorting", "action_index": 0, "text": "first five names",
                              "field": "last_name", "direction": "asc"}], final_response="deterministic")
    result = coerce_planned_turn(planned, static_catalog(), 49, question)
    expected = "last_name" if explicit == "last name" else "student_name"
    assert result.actions[0].sort_by == expected
    assert result.stated_requirements[0].field == expected
    assert coerce_stated_requirements(planned.stated_requirements, static_catalog(), 49, question) == result.stated_requirements


@pytest.mark.parametrize("count_field", ["count", "Count"])
def test_rank_count_order_is_not_an_unknown_source_field(count_field):
    planned = PlannedTurn(actions=[{"type": "query", "goal": "rank", "group_by": ["community"],
                                    "sort_by": count_field, "sort_direction": "desc"}],
        stated_requirements=[{"kind": "goal", "action_index": 0, "text": "rank communities", "goal": "rank"},
                             {"kind": "grouping", "action_index": 0, "text": "communities", "field": "community"},
                             {"kind": "sorting", "action_index": 0, "text": "largest count", "field": count_field, "direction": "desc"}],
        final_response="deterministic")
    result = coerce_planned_turn(planned, static_catalog(), 91)
    assert result.actions[0].group_by == ["community"]
    assert result.actions[0].sort_by is None
    assert result.actions[0].sort_direction == "desc"
    assert len(result.stated_requirements) == 2
    assert coerce_stated_requirements(planned.stated_requirements, static_catalog(), 91) == result.stated_requirements
    # An unrelated unknown field must still be rejected by the normal validator.
    other = planned.model_copy(update={"actions": [planned.actions[0].model_copy(update={"sort_by": "invented"})]})
    assert coerce_planned_turn(other, static_catalog(), 91).actions[0].sort_by == "invented"


def test_missing_cause_list_shows_the_missing_source_attribute():
    scope = AccessScope(principal_id="test", allowed_file_ids=(91,))
    turn = TurnPlan(actions=[QueryAction(datasets=[91], goal="list", requested_fields=["student_name"],
        filters=[{"field": "cause_of_death", "operator": "IS_UNKNOWN"}])],
        final_response="deterministic", confidence=1)
    plan = build_query_plan(turn, scope=scope, catalog=static_catalog())
    validate_query_plan(plan, scope, static_catalog())
    assert next(step for step in plan.steps if step.op.value == "PROJECT").fields == ["student_name", "cause_of_death"]
    assert next(step for step in plan.steps if step.op.value == "FILTER").where[0].operator.value == "IS_UNKNOWN"


def test_all_information_review_compacts_only_the_complete_source_projection():
    fields = ["student_name", "cause_of_death", "community"]
    planned = PlannedTurn(actions=[{"type": "query", "goal": "list", "requested_fields": fields,
        "filters": [{"field": "student_name", "operator": "STARTS_WITH", "value": "A"}]}],
        stated_requirements=[{"kind": "projection", "action_index": 0, "text": "all information", "requested_fields": fields}],
        final_response="deterministic")
    payload = json.loads(_compact_planned_turn_json(planned, source_fields=fields))
    assert payload["actions"][0]["requested_fields"] == ["*"]
    assert payload["stated_requirements"][0]["requested_fields"] == ["*"]
    assert payload["actions"][0]["filters"][0]["value"] == "A"
    assert planned.actions[0].requested_fields == fields
    subset = planned.model_copy(update={"actions": [planned.actions[0].model_copy(update={"requested_fields": fields[:2]})]})
    assert json.loads(_compact_planned_turn_json(subset, source_fields=fields))["actions"][0]["requested_fields"] == fields[:2]


@pytest.mark.asyncio
@pytest.mark.parametrize("recovers", [False, True])
async def test_review_transport_retry_is_bounded_and_never_skips_review(monkeypatch, recovers):
    from app.llm.base import StructuredOutputResult
    from app.llm.http_reasoner import TransientProviderError
    from app.planning.ai_planner import AIPlanner

    prompts = []
    delays = []

    async def pause(seconds):
        delays.append(seconds)

    monkeypatch.setattr("app.planning.ai_planner.asyncio.sleep", pause)

    class Provider:
        async def structured_output_with_metadata(self, *, system, user, response_model):
            prompts.append(user)
            if len(prompts) == 1 or not recovers:
                raise TransientProviderError("500 from reasoning provider")
            return StructuredOutputResult(value=response_model.model_validate({
                "verdict": "complete", "reconstructed_requirements": [
                    {"kind": "goal", "action_index": 0, "text": "count rows", "goal": "count"}
                ]
            }))

    planned = PlannedTurn(actions=[{"type": "query", "goal": "count"}], final_response="deterministic")
    result = await AIPlanner(Provider())._review("How many records?", planned, static_catalog(), 49)
    assert result.status == ("ok" if recovers else "unavailable")
    assert result.attempts == 2
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert delays == [0.25]
    if not recovers:
        assert not result.output_metadata


@pytest.mark.parametrize("goal,branches,logic,flatten", [
    ("rank", 1, "and", True), ("compare", 1, "and", False),
    ("rank", 2, "and", False), ("rank", 1, "or", False),
])
def test_single_rank_population_branch_matches_independent_reconstruction(goal, branches, logic, flatten):
    planned = PlannedTurn(actions=[{"type": "query", "goal": goal, "group_by": ["community"]}],
        stated_requirements=[{"kind": "goal", "action_index": 0, "text": "rank communities", "goal": goal},
                             {"kind": "filter_logic", "action_index": 0, "text": "population logic", "filter_logic": logic},
                             *[{"kind": "filter", "action_index": 0, "text": "TB population", "collection": "compare",
                                "branch_label": f"population {index}", "field": "cause_of_death", "operator": "CONTAINS",
                                "value": "tuberculosis"} for index in range(branches)]], final_response="deterministic")
    reconstructed = coerce_stated_requirements(planned.stated_requirements, static_catalog(), 91)
    conditions = [item for item in reconstructed if item.kind == "filter"]
    assert all(item.collection == ("filters" if flatten else "compare") for item in conditions)
    assert all(item.field == "cause_of_death" and item.value == "tuberculosis" for item in conditions)
    assert coerce_planned_turn(planned, static_catalog(), 91).stated_requirements == reconstructed


@pytest.mark.parametrize("field", [None, "cause_of_death", "invented"])
def test_ungrouped_count_aggregate_has_the_same_canonical_row_count(field):
    planned = PlannedTurn(actions=[{"type": "query", "goal": "aggregate",
        "aggregate": {"function": "count", "field": field}, "filters": [
            {"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}]}],
        stated_requirements=[{"kind": "goal", "action_index": 0, "text": "count cells", "goal": "aggregate"},
                             {"kind": "aggregate", "action_index": 0, "text": "count cells", "function": "count", "field": field}],
        final_response="deterministic")
    result = coerce_planned_turn(planned, static_catalog(), 91)
    assert result.actions[0].filters == planned.actions[0].filters
    if field != "invented":
        assert result.actions[0].goal == "count" and result.actions[0].aggregate is None
        assert len(result.stated_requirements) == 1 and result.stated_requirements[0].goal == "count"
    else:
        assert result.actions[0].aggregate.field == "invented"
    assert coerce_stated_requirements(planned.stated_requirements, static_catalog(), 91) == result.stated_requirements


def test_matching_cell_count_cannot_be_answered_by_distinct_values():
    from app.planning.ai_planner import _validate_explicit_request
    scope = AccessScope(principal_id="test", allowed_file_ids=(91,))
    turn = TurnPlan(actions=[QueryAction(datasets=[91], goal="aggregate",
        aggregate={"function": "count_distinct", "field": "cause_of_death"}, filters=[
            {"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}])],
        final_response="deterministic", confidence=1)
    plan = build_query_plan(turn, scope=scope, catalog=static_catalog())
    with pytest.raises(PlanValidationError, match="record count"):
        _validate_explicit_request("How many causes of death contain tuberculosis?", turn, plan, static_catalog())
    _validate_explicit_request("How many distinct causes of death contain tuberculosis?", turn, plan, static_catalog())
