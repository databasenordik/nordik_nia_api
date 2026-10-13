"""Named condition-loss cases from docs/pipeline-fixes.md must survive compilation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.ai_turn_schema import PlannedTurn, bind_scope
from app.planning.catalog import static_catalog
from app.planning.plan_schema import FilterOperator
from app.planning.query_plan_builder import build_query_plan
from app.planning.turn_validator import validate_turn_plan
from app.security.access_scope import AccessScope


def _scope(*file_ids: int) -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=tuple(file_ids))


def _compile(planned: PlannedTurn, file_id: int):
    catalog = static_catalog().for_scope(_scope(file_id))
    bound = bind_scope(planned, file_id)
    validated = validate_turn_plan(bound, scope=_scope(file_id), catalog=catalog, strict=True)
    return build_query_plan(validated, scope=_scope(file_id), catalog=catalog)


def _leaves(plan):
    found = []
    for step in plan.steps:
        for predicate in step.where:
            found.extend(predicate.leaves())
    return found


def _has(plan, field: str, operator: str, value=None) -> bool:
    want = FilterOperator(operator)
    for leaf in _leaves(plan):
        if leaf.field != field or leaf.operator != want:
            continue
        if value is None:
            return True
        if str(leaf.value).casefold() == str(value).casefold():
            return True
    return False


def test_q20_deceased_and_missing_community():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [
                        {"field": "deceased_status", "operator": "IS_TRUE"},
                        {"field": "community", "operator": "IS_UNKNOWN"},
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "deceased",
                    "field": "deceased_status",
                    "operator": "IS_TRUE",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "no community",
                    "field": "community",
                    "operator": "IS_UNKNOWN",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 49)
    assert _has(plan, "deceased_status", "IS_TRUE")
    assert _has(plan, "community", "IS_UNKNOWN")


def test_q37_deceased_and_missing_death_details():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "count",
                    "filters": [
                        {"field": "deceased_status", "operator": "IS_TRUE"},
                        {"field": "death_details", "operator": "IS_UNKNOWN"},
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "deceased",
                    "field": "deceased_status",
                    "operator": "IS_TRUE",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "no death details",
                    "field": "death_details",
                    "operator": "IS_UNKNOWN",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 49)
    assert _has(plan, "deceased_status", "IS_TRUE")
    assert _has(plan, "death_details", "IS_UNKNOWN")


def test_q44_deceased_parents_and_siblings():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "list",
                    "filters": [
                        {"field": "deceased_status", "operator": "IS_TRUE"},
                        {"field": "parents_names", "operator": "IS_KNOWN"},
                        {"field": "siblings", "operator": "IS_KNOWN"},
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "list", "goal": "list"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "deceased",
                    "field": "deceased_status",
                    "operator": "IS_TRUE",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "parents known",
                    "field": "parents_names",
                    "operator": "IS_KNOWN",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "siblings known",
                    "field": "siblings",
                    "operator": "IS_KNOWN",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 49)
    assert _has(plan, "deceased_status", "IS_TRUE")
    assert _has(plan, "parents_names", "IS_KNOWN")
    assert _has(plan, "siblings", "IS_KNOWN")


def test_q57_tuberculosis_grouped_by_community():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "rank",
                    "group_by": ["community"],
                    "filters": [{"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}],
                    "top_n": 8,
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "rank", "goal": "rank"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "tuberculosis",
                    "field": "cause_of_death",
                    "operator": "CONTAINS",
                    "value": "tuberculosis",
                },
                {
                    "kind": "grouping",
                    "action_index": 0,
                    "text": "community",
                    "field": "community",
                    "position": 0,
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 91)
    assert _has(plan, "cause_of_death", "CONTAINS", "tuberculosis")
    assert any(step.fields == ["community"] or "community" in step.fields for step in plan.steps)


def test_q62_unknown_cause_of_death():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "list",
                    "filters": [{"field": "cause_of_death", "operator": "IS_UNKNOWN"}],
                    "requested_fields": ["student_name", "cause_of_death"],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "list", "goal": "list"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "unknown cause",
                    "field": "cause_of_death",
                    "operator": "IS_UNKNOWN",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 91)
    assert _has(plan, "cause_of_death", "IS_UNKNOWN")


def test_q73_walpole_projection():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "list",
                    "filters": [{"field": "community", "operator": "CONTAINS", "value": "Walpole Island"}],
                    "requested_fields": ["student_name", "age_at_death", "death_date", "cause_of_death"],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "list", "goal": "list"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "Walpole Island",
                    "field": "community",
                    "operator": "CONTAINS",
                    "value": "Walpole Island",
                },
                {
                    "kind": "projection",
                    "action_index": 0,
                    "text": "age death cause",
                    "requested_fields": ["student_name", "age_at_death", "death_date", "cause_of_death"],
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 91)
    assert _has(plan, "community", "CONTAINS", "Walpole Island")
    projected = [name for step in plan.steps for name in step.fields]
    assert {"age_at_death", "death_date", "cause_of_death"} <= set(projected)


def test_q75_tb_communities_with_more_than_one():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "rank",
                    "group_by": ["community"],
                    "having_min_count": 2,
                    "filters": [{"field": "cause_of_death", "operator": "CONTAINS", "value": "tuberculosis"}],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "rank", "goal": "rank"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "tuberculosis",
                    "field": "cause_of_death",
                    "operator": "CONTAINS",
                    "value": "tuberculosis",
                },
                {
                    "kind": "having_min_count",
                    "action_index": 0,
                    "text": "more than one",
                    "having_min_count": 2,
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    )
    plan = _compile(planned, 91)
    assert _has(plan, "cause_of_death", "CONTAINS", "tuberculosis")
    assert any(step.having_min_count == 2 for step in plan.steps)


def test_three_condition_garden_river_admitted_tuberculosis():
    planned = PlannedTurn.model_validate(
        {
            "actions": [
                {
                    "type": "query",
                    "goal": "list",
                    "filters": [
                        {"field": "community", "operator": "EQUALS", "value": "Garden River"},
                        {"field": "first_admitted_date", "operator": "AFTER", "value": 1905},
                        {
                            "field": "cause_of_death",
                            "operator": "CONTAINS",
                            "value": "tuberculosis",
                        },
                    ],
                }
            ],
            "stated_requirements": [
                {"kind": "goal", "action_index": 0, "text": "list", "goal": "list"},
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "Garden River",
                    "field": "community",
                    "operator": "EQUALS",
                    "value": "Garden River",
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "after 1905",
                    "field": "first_admitted_date",
                    "operator": "AFTER",
                    "value": 1905,
                },
                {
                    "kind": "filter",
                    "action_index": 0,
                    "text": "died from tuberculosis",
                    "field": "cause_of_death",
                    "operator": "CONTAINS",
                    "value": "tuberculosis",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.93,
        }
    )
    plan = _compile(planned, 91)
    assert _has(plan, "community", "EQUALS", "Garden River")
    assert _has(plan, "first_admitted_date", "AFTER", 1905)
    assert _has(plan, "cause_of_death", "CONTAINS", "tuberculosis")
