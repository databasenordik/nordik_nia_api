import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, Predicate, QueryPlan


def test_plan_requires_unique_step_ids():
    with pytest.raises(ValueError):
        QueryPlan(
            scope=PlanScope(file_ids=(49,)),
            steps=[
                PlanStep(id="a", op=PlanOp.FILTER, input="current_records"),
                PlanStep(id="a", op=PlanOp.COUNT, input="a"),
            ],
        )


def test_plan_rejects_extra_keys():
    with pytest.raises(ValueError):
        QueryPlan.model_validate(
            {
                "scope": {"file_ids": [49]},
                "steps": [{"id": "a", "op": "COUNT", "input": "current_records", "sql": "SELECT 1"}],
            }
        )


def test_golden_plan_shape():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="current_records",
                where=[Predicate(field="student_name", operator="STARTS_WITH", value="S")],
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    assert plan.op_names() == ["FILTER", "COUNT"]
    assert plan.scope.file_ids == (49,)
