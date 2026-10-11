import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.catalog import static_catalog
from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, Predicate, QueryPlan
from app.planning.plan_validator import PlanValidationError, validate_query_plan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _plan(**kwargs) -> QueryPlan:
    defaults = dict(
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
    defaults.update(kwargs)
    return QueryPlan(**defaults)


def test_valid_plan_passes():
    validate_query_plan(_plan(), _scope(), static_catalog())


def test_rejects_out_of_scope_file():
    plan = _plan(scope=PlanScope(file_ids=(93,)))
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog())
    assert exc.value.code == "access_restricted"


def test_rejects_unknown_field():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="current_records",
                where=[Predicate(field="password_hash", operator="EQUALS", value="x")],
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog())
    assert exc.value.code == "unknown_field"


def test_rejects_disallowed_operator():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="current_records",
                where=[Predicate(field="deceased_status", operator="STARTS_WITH", value="t")],
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog())
    assert exc.value.code == "operator"


def test_rejects_broken_dag():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[PlanStep(id="result", op=PlanOp.COUNT, input="missing")],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog())
    assert exc.value.code == "dag"


def test_rejects_sql_in_step_id():
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(
            QueryPlan(
                scope=PlanScope(file_ids=(49,)),
                steps=[PlanStep(id="select star", op=PlanOp.COUNT, input="current_records")],
            ),
            _scope(),
            static_catalog(),
        )
    assert exc.value.code == "forbidden_sql"


def test_rejects_step_file_ids_outside_scope():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="current_records",
                where=[Predicate(field="student_name", operator="STARTS_WITH", value="S")],
                file_ids=(93,),
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog().for_scope(_scope()))
    assert exc.value.code == "access_restricted"


def test_search_value_may_contain_select():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(
                id="fts",
                op=PlanOp.FULL_TEXT_SEARCH,
                input="current_records",
                query="the select committee",
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="fts"),
        ],
    )
    validate_query_plan(plan, _scope(), static_catalog().for_scope(_scope()))


def test_filter_requires_predicates():
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[
            PlanStep(id="matches", op=PlanOp.FILTER, input="current_records"),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(plan, _scope(), static_catalog())
    assert exc.value.code == "op_contract"


def test_rejects_application_table_name():
    with pytest.raises(PlanValidationError) as exc:
        validate_query_plan(
            QueryPlan(
                scope=PlanScope(file_ids=(49,)),
                steps=[PlanStep(id="file_data", op=PlanOp.COUNT, input="current_records")],
            ),
            _scope(),
            static_catalog(),
        )
    assert exc.value.code == "forbidden_table"
