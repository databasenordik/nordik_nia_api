import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.execution.turn import TurnResult
from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, Predicate, QueryPlan
from benchmarks.regression import Question, _grade_case, load_suite


def _turn(*predicates: Predicate) -> TurnResult:
    return TurnResult(
        status="answered",
        answer="A non-empty but potentially wrong answer",
        selected_file_id=49,
        plan=QueryPlan(
            scope=PlanScope(file_ids=(49,)),
            steps=[
                PlanStep(
                    id="matches",
                    op=PlanOp.FILTER,
                    input="current_records",
                    where=list(predicates),
                    file_ids=(49,),
                ),
                PlanStep(
                    id="result",
                    op=PlanOp.COUNT,
                    input="matches",
                    file_ids=(49,),
                ),
            ],
        ),
    )


def test_semantic_gate_rejects_nonempty_answer_with_dropped_condition():
    item = Question(20, "question", "", "", "")
    case = {
        "selected_file_id": 49,
        "expected_plan": {
            "required_ops": ["COUNT"],
            "predicates": [
                {"field": "deceased_status", "operator": "IS_TRUE"},
                {"field": "community", "operator": "IS_UNKNOWN"},
            ],
        },
    }
    result = _grade_case(
        item,
        case,
        "There are 1,066 deceased students.",
        _turn(Predicate(field="deceased_status", operator="IS_TRUE")),
    )
    assert result.verdict == "fail"
    assert any("community" in reason for reason in result.reasons)


def test_semantic_gate_accepts_all_required_conditions():
    item = Question(20, "question", "", "", "")
    case = {
        "selected_file_id": 49,
        "expected_plan": {
            "required_ops": ["COUNT"],
            "predicates": [
                {"field": "deceased_status", "operator": "IS_TRUE"},
                {"field": "community", "operator": "IS_UNKNOWN"},
            ],
        },
    }
    result = _grade_case(
        item,
        case,
        "answer",
        _turn(
            Predicate(field="deceased_status", operator="IS_TRUE"),
            Predicate(field="community", operator="IS_UNKNOWN"),
        ),
    )
    assert result.verdict == "pass"


def test_semantic_gate_treats_rank_fields_as_grouping_contract():
    item = Question(57, "question", "", "", "")
    case = {
        "selected_file_id": 91,
        "expected_plan": {
            "required_ops": ["RANK"],
            "group_by": ["community"],
            "having_min_count": 2,
        },
    }
    turn = TurnResult(
        status="answered",
        answer="answer",
        selected_file_id=91,
        plan=QueryPlan(
            scope=PlanScope(file_ids=(91,)),
            steps=[
                PlanStep(
                    id="rank",
                    op=PlanOp.RANK,
                    input="current_records",
                    fields=["community"],
                    having_min_count=2,
                    file_ids=(91,),
                )
            ],
        ),
    )

    result = _grade_case(item, case, "answer", turn)

    assert result.verdict == "pass"


def test_matrix_aborts_on_connection_loss():
    from benchmarks.database_question_matrix import _connection_lost

    assert _connection_lost(ConnectionRefusedError("postgres"))
    assert _connection_lost(OSError(111, "Connection refused"))
    assert not _connection_lost(ValueError("count mismatch"))


def test_matrix_accepts_thousands_separators_and_synonym_known():
    from benchmarks.database_question_matrix import (
        Condition,
        _matches,
        _mentions_number,
    )

    assert _mentions_number("yes: 1,066\nno: 1,035", 1066)
    assert _mentions_number("yes: 1,066\nno: 1,035", 1035)
    assert not _mentions_number("yes: 1,066", 665)
    from benchmarks.database_question_matrix import _normalize_name

    assert _normalize_name("Example PERSON (Archive Reference)") == (
        "example person archive reference"
    )
    rules = {
        "value_synonyms": {
            "unknown": ["unknown"],
            "tuberculosis": ["tuberculosis", "consumption", "phthisis"],
        }
    }
    assert _matches("tuberculosis", Condition("cause_of_death", "known"), rules=rules)
    assert not _matches("unknown", Condition("cause_of_death", "known"), rules=rules)
    assert _matches("consumption", Condition("cause_of_death", "contains", "tuberculosis"), rules=rules)


def test_semantic_gate_accepts_an_authorized_equivalent_value():
    item = Question(1, "question", "", "", "")
    case = {
        "selected_file_id": 49,
        "expected_plan": {
            "predicates": [
                {
                    "field": "admitted_date",
                    "operator": "AFTER",
                    "values": [1905, "1905-12-31"],
                }
            ]
        },
    }

    result = _grade_case(
        item,
        case,
        "answer",
        _turn(Predicate(field="admitted_date", operator="AFTER", value="1905-12-31")),
    )

    assert result.verdict == "pass"


def test_suite_rejects_case_without_answer_or_semantic_contract(tmp_path: Path):
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "broken",
                        "source_question_id": 1,
                        "question": "question",
                        "selected_file_id": 49,
                        "category": "master-specific",
                        "expected_outcome": "count",
                        "critical": True,
                        "private": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_plan"):
        load_suite(path)
