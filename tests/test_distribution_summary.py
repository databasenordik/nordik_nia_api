"""A breakdown is answered as a summary a researcher can check, not a dump of every label.

Asked how many children a list holds and where they are from, NIA printed "Students: 2781
students." and then all 370 community labels one per line. The answer testers asked for leads
with the totals -- how many records, how many carry the field, how many are blank, how many
distinct values -- and then the most common values as a table. These pin that shape, and that a
count states the conditions it was computed under.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.protocol import GroupRow  # noqa: E402
from app.execution.executor import ActionResult, ComputedFact, ExecutionResult  # noqa: E402
from app.execution.formatter import format_deterministic_answer  # noqa: E402
from app.planning.catalog import static_catalog  # noqa: E402
from app.planning.plan_schema import (  # noqa: E402
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)


def _plan(file_id: int = 49) -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=(file_id,)),
        goals=("count",),
        steps=[PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION)],
    )


def _answer(*actions: ActionResult, file_id: int = 49) -> str:
    return format_deterministic_answer(
        "question", _plan(file_id), ExecutionResult(action_results=list(actions)), static_catalog()
    )


def _count(value: int, *predicates: Predicate, file_id: int = 49) -> ActionResult:
    return ActionResult(
        action_id="a0",
        goal="count",
        file_ids=(file_id,),
        facts=[ComputedFact(name="count", value=value)],
        predicates=list(predicates),
        total_count=value,
    )


def _spread(groups: list[GroupRow], *, top_n: int | None = None) -> ActionResult:
    return ActionResult(
        action_id="a1",
        goal="rank",
        file_ids=(49,),
        ranked_groups=groups,
        group_fields=("community", None),
        grouped_field="community",
        requested_top_n=top_n,
    )


def _places(n: int) -> list[GroupRow]:
    return [
        GroupRow(file_id=49, label=f"Place {index:02d}", secondary_label=None, count=100 - index)
        for index in range(n)
    ]


def test_a_count_and_a_breakdown_of_the_same_records_are_one_answer():
    groups = _places(40)
    recorded = sum(item.count for item in groups)
    answer = _answer(_count(recorded + 30), _spread(groups))
    assert answer.startswith(f"There are **{recorded + 30:,} student records** in the")
    assert f"**{recorded:,} have a value recorded** and **30 are blank**" in answer
    assert "**40 distinct values**" in answer
    assert "| First Nation / Community | Students |" in answer
    assert "| Place 14 | 86 |" in answer
    assert "Place 15" not in answer
    assert "…and 25 more" in answer


def test_a_short_breakdown_is_shown_whole_with_its_blank_rows():
    groups = [
        GroupRow(file_id=49, label="Garden River", secondary_label=None, count=12),
        GroupRow(file_id=49, label="Walpole Island", secondary_label=None, count=9),
        GroupRow(file_id=49, label=None, secondary_label=None, count=4),
    ]
    answer = _answer(_spread(groups))
    assert "**21 have a value recorded** and **4 are blank**" in answer
    assert "**2 distinct values**. All of them:" in answer
    assert "| Garden River | 12 |" in answer
    assert "| Not recorded | 4 |" in answer


def test_a_requested_top_n_stays_a_ranked_table():
    answer = _answer(_spread(_places(2), top_n=2))
    assert answer.startswith("By First Nation / Community in the")
    assert "| Place 00 | 100 |" in answer


def test_a_count_and_a_breakdown_under_different_conditions_are_kept_apart():
    """Merging is only honest when both actions counted the same records."""
    filtered = _count(5, Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"))
    answer = _answer(filtered, _spread(_places(3)))
    assert not answer.startswith("There are **")
    assert answer.startswith("Garden River: 5 students.\n\n")
    assert "| Place 00 | 100 |" in answer


def test_the_same_answer_from_two_identical_actions_is_said_once():
    first = _spread(_places(3))
    second = ActionResult(
        action_id="a2",
        goal="rank",
        file_ids=(49,),
        ranked_groups=_places(3),
        group_fields=("community", None),
        grouped_field="community",
    )
    answer = _answer(first, second)
    assert answer.count("| Place 00 | 100 |") == 1


def test_a_word_count_names_the_field_and_the_words_it_matched():
    """ "0 students" alone hid that the words were looked for in the wrong field."""
    predicate = Predicate(
        field="notes", operator=FilterOperator.CONTAINS_ANY, value=["transferred", "transfer"]
    )
    assert _answer(_count(0, predicate)) == (
        '**0 students** in the Student master list with "transferred" or "transfer" in notes.'
    )


def test_a_widened_word_count_names_every_field_searched():
    group = Predicate(
        op="or",
        items=[
            Predicate(field="additional_information", operator=FilterOperator.CONTAINS, value="transferred to"),
            Predicate(field="death_details", operator=FilterOperator.CONTAINS, value="transferred to"),
        ],
    )
    answer = _answer(_count(13, group))
    assert answer.endswith('with "transferred to" in additional information or death details.')


def test_a_count_of_blank_values_says_so():
    predicate = Predicate(field="notes", operator=FilterOperator.IS_UNKNOWN)
    assert _answer(_count(7, predicate)).endswith("with no recorded notes.")


def test_a_searched_name_is_not_repeated_in_the_sentence():
    predicate = Predicate(field="student_name", operator=FilterOperator.CONTAINS, value="Albert")
    assert _answer(_count(1, predicate)) == "**1 student** in the Student master list."
