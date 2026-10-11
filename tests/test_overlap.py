"""Records sharing a value are compared by when they were present, not only by how many share it.

Asked about families with several children at the school "during the same period", NIA ranked
parents' names by record count: siblings a decade apart counted exactly like siblings who overlapped.
These tests pin the comparison of periods, on invented families, including the dates that cannot
be placed -- a year alone covers the whole year, two admissions in one cell are not guessed at, and
the answer says how many groups could not be checked.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.overlap import overlapping_groups  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.memory.types import ConversationRecord  # noqa: E402
from app.planning.catalog import static_catalog  # noqa: E402
from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, QueryPlan  # noqa: E402
from app.planning.query_plan_builder import build_query_plan  # noqa: E402
from app.planning.turn_schema import QueryAction, TurnPlan  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402


def _exact(iso):
    return {"iso": iso, "raw": iso, "kind": "exact_date"}


def _year(year):
    return {"raw": str(year), "kind": "year", "year": year}


def _row(row_id, name, parents, admitted=None, discharged=None):
    dates = {}
    if admitted is not None:
        dates["admitted"] = admitted
    if discharged is not None:
        dates["discharged"] = discharged
    return {
        "id": row_id,
        "source_row_id": row_id,
        "file_id": 49,
        "version": 1,
        "canonical_name": name.lower(),
        "row_data_normalized": {
            "canonical": {"display_name": name, "parents_names": parents, "dates": dates}
        },
    }


def _rows():
    return [
        # Three children: the first two overlap in 1952-1953, the third came much later.
        _row(1, "Ann Birch", "Mary and John Birch", _exact("1950-09-01"), _exact("1953-06-30")),
        _row(2, "Ben Birch", "Mary and John Birch", _exact("1952-09-01"), _exact("1956-06-30")),
        _row(3, "Cal Birch", "Mary and John Birch", _exact("1965-09-01"), _exact("1968-06-30")),
        # Two children, one after the other: the first left the day before the second arrived.
        _row(4, "Dee Cedar", "Rose Cedar", _exact("1940-09-01"), _exact("1944-08-31")),
        _row(5, "Eve Cedar", "Rose Cedar", _exact("1944-09-01"), _exact("1947-06-30")),
        # Two children, but one date cell holds two admissions and cannot be placed.
        _row(6, "Fay Dunn", "Lena Dunn", _exact("1930-09-01"), _exact("1934-06-30")),
        _row(7, "Gus Dunn", "Lena Dunn", {"raw": "1931 and 1933", "kind": "range"}, _exact("1935-06-30")),
        # A year alone covers the whole year, so these two overlap.
        _row(8, "Hal Elm", "Ida Elm", _year(1920), _year(1922)),
        _row(9, "Ivy Elm", "Ida Elm", _exact("1922-11-01"), _exact("1925-06-30")),
        # Only one child.
        _row(10, "Jon Fir", "Kay Fir", _exact("1910-09-01"), _exact("1912-06-30")),
    ]


def _compare(rows=None, need=2):
    return overlapping_groups(
        rows if rows is not None else _rows(),
        static_catalog(),
        group_field="parents_names",
        start_field="admitted_date",
        end_field="discharged_date",
        min_count=need,
    )


def test_only_members_present_at_the_same_time_count():
    result = _compare()
    assert result.shared == 4
    assert result.checkable == 3
    assert [group.value for group in result.groups] == ["Ida Elm", "Mary and John Birch"]
    birch = next(group for group in result.groups if group.value == "Mary and John Birch")
    assert [stay.name for stay in birch.together] == ["Ann Birch", "Ben Birch"]
    assert (birch.first.isoformat(), birch.last.isoformat()) == ("1952-09-01", "1953-06-30")
    assert (birch.members, birch.dated) == (3, 3)


def test_separate_stretches_of_togetherness_stay_separate():
    """Two siblings together in the 1930s and two others in the 1950s are not one long overlap."""
    rows = [
        _row(1, "Amy Gale", "Nora Gale", _exact("1931-09-01"), _exact("1934-06-30")),
        _row(2, "Bea Gale", "Nora Gale", _exact("1933-09-01"), _exact("1936-06-30")),
        _row(3, "Cy Gale", "Nora Gale", _exact("1950-09-01"), _exact("1953-06-30")),
        _row(4, "Di Gale", "Nora Gale", _exact("1952-09-01"), _exact("1955-06-30")),
    ]
    [group] = _compare(rows).groups
    assert [(first.year, last.year) for first, last in group.episodes] == [(1933, 1934), (1952, 1953)]
    assert len(group.together) == 4


def test_back_to_back_periods_do_not_overlap():
    assert all(group.value != "Rose Cedar" for group in _compare().groups)


def test_a_year_alone_covers_the_whole_year():
    elm = next(group for group in _compare().groups if group.value == "Ida Elm")
    assert elm.first.year == 1922


def test_asking_for_more_at_once_raises_the_bar():
    result = _compare(need=3)
    assert result.shared == 1 and result.checkable == 1 and result.groups == []


def test_a_duplicates_question_with_two_dates_compiles_to_a_period_comparison():
    turn = TurnPlan(
        normalized_request="families with several children at the same time",
        actions=[
            QueryAction(
                datasets=[49],
                goal="duplicates",
                group_by=["parents_names"],
                interval_start="admitted_date",
                interval_end="discharged_date",
            )
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    plan = build_query_plan(
        turn, scope=AccessScope(principal_id="r", allowed_file_ids=(49,)), catalog=static_catalog()
    )
    step = next(item for item in plan.steps if item.op is PlanOp.DUPLICATES)
    assert (step.interval_start, step.interval_end) == ("admitted_date", "discharged_date")


def test_the_answer_says_how_much_could_be_checked_and_who_overlapped():
    catalog = static_catalog()
    service = AssistantTurnService(
        MemoryDataGateway(records=_rows(), catalog=catalog), catalog=catalog, store=InMemoryMemoryStore()
    )
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=("duplicates",),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(
                id="result",
                op=PlanOp.DUPLICATES,
                input="rows",
                fields=["parents_names"],
                having_min_count=2,
                interval_start="admitted_date",
                interval_end="discharged_date",
            ),
        ],
    )
    result = asyncio.run(
        service._run_plan(
            AccessScope(principal_id="r", allowed_file_ids=(49,)),
            "Are there families with several children attending during the same period?",
            plan,
            ConversationRecord(id="c1", principal_id="r", selected_file_id=49),
            mode="text",
            catalog=catalog,
        )
    )
    assert result.status == "answered"
    assert "**4** values of **Parents names** are each shared by 2 or more students" in result.answer
    assert "For **3** of them" in result.answer and "in **2** of those" in result.answer
    assert "The other **1** cannot be checked" in result.answer
    assert "| Mary and John Birch | 2 of 3 | Ann Birch (1950–1953); Ben Birch (1952–1956) | 1952–1953 |" in result.answer
