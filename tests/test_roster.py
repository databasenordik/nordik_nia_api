"""A count on a short list says how many people its rows hold, and why the two differ.

A tester counted 49 children on a 56-row list: blank rows, a placeholder, notes about a
family, and one child entered twice. NIA answered 56. The row count is still true and stays
the computed fact; these tests pin the people count beside it, on invented rows, including
the rule that two rows sharing a name are one person only when nothing recorded tells them
apart.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.data_gateway.protocol import GroupRow  # noqa: E402
from app.execution.executor import ActionResult, ComputedFact, ExecutionResult  # noqa: E402
from app.execution.formatter import format_deterministic_answer  # noqa: E402
from app.execution.roster import roster_from_rows, roster_sentence  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.memory.types import ConversationRecord  # noqa: E402
from app.planning.catalog import static_catalog  # noqa: E402
from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, QueryPlan  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402


def _row(row_id, name, *, first="", last="", flags="", number=None, community=None):
    canonical = {"display_name": name}
    parts = {key: value for key, value in {"first": first, "last": last, "flags": flags}.items() if value}
    if parts:
        canonical["name_parts"] = parts
    if number:
        canonical["student_number"] = number
    if community:
        canonical["community"] = community
    return {
        "id": row_id,
        "source_row_id": row_id,
        "file_id": 49,
        "version": 1,
        "canonical_name": name.lower(),
        "canonical_community": community.lower() if community else None,
        "row_data_normalized": {"canonical": canonical},
    }


def _rows():
    return [
        _row(1, "Ann Sample", first="Ann", last="Sample"),
        _row(2, "Ann Sample", first="Ann", last="Sample", community="River Bend"),
        _row(3, ""),
        _row(4, "TO BE REVIEWED", flags="placeholder"),
        _row(5, "The Sample Family: Ann was the second child", first="Ann", last="Sample", flags="narrative; review"),
        _row(6, "Bob Sample", first="Bob", last="Sample", number="1901.001"),
        _row(7, "Bob Sample", first="Bob", last="Sample", number="1901.002"),
    ]


def test_rows_that_hold_no_person_or_repeat_one_are_told_apart_from_people():
    roster = roster_from_rows(_rows(), static_catalog())
    assert (roster.rows, roster.unique) == (7, 3)
    assert (roster.blank, roster.placeholders, roster.notes) == (1, 1, 1)
    assert roster.repeats == ["Ann Sample"]
    assert roster_sentence(roster) == (
        "Counting people rather than rows, that is **3 unique named people**; the other 4 rows "
        "are 1 blank row, 1 placeholder row, 1 note about a family and 1 repeated entry (Ann Sample)."
    )


def test_a_shared_name_with_different_identifiers_is_two_people():
    roster = roster_from_rows(_rows()[5:], static_catalog())
    assert roster.unique == 2 and not roster.repeats
    assert roster_sentence(roster) == ""


def test_a_value_on_either_of_a_persons_rows_counts_once():
    roster = roster_from_rows(_rows(), static_catalog())
    assert roster.recorded_for("community", static_catalog()) == 1


def test_a_count_and_its_breakdown_say_how_many_people_have_a_value():
    catalog = static_catalog()
    roster = roster_from_rows(_rows(), catalog)
    count = ActionResult(
        action_id="a0", goal="count", file_ids=(49,), facts=[ComputedFact(name="count", value=7)], total_count=7
    )
    spread = ActionResult(
        action_id="a1",
        goal="rank",
        file_ids=(49,),
        ranked_groups=[
            GroupRow(file_id=49, label="River Bend", secondary_label=None, count=1),
            GroupRow(file_id=49, label=None, secondary_label=None, count=6),
        ],
        group_fields=("community", None),
        grouped_field="community",
    )
    plan = QueryPlan(scope=PlanScope(file_ids=(49,)), goals=("count",), steps=[PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION)])
    answer = format_deterministic_answer(
        "q", plan, ExecutionResult(action_results=[count, spread]), catalog, roster=roster
    )
    assert "**3 unique named people**" in answer
    assert "Counting each person once, **1 of the 3** have a value recorded and **2** do not." in answer
    assert "| River Bend | 1 |" in answer


def test_a_count_on_a_short_list_reports_people_beside_rows():
    catalog = static_catalog()
    service = AssistantTurnService(
        MemoryDataGateway(records=_rows(), catalog=catalog), catalog=catalog, store=InMemoryMemoryStore()
    )
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=("count",),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(id="result", op=PlanOp.COUNT, input="rows"),
        ],
    )
    result = asyncio.run(
        service._run_plan(
            AccessScope(principal_id="researcher", allowed_file_ids=(49,)),
            "How many children are on this list?",
            plan,
            ConversationRecord(id="c1", principal_id="researcher", selected_file_id=49),
            mode="text",
            catalog=catalog,
        )
    )
    assert next(fact["value"] for fact in result.facts if fact["name"] == "count") == 7
    assert result.answer.startswith("**7 students** in the Student master list.")
    assert "**3 unique named people**" in result.answer
