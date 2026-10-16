"""Matched records are read against what a word search could not check; code does the counting.

A tester counted 12 children "transferred to other schools" where NIA counted 13 records that
say "transferred to": one of them names a sanatorium. These tests pin the reading that closes
that gap, on invented records: only the unchecked part of the question is read, every record
gets exactly one verdict or the reading is dropped, and the exact count is never replaced.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.match_review import (  # noqa: E402
    MatchReview,
    Reading,
    read_matches,
    reading_summary,
    unchecked_words,
)
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.fake import FakeReasoningProvider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.memory.types import ConversationRecord  # noqa: E402
from app.planning.catalog import static_catalog  # noqa: E402
from app.planning.plan_schema import (  # noqa: E402
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)
from app.security.access_scope import AccessScope  # noqa: E402

QUESTION = "How many children were transferred to other schools?"


@pytest.mark.parametrize(
    "question,searched,labels,expected",
    [
        (QUESTION, ["transferred to"], ["Additional information", "Student master list"], "other schools"),
        ("How many notes contain student register?", ["student register"], ["Notes"], ""),
        ("How many records mention a transfer?", ["transferred"], ["Additional information"], ""),
        ("How many students ran away more than once?", ["ran away"], ["Notes"], "more than once"),
        (
            "How many students from Garden River were transferred to other schools?",
            ["transferred to"],
            ["Additional information"],
            "Garden River, other schools",
        ),
    ],
)
def test_only_the_words_the_search_did_not_use_are_left_to_read(question, searched, labels, expected):
    assert unchecked_words(question, searched, labels) == expected


class _Reviewer:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.calls = 0

    async def structured_output(self, *, system, user, response_model):
        self.calls += 1
        return response_model.model_validate({"verdicts": self.verdicts})


RECORDS = [("Ada Birch", "Transferred to Lakeside Residential School"), ("Cal Dunn", "Transferred to Weston Sanatorium")]


def test_every_record_gets_its_verdict_in_order():
    reviewer = _Reviewer(
        [
            {"record": 2, "fits": "no", "reason": "a sanatorium, not a school"},
            {"record": 1, "fits": "yes", "reason": "names a residential school"},
        ]
    )
    readings = asyncio.run(read_matches(reviewer, QUESTION, "other schools", RECORDS))
    assert [item.fits for item in readings] == ["yes", "no"]


@pytest.mark.parametrize(
    "verdicts",
    [
        [{"record": 1, "fits": "yes"}],
        [{"record": 1, "fits": "yes"}, {"record": 1, "fits": "no"}],
        [{"record": 1, "fits": "yes"}, {"record": 3, "fits": "no"}],
    ],
)
def test_an_incomplete_reading_is_discarded_whole(verdicts):
    """A count of fits that skipped a record would claim a check nobody made."""
    assert asyncio.run(read_matches(_Reviewer(verdicts), QUESTION, "other schools", RECORDS)) is None


def test_nothing_is_read_when_the_search_covered_the_whole_question():
    reviewer = _Reviewer([])
    assert asyncio.run(read_matches(reviewer, QUESTION, "", RECORDS)) is None
    assert reviewer.calls == 0


def test_the_summary_counts_and_names_what_does_not_fit():
    text = reading_summary(
        "other schools",
        ["Ada Birch", "Cal Dunn"],
        [Reading("yes", "names a residential school"), Reading("no", "a sanatorium, not a school")],
    )
    assert "**1** fits and **1** does not" in text
    assert "Cal Dunn (a sanatorium, not a school)" in text
    assert "the count above is exact" in text


def _record(row_id, name, number, additional):
    return {
        "id": row_id,
        "source_row_id": row_id,
        "file_id": 49,
        "version": 1,
        "canonical_name": name.lower(),
        "row_data_normalized": {
            "canonical": {"display_name": name, "student_number": number},
            "fields": {"Additional Information": additional},
        },
    }


def test_the_answer_keeps_the_exact_count_and_adds_the_reading():
    catalog = static_catalog()
    records = [
        _record(1, "Ada Birch", "1950-001", "Transferred to Lakeside Residential School in 1952."),
        _record(2, "Ben Cedar", "1950-002", "Transferred from Northern Residential School."),
        _record(3, "Cal Dunn", "1950-003", "Transferred to Weston Sanatorium."),
    ]
    provider = FakeReasoningProvider()
    provider.structured_queue = [
        MatchReview.model_validate(
            {
                "verdicts": [
                    {"record": 1, "fits": "yes", "reason": "names a residential school"},
                    {"record": 2, "fits": "no", "reason": "a sanatorium, not a school"},
                ]
            }
        )
    ]
    service = AssistantTurnService(
        MemoryDataGateway(records=records, catalog=catalog),
        catalog=catalog,
        store=InMemoryMemoryStore(),
        reasoner=provider,
    )
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=("count",),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="rows",
                where=[Predicate(field="additional_information", operator=FilterOperator.CONTAINS, value="transferred")],
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )
    result = asyncio.run(
        service._run_plan(
            AccessScope(principal_id="researcher", allowed_file_ids=(49,)),
            QUESTION,
            plan,
            ConversationRecord(id="c1", principal_id="researcher", selected_file_id=49),
            mode="text",
            catalog=catalog,
        )
    )
    assert next(fact["value"] for fact in result.facts if fact["name"] == "count") == 2
    assert "**2 students**" in result.answer
    assert "**1** fits and **1** does not" in result.answer
    assert "| Fits the question |" in result.answer
    assert result.inference is True
