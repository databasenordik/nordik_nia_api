"""A word match on narrative text is checked against what the records actually say.

Asked how many children were transferred to other schools, NIA searched Notes, found nothing
and answered "0 students" -- while the transfers sat in Additional Information, written both as
"transferred from" one school and "transferred to" another. These tests pin the behaviour that
answer needed, on invented records rather than the real list: the words are looked for in the
narrative fields that hold them, the direction the question gives a word is followed when the
records use it both ways, and a short result shows the sentence each record matched on.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.text_search import narrative_fields, refine_text_search  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
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

TRANSFER_QUESTION = "How many children were transferred to other schools?"


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49,))


def _catalog():
    """The fallback catalog, with the student number's source column as the registry records it."""
    base = static_catalog()
    fields = tuple(
        spec.model_copy(update={"raw_json_keys": ("Student Number",)})
        if spec.file_id == 49 and spec.semantic_field == "student_number"
        else spec
        for spec in base.fields
    )
    return base.model_copy(update={"fields": fields})


def _record(row_id, name, *, number=None, notes=None, additional=None):
    canonical = {"display_name": name}
    normalized = {"canonical": canonical, "fields": {}}
    if number is not None:
        # Normalized the way ingest stores it; the source cell keeps the list's own form.
        canonical["student_number"] = number.replace(".", "-")
        normalized["fields"]["Student Number"] = {"raw": number, "role": "identifier"}
    if additional is not None:
        normalized["fields"]["Additional Information"] = additional
    if notes is not None:
        normalized["chat"] = {"narrative_bundle": {"notes": notes}}
    return {
        "id": row_id,
        "source_row_id": row_id,
        "file_id": 49,
        "version": 1,
        "canonical_name": name.lower(),
        "row_data_normalized": normalized,
    }


def _records():
    return [
        _record(1, "Ada Birch", number="1950.001", additional="Transferred to Lakeside Residential School in 1952."),
        _record(2, "Ben Cedar", number="1950.002", additional="Transferred from Northern Residential School (1949)."),
        _record(3, "Cal Dunn", number="1950.003", additional="Transferred from Hill School. Transferred to Weston Sanatorium."),
        _record(4, "Dee Elm", number="1950.004", additional="Transferred from River School."),
        _record(5, "Eve Fir", notes="Admitted with her brother."),
        _record(6, "Fay Gum", additional="Discharged in 1955."),
    ]


def _plan(field, value, *, operator=FilterOperator.CONTAINS, goal="count") -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=(goal,),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input="rows",
                where=[Predicate(field=field, operator=operator, value=value)],
            ),
            PlanStep(id="result", op=PlanOp.COUNT, input="matches"),
        ],
    )


def _refine(plan, question, records=None):
    catalog = _catalog()
    gateway = MemoryDataGateway(records=records if records is not None else _records(), catalog=catalog)
    return asyncio.run(refine_text_search(gateway, _scope(), plan, catalog, question))


def test_narrative_fields_are_the_prose_fields_not_names_or_identifiers():
    names = [spec.semantic_field for spec in narrative_fields((49,), static_catalog())]
    assert {"notes", "additional_information", "death_details"} <= set(names)
    assert "student_name" not in names and "student_number" not in names


def test_words_missing_from_the_chosen_field_are_found_where_they_are_written():
    refinement = _refine(_plan("notes", "transferred"), "How many records mention being transferred?")
    leaf = refinement.plan.steps[1].where[0]
    assert (leaf.field, leaf.value) == ("additional_information", "transferred")
    assert 'No record has "transferred" in its notes' in refinement.compose("")
    assert "additional information" in refinement.compose("")


def test_the_direction_in_the_question_is_followed_when_records_use_both():
    """ "transferred to" is the question; "transferred from" is a different event."""
    refinement = _refine(_plan("notes", "transferred"), TRANSFER_QUESTION)
    leaf = refinement.plan.steps[1].where[0]
    assert (leaf.field, leaf.operator, leaf.value) == (
        "additional_information",
        FilterOperator.CONTAINS,
        "transferred to",
    )
    said = refinement.compose("")
    assert '4 records mention "transferred"' in said
    assert 'the 2 that say "transferred to"' in said
    assert '3 say "transferred from"' in said


def test_a_short_count_shows_what_each_record_says():
    """The researcher can see that one "transferred to" names a sanatorium, not a school."""
    refinement = _refine(_plan("notes", "transferred"), TRANSFER_QUESTION)
    assert [item.name for item in refinement.excerpts] == ["Ada Birch", "Cal Dunn"]
    assert [item.number for item in refinement.excerpts] == ["1950.001", "1950.003"]
    assert "**Transferred to** Weston Sanatorium" in refinement.excerpts[1].text
    assert refinement.excerpts[1].text.startswith("…")


def test_without_a_direction_in_the_question_every_mention_counts():
    refinement = _refine(
        _plan("additional_information", "transferred"), "How many records mention a transfer?"
    )
    assert refinement.plan.steps[1].where[0].value == "transferred"
    assert not refinement.before
    assert len(refinement.excerpts) == 4


def test_a_word_used_in_only_one_direction_is_not_narrowed():
    records = [item for item in _records() if item["source_row_id"] in {1, 6}]
    refinement = _refine(_plan("additional_information", "transferred"), TRANSFER_QUESTION, records)
    assert refinement.plan.steps[1].where[0].value == "transferred"
    assert not refinement.before


def test_a_word_no_narrative_field_holds_says_every_field_was_checked():
    refinement = _refine(_plan("notes", "expelled"), "How many students were expelled?")
    assert refinement.plan.steps[1].where[0].field == "notes"
    assert 'none of them mention "expelled" either' in refinement.compose("")
    assert not refinement.excerpts


def test_a_list_answer_is_not_given_a_second_list_of_excerpts():
    refinement = _refine(
        _plan("additional_information", "transferred", goal="list"), "List the transferred students"
    )
    assert not refinement.excerpts


def test_a_plan_with_no_narrative_word_match_is_left_alone():
    plan = _plan("community", "Garden River", operator=FilterOperator.EQUALS)
    assert _refine(plan, "How many students are from Garden River?") is None


def test_the_answer_names_where_it_searched_and_lists_each_record():
    catalog = _catalog()
    service = AssistantTurnService(
        MemoryDataGateway(records=_records(), catalog=catalog),
        catalog=catalog,
        store=InMemoryMemoryStore(),
    )
    conversation = ConversationRecord(id="c1", principal_id="researcher", selected_file_id=49)
    result = asyncio.run(
        service._run_plan(
            _scope(),
            TRANSFER_QUESTION,
            _plan("notes", "transferred"),
            conversation,
            mode="text",
            catalog=catalog,
        )
    )
    assert result.status == "answered"
    assert next(fact["value"] for fact in result.facts if fact["name"] == "count") == 2
    assert 'with "transferred to" in additional information' in result.answer
    assert "| Ada Birch | 1950.001 |" in result.answer
