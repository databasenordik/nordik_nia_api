import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.turn_schema import ConversationAction, TurnPlan
from app.planning.turn_validator import validate_turn_plan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


def _student_records(total: int) -> list[dict]:
    return [
        {
            "id": index,
            "source_row_id": 10_000 + index,
            "file_id": 49,
            "version": 1,
            "canonical_name": f"Student {index:04d}",
            "canonical_community": "Test Community",
            "canonical_school": None,
            "search_text": f"Student {index:04d}",
            "raw": {"Name": f"Student {index:04d}"},
            "row_data_normalized": {"canonical": {"name": f"Student {index:04d}"}},
        }
        for index in range(1, total + 1)
    ]


def test_low_confidence_greeting_is_not_clarified():
    turn = TurnPlan(
        normalized_request="hi",
        actions=[ConversationAction(intent="greeting")],
        final_response="compiler_response",
        confidence=0.2,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert validated.respond_actions()[0].intent == "greeting"
    assert not validated.clarify_actions()


def test_hi_never_asks_for_clarification():
    result = asyncio.run(_service().answer(_scope(), "hi"))
    assert result.status == "answered"
    assert "help" in result.answer.lower() or result.answer.lower().startswith("hi")
    assert "say that another way" not in result.answer.lower()


def test_who_are_you_is_assistant_identity():
    result = asyncio.run(_service().answer(_scope(), "who  are u"))
    assert result.status == "answered"
    assert "Nia" in result.answer
    assert "2781" not in result.answer
    assert "1." not in result.answer


def test_remembers_introduced_name():
    service = _service()
    hello = asyncio.run(service.answer(_scope(), "i am test user"))
    assert hello.status == "answered"
    assert "Test User" in hello.answer
    asked = asyncio.run(service.answer(_scope(), "whats my name", conversation_id=hello.conversation_id))
    assert asked.status == "answered"
    assert "Test User" in asked.answer
    assert "don't have access" not in asked.answer.lower()


def test_signed_in_display_name_answers_whats_my_name():
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91),
        display_name="researcher",
    )
    result = asyncio.run(_service().answer(scope, "whats my name"))
    assert result.status == "answered"
    assert "researcher" in result.answer.lower()


def test_correction_does_not_list_students():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    cid = listed.conversation_id
    asyncio.run(service.answer(_scope(), "who are u", conversation_id=cid))
    you_should = asyncio.run(service.answer(_scope(), "you should", conversation_id=cid))
    assert "1." not in you_should.answer
    assert "student" not in you_should.answer.lower()
    denied = asyncio.run(service.answer(_scope(), "i did not ask your name", conversation_id=cid))
    assert denied.status == "answered"
    assert "1." not in denied.answer
    assert "2781" not in denied.answer
    assert (
        "sorry" in denied.answer.lower()
        or "misunderstood" in denied.answer.lower()
        or "last answer" in denied.answer.lower()
    )


def test_spoken_name_correction_is_acknowledged():
    service = _service()
    introduced = asyncio.run(service.answer(_scope(), "tell me about yourself", mode="voice"))
    corrected = asyncio.run(
        service.answer(
            _scope(),
            "It should be Nia, not letter N, then letter I, then letter A.",
            conversation_id=introduced.conversation_id,
            mode="voice",
        )
    )
    assert "thanks for the correction" in corrected.answer.lower()
    assert "ask another way" not in corrected.answer.lower()


def test_general_question_uses_model_not_database():
    service = _service()
    result = asyncio.run(service.answer(_scope(), "why is the sky blue"))
    assert result.status == "answered"
    assert result.plan is None
    assert "sky" in result.answer.lower() or "general answer" in result.answer.lower()
    assert "Joseph" not in result.answer
    assert "file:" not in result.answer
    assert result.reasoning_calls >= 2


def test_unclear_single_word_gets_neutral_clarification():
    result = asyncio.run(_service().answer(_scope(), "hard", mode="voice"))
    assert result.answer == "I may have misheard that. Could you say it again?"
    assert "emoji" not in result.answer.lower()


def test_general_question_after_list_does_not_cite_students():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    result = asyncio.run(
        service.answer(_scope(), "write a short haiku about winter", conversation_id=listed.conversation_id)
    )
    assert result.plan is None
    assert "file:" not in result.answer
    assert "Joseph" not in result.answer
    assert "haiku" in result.answer.lower() or "winter" in result.answer.lower() or "general answer" in result.answer.lower()


def test_can_you_speak_is_voice_capability():
    result = asyncio.run(_service().answer(_scope(), "can you speak"))
    assert result.status == "answered"
    assert "voice" in result.answer.lower()
    assert "Tara" not in result.answer
    assert "1." not in result.answer


def test_common_baby_names_is_general_not_records():
    result = asyncio.run(_service().answer(_scope(), "what are common baby names"))
    assert result.plan is None
    assert "Joseph" not in result.answer
    assert "file:" not in result.answer


def test_what_after_list_explains_last_answer():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    asked = asyncio.run(service.answer(_scope(), "what is that", conversation_id=listed.conversation_id))
    assert "file:" not in asked.answer
    assert "last answer" in asked.answer.lower()
    assert "5 students" in asked.answer or "Student master list" in asked.answer


def test_yes_and_what_are_not_student_rows():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    cid = listed.conversation_id
    yes = asyncio.run(service.answer(_scope(), "yes", conversation_id=cid))
    assert "Joseph" not in yes.answer
    assert "file:" not in yes.answer
    assert "1." not in yes.answer
    what = asyncio.run(service.answer(_scope(), "what is that", conversation_id=cid))
    assert "file:" not in what.answer
    assert "Joseph" not in what.answer
    bare = asyncio.run(service.answer(_scope(), "what", conversation_id=cid))
    assert "file:" not in bare.answer


def test_hi_after_a_list_does_not_repeat_the_page():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    assert "1." in listed.answer
    greeted = asyncio.run(service.answer(_scope(), "hi", conversation_id=listed.conversation_id))
    assert greeted.status == "answered"
    assert "2781" not in greeted.answer
    assert "1." not in greeted.answer
    assert "help" in greeted.answer.lower() or greeted.answer.lower().startswith("hi")


def test_list_all_reports_total_and_numbers_rows():
    result = asyncio.run(_service().answer(_scope(), "list all students in the master list"))
    assert result.status == "answered"
    assert "5 students" in result.answer
    assert "1." in result.answer
    assert "Joseph E" in result.answer
    assert result.plan is not None
    assert "COUNT" in result.plan.op_names()
    assert "PROJECT" in result.plan.op_names()


def test_random_students_use_database_sampling_and_keep_citations():
    service = AssistantTurnService(
        MemoryDataGateway(_student_records(20)),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )

    result = asyncio.run(
        service.answer(_scope(), "Can you give me five random students in the master list?")
    )

    names = [row["canonical_name"] for row in result.rows]
    assert result.status == "answered"
    assert result.plan is not None
    assert "SAMPLE" in result.plan.op_names()
    assert "SORT" not in result.plan.op_names()
    assert len(names) == 5
    assert names != sorted(names)
    assert "Random sample of 5 from 20 matching students" in result.answer
    assert "Say \"next\"" not in result.answer
    assert len(result.citations) == 5


def test_random_students_with_both_dates_filters_and_projects_both_fields():
    records = _student_records(12)
    for index, record in enumerate(records, start=1):
        dates = {"admitted": f"19{index:02d}"}
        if index <= 7:
            dates["discharged"] = f"19{index + 10:02d}"
        record["row_data_normalized"]["canonical"]["dates"] = dates
    service = AssistantTurnService(
        MemoryDataGateway(records),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )

    result = asyncio.run(
        service.answer(
            _scope(),
            "Give me five random students that have admitted date and discharged date.",
        )
    )

    filters = result.plan.step_map()["matches"].where
    assert {(item.field, str(item.operator)) for item in filters} == {
        ("admitted_date", "IS_KNOWN"),
        ("discharged_date", "IS_KNOWN"),
    }
    assert len(result.rows) == 5
    assert "from 7 matching students" in result.answer
    assert all(
        {"admitted_date", "discharged_date"} <= set(row["values"])
        for row in result.list_results[0]["rows"]
    )


def test_incomplete_voice_fragment_does_not_execute_or_page():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list the students", mode="voice"))

    result = asyncio.run(
        service.answer(
            _scope(),
            "Give me five st-",
            conversation_id=listed.conversation_id,
            mode="voice",
        )
    )

    assert result.status == "clarification"
    assert result.plan is None
    assert result.rows == []
    assert result.answer == "I heard an incomplete phrase. Please finish the request."


def test_non_pagination_request_cannot_inherit_compiler_next_page():
    from app.planning.trusted_overrides import apply_trusted_overrides
    from app.planning.turn_schema import ModifyPreviousAction

    mistaken = TurnPlan(
        normalized_request="give me five students",
        actions=[ModifyPreviousAction(goal="list", page="next", limit=5)],
        final_response="deterministic",
        confidence=0.9,
    )

    corrected = apply_trusted_overrides(
        mistaken,
        "give me five students",
        has_active_query=True,
    )

    assert corrected.modify_actions()[0].page == "first"
    assert corrected.modify_actions()[0].offset == 0


def test_count_then_list_them_switches_goal():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students are in the master list?"))
    assert first.status == "answered"
    assert "5 students" in first.answer
    second = asyncio.run(
        service.answer(
            _scope(),
            "list those students one by one in organized way",
            conversation_id=first.conversation_id,
        )
    )
    assert second.status == "answered"
    assert "1." in second.answer
    assert "There are 5 students" in second.answer
    assert second.plan is not None
    assert "list" in second.plan.goals


def test_challenge_after_partial_list_admits_truncation():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    challenged = asyncio.run(
        service.answer(
            _scope(),
            "I believe there more than that",
            conversation_id=listed.conversation_id,
        )
    )
    assert challenged.status == "answered"
    assert "5 matching records" in challenged.answer or "There are 5 students" in challenged.answer
    assert challenged.plan is not None
    assert "verify" in challenged.plan.goals


def test_next_advances_offset():
    service = _service()
    first = asyncio.run(
        service.answer(_scope(), "list all students in the master list")
    )
    # Seed only has 5 rows, so page 2 is empty; still a list goal with a later offset.
    nxt = asyncio.run(
        service.answer(_scope(), "next", conversation_id=first.conversation_id)
    )
    assert nxt.status == "answered"
    assert nxt.plan is not None
    limited = nxt.plan.step_map().get("limited")
    assert limited is not None
    assert limited.offset == 25


def test_yes_after_next_offer_pages_forward():
    from app.planning.trusted_overrides import apply_trusted_overrides

    turn = TurnPlan(
        normalized_request="yes",
        actions=[ConversationAction(intent="acknowledgement")],
        final_response="compiler_response",
        confidence=1.0,
    )
    out = apply_trusted_overrides(
        turn,
        "yes",
        has_active_query=True,
        last_assistant_text='There are 2781 students in the Student master list.\nShowing 1–25 of 2781:\n1. Ada\n\nSay "next" to continue.',
    )
    assert out.modify_actions()
    assert out.modify_actions()[0].page == "next"


def test_previous_pages_backward():
    from app.planning.trusted_overrides import apply_trusted_overrides

    turn = TurnPlan(
        normalized_request="previous",
        actions=[ConversationAction(intent="acknowledgement")],
        final_response="compiler_response",
        confidence=0.5,
    )
    out = apply_trusted_overrides(turn, "previous", has_active_query=True)
    assert out.modify_actions()
    assert out.modify_actions()[0].page == "previous"


def test_tell_me_about_name_after_list_looks_up_the_person():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    asked = asyncio.run(
        service.answer(_scope(), "tell me about Joseph", conversation_id=listed.conversation_id)
    )
    assert asked.status == "answered"
    assert "Joseph E" in asked.answer
    assert "Samuel A" not in asked.answer
    assert "file:" not in asked.answer


def test_who_is_the_president_after_list_stays_general():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list all students in the master list"))
    asked = asyncio.run(
        service.answer(_scope(), "who is the president", conversation_id=listed.conversation_id)
    )
    assert asked.plan is None
    assert "Joseph" not in asked.answer
    assert "1." not in asked.answer


def test_spoken_page_does_not_read_every_name():
    from app.execution.executor import ActionResult, ExecutionResult
    from app.execution.formatter import format_spoken_answer
    from app.planning.catalog import static_catalog
    from app.planning.plan_schema import PlanOp, PlanScope, PlanStep, QueryPlan

    names = [f"Student {index}" for index in range(1, 26)]
    plan = QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        steps=[PlanStep(id="rows", op=PlanOp.PROJECT, input="current_records")],
        goals=["list"],
    )
    spoken = format_spoken_answer(
        plan,
        ExecutionResult(
            action_results=[
                ActionResult(
                    action_id="a0",
                    goal="list",
                    rows=[{"canonical_name": name} for name in names],
                    total_count=2781,
                    offset=0,
                    page_size=25,
                    has_more=True,
                )
            ]
        ),
        static_catalog(),
    )
    assert spoken is not None
    assert "2781 students" in spoken
    assert "Starting with Student 1" in spoken
    assert "Say next to continue" in spoken
    assert "Student 25" not in spoken


@pytest.mark.parametrize(
    ("total", "size", "expected_offset", "expected_count"),
    [
        (0, 100, 0, 0),
        (1, 100, 0, 1),
        (49, 100, 0, 49),
        (50, 100, 0, 50),
        (51, 50, 1, 50),
        (99, 50, 49, 50),
        (100, 100, 0, 50),
        (101, 100, 1, 50),
    ],
)
def test_end_anchored_window_is_calculated_from_total(
    total: int,
    size: int,
    expected_offset: int,
    expected_count: int,
):
    gateway = MemoryDataGateway(_student_records(total))
    page = asyncio.run(
        gateway.list_records_window(
            _scope(),
            (49,),
            [],
            anchor="end",
            window_size=size,
            cursor=0,
            page_size=50,
            sort_field="student_name",
        )
    )
    assert page.offset == expected_offset
    assert len(page.rows) == expected_count
    assert [row["canonical_name"] for row in page.rows] == [
        row["canonical_name"] for row in _student_records(total)[expected_offset:expected_offset + 50]
    ]


def test_last_100_pages_exactly_the_tail_without_duplicates():
    service = AssistantTurnService(
        MemoryDataGateway(_student_records(101)),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )
    first = asyncio.run(service.answer(_scope(), "list the last 100 students"))
    second = asyncio.run(
        service.answer(_scope(), "next", conversation_id=first.conversation_id)
    )

    assert [row["canonical_name"] for row in first.rows] == [
        f"Student {index:04d}" for index in range(2, 52)
    ]
    assert [row["canonical_name"] for row in second.rows] == [
        f"Student {index:04d}" for index in range(52, 102)
    ]
    combined = [row["source_row_id"] for row in [*first.rows, *second.rows]]
    assert len(combined) == len(set(combined)) == 100
    assert first.list_results[0]["window"] == {
        "anchor": "end",
        "size": 100,
        "count": 100,
        "cursor": 0,
    }
    assert second.list_results[0]["has_more"] is False
    assert "Showing 2-51 of 101" in first.answer
    assert "Showing 52-101 of 101" in second.answer


def test_last_n_correction_modifies_the_active_query_without_a_guessed_offset():
    service = AssistantTurnService(
        MemoryDataGateway(_student_records(101)),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )
    listed = asyncio.run(service.answer(_scope(), "list the students"))
    corrected = asyncio.run(
        service.answer(_scope(), "I said last 100 only", conversation_id=listed.conversation_id)
    )

    limited = corrected.plan.step_map()["limited"]
    assert limited.offset == 0
    assert limited.window_anchor == "end"
    assert limited.window_size == 100
    assert corrected.rows[0]["canonical_name"] == "Student 0002"


def test_unavailable_requested_fields_are_explained_without_clarifying():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list the students"))
    result = asyncio.run(
        service.answer(
            _scope(),
            "put the date of birth and death besides each one",
            conversation_id=listed.conversation_id,
        )
    )

    assert result.status == "unsupported_fields"
    assert "Date of birth" in result.answer
    assert "Date of death" in result.answer
    assert "Admitted date" in result.answer
    assert "say that another way" not in result.answer.lower()


def test_supported_requested_field_is_projected_and_structured():
    service = _service()
    listed = asyncio.run(service.answer(_scope(), "list the students"))
    result = asyncio.run(
        service.answer(
            _scope(),
            "put the admitted date beside each one",
            conversation_id=listed.conversation_id,
        )
    )

    assert result.status == "answered"
    assert "Admitted date:" in result.answer
    assert [field["name"] for field in result.list_results[0]["fields"]] == [
        "student_name",
        "admitted_date",
    ]
    assert "values" in result.list_results[0]["rows"][0]


def test_available_information_question_is_catalog_driven():
    result = asyncio.run(
        _service().answer(
            _scope(),
            "What information is available in the research records I can access?",
        )
    )
    assert result.status == "answered"
    assert "Student master list" in result.answer
    assert "Confirmed deaths" not in result.answer
    assert "say that another way" not in result.answer.lower()
