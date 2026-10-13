"""A researcher pasting back the answer they expected gets the earlier question checked again.

Two testers pasted corrected figures -- a sentence and a table -- after a wrong answer. One got
"NIA couldn't produce a usable plan", because the planner restated the paste into fields with
a length limit; the other got an unrelated "no matching students". The message asks one thing:
check that answer again. These tests use invented tables, not the testers' records.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.turn import AssistantTurnService, _question_being_corrected  # noqa: E402
from app.llm.fake import FakeReasoningProvider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.memory.types import ConversationRecord, QueryFrame, QueryState  # noqa: E402
from app.memory.updates import _frame_question  # noqa: E402
from app.planning.ai_turn_schema import PlannedConversationAction, clip_free_text  # noqa: E402
from app.planning.corrections import is_pasted_correction  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

CORRECTION = """The answer should be:
The list holds **8 stored rows**, and after removing one blank row there are **7 people**.

| Place          | People |
| -------------- | -----: |
| River Bend     |      4 |
| Stony Point    |      3 |
"""

FOUND = """I found these records, and you missed them:

| # | Name        | Number   |
| - | ----------- | -------- |
| 1 | Example One | 1900.001 |
| 2 | Example Two | 1900.002 |
"""


@pytest.fixture
def ai_settings(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("PLANNER_MODE", raising=False)
    get_settings.cache_clear()


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


@pytest.mark.parametrize("text", [CORRECTION, FOUND])
def test_a_corrective_statement_with_a_table_is_a_correction(text):
    assert is_pasted_correction(text)


@pytest.mark.parametrize(
    "text",
    [
        "How many students were admitted in 1950?",
        "The answer should be higher, can you check the admitted dates again?",
        "Please look these up:\n\n| Name | Number |\n| --- | --- |\n| Example One | 1 |\n| Example Two | 2 |\n| Example Three | 3 |",
    ],
)
def test_a_question_a_complaint_or_a_bare_table_is_not(text):
    assert not is_pasted_correction(text)


def test_an_overlong_restatement_is_cut_at_a_sentence_not_refused():
    long_text = "This sentence restates the request. " * 60
    clipped = clip_free_text(long_text, 1000)
    assert len(clipped) <= 1000 and clipped.endswith(".")
    action = PlannedConversationAction.model_validate(
        {"type": "respond", "intent": "acknowledgement", "response": long_text}
    )
    assert len(action.response) <= 1000


def test_a_correction_never_becomes_the_question_a_frame_remembers():
    assert _frame_question(CORRECTION) is None
    assert _frame_question("  How many  students?  ") == "How many students?"


def test_the_corrected_question_is_the_one_the_answer_came_from():
    conversation = ConversationRecord(id="c1", principal_id="researcher", selected_file_id=49)
    conversation.frames["master"] = QueryFrame(
        frame_key="master",
        topic="master-list students",
        query=QueryState(file_ids=(49,)),
        question="How many students came from River Bend?",
    )
    conversation.active_frame_key = "master"
    assert _question_being_corrected(conversation, 49, CORRECTION) == (
        "How many students came from River Bend?"
    )
    assert _question_being_corrected(conversation, 49, "How many from Stony Point?") is None


def test_an_older_conversation_falls_back_to_the_last_thing_asked():
    conversation = ConversationRecord(id="c2", principal_id="researcher", selected_file_id=49)
    conversation.recent_turns = [
        {"role": "user", "text": "How many students?"},
        {"role": "assistant", "text": "There are 3 students."},
    ]
    assert _question_being_corrected(conversation, 49, CORRECTION) == "How many students?"


@pytest.mark.asyncio
async def test_a_pasted_correction_asks_the_earlier_question_again(ai_settings):
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    first = await service.answer(_scope(), "How many students?", selected_file_id=49)
    assert first.status == "answered"
    second = await service.answer(
        _scope(), CORRECTION, conversation_id=first.conversation_id, selected_file_id=49
    )
    assert second.status == "answered"
    assert second.answer.startswith("Thank you for checking.")
    assert "How many students?" in second.answer
    conversation = await store.get_or_create(_scope(), first.conversation_id)
    user_turns = [item["text"] for item in conversation.recent_turns if item["role"] == "user"]
    assert user_turns[-1] == CORRECTION
    assert conversation.active_frame().question == "How many students?"
