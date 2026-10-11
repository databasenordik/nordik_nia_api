"""Follow-up questions, on the AI planner.

The same contracts as ``test_conversation_memory.py`` -- which runs on the legacy planner,
because the suite pins ``PLANNER_MODE=legacy`` unless a test opts in. That left the most
common way anyone uses a research assistant ("which of those died at school?") verified
only against the path Phase 3 deletes.

Every test here asks a question, then asks a second question that only makes sense in the
light of the first.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.fake import FakeReasoningProvider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

MASTER = 49


@pytest.fixture(autouse=True)
def _ai_planner(monkeypatch):
    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _scope(principal: str = "principal-researcher") -> AccessScope:
    return AccessScope(principal_id=principal, allowed_file_ids=(49, 91))


def _service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


def _ask(service, question, *, conversation_id=None, scope=None):
    return asyncio.run(
        service.answer(
            scope or _scope(),
            question,
            selected_file_id=MASTER,
            conversation_id=conversation_id,
        )
    )


def _where_fields(result) -> set[str]:
    return {
        leaf.field
        for step in (result.plan.steps if result.plan else [])
        for predicate in step.where
        for leaf in predicate.leaves()
        if leaf.field
    }


# ---------------------------------------------------------------- editing the last question


def test_a_bare_year_edits_the_previous_question():
    """"What about 1911?" is not a new topic; it is the same question, one year over."""
    service = _service()
    first = _ask(service, "How many students were admitted in 1910?")
    assert first.facts[0]["value"] == 1

    second = _ask(service, "What about 1911?", conversation_id=first.conversation_id)
    assert second.status == "answered"
    years = [
        leaf.value
        for step in second.plan.steps
        for predicate in step.where
        for leaf in predicate.leaves()
        if leaf.field == "admitted_date"
    ]
    assert years == [1911]


def test_an_added_constraint_keeps_the_one_already_there():
    """"Only from Garden River" narrows the previous set rather than replacing it."""
    service = _service()
    first = _ask(service, "How many students have names starting with S?")
    assert first.facts[0]["value"] == 2

    second = _ask(service, "Only from Garden River.", conversation_id=first.conversation_id)
    assert second.status == "answered"
    assert {"student_name", "community"} <= _where_fields(second)


def test_which_of_those_narrows_the_previous_answer():
    service = _service()
    first = _ask(service, "How many students have names starting with S?")
    assert first.facts[0]["value"] == 2

    second = _ask(
        service,
        "Which of those are recorded as deceased?",
        conversation_id=first.conversation_id,
    )
    assert second.facts[0]["value"] == 1


# ---------------------------------------------------------------- pointing at a listed row


def test_an_ordinal_resolves_against_the_list_just_shown():
    service = _service()
    listed = _ask(service, "List students whose names start with S.")
    assert [row["canonical_name"] for row in listed.rows] == ["Samuel A", "Sarah B"]

    second = _ask(service, "Tell me about the second one.", conversation_id=listed.conversation_id)
    assert second.status == "answered"
    assert second.rows[0]["canonical_name"] == "Sarah B"


# ---------------------------------------------------------------- carried across turns


def test_a_follow_up_stays_on_the_selected_list():
    """Scope is the UI's to set, and a second turn must not widen it."""
    service = _service()
    first = _ask(service, "How many students have names starting with S?")
    second = _ask(service, "What about 1911?", conversation_id=first.conversation_id)
    assert second.selected_file_id == MASTER
    if second.plan is not None:
        assert tuple(second.plan.scope.file_ids) == (MASTER,)


def test_one_researcher_cannot_continue_another_s_conversation():
    """Refused outright, not quietly restarted.

    A second researcher handing over someone else's conversation id must not be given a
    fresh conversation either -- that would look like it worked. The store fails closed.
    """
    service = _service()
    first = _ask(service, "How many students have names starting with S?")
    with pytest.raises(PermissionError):
        _ask(
            service,
            "What about 1911?",
            conversation_id=first.conversation_id,
            scope=_scope("principal-someone-else"),
        )
