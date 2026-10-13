import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


def test_year_followup_edits_active_query_not_new_topic():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students were admitted in 1910?"))
    assert first.facts[0]["value"] == 1
    second = asyncio.run(
        service.answer(_scope(), "What about 1911?", conversation_id=first.conversation_id)
    )
    assert second.status == "answered"
    assert second.reasoning_calls == 1
    assert second.facts[0]["value"] == 1
    assert second.plan is not None
    year = second.plan.step_map()["matches"].where[0]
    assert year.field == "admitted_date"
    assert year.value == 1911


def test_constraint_edit_keeps_prior_filter():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    assert first.facts[0]["value"] == 2
    second = asyncio.run(
        service.answer(_scope(), "Only from Garden River.", conversation_id=first.conversation_id)
    )
    assert second.facts[0]["value"] == 2
    fields = {item.field for item in second.plan.step_map()["matches"].where}
    assert "student_name" in fields
    assert "community" in fields


def test_those_deceased_followup():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    second = asyncio.run(
        service.answer(
            _scope(),
            "Which of those are recorded as deceased?",
            conversation_id=first.conversation_id,
        )
    )
    assert second.facts[0]["value"] == 1


def test_ordinal_resolves_against_result_set():
    service = _service()
    listed = asyncio.run(
        service.answer(_scope(), "List students whose names start with S.")
    )
    assert listed.rows
    second = asyncio.run(
        service.answer(_scope(), "Tell me about the second one.", conversation_id=listed.conversation_id)
    )
    assert second.status == "answered"
    assert second.reasoning_calls == 1
    assert second.planner_type == "semantic_compiler"
    assert second.rows[0]["canonical_name"] == "Sarah B"


def test_topic_switch_requires_a_new_dataset_scoped_conversation():
    service = _service()
    master = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    assert master.facts[0]["value"] == 2
    confirmed = asyncio.run(
        service.answer(
            _scope(),
            "How many records are in the confirmed deaths dataset?",
            conversation_id=master.conversation_id,
        )
    )
    assert confirmed.status == "dataset_mismatch"
    assert confirmed.suggested_file_id == 91
    conflict = asyncio.run(
        service.answer(
            _scope(),
            "How many records are there?",
            conversation_id=master.conversation_id,
            selected_file_id=91,
        )
    )
    assert conflict.status == "dataset_scope_conflict"


def test_exclude_unknown_discharge_is_deterministic():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    second = asyncio.run(
        service.answer(
            _scope(),
            "Exclude records with unknown discharge dates.",
            conversation_id=first.conversation_id,
        )
    )
    assert second.status == "answered"
    assert any(item.operator == "IS_KNOWN" for item in second.plan.step_map()["matches"].where)


def test_conversations_are_isolated_by_principal():
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store)
    first = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    other = AccessScope(principal_id="someone-else", allowed_file_ids=(49, 91))
    try:
        asyncio.run(service.answer(other, "What about 1911?", conversation_id=first.conversation_id))
    except PermissionError:
        return
    raise AssertionError("expected principal isolation")
