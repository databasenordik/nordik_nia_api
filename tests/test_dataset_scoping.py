import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.filters import predicate_matches
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.memory.types import ConversationRecord
from app.planning.catalog import static_catalog
from app.planning.plan_schema import FilterOperator, Predicate
from app.planning.trusted_overrides import apply_trusted_overrides
from app.planning.turn_schema import QueryAction, TurnPlan
from app.security.access_scope import AccessScope
from app.execution.turn import _field_dataset_suggestion
from app.security.livekit_token import (
    decode_livekit_token,
    mint_livekit_token,
    scope_from_metadata,
    selected_file_id_from_metadata,
)


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


def test_selected_dataset_long_field_label_beats_other_dataset_short_alias():
    catalog = static_catalog()
    assert _field_dataset_suggestion("What is the most frequent location of death?", catalog, 91) is None
    assert _field_dataset_suggestion("How many records contain investigative notes?", catalog, 91) is None


def _service(store=None) -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(),
        store=store or InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )


def test_default_conversation_is_bound_to_master_list():
    store = InMemoryMemoryStore()
    result = asyncio.run(_service(store).answer(_scope(), "How many students are there?"))
    record = asyncio.run(store.get(_scope(), result.conversation_id))

    assert result.selected_file_id == 49
    assert result.plan.scope.file_ids == (49,)
    assert record is not None and record.selected_file_id == 49


def test_confirmed_deaths_selection_constrains_plan_and_distinct_causes():
    result = asyncio.run(
        _service().answer(
            _scope(),
            "What different causes of death are recorded?",
            selected_file_id=91,
        )
    )

    assert result.status == "answered"
    assert result.plan.scope.file_ids == (91,)
    assert "cause" in result.answer.lower()


def test_conversation_cannot_be_rebound_to_another_list():
    service = _service()
    first = asyncio.run(
        service.answer(_scope(), "How many students are there?", selected_file_id=49)
    )
    second = asyncio.run(
        service.answer(
            _scope(),
            "How many records are there?",
            conversation_id=first.conversation_id,
            selected_file_id=91,
        )
    )

    assert second.status == "dataset_scope_conflict"
    assert second.selected_file_id == 49
    assert second.suggested_file_id == 91


def test_naming_another_list_returns_a_switch_suggestion():
    """Explicitly naming the other list still asks the researcher to switch."""
    named = asyncio.run(
        _service().answer(
            _scope(),
            "How many records are in confirmed deaths?",
            selected_file_id=49,
        )
    )

    assert named.status == "dataset_mismatch" and named.suggested_file_id == 91


def test_field_only_on_the_other_list_is_explained_not_bounced():
    """A field the selected list lacks is explained, naming the list that has it.

    Bouncing on a field mention produced false positives - "stayed at the school",
    "burial location" - that stopped answerable Master List questions before any
    planning happened.
    """
    result = asyncio.run(
        _service().answer(
            _scope(),
            "What different causes of death are recorded?",
            selected_file_id=49,
        )
    )

    assert result.status != "dataset_mismatch"
    assert result.suggested_file_id is None


def test_cross_list_request_is_answered_once_per_list():
    """Two named lists produce two single-list executions and one merged answer."""
    result = asyncio.run(
        _service().answer(
            _scope(),
            "Compare the master list and confirmed deaths.",
            selected_file_id=49,
        )
    )

    assert result.status == "answered"
    assert result.plan is None
    assert "Student master list" in result.answer
    assert "Confirmed deaths" in result.answer


def test_legacy_conversation_is_read_only_for_continuation():
    store = InMemoryMemoryStore()
    legacy = ConversationRecord(
        id="00000000-0000-0000-0000-000000000001",
        principal_id="researcher",
        recent_turns=[{"role": "user", "text": "old mixed-list question"}],
    )
    asyncio.run(store.save(legacy))

    result = asyncio.run(
        _service(store).answer(
            _scope(),
            "Continue",
            conversation_id=legacy.id,
            selected_file_id=49,
        )
    )

    assert result.status == "legacy_unscoped_conversation"
    assert legacy.selected_file_id is None


def test_all_information_expands_from_only_the_scoped_catalog():
    catalog = static_catalog().for_scope(_scope().narrow((91,)))
    turn = TurnPlan(
        normalized_request="all information for Albert Penance",
        actions=[QueryAction(datasets=[91], goal="search", search_text="Albert Penance")],
        final_response="deterministic",
        confidence=0.99,
    )

    updated = apply_trusted_overrides(
        turn,
        "Show all information for Albert Penance",
        has_active_query=False,
        catalog=catalog,
    )
    action = updated.query_actions()[0]
    expected = {field.semantic_field for field in catalog.fields_for(91)}

    assert set(action.requested_fields) == expected
    assert len(action.requested_fields) > 16
    assert "death_date" in action.requested_fields
    assert "death_details" not in action.requested_fields


def test_typed_numeric_and_date_ranges_are_supported_in_memory():
    catalog = static_catalog()
    age = catalog.resolve_field(91, "age_at_death")
    death = catalog.resolve_field(91, "death_date")
    record = {
        "file_id": 91,
        "row_data_normalized": {
            "canonical": {"dates": {"death": "1903-07-30"}},
            "fields": {"AGE AT DEATH": "14 years"},
        },
    }

    assert age is not None and predicate_matches(
        record, Predicate(field="age_at_death", operator=FilterOperator.GREATER_THAN, value=10), age
    )
    assert death is not None and predicate_matches(
        record, Predicate(field="death_date", operator=FilterOperator.DATE_RANGE, value=[1900, 1905]), death
    )


def test_livekit_metadata_is_narrowed_to_selected_list():
    token = mint_livekit_token(
        _scope().narrow((91,)),
        room_name="nia-researcher-test",
        identity="user-researcher",
        selected_file_id=91,
    )
    metadata = decode_livekit_token(token)["metadata"]

    assert scope_from_metadata(metadata).allowed_file_ids == (91,)
    assert selected_file_id_from_metadata(metadata) == 91
