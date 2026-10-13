"""Answering a small list from its rows instead of by compiling a query.

The three death lists are tiny and messy. Compressing them into columns a query planner
can filter on is what produced every hard normalization problem in this project, so a list
small enough to read is handed to the model whole.

These lock the boundaries of that: when it takes over, when it must not, and that it
cannot widen what the reader is allowed to see.
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
from app.retrieval.context_builder import build_whole_list_packet  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402


@pytest.fixture(autouse=True)
def whole_list_enabled(monkeypatch):
    """The flag is off by default so the suite still exercises the query pipeline."""
    monkeypatch.setenv("WHOLE_LIST_ANSWER", "true")
    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service() -> tuple[AssistantTurnService, FakeReasoningProvider]:
    provider = FakeReasoningProvider()
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider
    )
    return service, provider


def _answer(service, question: str, file_id: int = 49):
    return asyncio.run(service.answer(_scope(), question, selected_file_id=file_id))


# ------------------------------------------------------------------ taking over


def test_a_data_question_on_a_small_list_is_answered_from_the_rows():
    service, _ = _service()
    result = _answer(service, "How many students are recorded as deceased?")
    assert result.status == "answered"
    assert result.planner_type == "whole_list"
    # The whole point: no query was compiled for this turn.
    assert result.plan is None


def test_the_answer_carries_the_rows_it_rests_on():
    service, _ = _service()
    result = _answer(service, "Which students are recorded as deceased?")
    assert result.evidence, "the rows the model read must travel with the answer"
    for item in result.evidence:
        assert item["source_id"].startswith("file:")


# ------------------------------------------------------------------ standing aside


def test_a_greeting_never_ships_the_list():
    service, _ = _service()
    result = _answer(service, "hello")
    assert result.status == "answered"
    assert result.planner_type != "whole_list"


def test_a_list_above_the_row_threshold_uses_the_query_path(monkeypatch):
    monkeypatch.setenv("WHOLE_LIST_MAX_ROWS", "1")
    get_settings.cache_clear()
    service, _ = _service()
    result = _answer(service, "How many students are recorded as deceased?")
    assert result.planner_type != "whole_list"


def test_an_oversized_payload_falls_back_even_under_the_row_threshold(monkeypatch):
    """The row count is a cheap pre-check; the rendered size is the real guard.

    Driven by making the packet builder decline, which is exactly what it does when the
    payload will not fit -- the alternative, a config small enough to trip on five fixture
    rows, would be below the floor the setting sensibly enforces.
    """
    import app.execution.turn as turn_module

    monkeypatch.setattr(turn_module, "build_whole_list_packet", lambda *a, **k: None)
    service, _ = _service()
    result = _answer(service, "How many students are recorded as deceased?")
    assert result.planner_type != "whole_list"
    assert result.plan is not None, "the query path must still produce an answer"


def test_the_flag_off_restores_the_query_path(monkeypatch):
    monkeypatch.setenv("WHOLE_LIST_ANSWER", "false")
    get_settings.cache_clear()
    service, _ = _service()
    result = _answer(service, "How many students are recorded as deceased?")
    assert result.planner_type != "whole_list"


# ------------------------------------------------------------------ the packet


def _rows(count: int) -> list[dict]:
    """Rows shaped the way the gateway returns them, so field projection resolves."""
    return [
        {
            "file_id": 49,
            "version": 1,
            "source_row_id": index,
            "canonical_name": f"person {index}",
            "canonical_community": "",
            "canonical_school": "",
            "row_data_normalized": {
                "canonical": {"display_name": f"Person {index}", "community": "Walpole Island"}
            },
        }
        for index in range(count)
    ]


def test_the_packet_refuses_rather_than_truncating_when_it_will_not_fit():
    from app.planning.catalog import static_catalog

    packet = build_whole_list_packet(
        "q", _rows(50), catalog=static_catalog(), fields=["student_name", "community"], max_chars=40
    )
    # None means "use the query path", never a silently shortened list -- a trimmed list
    # would turn "how many" into a count over a subset.
    assert packet is None


def test_a_packet_that_fits_keeps_every_row():
    from app.planning.catalog import static_catalog

    packet = build_whole_list_packet(
        "q", _rows(40), catalog=static_catalog(), fields=["student_name", "community"], max_chars=100_000
    )
    assert packet is not None
    assert len(packet.items) == 40
    assert packet.truncated is False


def test_derived_columns_are_not_sent_to_the_model():
    """Name and date parts exist for the query planner; the model reads the source."""
    service, _ = _service()
    from app.planning.catalog import static_catalog

    fields = asyncio.run(service._whole_list_fields(static_catalog(), 49))
    assert fields, "the source fields must still be projected"
    assert not any(name.endswith(("_year", "_month", "_day")) for name in fields)
    assert "name_residue" not in fields


# ------------------------------------------------------------------ authorization


def test_an_unauthorized_list_yields_nothing_through_the_gateway():
    service, _ = _service()
    denied = AccessScope(principal_id="principal-researcher", allowed_file_ids=(49,))
    rows = asyncio.run(service._gateway.list_records(denied, (91,), [], limit=100, offset=0))
    assert rows == []


def test_a_citation_outside_the_packet_is_stripped():
    from app.llm.citations import validate_synthesis
    from app.llm.schemas import MemoryUpdate, SynthesisAnswer
    from app.planning.catalog import static_catalog

    packet = build_whole_list_packet(
        "q", _rows(3), catalog=static_catalog(), fields=["student_name", "community"], max_chars=100_000
    )
    assert packet is not None
    answer = SynthesisAnswer(
        answer="…",
        citations=["file:49:v1:row:0", "file:93:v1:row:999"],
        inference=False,
        memory_update=MemoryUpdate(),
    )
    cleaned = validate_synthesis(answer, packet, _scope())
    assert cleaned.citations == ["file:49:v1:row:0"]
