import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.retrieval.context_builder import build_evidence_packet
from app.retrieval.hybrid import HybridRetriever
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def test_mention_plans_fts_and_fuzzy():
    parsed = parse_deterministic("Which records mention tuberculosis?", _scope(), static_catalog())
    assert parsed.plan is not None
    assert parsed.reasoning_calls == 0
    assert "FULL_TEXT_SEARCH" in parsed.plan.op_names()
    assert "FUZZY_SEARCH" in parsed.plan.op_names()
    assert "GET_EVIDENCE" in parsed.plan.op_names()


def test_mention_tuberculosis_returns_authorized_hits_only():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = asyncio.run(service.answer(_scope(), "Which records mention tuberculosis?"))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    names = {row.get("canonical_name") for row in result.rows}
    assert "Sarah B" in names
    assert "Private Name" not in names
    source_ids = {item["source_id"] for item in result.evidence}
    assert any(item.startswith("file:49:") or item.startswith("file:91:") for item in source_ids)
    assert not any("file:93:" in item for item in source_ids)
    assert "row_data" not in str(result.evidence)
    assert "row_data_normalized" not in str(result.evidence)


def test_exact_unique_skips_fuzzy_branch():
    gateway = MemoryDataGateway()
    retriever = HybridRetriever(gateway)
    result = asyncio.run(retriever.retrieve(_scope(), (49, 91), "Samuel A", want_fts=True, want_fuzzy=True))
    assert result.methods_run == ["exact"]
    assert "fuzzy" in result.cancelled
    assert "fts" in result.cancelled
    assert "fuzzy" not in gateway.calls
    assert len(result.hits) == 1
    assert result.hits[0].canonical_name == "Samuel A"


def _turn_service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


def test_fuzzy_resolves_community_typo():
    result = asyncio.run(_turn_service().answer(_scope(), "How many students are from Gardin River?"))
    assert result.status == "answered"
    assert result.facts[0]["value"] == 4


def test_named_lookup_finds_student():
    result = asyncio.run(_turn_service().answer(_scope(), "Find Samuel A"))
    assert result.status == "answered"
    assert any(row.get("canonical_name") == "Samuel A" for row in result.rows)


def test_context_builder_respects_token_budget_and_omits_full_rows():
    gateway = MemoryDataGateway()
    retriever = HybridRetriever(gateway)
    hits = asyncio.run(retriever.retrieve(_scope(), (49,), "tuberculosis", want_exact=False)).hits
    packet = build_evidence_packet("tuberculosis", hits, requested_fields=["student_name", "cause_of_death"])
    assert packet.items
    dumped = packet.model_dump()
    assert "row_data_normalized" not in str(dumped)
    assert all("student_name" in item["fields"] for item in dumped["items"])


def test_raw_fields_only_after_ids_selected():
    gateway = MemoryDataGateway()
    rows = asyncio.run(gateway.get_raw_fields(_scope(), [1102], ["Notes", "Cause of Death"]))
    assert len(rows) == 1
    assert rows[0]["selected"]["Cause of Death"] == "tuberculosis"
    assert set(rows[0]["selected"]) <= {"Notes", "Cause of Death"}


def test_private_mention_does_not_surface_restricted_row():
    result = asyncio.run(_turn_service().answer(_scope(), "Which records mention restricted?"))
    names = {row.get("canonical_name") for row in result.rows}
    assert "Private Name" not in names
