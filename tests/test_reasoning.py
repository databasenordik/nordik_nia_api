import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.citations import validate_synthesis
from app.llm.fake import FakeReasoningProvider
from app.llm.schemas import MemoryUpdate, SynthesisAnswer
from app.memory.store import InMemoryMemoryStore
from app.retrieval.types import EvidenceItem, EvidencePacket
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service(reasoner: FakeReasoningProvider | None = None) -> tuple[AssistantTurnService, FakeReasoningProvider]:
    provider = reasoner or FakeReasoningProvider()
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    return service, provider


def test_exact_count_uses_fast_path_and_skips_synthesis():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert provider.normalize_calls == 0
    assert provider.plan_calls == 0
    assert provider.synthesis_calls == 0
    assert result.facts[0]["value"] == 2


def test_synthesis_question_uses_planner_and_one_synthesis_call():
    service, provider = _service()
    result = asyncio.run(
        service.answer(
            _scope(),
            "Summarize what the records say about causes of death.",
            selected_file_id=91,
        )
    )
    assert result.status == "answered"
    assert result.planner_type == "semantic_compiler"
    assert provider.normalize_calls == 1
    assert provider.plan_calls == 0
    assert provider.synthesis_calls == 1
    assert result.reasoning_calls == 2
    assert "Sarah B" in result.answer or "cause" in result.answer.lower()
    assert result.citations
    assert "file:93:" not in " ".join(result.citations)
    assert all(item.startswith("file:91:") for item in result.citations)


def test_mixed_count_and_synthesis_keeps_executor_count():
    service, provider = _service()
    result = asyncio.run(
        service.answer(
            _scope(),
            "How many students whose names start with S are deceased, and what do the records say?",
        )
    )
    assert result.status == "answered"
    assert result.reasoning_calls == 2
    assert provider.normalize_calls == 1
    assert provider.plan_calls == 0
    assert any(item["name"] == "count" and item["value"] == 1 for item in result.facts)
    assert "count=1" in result.answer.lower() or "1" in result.answer


def test_citation_validator_strips_unknown_and_private_ids():
    packet = EvidencePacket(
        question="q",
        items=(
            EvidenceItem(
                source_id="file:49:v3:row:1102",
                file_id=49,
                version=3,
                source_row_id=1102,
                fields={"student_name": "Sarah B"},
            ),
        ),
    )
    raw = SynthesisAnswer(
        answer="text",
        citations=["file:49:v3:row:1102", "file:93:v1:row:1108", "file:49:v3:row:999"],
        memory_update=MemoryUpdate(active_file_ids=[49, 93], last_source_ids=["file:93:v1:row:1108"]),
    )
    cleaned = validate_synthesis(raw, packet, _scope())
    assert cleaned.citations == ["file:49:v3:row:1102"]
    assert cleaned.memory_update.last_source_ids == []
    assert cleaned.memory_update.active_file_ids == [49]


def test_without_reasoner_synthesis_stays_unplanned():
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore())
    result = asyncio.run(
        service.answer(
            _scope(),
            "Summarize what the records say about causes of death.",
            selected_file_id=91,
        )
    )
    assert result.status == "needs_reasoning"
    assert result.reasoning_calls == 0


def test_xai_reasoner_client_helpers_do_not_shadow_imported_roles(monkeypatch):
    import types

    chat = types.ModuleType("xai_sdk.chat")
    chat.system = lambda text: ("system", text)
    chat.user = lambda text: ("user", text)
    sdk = types.ModuleType("xai_sdk")

    class AsyncClient:
        def __init__(self, api_key):
            self.api_key = api_key

    sdk.AsyncClient = AsyncClient
    sdk.chat = chat
    monkeypatch.setitem(sys.modules, "xai_sdk", sdk)
    monkeypatch.setitem(sys.modules, "xai_sdk.chat", chat)

    from app.llm.xai_reasoner import XAIReasoningProvider

    client, mods = XAIReasoningProvider(api_key="test-key")._client()
    assert client.api_key == "test-key"
    assert mods.system("s") == ("system", "s")
    assert mods.user("u") == ("user", "u")
