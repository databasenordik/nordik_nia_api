import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.api.assistant import _memory_store, _turn_service
from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.main import create_app
from app.memory.postgres import record_from_snapshot, snapshot_record
from app.memory.store import InMemoryMemoryStore
from app.realtime.agent import ResearchVoiceAgent
from app.realtime.turn_hook import AssistantTurnHook
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _client(store: InMemoryMemoryStore) -> TestClient:
    app = create_app()

    def scope() -> AccessScope:
        return _scope()

    def service() -> AssistantTurnService:
        return AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())

    app.dependency_overrides[get_access_scope] = scope
    app.dependency_overrides[_memory_store] = lambda: store
    app.dependency_overrides[_turn_service] = service
    return TestClient(app)


def test_text_then_voice_share_one_thread():
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    text = asyncio.run(service.answer(_scope(), "How many students have names starting with S?", mode="text"))
    voice = asyncio.run(
        service.answer(
            _scope(),
            "Which of those are recorded as deceased?",
            mode="voice",
            conversation_id=text.conversation_id,
        )
    )
    assert voice.conversation_id == text.conversation_id
    record = asyncio.run(store.get(_scope(), text.conversation_id or ""))
    assert record is not None
    modes = [turn.get("mode") for turn in record.recent_turns]
    assert "text" in modes
    assert "voice" in modes
    assert record.active_frame_key == "master"


def test_partial_transcript_is_not_persisted():
    store = InMemoryMemoryStore()
    hook = AssistantTurnHook(
        AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    )
    events: list[dict] = []

    async def notify(payload: dict) -> None:
        events.append(payload)

    voice = ResearchVoiceAgent(hook, notify=notify)

    async def run() -> None:
        await voice.publish_transcript("How many students admitted before", final=False)
        await voice.publish_transcript("How many students have names starting with S?", final=True)
        await voice.handle_final_transcript("How many students have names starting with S?", _scope())

    asyncio.run(run())
    statuses = [item.get("status") for item in events if item.get("type") == "transcript"]
    assert "partial" in statuses
    assert "final" in statuses
    records = asyncio.run(store.list_for_principal(_scope()))
    assert records
    texts = [turn["text"] for turn in records[0].recent_turns if turn["role"] == "user"]
    assert "How many students have names starting with S?" in texts
    assert "How many students admitted before" not in texts


def test_conversation_api_returns_mode_and_citations():
    store = InMemoryMemoryStore()
    client = _client(store)
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    conversation_id = asked.json()["conversation_id"]
    detail = client.get(f"/api/assistant/conversations/{conversation_id}")
    assert detail.status_code == 200
    turns = detail.json()["turns"]
    assert all("mode" in turn for turn in turns)
    assert any(turn["mode"] == "text" for turn in turns)
    voice = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    asyncio.run(
        voice.answer(
            _scope(),
            "Summarize what the records say about causes of death.",
            mode="voice",
            conversation_id=conversation_id,
        )
    )
    again = client.get(f"/api/assistant/conversations/{conversation_id}")
    modes = {turn["mode"] for turn in again.json()["turns"]}
    assert "text" in modes
    assert "voice" in modes


def test_reconnect_snapshot_restores_frames():
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    first = asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    record = asyncio.run(store.get(_scope(), first.conversation_id or ""))
    assert record is not None
    payload = snapshot_record(record)
    restored = record_from_snapshot(payload, record.principal_id)
    assert restored.id == record.id
    assert restored.active_frame_key == record.active_frame_key
    assert restored.frames.keys() == record.frames.keys()
    assert restored.recent_turns[-1]["mode"] == "text"


def test_voice_reply_emits_citations_while_speaking():
    events: list[dict] = []

    async def notify(payload: dict) -> None:
        events.append(payload)

    hook = AssistantTurnHook(
        AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider())
    )
    voice = ResearchVoiceAgent(hook, notify=notify)
    asyncio.run(
        voice.handle_final_transcript(
            "Summarize what the records say about causes of death.",
            _scope(),
            selected_file_id=91,
        )
    )
    kinds = [item.get("type") for item in events]
    assert "assistant_delta" in kinds or "assistant_reply" in kinds
    assert "citation" in kinds or any(item.get("citations") for item in events if item.get("type") == "assistant_reply")


def test_spoken_voice_switch_skips_assistant_core_and_greets_with_clean_text():
    store = InMemoryMemoryStore()
    events: list[dict] = []
    selected: list[str] = []
    spoken: list[str] = []

    async def notify(payload: dict) -> None:
        events.append(payload)

    async def switch_voice(preference: str) -> None:
        selected.append(preference)

    async def speak(text: str) -> None:
        spoken.append(text)

    hook = AssistantTurnHook(
        AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    )
    voice = ResearchVoiceAgent(hook, notify=notify, switch_voice=switch_voice)
    reply = asyncio.run(
        voice.handle_final_transcript(
            "Please use a woman's voice.",
            _scope(),
            conversation_id="conversation-1",
            on_spoken_phrase=speak,
        )
    )

    assert selected == ["female"]
    assert spoken == ["Hi, I'm here."]
    assert reply == "Hi, I'm here."
    assert not any(tag in reply for tag in ("<soft>", "[pause]", "<emphasis>"))
    assistant_events = [item for item in events if item.get("type", "").startswith("assistant_")]
    assert assistant_events
    assert all("<" not in item.get("text", "") and "[pause]" not in item.get("text", "") for item in assistant_events)
    assert asyncio.run(store.list_for_principal(_scope())) == []
