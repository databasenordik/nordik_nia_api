import asyncio
import sys

import pytest

from repo_paths import BACKEND_ROOT  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient

from app.api.assistant import _turn_service
from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.main import create_app
from app.memory.store import InMemoryMemoryStore
from app.realtime.agent import (
    PREEMPTIVE_GENERATION,
    build_turn_service,
    message_transcript,
    session_options,
)
from app.realtime.states import VoiceState, VoiceStateMachine
from app.realtime.turn_hook import AssistantTurnHook
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope
from app.security.livekit_token import (
    AGENT_NAME,
    assigned_room_name,
    conversation_id_from_metadata,
    decode_livekit_token,
    mint_livekit_token,
    principal_room_prefix,
    scope_from_metadata,
)


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _client() -> TestClient:
    app = create_app()

    def scope() -> AccessScope:
        return _scope()

    def service() -> AssistantTurnService:
        return AssistantTurnService(
            MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
        )

    app.dependency_overrides[get_access_scope] = scope
    app.dependency_overrides[_turn_service] = service
    return TestClient(app)


def test_token_requires_auth():
    response = TestClient(create_app()).post("/api/livekit/token", json={})
    assert response.status_code == 401


def test_token_is_short_lived_and_carries_scope():
    client = _client()
    response = client.post(
        "/api/livekit/token",
        json={"conversation_id": "conv-1"},
        headers={"Authorization": "Bearer unused"},
    )
    # override supplies scope without parsing bearer; mint still runs
    assert response.status_code == 200
    body = response.json()
    assert body["ttl_seconds"] <= 15 * 60
    assert body["agent_name"] == "nia-research-assistant"
    assert body["server_url"].startswith("ws")
    decoded = decode_livekit_token(body["participant_token"])
    assert decoded["video"]["roomJoin"] is True
    assert decoded["video"]["canPublish"] is True
    scope = scope_from_metadata(decoded["metadata"])
    assert scope.principal_id == "principal-researcher"
    assert scope.allowed_file_ids == (49,)
    assert body["selected_file_id"] == 49
    assert decoded["exp"] - decoded["nbf"] == 15 * 60
    assert decoded["roomConfig"]["agents"][0]["agentName"] == AGENT_NAME
    assert body["room_name"].startswith(principal_room_prefix(_scope()))


def test_token_rejects_foreign_room_name():
    client = _client()
    response = client.post(
        "/api/livekit/token",
        json={"room_name": "nia-someone-else-room"},
        headers={"Authorization": "Bearer unused"},
    )
    assert response.status_code == 400


def test_token_accepts_own_room_prefix():
    room = assigned_room_name(_scope(), conversation_id="conv-reconnect")
    client = _client()
    response = client.post(
        "/api/livekit/token",
        json={"room_name": room, "conversation_id": "conv-reconnect"},
        headers={"Authorization": "Bearer unused"},
    )
    assert response.status_code == 200
    assert response.json()["room_name"] == room


def test_token_cannot_be_minted_with_broader_client_permissions():
    token = mint_livekit_token(_scope(), room_name="r", identity="u")
    decoded = decode_livekit_token(token)
    scope = scope_from_metadata(decoded["metadata"])
    assert 93 not in scope.allowed_file_ids


def test_preemptive_generation_is_disabled():
    assert PREEMPTIVE_GENERATION is False
    options = session_options()
    assert options["preemptive_generation"]["enabled"] is False
    assert options["stt"] == "whisper.cpp"
    assert options["vad"] == "silero"
    assert options["turn_detection"] == "vad"
    assert options["tts"] == "xai"
    assert "assemblyai" not in str(options).lower()
    assert "fishaudio" not in str(options).lower()


def test_turn_hook_runs_core_before_any_reply():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    hook = AssistantTurnHook(service)
    assert hook.started_replies == []
    result = asyncio.run(hook.on_user_turn_completed("How many students have names starting with S?", _scope()))
    assert result.retrieval_completed is True
    assert hook.started_replies
    assert result.reply == hook.started_replies[0]
    assert "2" in result.reply
    assert result.core.reasoning_calls == 0


def test_voice_state_machine_happy_path_and_interrupt():
    machine = VoiceStateMachine()
    assert machine.state is VoiceState.DISCONNECTED
    machine.transition(VoiceState.CONNECTING)
    machine.transition(VoiceState.READY)
    machine.transition(VoiceState.LISTENING)
    machine.transition(VoiceState.USER_SPEAKING)
    machine.transition(VoiceState.PROCESSING)
    machine.transition(VoiceState.ASSISTANT_SPEAKING)
    machine.transition(VoiceState.INTERRUPTED)
    machine.transition(VoiceState.LISTENING)
    try:
        machine.transition(VoiceState.CONNECTING)
    except ValueError:
        return
    raise AssertionError("illegal transition should fail")


def test_agent_module_does_not_enable_cloud_inference():
    source = (BACKEND_ROOT / "app" / "realtime" / "agent.py").read_text(
        encoding="utf-8"
    )
    lowered = source.lower()
    assert "assemblyai" not in lowered
    assert "fishaudio" not in lowered
    assert "gemma" not in lowered
    assert "inference.turndetector" not in lowered
    assert "inference.stt" not in lowered
    assert "preemptive_generation=false" in lowered.replace(" ", "")
    assert "silero" in lowered
    assert "whisper" in lowered
    assert "create_livekit_stt" in source
    assert "async def _agent_entry" in source
    assert "def entry(" not in source


def test_livekit_entrypoint_is_pickleable():
    import pickle

    from app.realtime.agent import _agent_entry

    restored = pickle.loads(pickle.dumps(_agent_entry))
    assert restored.__name__ == "_agent_entry"
    assert restored.__module__ == "app.realtime.agent"


def test_scope_is_read_from_participant_metadata():
    token = mint_livekit_token(
        _scope(),
        room_name="nia-principal-researcher-conv-1",
        identity="user-1",
        conversation_id="conv-1",
    )
    metadata = decode_livekit_token(token)["metadata"]
    scope = scope_from_metadata(metadata)
    assert scope.principal_id == "principal-researcher"
    assert conversation_id_from_metadata(metadata) == "conv-1"


def test_message_transcript_prefers_text_content():
    class _Msg:
        text_content = "How many students"
        content = ["ignored"]

    assert message_transcript(_Msg()) == "How many students"


@pytest.mark.asyncio
async def test_production_voice_fails_closed_when_database_is_unavailable(monkeypatch):
    from app.config import get_settings

    async def unavailable():
        raise RuntimeError("database unavailable")

    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setattr("app.db.pool.init_pool", unavailable)
    get_settings.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="database unavailable"):
            await build_turn_service()
    finally:
        get_settings.cache_clear()
