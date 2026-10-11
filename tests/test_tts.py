import asyncio
import base64
import json
import sys

import pytest

from repo_paths import BACKEND_ROOT, monorepo_file  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from app.concurrency.cancellation import CancellationToken
from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.realtime.agent import session_options
from app.realtime.turn_hook import AssistantTurnHook
from app.security.access_scope import AccessScope
from app.tts.factory import get_tts_provider
from app.tts.fake import FakeTTSProvider
from app.tts.expressive import add_contextual_voice_tags, strip_voice_tags
from app.tts.humanize import humanize_for_speech
from app.tts.kokoro import KokoroTTSProvider
from app.tts.livekit_adapter import prepare_speech_text
from app.tts.phrases import PhraseBuffer, PhraseQueue, SpokenPhrase
from app.tts.session import BargeInController, SpeechSession
from app.tts.xai_tts import XAITTSProvider
from app.tts.voice_control import requested_voice_preference


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


class ScriptedTTSSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._outgoing: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)
        payload = json.loads(message)
        if payload.get("type") == "text.done":
            pcm = base64.b64encode(b"PCM-AUDIO").decode("ascii")
            await self._outgoing.put(json.dumps({"type": "audio.delta", "delta": pcm}))
            await self._outgoing.put(json.dumps({"type": "audio.done"}))

    async def recv(self) -> str:
        return await self._outgoing.get()

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_xai_tts_streams_pcm_and_preconnects():
    socket = ScriptedTTSSocket()

    async def open_socket(url: str, headers: dict, timeout: float):
        assert url.startswith("wss://api.x.ai/v1/tts")
        assert "rex" in url
        return socket

    provider = XAITTSProvider(api_key="test-key", open_socket=open_socket)
    await provider.connect_session()
    assert provider.connected is True
    chunks = [chunk async for chunk in provider.synthesize_stream("There are two students.")]
    assert chunks == [b"PCM-AUDIO"]
    assert provider.first_audio_ms is not None
    assert any("text.delta" in item for item in socket.sent)
    await provider.cancel_generation()
    assert socket.closed is True


def test_humanize_strips_filler_and_source_ids():
    spoken = humanize_for_speech(
        "Certainly! There are 2 students [file:49:v3:row:1102] in the file."
    )
    assert spoken.startswith("There are 2 students")
    assert "file:49" not in spoken
    assert "Certainly" not in spoken


def test_contextual_xai_tags_are_spoken_only_and_restrained():
    sensitive = add_contextual_voice_tags("The record says this student is deceased.")
    count = add_contextual_voice_tags("There are 2,781 students in the master list.")
    ordinary = add_contextual_voice_tags("Nia can help with the authorized records.")

    assert sensitive == "<soft>The record says this student is deceased.</soft>"
    assert count == "There are <emphasis>2,781</emphasis> students in the master list."
    assert ordinary == "Nia can help with the authorized records."
    combined = " ".join((sensitive, count, ordinary))
    assert not any(tag in combined for tag in ("[laugh]", "[giggle]", "<sing-song>"))
    assert strip_voice_tags(combined) == (
        "The record says this student is deceased. "
        "There are 2,781 students in the master list. "
        "Nia can help with the authorized records."
    )


def test_prepare_speech_text_adds_tags_only_when_explicitly_enabled():
    displayed = "There are 12 students in NIA."
    assert prepare_speech_text(displayed) == "There are 12 students in Nia."
    assert prepare_speech_text(displayed, expressive=True) == (
        "There are <emphasis>12</emphasis> students in Nia."
    )
    assert displayed == "There are 12 students in NIA."


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("switch to female", "female"),
        ("Use a woman's voice", "female"),
        ("Make the voice female", "female"),
        ("Could you sound like a woman?", "female"),
        ("female voice please", "female"),
        ("change the voice to a man", "male"),
        ("use a male voice", "male"),
        ("Tell me about female voices", None),
        ("What voices are available?", None),
    ],
)
def test_voice_preference_intent_detection(utterance, expected):
    assert requested_voice_preference(utterance) == expected


@pytest.mark.asyncio
async def test_xai_voice_switch_reconnects_before_next_utterance():
    sockets = [ScriptedTTSSocket(), ScriptedTTSSocket()]
    urls: list[str] = []

    async def open_socket(url: str, headers: dict, timeout: float):
        urls.append(url)
        return sockets[len(urls) - 1]

    provider = XAITTSProvider(api_key="test-key", voice="rex", open_socket=open_socket)
    await provider.connect_session()
    await provider.set_voice("ara")
    assert sockets[0].closed is True
    chunks = [chunk async for chunk in provider.synthesize_stream("Hi, I'm here.")]
    assert chunks == [b"PCM-AUDIO"]
    assert "voice=rex" in urls[0]
    assert "voice=ara" in urls[1]


def test_phrase_buffer_emits_sentences_not_words():
    buffer = PhraseBuffer()
    buffer.start("g1")
    assert buffer.push("John") == []
    first = buffer.push(" Smith joined in 2023. He later")
    assert len(first) == 1
    assert first[0].sequence_id == 1
    assert "2023" in first[0].text
    rest = buffer.flush()
    assert rest and "later" in rest[0].text


def test_phrase_queue_bounds_and_flush():
    queue = PhraseQueue(max_phrases=2)
    gen = queue.start_generation()
    assert queue.put(SpokenPhrase(1, gen, "One."))
    assert queue.put(SpokenPhrase(2, gen, "Two."))
    assert queue.put(SpokenPhrase(3, gen, "Three.")) is False
    dropped = queue.flush()
    assert [item.text for item in dropped] == ["One.", "Two."]
    assert len(queue) == 0


@pytest.mark.asyncio
async def test_speech_session_refuses_text_mode():
    with pytest.raises(RuntimeError, match="text mode"):
        SpeechSession(FakeTTSProvider(), mode="text")


@pytest.mark.asyncio
async def test_barge_in_cancels_tts_and_keeps_query_state():
    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    first = await service.answer(_scope(), "How many students have names starting with S?")
    assert first.status == "answered"
    conversation = await store.get(_scope(), first.conversation_id)
    assert conversation is not None
    frames_before = dict(conversation.frames)

    provider = FakeTTSProvider()
    token = CancellationToken()
    speech = SpeechSession(provider, cancellation=token)
    await speech.connect()
    controller = BargeInController(speech, token)
    speech.begin_turn()
    await speech.enqueue_text("One sentence. Two sentence. Three sentence.")
    assert len(speech.queue) >= 2
    await controller.interrupt()
    leftover = [chunk async for chunk in speech.play()]
    assert leftover == []
    assert controller.interrupted is True
    assert speech.interrupted is True
    assert provider.cancelled is True
    assert token.cancelled is True
    assert controller.query_state_kept is True
    after = await store.get(_scope(), first.conversation_id)
    assert after is not None
    assert after.frames == frames_before
    assert after.active_frame_key == conversation.active_frame_key


@pytest.mark.asyncio
async def test_voice_phrases_start_after_retrieval_and_skip_ids():
    spoken: list[str] = []

    async def on_phrase(text: str) -> None:
        spoken.append(text)

    hook = AssistantTurnHook(
        AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider())
    )
    result = await hook.on_user_turn_completed(
        "How many students have names starting with S?",
        _scope(),
        on_spoken_phrase=on_phrase,
    )
    assert result.retrieval_completed is True
    assert spoken
    assert all("file:" not in item for item in spoken)
    assert result.core.spoken_text
    assert "file:" not in (result.core.spoken_text or "")


@pytest.mark.asyncio
async def test_voice_synthesis_streams_phrases_before_join():
    spoken: list[str] = []

    async def on_phrase(text: str) -> None:
        spoken.append(text)

    provider = FakeReasoningProvider()
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    result = await service.answer(
        _scope(),
        "Summarize what the records say about causes of death.",
        mode="voice",
        selected_file_id=91,
        on_spoken_phrase=on_phrase,
    )
    assert provider.stream_calls == 1
    assert spoken
    assert result.spoken_text
    assert all("file:" not in item for item in spoken)


def test_factory_switch_does_not_import_retrieval():
    xai = get_tts_provider("fake")
    assert isinstance(xai, FakeTTSProvider)
    kokoro = get_tts_provider("kokoro")
    assert isinstance(kokoro, KokoroTTSProvider)
    source = BACKEND_ROOT / "app" / "tts" / "factory.py"
    text = source.read_text(encoding="utf-8")
    assert "retrieval" not in text
    assert "planning" not in text


def test_session_options_follow_tts_provider_setting():
    options = session_options()
    assert options["tts"] in {"xai", "kokoro", "fake"}
    assert options["preemptive_generation"]["enabled"] is False


@pytest.mark.asyncio
async def test_kokoro_streams_pcm_from_openai_speech_endpoint():
    httpx = pytest.importorskip("httpx")
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    seen: list[dict] = []

    async def health(_request):
        return JSONResponse({"status": "ok"})

    async def speech(request):
        body = await request.json()
        seen.append(body)

        async def chunks():
            yield b"KOKORO-PCM-1"
            yield b"KOKORO-PCM-2"

        return StreamingResponse(chunks(), media_type="audio/pcm")

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/audio/speech", speech, methods=["POST"]),
        ]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://kokoro") as client:
        provider = KokoroTTSProvider(base_url="http://kokoro", client=client)
        await provider.connect_session()
        chunks = [chunk async for chunk in provider.synthesize_stream("There are two students.")]
    assert provider.connected is True
    joined = b"".join(chunks)
    assert b"KOKORO-PCM-1" in joined
    assert b"KOKORO-PCM-2" in joined
    assert seen[0]["model"] == "kokoro"
    assert seen[0]["input"] == "There are two students."
    assert seen[0]["response_format"] == "pcm"
    assert seen[0]["stream"] is True


@pytest.mark.asyncio
async def test_kokoro_cancel_stops_stream():
    httpx = pytest.importorskip("httpx")
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse({"status": "ok"})

    async def speech(_request):
        async def chunks():
            yield b"ONE"
            await asyncio.sleep(0)
            yield b"TWO"

        return StreamingResponse(chunks(), media_type="audio/pcm")

    app = Starlette(
        routes=[
            Route("/health", health),
            Route("/v1/audio/speech", speech, methods=["POST"]),
        ]
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://kokoro") as client:
        provider = KokoroTTSProvider(base_url="http://kokoro", client=client)
        await provider.connect_session()
        received: list[bytes] = []
        async for chunk in provider.synthesize_stream("Keep talking."):
            received.append(chunk)
            await provider.cancel_generation()
    assert provider.cancelled is True
    assert received


def test_kokoro_switch_leaves_retrieval_and_reasoning_untouched():
    root = BACKEND_ROOT / "app"
    for directory in ("retrieval", "planning", "data_gateway", "llm"):
        text = "\n".join(path.read_text(encoding="utf-8") for path in (root / directory).rglob("*.py"))
        assert "kokoro" not in text.lower()
        assert "tts_provider" not in text.lower()
    turn = (root / "execution" / "turn.py").read_text(encoding="utf-8")
    assert "KokoroTTSProvider" not in turn
    assert "XAITTSProvider" not in turn
    compose = monorepo_file("infra", "docker-compose.kokoro.yml")
    overlay = compose.read_text(encoding="utf-8")
    assert "TTS_PROVIDER: kokoro" in overlay
    assert "KOKORO_URL: http://kokoro:8880" in overlay


def test_speech_session_accepts_kokoro_provider():
    provider = get_tts_provider("kokoro")
    assert isinstance(provider, KokoroTTSProvider)
    session = SpeechSession(provider, mode="voice")
    assert session.provider is provider


def test_tts_layer_has_no_jwt():
    tts_dir = BACKEND_ROOT / "app" / "tts"
    text = "\n".join(path.read_text(encoding="utf-8") for path in tts_dir.glob("*.py"))
    assert "standalone_jwt" not in text
    assert "jwt.encode" not in text
def test_product_name_is_natural_for_speech_but_display_text_is_unchanged():
    displayed = "I'm NIA. NIA can help with the authorized records."
    assert prepare_speech_text(displayed) == (
        "I'm Nia. Nia can help with the authorized records."
    )
    assert displayed == "I'm NIA. NIA can help with the authorized records."
