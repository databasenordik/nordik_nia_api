import array
import sys
from pathlib import Path

# Imports below intentionally follow the local source-path setup.
# ruff: noqa: E402

import pytest

from repo_paths import BACKEND_ROOT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(ROOT / "services" / "whisper" / "src"))

pytest.importorskip(
    "whisper_service",
    reason="the whisper service lives in the standalone repository, not in the API repository",
)

from app.stt.livekit_adapter import LiveKitStreamPump, LiveKitWhisperAdapter
from app.stt.whisper_cpp import WhisperCppSTTProvider
from whisper_service.engine import ScriptedEngine
from whisper_service.server import WhisperService, create_app


def _speech(ms: int = 500) -> bytes:
    samples = int(16000 * ms / 1000)
    return array.array("h", [1500 if index % 2 == 0 else -1500 for index in range(samples)]).tobytes()


@pytest.mark.asyncio
async def test_provider_streams_partial_and_final_from_service():
    httpx = pytest.importorskip("httpx")
    service = WhisperService(ScriptedEngine())
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://whisper") as client:
        provider = WhisperCppSTTProvider(base_url="http://whisper", client=client)

        async def audio():
            yield _speech(500)
            yield _speech(500)

        events = [event async for event in provider.transcribe_stream(audio())]
        assert len(service.sessions) == 1
        await provider.aclose()
        assert service.sessions == {}
    kinds = [event.type for event in events]
    assert "partial" in kinds
    assert kinds[-1] == "final"
    assert events[-1].text


def test_livekit_adapter_maps_partial_and_final():
    class _Stub:
        async def transcribe_stream(self, audio, *, sample_rate=16000):
            if False:
                yield None

        async def reset_session(self) -> None:
            return None

        async def finalize_turn(self):
            return None

    adapter = LiveKitWhisperAdapter(_Stub())
    from app.stt.base import TranscriptEvent

    interim = adapter.to_livekit_payload(
        TranscriptEvent(type="partial", text="How many", start_ms=0, end_ms=400, confidence=0.6)
    )
    final = adapter.to_livekit_payload(
        TranscriptEvent(type="final", text="How many students", start_ms=0, end_ms=1200, confidence=0.9)
    )
    assert interim["type"] == "interim_transcript"
    assert final["type"] == "final_transcript"
    assert final["alternatives"][0]["text"] == "How many students"


@pytest.mark.asyncio
async def test_livekit_stream_pump_emits_partial_then_final():
    httpx = pytest.importorskip("httpx")
    service = WhisperService(ScriptedEngine())
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://whisper") as client:
        provider = WhisperCppSTTProvider(base_url="http://whisper", client=client)
        pump = LiveKitStreamPump(provider)
        pushed = await pump.push(_speech(500))
        pushed.extend(await pump.push(_speech(500)))
        finals = await pump.flush()
    kinds = [item["type"] for item in pushed + finals]
    assert "interim_transcript" in kinds or "final_transcript" in kinds
    assert finals[-1]["type"] == "final_transcript"
    assert finals[-1]["alternatives"][0]["text"]


def test_livekit_stt_uses_vad_backed_batch_recognition():
    pytest.importorskip("livekit.agents")

    class _Stub:
        async def transcribe_stream(self, audio, *, sample_rate=16000):
            if False:
                yield None

        async def reset_session(self) -> None:
            return None

        async def finalize_turn(self):
            return None

    from app.stt.livekit_adapter import create_livekit_stt

    implementation = create_livekit_stt(_Stub())
    assert implementation.capabilities.streaming is False
    assert implementation.capabilities.interim_results is False


def test_backend_stt_has_no_whisper_subprocess():
    stt_dir = ROOT / "backend" / "app" / "stt"
    text = "\n".join(path.read_text(encoding="utf-8") for path in stt_dir.glob("*.py"))
    assert "import subprocess" not in text
    assert "asyncio.create_subprocess" not in text
    assert "whisper_service.engine" not in text
