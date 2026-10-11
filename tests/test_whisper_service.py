import array
import sys
from pathlib import Path
from types import SimpleNamespace

# Imports below intentionally follow the local source-path setup.
# ruff: noqa: E402

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "services" / "whisper" / "src"))

pytest.importorskip(
    "whisper_service",
    reason="the whisper service lives in the standalone repository, not in the API repository",
)

from whisper_service.engine import FasterWhisperEngine, ScriptedEngine
from whisper_service.server import WhisperService, create_app
from whisper_service.session import WhisperSession


def _speech(ms: int = 500, sample_rate: int = 16000) -> bytes:
    samples = int(sample_rate * ms / 1000)
    return array.array("h", [1200 if index % 2 == 0 else -1200 for index in range(samples)]).tobytes()


@pytest.mark.asyncio
async def test_session_emits_partial_then_final():
    engine = ScriptedEngine(
        partials=["How many", "How many students"],
        final="How many students have names starting with S",
    )
    session = WhisperSession(engine, hop_ms=200, window_ms=800)
    events = await session.push_pcm(_speech(250))
    assert events
    assert events[0].type == "partial"
    assert events[0].text.startswith("How many")
    final = await session.finalize_turn()
    assert final.type == "final"
    assert "students" in final.text
    assert final.end_ms > 0


@pytest.mark.asyncio
async def test_http_session_roundtrip():
    httpx = pytest.importorskip("httpx")
    service = WhisperService(ScriptedEngine())
    app = create_app(service)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://whisper") as client:
        health = await client.get("/health")
        assert health.json()["engine"] == "scripted"
        opened = await client.post("/v1/session")
        session_id = opened.json()["session_id"]
        pushed = await client.post(f"/v1/session/{session_id}/audio", content=_speech(500))
        assert pushed.status_code == 200
        final = await client.post(f"/v1/session/{session_id}/finalize")
        assert final.json()["type"] == "final"
        assert final.json()["text"]


@pytest.mark.asyncio
async def test_http_session_delete_releases_capacity():
    httpx = pytest.importorskip("httpx")
    service = WhisperService(ScriptedEngine(), max_sessions=1)
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://whisper") as client:
        opened = await client.post("/v1/session")
        session_id = opened.json()["session_id"]
        assert len(service.sessions) == 1

        closed = await client.delete(f"/v1/session/{session_id}")
        assert closed.status_code == 200
        assert service.sessions == {}

        replacement = await client.post("/v1/session")
        assert replacement.status_code == 200
        assert replacement.json()["session_id"] != session_id


@pytest.mark.asyncio
async def test_idle_http_session_expires_without_graceful_client_shutdown():
    httpx = pytest.importorskip("httpx")
    now = [100.0]
    service = WhisperService(
        ScriptedEngine(),
        max_sessions=1,
        session_ttl_seconds=30,
        clock=lambda: now[0],
    )
    transport = httpx.ASGITransport(app=create_app(service))
    async with httpx.AsyncClient(transport=transport, base_url="http://whisper") as client:
        opened = await client.post("/v1/session")
        stale_id = opened.json()["session_id"]
        assert len(service.sessions) == 1

        now[0] += 31
        health = await client.get("/health")
        assert health.json()["sessions"] == 0
        assert stale_id not in service.sessions

        replacement = await client.post("/v1/session")
        assert replacement.status_code == 200
        assert replacement.json()["session_id"] != stale_id


def test_whisper_service_does_not_import_assistant_core():
    root = ROOT / "services" / "whisper" / "src"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from app." in text or "import app." in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


@pytest.mark.asyncio
async def test_faster_whisper_receives_accuracy_and_vocabulary_options(monkeypatch):
    captured: dict = {}

    class _Model:
        def __init__(self, model_size, *, device, compute_type):
            captured["model_size"] = model_size
            captured["device"] = device
            captured["compute_type"] = compute_type

        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return iter([SimpleNamespace(text=" Nia")]), SimpleNamespace()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=_Model))
    engine = FasterWhisperEngine(
        "small.en",
        initial_prompt="Nia may be spelled N I A.",
        hotwords="Nia, N I A, hi, hello, hey",
        beam_size=5,
        vad_filter=False,
    )
    result = await engine.transcribe(_speech(500), 16000)

    assert result.text == "Nia"
    assert captured["model_size"] == "small.en"
    assert captured["beam_size"] == 5
    assert captured["initial_prompt"] == "Nia may be spelled N I A."
    assert captured["hotwords"] == "Nia, N I A, hi, hello, hey"
    assert captured["vad_filter"] is False
