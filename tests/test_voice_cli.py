import importlib.util
import sys

from fastapi.testclient import TestClient

from repo_paths import BACKEND_ROOT, monorepo_file  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from app.api.speech import _tts_provider
from app.main import create_app
from app.security.access_scope import AccessScope
from app.security.deps import get_access_scope


def _load_cli():
    path = monorepo_file("voice_cli.py")
    spec = importlib.util.spec_from_file_location("nia_voice_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CapturingTTS:
    supports_expressive_tags = True

    def __init__(self) -> None:
        self.text = ""
        self.voice_config = None
        self.closed = False

    async def synthesize_stream(self, text, voice_config=None):
        self.text = text
        self.voice_config = voice_config
        yield b"PCM-AUDIO"

    async def close_session(self):
        self.closed = True


def test_authenticated_speech_uses_female_voice_and_spoken_only_tags():
    provider = CapturingTTS()
    app = create_app()
    app.dependency_overrides[get_access_scope] = lambda: AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91),
    )
    app.dependency_overrides[_tts_provider] = lambda: provider

    with TestClient(app) as client:
        response = client.post(
            "/api/voice/speech",
            json={
                "text": "There are 2,781 students in the master list.",
                "preference": "female",
            },
        )

    assert response.status_code == 200
    assert response.content == b"PCM-AUDIO"
    assert response.headers["x-nia-voice"] == "ara"
    assert provider.voice_config == {"voice": "ara"}
    assert provider.text == (
        "There are <emphasis>2,781</emphasis> students in the master list."
    )
    assert provider.closed is True


def test_speech_endpoint_requires_an_authenticated_user():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/voice/speech",
            json={"text": "Hi, I'm here.", "preference": "male"},
        )
    assert response.status_code == 401


def test_cli_voice_control_switches_without_an_assistant_query():
    cli = _load_cli()

    class Speaker:
        def __init__(self):
            self.preference = "male"
            self.spoken = []

        def set_preference(self, preference):
            self.preference = preference

        def say(self, text):
            self.spoken.append(text)

    speaker = Speaker()
    assert cli.handle_voice_control("Please use a woman's voice", speaker) is True
    assert speaker.preference == "female"
    assert speaker.spoken == ["Hi, I'm here."]
    assert cli.handle_voice_control("Tell me about women in the records", speaker) is False


def test_cli_rejects_a_session_for_the_wrong_principal():
    cli = _load_cli()

    class Response:
        ok = True

        def raise_for_status(self):
            return None

        def json(self):
            return {"principal_id": "principal-someone-else"}

    class Session:
        def __init__(self):
            self.cookies = self

        def post(self, *args, **kwargs):
            return Response()

        def clear(self):
            return None

    client = cli.AssistantClient(
        "http://assistant",
        "researcher",
        "configured-password",
        expected_principal="principal-researcher",
    )
    client._http = Session()
    try:
        client.login()
    except RuntimeError as exc:
        assert "principal mismatch" in str(exc)
    else:
        raise AssertionError("wrong principal was accepted")

