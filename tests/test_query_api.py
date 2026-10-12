import asyncio
import base64
import sys

from repo_paths import BACKEND_ROOT  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient

from app.api.assistant import (
    QueryRequest,
    _feedback_store,
    _memory_store,
    _stream_query,
    _turn_service,
)
from app.config import Settings
from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService, TurnResult
from app.llm.fake import FakeReasoningProvider
from app.main import create_app
from app.memory.feedback import InMemoryFeedbackStore
from app.memory.store import InMemoryMemoryStore
from app.security.access_scope import AccessScope
from app.security.auth_adapter import get_auth_adapter
from app.security.deps import get_access_scope
from app.security.standalone_jwt import StandaloneJwtAuthAdapter
from app.tts.base import ensure_no_tts_in_text_mode


def _client(store: InMemoryMemoryStore | None = None) -> TestClient:
    app = create_app()
    memory = store or InMemoryMemoryStore()
    reports = InMemoryFeedbackStore()

    def scope() -> AccessScope:
        return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))

    def service() -> AssistantTurnService:
        return AssistantTurnService(MemoryDataGateway(), store=memory, reasoner=FakeReasoningProvider())

    app.dependency_overrides[get_access_scope] = scope
    app.dependency_overrides[_memory_store] = lambda: memory
    app.dependency_overrides[_turn_service] = service
    app.dependency_overrides[_feedback_store] = lambda: reports
    client = TestClient(app)
    client.reports = reports  # type: ignore[attr-defined]
    return client


def test_query_golden_path_json():
    response = _client().post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reasoning_calls"] == 0
    assert body["plan"]["scope"]["file_ids"] == [49]
    assert body["facts"][0]["value"] == 2


def test_list_query_returns_structured_rows_and_range_metadata():
    response = _client().post(
        "/api/assistant/query",
        json={"question": "List the last 3 students", "mode": "text"},
    )
    assert response.status_code == 200
    body = response.json()
    result = body["list_results"][0]
    assert result["window"] == {"anchor": "end", "size": 3, "count": 3, "cursor": 0}
    assert result["offset"] == 2
    assert result["has_more"] is False
    assert result["fields"][0] == {"name": "student_name", "label": "Student name"}
    assert result["rows"][0]["values"]["student_name"] == "Samuel A"


def test_query_requires_auth():
    app = create_app()
    client = TestClient(app)
    response = client.post("/api/assistant/query", json={"question": "How many students?"})
    assert response.status_code == 401


def test_cookie_authenticated_mutation_rejects_untrusted_browser_origin():
    settings = Settings(
        jwt_secret="unit-test-secret-that-is-long-enough-32",
        standalone_demo_username="test-user",
        standalone_demo_password="unit-test-password",
        standalone_demo_principal_id="test-principal",
    )
    adapter = StandaloneJwtAuthAdapter(settings)
    app = create_app()
    app.dependency_overrides[get_auth_adapter] = lambda: adapter

    with TestClient(app) as client:
        signed_in = client.post(
            "/api/auth/login",
            json={"username": "test-user", "password": "unit-test-password"},
        )
        response = client.post(
            "/api/assistant/query",
            headers={"Origin": "https://attacker.invalid", "Sec-Fetch-Site": "cross-site"},
            json={"question": "How many students?"},
        )

    assert signed_in.status_code == 200
    assert response.status_code == 403


def test_query_sse_emits_required_events():
    response = _client().post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    for event in ("plan_started", "plan_validated", "computed_fact", "answer_delta", "answer_completed"):
        assert f"event: {event}" in body
    assert body.count("event: plan_started") == 1
    assert "event: citation" not in body or "file:93" not in body


def test_sse_starts_before_assistant_computation_finishes():
    class DelayedService:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def answer(self, *_args, **_kwargs):
            self.started.set()
            await self.release.wait()
            return TurnResult(status="answered", answer="Finished")

    async def run():
        service = DelayedService()
        stream = _stream_query(QueryRequest(question="wait"), _client_scope(), service)

        first = await stream.__anext__()
        assert "event: plan_started" in first
        assert not service.started.is_set()

        next_event = asyncio.create_task(stream.__anext__())
        await service.started.wait()
        assert not next_event.done()
        service.release.set()
        assert "event: answer_delta" in await next_event
        await stream.aclose()

    asyncio.run(run())


def test_turn_service_emits_execution_progress_in_order():
    async def run():
        events: list[str] = []

        async def capture(event: str, _data: dict) -> None:
            events.append(event)

        service = AssistantTurnService(
            MemoryDataGateway(),
            store=InMemoryMemoryStore(),
            reasoner=FakeReasoningProvider(),
        )
        await service.answer(
            _client_scope(),
            "How many students have names starting with S?",
            on_progress=capture,
        )
        assert events.index("plan_validated") < events.index("retrieval_completed")
        assert events.index("retrieval_completed") < events.index("computed_fact")
        assert events.index("computed_fact") < events.index("answer_delta")

    asyncio.run(run())


def test_closing_sse_cancels_inflight_turn(monkeypatch):
    import app.api.assistant as assistant_api

    class NeverFinishes:
        def __init__(self) -> None:
            self.cancellation = None

        async def answer(self, *_args, cancellation=None, **_kwargs):
            self.cancellation = cancellation
            await asyncio.Event().wait()

    async def run():
        service = NeverFinishes()
        stream = _stream_query(QueryRequest(question="wait"), _client_scope(), service)
        await stream.__anext__()
        assert await stream.__anext__() == ": keep-alive\n\n"
        await stream.aclose()
        assert service.cancellation is not None
        assert service.cancellation.cancelled
        assert service.cancellation.reason == "client_disconnected"

    monkeypatch.setattr(assistant_api, "_SSE_HEARTBEAT_SECONDS", 0.001)
    asyncio.run(run())


def test_synthesis_sse_includes_citations():
    response = _client().post(
        "/api/assistant/query",
        json={
            "question": "Summarize what the records say about causes of death.",
            "mode": "text",
            "selected_file_id": 91,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert "event: citation" in response.text
    assert "event: answer_delta" in response.text
    assert "file:93" not in response.text


def test_conversation_list_and_get():
    store = InMemoryMemoryStore()
    client = _client(store)
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    conversation_id = asked.json()["conversation_id"]
    listed = client.get("/api/assistant/conversations")
    assert listed.status_code == 200
    ids = {item["id"] for item in listed.json()["conversations"]}
    assert conversation_id in ids
    detail = client.get(f"/api/assistant/conversations/{conversation_id}")
    assert detail.status_code == 200
    roles = [turn["role"] for turn in detail.json()["turns"]]
    assert "user" in roles and "assistant" in roles


def test_text_mode_never_starts_tts():
    ensure_no_tts_in_text_mode("text")
    source = BACKEND_ROOT / "app" / "api" / "assistant.py"
    text = source.read_text(encoding="utf-8")
    assert "ensure_no_tts_in_text_mode" in text
    assert "synthesize_stream" not in text


def _client_scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def test_a_conversation_can_be_deleted_and_stops_being_listed():
    store = InMemoryMemoryStore()
    client = _client(store)
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    conversation_id = asked.json()["conversation_id"]

    removed = client.delete(f"/api/assistant/conversations/{conversation_id}")
    assert removed.status_code == 204

    listed = client.get("/api/assistant/conversations")
    assert conversation_id not in {item["id"] for item in listed.json()["conversations"]}
    assert client.get(f"/api/assistant/conversations/{conversation_id}").status_code == 404


def test_deleting_the_same_conversation_twice_is_a_404_not_a_crash():
    store = InMemoryMemoryStore()
    client = _client(store)
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    conversation_id = asked.json()["conversation_id"]
    assert client.delete(f"/api/assistant/conversations/{conversation_id}").status_code == 204
    assert client.delete(f"/api/assistant/conversations/{conversation_id}").status_code == 404


def test_a_conversation_id_that_was_never_real_is_a_404():
    client = _client()
    assert client.delete("/api/assistant/conversations/not-a-uuid").status_code == 404
    assert (
        client.delete("/api/assistant/conversations/11111111-2222-3333-4444-555555555555").status_code
        == 404
    )


def test_one_researcher_cannot_delete_another_researchers_conversation():
    """404 rather than 403: a different status would say which ids are real."""
    from app.memory.types import ConversationRecord

    store = InMemoryMemoryStore()
    someone_else = ConversationRecord(
        id="99999999-8888-7777-6666-555555555555", principal_id="principal-someone-else"
    )
    asyncio.run(store.save(someone_else))

    client = _client(store)
    assert client.delete(f"/api/assistant/conversations/{someone_else.id}").status_code == 404
    # And it is still there for the principal it belongs to.
    other_scope = AccessScope(principal_id="principal-someone-else", allowed_file_ids=(49,))
    assert asyncio.run(store.get(other_scope, someone_else.id)) is not None


def test_a_tester_can_report_a_problem():
    """The sign-in screen tells testers to click Report and submit; this is where it lands."""
    client = _client()
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    conversation_id = asked.json()["conversation_id"]

    sent = client.post(
        "/api/assistant/feedback",
        json={
            "comment": "The count looked wrong for names starting with S.",
            "reporter_name": "Sam Tester",
            "conversation_id": conversation_id,
        },
    )
    assert sent.status_code == 204
    [report] = client.reports.reports
    assert report.reporter_name == "Sam Tester"
    assert "count looked wrong" in report.comment
    assert report.conversation_id == conversation_id


def test_a_report_needs_a_name_but_not_a_comment():
    """Whoever filed it must be askable; what they saw can be left to the chat and a screenshot."""
    client = _client()
    assert client.post("/api/assistant/feedback", json={"comment": "Voice cut out."}).status_code == 422
    blank = client.post("/api/assistant/feedback", json={"reporter_name": "   ", "comment": "x"})
    assert blank.status_code == 422
    assert client.reports.reports == []
    sent = client.post("/api/assistant/feedback", json={"reporter_name": "Sam Tester"})
    assert sent.status_code == 204
    [report] = client.reports.reports
    assert (report.reporter_name, report.comment) == ("Sam Tester", "")
    assert report.conversation_id is None


_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + bytes(32)).decode()


def test_a_report_can_carry_a_screenshot():
    """Testers asked to add "a picture of the issue"; the path it was taken from is dropped."""
    client = _client()
    sent = client.post(
        "/api/assistant/feedback",
        json={
            "comment": "See the screenshot.",
            "reporter_name": "Sam Tester",
            "attachments": [
                {"filename": "C:\\Users\\tester\\shot.png", "content_type": "image/png", "data_base64": _PNG}
            ],
        },
    )
    assert sent.status_code == 204
    [report] = client.reports.reports
    [attachment] = report.attachments
    assert (attachment.filename, attachment.content_type) == ("shot.png", "image/png")
    assert attachment.content.startswith(b"\x89PNG")


def test_a_file_that_is_not_what_it_claims_is_refused_and_nothing_is_kept():
    client = _client()
    disguised = base64.b64encode(b"MZ\x90\x00 an executable, not an image").decode()
    sent = client.post(
        "/api/assistant/feedback",
        json={
            "comment": "Attached.",
            "reporter_name": "Sam Tester",
            "attachments": [{"filename": "shot.png", "content_type": "image/png", "data_base64": disguised}],
        },
    )
    assert sent.status_code == 422
    assert "does not look like" in sent.json()["detail"]
    assert client.reports.reports == []


def test_an_unsupported_kind_of_file_is_refused_with_a_reason():
    client = _client()
    sent = client.post(
        "/api/assistant/feedback",
        json={
            "comment": "Attached.",
            "reporter_name": "Sam Tester",
            "attachments": [
                {"filename": "tool.exe", "content_type": "application/x-msdownload", "data_base64": _PNG}
            ],
        },
    )
    assert sent.status_code == 422
    assert "can't be attached" in sent.json()["detail"]


def test_more_than_the_allowed_number_of_files_is_refused():
    client = _client()
    one = {"filename": "shot.png", "content_type": "image/png", "data_base64": _PNG}
    sent = client.post(
        "/api/assistant/feedback",
        json={"comment": "Attached.", "reporter_name": "Sam Tester", "attachments": [one] * 4},
    )
    assert sent.status_code == 422
    assert client.reports.reports == []


def test_a_report_without_a_name_is_refused_whatever_it_says():
    client = _client()
    assert client.post("/api/assistant/feedback", json={"comment": "   "}).status_code == 422
    assert client.post("/api/assistant/feedback", json={"reporter_name": ""}).status_code == 422
    assert client.reports.reports == []


def test_a_report_cannot_be_used_to_dump_unbounded_text():
    client = _client()
    huge = client.post(
        "/api/assistant/feedback", json={"comment": "x" * 4001, "reporter_name": "Sam Tester"}
    )
    assert huge.status_code == 422
    long_name = client.post(
        "/api/assistant/feedback", json={"comment": "fine", "reporter_name": "n" * 121}
    )
    assert long_name.status_code == 422


def _make_conversation(client) -> str:
    asked = client.post(
        "/api/assistant/query",
        json={"question": "How many students have names starting with S?", "mode": "text"},
    )
    return asked.json()["conversation_id"]


def test_a_conversation_can_be_renamed():
    store = InMemoryMemoryStore()
    client = _client(store)
    conversation_id = _make_conversation(client)

    renamed = client.patch(
        f"/api/assistant/conversations/{conversation_id}", json={"title": "Names beginning with S"}
    )
    assert renamed.status_code == 204

    listed = client.get("/api/assistant/conversations").json()["conversations"]
    titles = {item["id"]: item["title"] for item in listed}
    assert titles[conversation_id] == "Names beginning with S"


def test_renaming_leaves_the_turns_alone():
    """A title is what the sidebar shows; the conversation underneath is untouched."""
    store = InMemoryMemoryStore()
    client = _client(store)
    conversation_id = _make_conversation(client)
    before = client.get(f"/api/assistant/conversations/{conversation_id}").json()["turns"]

    client.patch(f"/api/assistant/conversations/{conversation_id}", json={"title": "Renamed"})

    after = client.get(f"/api/assistant/conversations/{conversation_id}").json()
    assert after["title"] == "Renamed"
    assert [turn["text"] for turn in after["turns"]] == [turn["text"] for turn in before]


def test_a_blank_or_oversized_title_is_refused():
    client = _client()
    conversation_id = _make_conversation(client)
    for title in ("", "   ", "x" * 121):
        response = client.patch(
            f"/api/assistant/conversations/{conversation_id}", json={"title": title}
        )
        assert response.status_code == 422, title
    # The original title survived every refusal.
    assert client.get(f"/api/assistant/conversations/{conversation_id}").json()["title"]


def test_renaming_a_conversation_that_is_not_yours_is_a_404():
    """404 rather than 403, for the same reason delete answers 404: ids stay unguessable."""
    from app.memory.types import ConversationRecord

    store = InMemoryMemoryStore()
    theirs = ConversationRecord(
        id="88888888-7777-6666-5555-444444444444", principal_id="principal-someone-else"
    )
    theirs.title = "Their conversation"
    asyncio.run(store.save(theirs))

    client = _client(store)
    assert (
        client.patch(f"/api/assistant/conversations/{theirs.id}", json={"title": "mine now"}).status_code
        == 404
    )
    other_scope = AccessScope(principal_id="principal-someone-else", allowed_file_ids=(49,))
    assert asyncio.run(store.get(other_scope, theirs.id)).title == "Their conversation"


def test_renaming_something_that_does_not_exist_is_a_404():
    client = _client()
    assert (
        client.patch("/api/assistant/conversations/not-a-uuid", json={"title": "x"}).status_code
        == 404
    )


def test_reporting_a_chat_renames_it_so_it_can_be_found_again():
    """A sidebar of testers' questions all look alike; the reported one should not."""
    store = InMemoryMemoryStore()
    client = _client(store)
    conversation_id = _make_conversation(client)

    sent = client.post(
        "/api/assistant/feedback",
        json={
            "comment": "The count looked wrong.",
            "reporter_name": "Sam Tester",
            "conversation_id": conversation_id,
            "tz_offset_minutes": 0,
        },
    )
    assert sent.status_code == 204

    listed = client.get("/api/assistant/conversations").json()["conversations"]
    title = {item["id"]: item["title"] for item in listed}[conversation_id]
    assert title.startswith("Reported - Sam Tester - ")
    # The turns are still there to read; only the name changed.
    assert client.get(f"/api/assistant/conversations/{conversation_id}").json()["turns"]


def test_a_refused_report_leaves_the_chat_as_it_was():
    """A report turned away for having no name must not mark the chat as reported."""
    store = InMemoryMemoryStore()
    client = _client(store)
    conversation_id = _make_conversation(client)

    refused = client.post(
        "/api/assistant/feedback",
        json={"comment": "Voice cut out.", "conversation_id": conversation_id},
    )
    assert refused.status_code == 422
    listed = client.get("/api/assistant/conversations").json()["conversations"]
    title = {item["id"]: item["title"] for item in listed}[conversation_id]
    assert not title.startswith("Reported - ")


def test_a_report_with_no_chat_attached_renames_nothing():
    client = _client()
    sent = client.post("/api/assistant/feedback", json={"reporter_name": "Sam Tester"})
    assert sent.status_code == 204
    assert client.get("/api/assistant/conversations").json()["conversations"] == []


def test_the_reported_title_reads_in_the_reporters_own_time():
    """Stamped from the offset the browser sent, not from wherever the server sits."""
    from datetime import UTC, datetime

    from app.api.assistant import reported_title

    filed = datetime(2026, 9, 8, 14, 16, tzinfo=UTC)
    assert reported_title("Eyad", filed) == "Reported - Eyad - 08/09 14:16"
    assert reported_title("  Eyad  ", filed) == "Reported - Eyad - 08/09 14:16"
    assert reported_title("", filed) == "Reported - 08/09 14:16"
    assert reported_title("   ", filed) == "Reported - 08/09 14:16"
