import asyncio
import json

import grpc
import pytest

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService, TurnResult
from app.grpc_api.community_gateway import CommunityGateway
from app.grpc_api.generated import nia_pb2, nia_pb2_grpc
from app.grpc_api.server import AssistantServicer, initial_scope
from app.config import Settings, get_settings
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.security.access_scope import AccessScope


@pytest.fixture
def scope():
    return AccessScope(principal_id="grpc-test", allowed_file_ids=(49, 91))


@pytest.fixture
async def grpc_client(scope, monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("PLANNER_MODE", "legacy")
    monkeypatch.setenv("REASONING_PROVIDER", "fake")
    get_settings.cache_clear()
    store = InMemoryMemoryStore()
    def factory(scope, communities):
        return MemoryDataGateway(), store, FakeReasoningProvider()
    server = grpc.aio.server()
    nia_pb2_grpc.add_AssistantServicer_to_server(AssistantServicer(scope, factory), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield nia_pb2_grpc.AssistantStub(channel)
    finally:
        await channel.close()
        await server.stop(0)
        get_settings.cache_clear()


async def test_query_over_real_grpc_without_credentials(grpc_client):
    reply = await grpc_client.Query(nia_pb2.QueryRequest(user_id=1,
        question="How many records are in this list?", selected_file_id=49
    ), timeout=5)
    assert reply.status == "answered"
    assert reply.answer
    assert reply.selected_file_id == 49
    assert reply.conversation_id
    body = json.loads(reply.result_json)
    assert body["answer"] == reply.answer


async def test_health_over_grpc(grpc_client):
    reply = await grpc_client.Health(nia_pb2.HealthRequest(), timeout=5)
    assert reply.status == "ok"


async def test_user_file_conversation_reuse_and_isolation(grpc_client):
    async def ask(user, file=49, conversation=""):
        return await grpc_client.Query(nia_pb2.QueryRequest(
            user_id=user, selected_file_id=file, conversation_id=conversation,
            question="How many records are in this list?"), timeout=5)
    first = await ask(10)
    resumed = await ask(10)
    other_user = await ask(11)
    other_file = await ask(10, 91)
    assert resumed.conversation_id == first.conversation_id
    assert first.exchange_id and resumed.exchange_id != first.exchange_id
    assert len({first.conversation_id, other_user.conversation_id, other_file.conversation_id}) == 3
    for user, file in [(11, 49), (10, 91)]:
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await ask(user, file, first.conversation_id)
        assert exc.value.code() == grpc.StatusCode.NOT_FOUND


async def test_grpc_history_returns_owned_pairs_and_validates_limit(grpc_client):
    for _ in range(12):
        reply = await grpc_client.Query(nia_pb2.QueryRequest(
            user_id=10, selected_file_id=49, question="How many records are in this list?"), timeout=5)
    history = await grpc_client.History(nia_pb2.HistoryRequest(user_id=10, selected_file_id=49, limit=2))
    body = json.loads(history.result_json)
    assert body["conversation_id"] == reply.conversation_id
    assert len(body["entries"]) == 2 and body["has_more"]
    assert body["entries"][-1]["exchange_id"] == reply.exchange_id
    assert all(entry["question_id"] != entry["answer_id"] for entry in body["entries"])
    default = await grpc_client.History(nia_pb2.HistoryRequest(user_id=10, selected_file_id=49))
    assert len(json.loads(default.result_json)["entries"]) == 10
    empty = await grpc_client.History(nia_pb2.HistoryRequest(user_id=11, selected_file_id=49))
    assert json.loads(empty.result_json)["entries"] == []
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.History(nia_pb2.HistoryRequest(user_id=11, selected_file_id=49, conversation_id=reply.conversation_id))
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND
    for limit in [-1, 101]:
        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await grpc_client.History(nia_pb2.HistoryRequest(user_id=10, selected_file_id=49, limit=limit))
        assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_grpc_requires_user_identity(grpc_client):
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.Query(nia_pb2.QueryRequest(selected_file_id=49, question="count"))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_llm_context_comes_only_from_the_user_file_conversation(grpc_client, monkeypatch):
    import app.execution.turn as turn_module

    captured = []
    original = turn_module.plan_user_turn

    async def capture(question, scope, *args, **kwargs):
        captured.append((scope.user_id, scope.allowed_file_ids, kwargs.get("memory_text", "")))
        return await original(question, scope, *args, **kwargs)

    monkeypatch.setattr(turn_module, "plan_user_turn", capture)
    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    first_question = "How many students have names starting with S?"
    first = await grpc_client.Query(nia_pb2.QueryRequest(
        user_id=10, selected_file_id=49, question=first_question))
    await grpc_client.Query(nia_pb2.QueryRequest(
        user_id=11, selected_file_id=49, question="How many records are in this list?"))
    await grpc_client.Query(nia_pb2.QueryRequest(
        user_id=10, selected_file_id=91, question="How many records are in this list?"))
    resumed = await grpc_client.Query(nia_pb2.QueryRequest(
        user_id=10, selected_file_id=49, question="What about 1911?"))
    assert resumed.conversation_id == first.conversation_id
    assert first_question in captured[-1][2]
    assert all(first_question not in memory for user, files, memory in captured if user == 11 or files == (91,))


@pytest.mark.parametrize("question,file_id", [("", 49), ("x" * 4001, 49), ("count", 0)],
                         ids=["empty-question", "long-question", "zero-file"])
async def test_invalid_requests_return_invalid_argument(grpc_client, question, file_id):
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.Query(nia_pb2.QueryRequest(user_id=1, question=question, selected_file_id=file_id), timeout=5)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_community_constraint_changes_count(grpc_client):
    all_rows = await grpc_client.Query(nia_pb2.QueryRequest(user_id=1,
        question="How many records are in this list?", selected_file_id=49
    ), timeout=5)
    filtered = await grpc_client.Query(nia_pb2.QueryRequest(user_id=1,
        question="How many records are in this list?", selected_file_id=49,
        communities=["Garden River"],
    ), timeout=5)
    assert filtered.status == "answered"
    assert filtered.answer != all_rows.answer
    assert filtered.conversation_id == all_rows.conversation_id
    assert json.loads(filtered.result_json)["conversation_id"] == all_rows.conversation_id


async def test_community_filtered_followup_reuses_owned_conversation(grpc_client):
    initial = await grpc_client.Query(nia_pb2.QueryRequest(user_id=1,
        question="How many records are in this list?", selected_file_id=49,
        communities=["Garden River"],
    ), timeout=5)
    reply = await grpc_client.Query(nia_pb2.QueryRequest(user_id=1,
        question="How many records are in this list?", selected_file_id=49,
        conversation_id=initial.conversation_id, communities=["Garden River"],
    ), timeout=5)
    assert reply.conversation_id == initial.conversation_id
    assert reply.exchange_id != initial.exchange_id
    history = await grpc_client.History(nia_pb2.HistoryRequest(user_id=1, selected_file_id=49))
    pairs = json.loads(history.result_json)["entries"]
    assert len(pairs) == 2
    assert all(pair["communities"] == ["Garden River"] for pair in pairs)


async def test_community_constraint_applies_to_rows_candidates_and_raw(scope):
    gateway = MemoryDataGateway()
    catalog = await gateway.load_catalog(scope)
    filtered = CommunityGateway(gateway, catalog, ["Garden River"])
    expected = await filtered.list_records(scope, (49,), [], limit=100)
    assert expected
    assert len(expected) < len(await gateway.list_records(scope, (49,), [], limit=100))
    assert await filtered.count_records(scope, (49,), []) == len(expected)
    candidates = await filtered.retrieve_candidates(scope, (49,), "Samuel", "exact")
    assert all(row["canonical_community"] == "Garden River" for row in candidates)
    all_rows = await gateway.list_records(scope, (49,), [], limit=100)
    raw = await filtered.get_raw_fields(scope, [r["source_row_id"] for r in all_rows], ["Name"])
    assert {r["id"] for r in raw} == {r["source_row_id"] for r in expected}


async def test_deadline_cancels_server_work(grpc_client, monkeypatch):
    canceled = asyncio.Event()
    async def slow_answer(self, *args, **kwargs):
        try:
            await asyncio.sleep(10)
            return TurnResult(status="answered", answer="late")
        finally:
            canceled.set()
    monkeypatch.setattr(AssistantTurnService, "answer", slow_answer)
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.Query(nia_pb2.QueryRequest(user_id=1, question="count", selected_file_id=49), timeout=0.05)
    assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    await asyncio.wait_for(canceled.wait(), 1)


async def test_query_scopes_to_the_selected_file_without_a_startup_allow_list():
    captured_scope = None

    def factory(scope, _communities):
        nonlocal captured_scope
        captured_scope = scope
        return MemoryDataGateway(), InMemoryMemoryStore(), FakeReasoningProvider()

    service = AssistantServicer(
        AccessScope(principal_id="grpc-test", allowed_file_ids=()), factory
    )
    reply = await service.Query(
        nia_pb2.QueryRequest(user_id=1, question="How many records are in this list?", selected_file_id=49),
        None,
    )

    assert reply.status == "answered"
    assert captured_scope is not None
    assert captured_scope.allowed_file_ids == (49,)


def test_scope_configuration_needs_no_jwt_or_fixed_file_list():
    settings = Settings(_env_file=None, nia_grpc_principal_id="configured-data-principal",
                        jwt_secret="")
    assert initial_scope(settings).allowed_file_ids == ()
    with pytest.raises(ValueError, match="NIA_GRPC_PRINCIPAL_ID"):
        initial_scope(Settings(_env_file=None, nia_grpc_principal_id="",
                               standalone_demo_principal_id=""))
