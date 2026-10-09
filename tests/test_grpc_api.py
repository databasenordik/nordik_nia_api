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
    reply = await grpc_client.Query(nia_pb2.QueryRequest(
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


@pytest.mark.parametrize("question,file_id", [("", 49), ("x" * 4001, 49), ("count", 0), ("count", 999)],
                         ids=["empty-question", "long-question", "zero-file", "disabled-file"])
async def test_invalid_requests_return_invalid_argument(grpc_client, question, file_id):
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.Query(nia_pb2.QueryRequest(question=question, selected_file_id=file_id), timeout=5)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_community_constraint_changes_count(grpc_client):
    all_rows = await grpc_client.Query(nia_pb2.QueryRequest(
        question="How many records are in this list?", selected_file_id=49
    ), timeout=5)
    filtered = await grpc_client.Query(nia_pb2.QueryRequest(
        question="How many records are in this list?", selected_file_id=49,
        communities=["Garden River"],
    ), timeout=5)
    assert filtered.status == "answered"
    assert filtered.answer != all_rows.answer
    assert not filtered.conversation_id
    assert not json.loads(filtered.result_json)["conversation_id"]


async def test_community_filtered_followup_is_rejected(grpc_client):
    initial = await grpc_client.Query(nia_pb2.QueryRequest(
        question="How many records are in this list?", selected_file_id=49
    ), timeout=5)
    with pytest.raises(grpc.aio.AioRpcError) as exc:
        await grpc_client.Query(nia_pb2.QueryRequest(
            question="Show those records", selected_file_id=49,
            conversation_id=initial.conversation_id, communities=["Garden River"],
        ), timeout=5)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


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
        await grpc_client.Query(nia_pb2.QueryRequest(question="count", selected_file_id=49), timeout=0.05)
    assert exc.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
    await asyncio.wait_for(canceled.wait(), 1)


def test_scope_configuration_needs_no_jwt():
    settings = Settings(_env_file=None, nia_grpc_principal_id="configured-data-principal",
                        nia_grpc_allowed_file_ids="49,91", jwt_secret="")
    assert initial_scope(settings).allowed_file_ids == (49, 91)
    with pytest.raises(ValueError, match="NIA_GRPC_PRINCIPAL_ID"):
        initial_scope(Settings(_env_file=None, nia_grpc_principal_id="",
                               standalone_demo_principal_id=""))
