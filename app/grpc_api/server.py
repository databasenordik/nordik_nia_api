import asyncio
import json
import logging
import signal

import grpc
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.api.assistant import QueryRequest, _serialize_turn
from app.config import Settings
from app.data_gateway.memory import MemoryDataGateway
from app.data_gateway.repository import AssistantDataGateway
from app.db.pool import close_pool, get_pool, init_pool
from app.execution.turn import AssistantTurnService
from app.grpc_api.community_gateway import CommunityGateway
from app.grpc_api.generated import nia_pb2, nia_pb2_grpc
from app.llm.factory import get_reasoning_provider
from app.memory.store import default_memory_store
from app.security.access_scope import AccessScope

logger = logging.getLogger("nia.grpc")


def initial_scope(settings: Settings) -> AccessScope:
    principal_id = settings.nia_grpc_principal_id or settings.standalone_demo_principal_id
    if not principal_id:
        raise ValueError("NIA_GRPC_PRINCIPAL_ID is required for the initial gRPC data scope")
    ids = tuple(int(v.strip()) for v in settings.nia_grpc_allowed_file_ids.split(",") if v.strip())
    if not ids or any(v <= 0 for v in ids):
        raise ValueError("NIA_GRPC_ALLOWED_FILE_IDS must contain positive file IDs")
    return AccessScope(principal_id=principal_id, allowed_file_ids=ids,
                       can_use_private_files=settings.nia_grpc_can_use_private_files)


def build_turn_service(scope, communities):
    try:
        gateway = AssistantDataGateway(get_pool())
    except RuntimeError:
        from app.config import get_settings
        if get_settings().app_env != "development":
            raise
        gateway = MemoryDataGateway()
    return gateway, default_memory_store(), get_reasoning_provider()


class AssistantServicer(nia_pb2_grpc.AssistantServicer):
    def __init__(self, scope: AccessScope, service_factory=build_turn_service):
        self.scope = scope
        self.service_factory = service_factory

    async def Health(self, request, context):
        return nia_pb2.HealthResponse(status="ok")

    async def Query(self, request, context):
        try:
            body = QueryRequest(question=request.question,
                                selected_file_id=request.selected_file_id,
                                conversation_id=request.conversation_id or None,
                                model=request.model or None, mode="text")
            if request.selected_file_id <= 0:
                raise ValueError("selected_file_id must be positive")
            communities = tuple(sorted({c.strip() for c in request.communities if c.strip()}))
            if len(communities) > 100 or any(len(c) > 200 for c in communities):
                raise ValueError("too many communities or an overlong community name")
            if not self.scope.allows_file(request.selected_file_id):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "dataset is not enabled for gRPC")
            scope = self.scope.narrow((request.selected_file_id,))
            gateway, store, reasoner = self.service_factory(scope, communities)
            if communities:
                catalog = await gateway.load_catalog(scope)
                if catalog.resolve_field(request.selected_file_id, "community") is None:
                    raise ValueError("selected dataset does not support community filtering")
                gateway = CommunityGateway(gateway, catalog, communities)
            # Legacy callers do not send conversation IDs. New follow-ups must retain
            # the same dataset and filter; prevent an unfiltered old result being reused.
            if communities and body.conversation_id:
                raise ValueError("community-filtered follow-up conversations are not supported yet")
            service = AssistantTurnService(gateway, store=store, reasoner=reasoner)
            result = await service.answer(scope, body.question, mode="text",
                                          selected_file_id=body.selected_file_id,
                                          conversation_id=body.conversation_id, model=body.model)
            serialized = jsonable_encoder(_serialize_turn(result))
            if communities:
                serialized["conversation_id"] = None
            return nia_pb2.QueryResponse(
                status=result.status, answer=result.answer,
                conversation_id="" if communities else result.conversation_id or "",
                selected_file_id=result.selected_file_id or request.selected_file_id,
                error_code=result.error_code or "", detail=result.detail or "",
                result_json=json.dumps(serialized).encode(),
            )
        except (ValidationError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except grpc.aio.AbortError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("NIA gRPC query failed")
            await context.abort(grpc.StatusCode.INTERNAL, "NIA query failed")


async def serve(settings: Settings, port: int) -> None:
    scope = initial_scope(settings)
    try:
        try:
            await init_pool()
            if settings.app_env == "production" or settings.fail_closed_on_privilege_error:
                from app.db.privilege_selftest import assert_runtime_isolation
                await assert_runtime_isolation()
        except Exception:
            await close_pool()
            if settings.app_env != "development":
                raise
            logger.warning("gRPC running with development fixture data; database unavailable")
        server = grpc.aio.server()
        nia_pb2_grpc.add_AssistantServicer_to_server(AssistantServicer(scope), server)
        if not server.add_insecure_port(f"{settings.app_host}:{port}"):
            raise RuntimeError("could not bind NIA gRPC port")
        await server.start()
        logger.info("NIA gRPC listening on %s:%s", settings.app_host, port)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass
        try:
            await stop.wait()
        finally:
            await server.stop(5)
    finally:
        await close_pool()


if __name__ == "__main__":
    from app.config import get_settings
    import os
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(get_settings(), int(os.environ.get("PORT", "8080"))))
