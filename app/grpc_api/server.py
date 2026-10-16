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
    # The Go adapter supplies one selected file ID for every request.  The request
    # scope is created in Query so this service is not constrained by a stale,
    # manually maintained environment-variable allow-list.
    return AccessScope(principal_id=principal_id, allowed_file_ids=(),
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

    def user_scope(self, request):
        if request.user_id <= 0 or request.selected_file_id <= 0:
            raise ValueError("user_id and selected_file_id must be positive")
        return self.scope.model_copy(update={"user_id": request.user_id,
                                            "allowed_file_ids": (request.selected_file_id,)})

    async def History(self, request, context):
        try:
            scope = self.user_scope(request)
            limit = request.limit or 10
            if limit < 1 or limit > 100:
                raise ValueError("limit must be between 1 and 100")
            gateway, store, _ = self.service_factory(scope, ())
            catalog = await gateway.load_catalog(scope)
            if catalog.dataset(request.selected_file_id) is None:
                raise PermissionError("file not available")
            history = await store.history_for_file(scope, request.selected_file_id,
                                                   request.conversation_id or None, limit)
            return nia_pb2.HistoryResponse(result_json=json.dumps(jsonable_encoder(history)).encode())
        except (ValidationError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except PermissionError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "conversation not found")
        except grpc.aio.AbortError:
            raise
        except Exception:
            logger.exception("NIA gRPC history failed")
            await context.abort(grpc.StatusCode.INTERNAL, "NIA history failed")

    async def Query(self, request, context):
        try:
            body = QueryRequest(question=request.question,
                                selected_file_id=request.selected_file_id,
                                conversation_id=request.conversation_id or None,
                                model=request.model or None, mode="text")
            scope = self.user_scope(request)
            communities = tuple(sorted({c.strip() for c in request.communities if c.strip()}))
            if len(communities) > 100 or any(len(c) > 200 for c in communities):
                raise ValueError("too many communities or an overlong community name")
            # Restrict this turn to Go's selected dataset.  The database functions
            # enforce whether that dataset is enabled and whether this principal may
            # use it; no fixed NIA_GRPC_ALLOWED_FILE_IDS deployment setting is used.
            gateway, store, reasoner = self.service_factory(scope, communities)
            if communities:
                catalog = await gateway.load_catalog(scope)
                if catalog.resolve_field(request.selected_file_id, "community") is None:
                    raise ValueError("selected dataset does not support community filtering")
                gateway = CommunityGateway(gateway, catalog, communities)
            catalog = await gateway.load_catalog(scope)
            if catalog.dataset(request.selected_file_id) is None:
                raise PermissionError("file not available")
            conversation = await store.for_user_file(scope, request.selected_file_id, body.conversation_id)
            if conversation.community_filter != communities:
                # Keep all persisted exchanges in the same user/file history, but
                # do not reuse model context produced under a different row filter.
                conversation.community_filter = communities
                conversation.recent_turns.clear()
                conversation.frames.clear()
                conversation.result_sets.clear()
                conversation.rolling_summary = ""
                conversation.active_frame_key = None
                conversation.last_action_keys.clear()
                await store.save(conversation)
            body.conversation_id = conversation.id
            service = AssistantTurnService(gateway, store=store, reasoner=reasoner)
            result = await service.answer(scope, body.question, mode="text",
                                          selected_file_id=body.selected_file_id,
                                          conversation_id=body.conversation_id, model=body.model)
            serialized = jsonable_encoder(_serialize_turn(result))
            return nia_pb2.QueryResponse(
                status=result.status, answer=result.answer,
                conversation_id=result.conversation_id or "",
                exchange_id=result.exchange_id or "",
                selected_file_id=result.selected_file_id or request.selected_file_id,
                error_code=result.error_code or "", detail=result.detail or "",
                result_json=json.dumps(serialized).encode(),
            )
        except (ValidationError, ValueError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except PermissionError:
            await context.abort(grpc.StatusCode.NOT_FOUND, "conversation not found")
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
