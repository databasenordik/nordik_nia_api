import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.assistant import router as assistant_router
from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.livekit_tokens import router as livekit_router
from app.api.speech import router as speech_router
from app.config import get_settings, validate_production_settings
from app.db.pool import close_pool, init_pool
from app.llm.factory import describe_reasoning
from app.security.rate_limit import RateLimitMiddleware

logger = logging.getLogger("nia")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    validate_production_settings(settings)
    reasoning = describe_reasoning()
    logger.info(
        "semantic_interpreter status=%s provider=%s model=%s detail=%s",
        reasoning["status"],
        reasoning["provider"],
        reasoning["model"],
        reasoning["detail"],
    )
    try:
        await init_pool()
        if settings.app_env == "production" or (
            settings.fail_closed_on_privilege_error and settings.app_env != "test"
        ):
            from app.db.privilege_selftest import assert_runtime_isolation

            await assert_runtime_isolation()
    except Exception:
        await close_pool()
        if settings.app_env == "production":
            raise
        if settings.app_env == "development":
            # Allow the API to boot for UI/auth work even if Postgres is down.
            pass
        else:
            raise
    yield
    await close_pool()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="NIA Standalone Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(RateLimitMiddleware, settings=settings)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(assistant_router)
    application.include_router(livekit_router)
    application.include_router(speech_router)
    return application


app = create_app()
