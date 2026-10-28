import asyncio

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.pool import get_pool
from app.llm.factory import describe_reasoning

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    reasoning = describe_reasoning()
    return {
        "status": "ok",
        "env": settings.app_env,
        "auth_adapter": settings.auth_adapter,
        "semantic_interpreter": reasoning["status"],
        "reasoning_provider": reasoning["provider"],
        "reasoning_model": reasoning["model"],
        "reasoning_detail": reasoning["detail"],
    }


@router.get("/ready")
async def readiness():
    settings = get_settings()
    database, whisper, livekit = await asyncio.gather(
        _check_database(),
        _check_http(f"{settings.whisper_service_url.rstrip('/')}/health", settings.readiness_timeout_ms),
        _check_http(settings.livekit_health_url, settings.readiness_timeout_ms),
    )
    reasoning = describe_reasoning()
    reasoning_required = settings.reasoning_provider.strip().lower() not in {
        "off",
        "none",
        "disabled",
    }
    checks = {
        "database": database,
        "whisper": whisper,
        "livekit": livekit,
        # This is configuration readiness, not a paid provider probe.
        "reasoning": {
            "ok": reasoning["status"] == "connected" or not reasoning_required,
            "status": reasoning["status"],
        },
    }
    ready = all(item["ok"] for item in checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status_code=200 if ready else 503,
    )


async def _check_database() -> dict[str, object]:
    try:
        async with get_pool().acquire() as conn:
            value = await conn.fetchval("SELECT 1")
        return {"ok": value == 1}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


async def _check_http(url: str, timeout_ms: int) -> dict[str, object]:
    try:
        async with httpx.AsyncClient(timeout=max(timeout_ms, 100) / 1000) as client:
            response = await client.get(url)
        # A 404 still proves the internal service is accepting requests.
        return {"ok": response.status_code < 500, "status_code": response.status_code}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}
