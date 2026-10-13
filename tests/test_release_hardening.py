import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import Settings, validate_production_settings
from app.security.rate_limit import RateLimitMiddleware


def test_unsafe_production_defaults_are_rejected():
    with pytest.raises(RuntimeError, match="unsafe production configuration"):
        validate_production_settings(Settings(app_env="production"))


def test_secure_website_production_configuration_is_accepted():
    validate_production_settings(
        Settings(
            app_env="production",
            auth_adapter="website",
            jwt_secret="assistant-session-secret-that-is-long-enough",
            website_jwt_secret="website-signing-secret-that-is-long-enough",
            auth_cookie_secure=True,
            livekit_api_secret="livekit-secret-that-is-not-the-development-value",
        )
    )


def test_expensive_routes_are_rate_limited_per_client():
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        settings=Settings(auth_rate_limit_per_minute=1),
    )

    @app.post("/api/auth/login")
    async def login():
        return {"ok": True}

    client = TestClient(app)
    assert client.post("/api/auth/login").status_code == 200
    limited = client.post("/api/auth/login")
    assert limited.status_code == 429
    assert int(limited.headers["retry-after"]) >= 1


def test_readiness_reports_dependency_failure(monkeypatch):
    health_module = importlib.import_module("app.api.health")

    async def database_ok():
        return {"ok": True}

    checked = 0

    async def dependency_check(_url, _timeout):
        nonlocal checked
        checked += 1
        return {"ok": checked != 1}

    monkeypatch.setattr(health_module, "_check_database", database_ok)
    monkeypatch.setattr(health_module, "_check_http", dependency_check)
    monkeypatch.setattr(
        health_module,
        "describe_reasoning",
        lambda: {"status": "connected", "provider": "fake", "model": "fake", "detail": "ok"},
    )
    app = FastAPI()
    app.include_router(health_module.router)

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["checks"]["whisper"]["ok"] is False
