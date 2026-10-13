"""Test defaults. Production/application default is PLANNER_MODE=ai (Phase 2).

Legacy planner tests stay on the rollback path unless a test opts into AI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings


@pytest.fixture(autouse=True)
def _legacy_planner_unless_ai_requested(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    wants_ai = "ai_settings" in request.fixturenames or request.node.get_closest_marker("planner_ai")
    monkeypatch.setenv("PLANNER_MODE", "ai" if wants_ai else "legacy")
    monkeypatch.setenv("LIVEKIT_API_KEY", "unit-test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unit-test-livekit-secret-that-is-long-enough-32")
    # Settings reads ../.env, so whatever the developer has deployed against leaks in here.
    # Pointing a backend at a tunnel -- AUTH_COOKIE_SECURE=true for the cross-site cookie --
    # failed three tests that had nothing to do with the change: TestClient speaks http, so a
    # Secure cookie is never sent back and authenticated calls 401, and the hardening test's
    # "unsafe production default" was no longer unsafe. The suite decides its own
    # environment; a test that wants another value passes it to Settings explicitly.
    for name, value in (
        ("AUTH_COOKIE_SECURE", "false"),
        ("AUTH_COOKIE_SAMESITE", "lax"),
        ("CORS_ORIGINS", "http://localhost:3000"),
        ("FRONTEND_ORIGIN", "http://localhost:3000"),
        ("STANDALONE_EXTRA_ACCOUNTS", ""),
    ):
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
