from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from starlette.responses import JSONResponse

from app.config import Settings


class RateLimitMiddleware:
    """Small per-instance sliding-window limiter for expensive public routes.

    A shared gateway/Redis limiter can be added for a multi-instance deployment;
    this layer still protects each process from accidental and direct abuse.
    """

    def __init__(self, app, *, settings: Settings) -> None:
        self.app = app
        self.settings = settings
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._limits = {
            ("POST", "/api/auth/login"): settings.auth_rate_limit_per_minute,
            ("POST", "/api/auth/session"): settings.auth_rate_limit_per_minute,
            ("POST", "/api/assistant/query"): settings.query_rate_limit_per_minute,
            ("POST", "/api/livekit/token"): settings.livekit_rate_limit_per_minute,
            ("POST", "/api/voice/speech"): settings.tts_rate_limit_per_minute,
        }

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self.settings.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        route = (scope["method"].upper(), scope["path"])
        limit = self._limits.get(route, 0)
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        client = _client_address(scope, trust_proxy=self.settings.trust_proxy_headers)
        key = (client, f"{route[0]} {route[1]}")
        now = time.monotonic()
        cutoff = now - 60.0
        async with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(60.0 - (now - events[0])))
                response = JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
                await response(scope, receive, send)
                return
            events.append(now)

        await self.app(scope, receive, send)


def _client_address(scope, *, trust_proxy: bool) -> str:
    if trust_proxy:
        for name, value in scope.get("headers") or []:
            if name.lower() == b"x-forwarded-for":
                forwarded = value.decode("latin-1").split(",", 1)[0].strip()
                if forwarded:
                    return forwarded
    client = scope.get("client")
    return str(client[0]) if client else "unknown"
