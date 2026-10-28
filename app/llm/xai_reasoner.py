from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from app.config import get_settings
from app.llm.base import StructuredOutputResult, TransientReasoningError
from app.llm.schemas import PlannedQuery, SynthesisAnswer

_GLOBAL_SEMAPHORE: asyncio.Semaphore | None = None
_GLOBAL_SEMAPHORE_LIMIT: int | None = None


def _global_semaphore() -> asyncio.Semaphore:
    global _GLOBAL_SEMAPHORE, _GLOBAL_SEMAPHORE_LIMIT
    limit = max(1, get_settings().model_max_concurrency_global)
    if _GLOBAL_SEMAPHORE is None or _GLOBAL_SEMAPHORE_LIMIT != limit:
        _GLOBAL_SEMAPHORE = asyncio.Semaphore(limit)
        _GLOBAL_SEMAPHORE_LIMIT = limit
    return _GLOBAL_SEMAPHORE


class XAIReasoningProvider:
    """Async xAI Grok client. Import is lazy so tests do not need the SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrency: int = 2,
        request_timeout_ms: int | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.xai_api_key
        self._model = model or settings.reasoning_model
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._timeout_s = (
            request_timeout_ms if request_timeout_ms is not None else settings.xai_request_timeout_ms
        ) / 1000
        self.provider_name = "xai"
        self.model_name = self._model
        self._client_instance = None
        self._chat_module = None

    async def stream_answer(self, *, system: str, user: str) -> AsyncIterator[str]:
        client, chat_mod = self._client()
        async with _global_semaphore(), self._semaphore:
            chat = client.chat.create(
                model=self._model,
                messages=[chat_mod.system(system)],
            )
            chat.append(chat_mod.user(user))
            stream = chat.stream()
            while True:
                try:
                    _response, chunk = await asyncio.wait_for(
                        stream.__anext__(), timeout=self._timeout_s
                    )
                except StopAsyncIteration:
                    return
                except TimeoutError as exc:
                    raise TransientReasoningError(
                        f"xAI stream exceeded its {self._timeout_s:.0f}s deadline"
                    ) from exc
                except Exception as exc:
                    if _is_transient_xai_error(exc):
                        raise TransientReasoningError(
                            f"transient xAI stream failure: {type(exc).__name__}: {exc}"[:300]
                        ) from exc
                    raise
                text = getattr(chunk, "content", None)
                if text:
                    yield text

    async def answer_structured(self, *, system: str, user: str) -> SynthesisAnswer:
        parsed, _meta = await self._parse(SynthesisAnswer, system=system, user=user)
        return parsed

    async def plan_structured(self, *, system: str, user: str) -> PlannedQuery:
        parsed, _meta = await self._parse(PlannedQuery, system=system, user=user)
        return parsed

    async def structured_output(self, *, system: str, user: str, response_model):
        """Generic structured call, used by the semantic normalizer.

        Planning depends on this shape rather than on the xAI SDK, so the
        normalizer stays provider-agnostic.
        """
        result = await self.structured_output_with_metadata(
            system=system, user=user, response_model=response_model
        )
        return result.value

    async def structured_output_with_metadata(
        self, *, system: str, user: str, response_model
    ) -> StructuredOutputResult:
        parsed, meta = await self._parse(response_model, system=system, user=user)
        return StructuredOutputResult(value=parsed, **meta)

    async def _parse(self, model_type, *, system: str, user: str):
        client, chat_mod = self._client()
        started = time.perf_counter()
        settings = get_settings()
        options: dict[str, object] = {"temperature": settings.model_temperature}
        if settings.model_seed is not None:
            options["seed"] = settings.model_seed
        async with _global_semaphore(), self._semaphore:
            chat = client.chat.create(
                model=self._model,
                messages=[chat_mod.system(system)],
                **options,
            )
            chat.append(chat_mod.user(user))
            try:
                response, parsed = await asyncio.wait_for(
                    chat.parse(model_type), timeout=self._timeout_s
                )
            except TimeoutError as exc:
                raise TransientReasoningError(
                    f"xAI structured call exceeded its {self._timeout_s:.0f}s deadline"
                ) from exc
            except Exception as exc:
                if _is_transient_xai_error(exc):
                    raise TransientReasoningError(
                        f"transient xAI structured failure: {type(exc).__name__}: {exc}"[:300]
                    ) from exc
                raise
        duration_ms = (time.perf_counter() - started) * 1000
        return parsed, _usage_metadata(response, self.provider_name, self.model_name, duration_ms)

    def _client(self):
        if not self._api_key:
            raise RuntimeError("XAI_API_KEY is not configured")
        if self._client_instance is not None and self._chat_module is not None:
            return self._client_instance, self._chat_module
        try:
            from types import SimpleNamespace

            from xai_sdk import AsyncClient
            from xai_sdk.chat import system as chat_system
            from xai_sdk.chat import user as chat_user
        except ImportError as exc:
            raise RuntimeError("xai-sdk is not installed") from exc

        # Alias the SDK helpers. A nested class named `system`/`user` would
        # shadow the imported names and raise NameError.
        self._client_instance = AsyncClient(api_key=self._api_key)
        self._chat_module = SimpleNamespace(
            system=chat_system,
            user=chat_user,
        )
        return self._client_instance, self._chat_module


_TRANSIENT_XAI_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 522, 524})
_TRANSIENT_XAI_WORDS = (
    "timeout",
    "timed out",
    "deadline",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "connection reset",
    "connection refused",
)


def _is_transient_xai_error(exc: Exception) -> bool:
    """Classify SDK transport failures without importing vendor exception classes.

    The xAI SDK has changed exception types across releases. The stable signals are HTTP/gRPC
    status attributes and status text; schema/validation exceptions have neither and must keep
    surfacing as final invalid output rather than being retried as infrastructure failures.
    """
    for attr in ("status_code", "status", "http_status", "code"):
        value = getattr(exc, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                value = None
        if value is None:
            continue
        try:
            if int(value) in _TRANSIENT_XAI_STATUS:
                return True
        except (TypeError, ValueError):
            text = str(value).casefold()
            if any(token in text for token in ("unavailable", "deadline", "resource_exhausted")):
                return True
    text = f"{type(exc).__name__}: {exc}".casefold()
    if any(word in text for word in _TRANSIENT_XAI_WORDS):
        return True
    return any(f" {status} " in f" {text} " for status in _TRANSIENT_XAI_STATUS)


def _usage_metadata(response, provider: str, model: str, duration_ms: float) -> dict:
    usage = getattr(response, "usage", None)
    prompt = _optional_int(
        getattr(usage, "prompt_tokens", None)
        or getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_text_tokens", None)
    )
    completion = _optional_int(
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_text_tokens", None)
    )
    total = _optional_int(getattr(usage, "total_tokens", None))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    cost = getattr(usage, "cost_usd", None)
    if cost is None:
        cost = getattr(response, "cost_usd", None)
    try:
        cost_usd = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_usd = None
    return {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_usd": cost_usd,
        "duration_ms": duration_ms,
    }


def _optional_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

