from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.llm.base import ReasoningProvider
from app.llm.fake import FakeReasoningProvider

logger = logging.getLogger("nia.llm")


@lru_cache(maxsize=16)
def _http_provider(
    api_key: str,
    base_url: str,
    model: str,
    max_concurrency: int,
    request_timeout_ms: int,
) -> ReasoningProvider:
    from app.llm.http_reasoner import HttpReasoningProvider

    return HttpReasoningProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_concurrency=max_concurrency,
        request_timeout_ms=request_timeout_ms,
    )


# Cached per model, because the researcher can switch models per turn and each distinct
# choice would otherwise rebuild a client. Sized well above the offered list so a switch
# never evicts the one in use.
@lru_cache(maxsize=16)
def _xai_provider(
    api_key: str,
    model: str,
    max_concurrency: int,
    request_timeout_ms: int,
) -> ReasoningProvider:
    from app.llm.xai_reasoner import XAIReasoningProvider

    return XAIReasoningProvider(
        api_key=api_key,
        model=model,
        max_concurrency=max_concurrency,
        request_timeout_ms=request_timeout_ms,
    )


def reasoning_unavailable_reason() -> str | None:
    """Why the Turn Interpreter cannot call a model on this process, or None."""
    settings = get_settings()
    name = settings.reasoning_provider.strip().lower()
    if name in {"off", "none", "disabled"}:
        return "REASONING_PROVIDER is disabled"
    if name == "fake":
        return None
    if not (settings.xai_api_key or "").strip():
        return (
            "XAI_API_KEY is not configured on this process; "
            "the semantic compiler cannot run"
        )
    return None


def describe_reasoning() -> dict[str, Any]:
    settings = get_settings()
    reason = reasoning_unavailable_reason()
    return {
        "provider": settings.reasoning_provider,
        "model": settings.reasoning_model,
        "status": "unavailable" if reason else "connected",
        "detail": reason or "ok",
        "api_key_configured": bool((settings.xai_api_key or "").strip()),
    }


def allowed_models() -> tuple[str, ...]:
    """The models a researcher may choose, default first."""
    settings = get_settings()
    choices = tuple(dict.fromkeys(settings.reasoning_model_choices))
    default = settings.reasoning_model
    if default in choices:
        return (default, *(item for item in choices if item != default))
    return (default, *choices)


def resolve_model(requested: str | None) -> str:
    """The model to use for a turn.

    An unrecognised name falls back to the default rather than reaching the provider: the
    string is passed straight through to the host, so accepting it unchecked would let a
    caller select any model the account can see.
    """
    wanted = (requested or "").strip()
    if not wanted:
        return get_settings().reasoning_model
    return wanted if wanted in allowed_models() else get_settings().reasoning_model


def timeout_for(model: str) -> int:
    """How long this model is allowed, which depends on whether it reasons.

    A reasoning model spends thousands of tokens thinking before it answers -- measured p95
    87s against 21s for the direct one. One shared ceiling cannot serve both: set low enough
    for the fast model and the slow ones fail outright, set high enough for the slow ones
    and a hung request blocks the common case for minutes.
    """
    settings = get_settings()
    if model in settings.direct_answer_models:
        return settings.xai_request_timeout_ms
    return settings.reasoning_request_timeout_ms


def get_reasoning_provider(model_override: str | None = None) -> ReasoningProvider | None:
    settings = get_settings()
    name = settings.reasoning_provider.strip().lower()
    if name in {"off", "none", "disabled"}:
        logger.warning("reasoning provider disabled (%s)", name)
        return None
    if name == "fake":
        return FakeReasoningProvider()
    if not (settings.xai_api_key or "").strip():
        logger.warning(
            "REASONING_PROVIDER=%s but XAI_API_KEY is empty; "
            "semantic compiler will be unavailable",
            settings.reasoning_provider,
        )
        return None
    model = resolve_model(model_override)
    timeout_ms = timeout_for(model)
    if name == "xai":
        return _xai_provider(
            settings.xai_api_key,
            model,
            settings.model_max_concurrency_per_turn,
            timeout_ms,
        )
    return _http_provider(
        settings.xai_api_key,
        settings.reasoning_base_url,
        model,
        settings.model_max_concurrency_per_turn,
        timeout_ms,
    )
