from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.planning.canonical_query import CanonicalQuery, CanonicalTurn
from app.planning.catalog import FieldCatalog
from app.planning.semantic_normalizer import (
    SEMANTIC_NORMALIZER_SYSTEM,
    semantic_normalizer_user_prompt,
)
from app.security.access_scope import AccessScope

logger = logging.getLogger("nia.planning.interpreter")


class CanonicalStructuredCall(Protocol):
    async def __call__(
        self,
        *,
        system: str,
        user: str,
        response_model: type[CanonicalTurn],
    ) -> CanonicalTurn: ...


@dataclass
class InterpreterResult:
    """One attempt to interpret a turn. Always emitted, even when the LLM is down."""

    status: str
    provider: str
    model: str
    duration_ms: float
    reasoning_calls: int
    request: dict[str, Any]
    output: dict[str, Any] | None = None
    canonical: CanonicalQuery | None = None
    error: str | None = None

    def as_layer(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": "semantic_interpreter",
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "reasoning_calls": self.reasoning_calls,
            "request": self.request,
            "output": self.output,
        }
        if self.error:
            payload["error"] = self.error
        return payload


class TurnInterpreterProtocol(Protocol):
    provider_name: str
    model_name: str

    async def interpret(
        self,
        user_text: str,
        *,
        normalized_text: str,
        conversation_state: str,
        mode: str,
        catalog: FieldCatalog,
        scope: AccessScope,
    ) -> InterpreterResult: ...


class LLMTurnInterpreter:
    """Paid xAI (or injected) structured call: utterance → CanonicalTurn.

    This is the natural-language layer. Deterministic parsing must abstain
    before this runs. It is never a regex table and never a silent placeholder.
    """

    def __init__(
        self,
        structured_call: CanonicalStructuredCall,
        *,
        provider_name: str,
        model_name: str,
    ) -> None:
        self._structured_call = structured_call
        self.provider_name = provider_name
        self.model_name = model_name

    async def interpret(
        self,
        user_text: str,
        *,
        normalized_text: str,
        conversation_state: str,
        mode: str,
        catalog: FieldCatalog,
        scope: AccessScope,
    ) -> InterpreterResult:
        scoped = catalog.for_scope(scope)
        user_payload = semantic_normalizer_user_prompt(
            question=user_text,
            normalized_text=normalized_text,
            catalog=scoped,
            memory_text=conversation_state,
        )
        request = {
            "user_utterance": user_text,
            "normalized_text": normalized_text,
            "mode": mode,
            "catalog_file_ids": [item.file_id for item in scoped.datasets],
            "memory_present": bool(conversation_state.strip()),
            "response_model": "CanonicalTurn",
        }
        started = time.perf_counter()
        try:
            result = await self._structured_call(
                system=SEMANTIC_NORMALIZER_SYSTEM,
                user=user_payload,
                response_model=CanonicalTurn,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            canonical = result if isinstance(result, CanonicalQuery) else CanonicalTurn.model_validate(result)
            output = _public_canonical(canonical)
            logger.info(
                "semantic_interpreter provider=%s model=%s status=ok duration_ms=%.1f "
                "reasoning_calls=1 type=%s intent=%s goal=%s",
                self.provider_name,
                self.model_name,
                duration_ms,
                output.get("type"),
                output.get("intent"),
                output.get("goal"),
            )
            return InterpreterResult(
                status="ok",
                provider=self.provider_name,
                model=self.model_name,
                duration_ms=duration_ms,
                reasoning_calls=1,
                request=request,
                output=output,
                canonical=canonical,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "semantic_interpreter provider=%s model=%s status=error duration_ms=%.1f",
                self.provider_name,
                self.model_name,
                duration_ms,
            )
            return InterpreterResult(
                status="error",
                provider=self.provider_name,
                model=self.model_name,
                duration_ms=duration_ms,
                reasoning_calls=1,
                request=request,
                error=str(exc) or exc.__class__.__name__,
            )


class UnavailableTurnInterpreter:
    """Always-present layer when xAI is not configured. Never pretends to have run."""

    def __init__(self, reason: str) -> None:
        self.provider_name = "none"
        self.model_name = ""
        self._reason = reason

    async def interpret(
        self,
        user_text: str,
        *,
        normalized_text: str,
        conversation_state: str,
        mode: str,
        catalog: FieldCatalog,
        scope: AccessScope,
    ) -> InterpreterResult:
        scoped = catalog.for_scope(scope)
        request = {
            "user_utterance": user_text,
            "normalized_text": normalized_text,
            "mode": mode,
            "catalog_file_ids": [item.file_id for item in scoped.datasets],
            "memory_present": bool(conversation_state.strip()),
            "response_model": "CanonicalTurn",
        }
        logger.warning("semantic_interpreter unavailable: %s", self._reason)
        return InterpreterResult(
            status="unavailable",
            provider="none",
            model="",
            duration_ms=0.0,
            reasoning_calls=0,
            request=request,
            error=self._reason,
        )


def _public_canonical(canonical: CanonicalQuery) -> dict[str, Any]:
    return {
        "type": getattr(canonical.turn_type, "value", str(canonical.turn_type)),
        "intent": canonical.intent,
        "file_ids": list(canonical.file_ids),
        "goal": getattr(canonical.goal, "value", canonical.goal),
        "filters": [item.model_dump(mode="json") for item in canonical.filters],
        "group_by": list(canonical.group_by),
        "search_text": canonical.search_text,
        "unresolved": list(canonical.unresolved),
        "clarification": canonical.clarification_question,
        "answer_directly": canonical.answer_directly,
        "direct_answer": canonical.direct_answer,
        "canonical_text": canonical.canonical_text,
        "confidence": canonical.confidence,
        "requires_full_planner": canonical.requires_full_planner,
    }
