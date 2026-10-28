from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from app.llm.schemas import PlannedQuery, SynthesisAnswer


@dataclass(frozen=True)
class StructuredOutputResult[T: BaseModel]:
    """Parsed structured call plus optional provider accounting."""

    value: T
    provider: str = ""
    model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # Prompt tokens the host served from cache. The gRPC path never reported these, so the
    # value of keeping the stable part of a prompt first was unmeasurable; over REST it is
    # reported and worth recording.
    cached_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: float = 0.0


class TransientReasoningError(RuntimeError):
    """The provider did not answer, and might on the next attempt.

    Kept here rather than beside a provider because the planner has to be able to tell a
    timeout from a refusal without importing one, and the distinction decides whether a turn
    is retried or lost. A read timeout on the York thread cost the researcher a whole answer
    that a second call would have given them.
    """


class ReasoningProvider(Protocol):
    """Paid reasoning boundary. Planning/retrieval/gateway must not import a vendor SDK."""

    async def stream_answer(self, *, system: str, user: str) -> AsyncIterator[str]: ...

    async def answer_structured(self, *, system: str, user: str) -> SynthesisAnswer: ...

    async def plan_structured(self, *, system: str, user: str) -> PlannedQuery: ...

    async def structured_output[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> T: ...

    async def structured_output_with_metadata[T: BaseModel](
        self, *, system: str, user: str, response_model: type[T]
    ) -> StructuredOutputResult[T]: ...
