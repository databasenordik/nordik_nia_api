from __future__ import annotations

import asyncio
from collections.abc import Awaitable


class StepTimeoutError(TimeoutError):
    def __init__(self, step_id: str, timeout_ms: int) -> None:
        super().__init__(f"step {step_id} timed out after {timeout_ms}ms")
        self.step_id = step_id
        self.timeout_ms = timeout_ms


async def with_timeout[T](awaitable: Awaitable[T], timeout_ms: int, step_id: str) -> T:
    try:
        return await asyncio.wait_for(awaitable, timeout=max(timeout_ms, 1) / 1000)
    except TimeoutError as exc:
        raise StepTimeoutError(step_id, timeout_ms) from exc
