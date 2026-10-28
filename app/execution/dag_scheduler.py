from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from app.concurrency.cancellation import CancellationToken
from app.concurrency.limits import ResourceGate, ResourceLimits
from app.concurrency.singleflight import Singleflight
from app.concurrency.timeouts import with_timeout


@dataclass
class ScheduledStep:
    step_id: str
    operation: str
    dependencies: list[str]
    resource_class: str = "local"
    timeout_ms: int = 2000
    cancel_policy: str = "skip_optional"
    idempotency_key: str | None = None
    branch_group: str | None = None
    scope_fingerprint: str = ""
    payload: Any = None


@dataclass
class StepResult:
    value: Any = None
    cancelled: bool = False
    early_complete: bool = False
    error: str | None = None
    result_count: int | None = None


@dataclass
class StepTrace:
    step_id: str
    operation: str
    branch_group: str | None
    scope_fingerprint: str
    ready_at: float | None = None
    queued_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    worker_id: str | None = None
    timeout_ms: int = 0
    cancelled: bool = False
    idempotency_key: str | None = None
    result_count: int | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        if self.started_at is None or self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at) * 1000


@dataclass
class RunTrace:
    access_scope_fingerprint: str
    started_at: float
    ended_at: float | None = None
    critical_path_ms: float = 0.0
    parallel_step_peak: int = 0
    queued_ms: float = 0.0
    cancel_reason: str | None = None
    steps: dict[str, StepTrace] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        if self.ended_at is None:
            return 0.0
        return (self.ended_at - self.started_at) * 1000


ExecuteFn = Callable[[ScheduledStep, dict[str, StepResult]], Awaitable[StepResult]]


class DagScheduler:
    """Ready-set scheduler. Independent steps overlap; dependents wait."""

    def __init__(
        self,
        *,
        limits: ResourceLimits | None = None,
        cancellation: CancellationToken | None = None,
        singleflight: Singleflight | None = None,
    ) -> None:
        self._gate = ResourceGate(limits)
        self._cancel = cancellation or CancellationToken()
        self._singleflight = singleflight or Singleflight()
        self._worker = 0

    async def run(
        self,
        steps: list[ScheduledStep],
        execute: ExecuteFn,
        *,
        scope_fingerprint: str,
    ) -> tuple[dict[str, StepResult], RunTrace]:
        by_id = {step.step_id: step for step in steps}
        remaining = {step.step_id: set(step.dependencies) for step in steps}
        pending = set(by_id)
        results: dict[str, StepResult] = {}
        traces = {
            step.step_id: StepTrace(
                step_id=step.step_id,
                operation=step.operation,
                branch_group=step.branch_group,
                scope_fingerprint=step.scope_fingerprint or scope_fingerprint,
                timeout_ms=step.timeout_ms,
                idempotency_key=step.idempotency_key,
            )
            for step in steps
        }
        run = RunTrace(access_scope_fingerprint=scope_fingerprint, started_at=time.perf_counter())
        running: dict[str, asyncio.Task[StepResult]] = {}

        def mark_ready() -> None:
            now = time.perf_counter()
            for step_id, deps in remaining.items():
                if not deps and traces[step_id].ready_at is None and step_id in pending:
                    traces[step_id].ready_at = now

        mark_ready()

        while pending or running:
            self._cancel.raise_if_cancelled()
            launched = False
            for step_id in sorted(pending):
                if remaining[step_id]:
                    continue
                if len(running) >= self._gate.limits.global_concurrency:
                    break
                pending.remove(step_id)
                traces[step_id].queued_at = time.perf_counter()
                running[step_id] = asyncio.create_task(
                    self._run_one(by_id[step_id], results, traces[step_id], execute)
                )
                launched = True
            run.parallel_step_peak = max(run.parallel_step_peak, len(running))
            if not running:
                if pending and not launched:
                    raise RuntimeError(f"DAG deadlock remaining={sorted(pending)}")
                break
            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                step_id = next(key for key, value in running.items() if value is task)
                del running[step_id]
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    result = StepResult(cancelled=True)
                    traces[step_id].cancelled = True
                results[step_id] = result
                traces[step_id].completed_at = time.perf_counter()
                traces[step_id].cancelled = result.cancelled
                traces[step_id].error = result.error
                traces[step_id].result_count = result.result_count
                if result.early_complete:
                    self._cancel_group(
                        by_id[step_id], pending, running, remaining, results, traces
                    )
                for other_deps in remaining.values():
                    other_deps.discard(step_id)
                mark_ready()

        run.ended_at = time.perf_counter()
        run.steps = traces
        run.queued_ms = sum(
            ((trace.started_at or 0) - (trace.ready_at or 0)) * 1000
            for trace in traces.values()
            if trace.started_at and trace.ready_at
        )
        run.critical_path_ms = _critical_path_ms(steps, traces)
        run.cancel_reason = self._cancel.reason
        return results, run

    async def _run_one(
        self,
        step: ScheduledStep,
        results: dict[str, StepResult],
        trace: StepTrace,
        execute: ExecuteFn,
    ) -> StepResult:
        self._worker += 1
        worker = f"worker-{self._worker}"
        trace.worker_id = worker
        await self._gate.acquire(step.resource_class)
        try:
            self._cancel.raise_if_cancelled()
            trace.started_at = time.perf_counter()

            async def invoke() -> StepResult:
                return await execute(step, results)

            if step.idempotency_key:
                return await self._singleflight.do(
                    step.idempotency_key,
                    lambda: with_timeout(invoke(), step.timeout_ms, step.step_id),
                )
            return await with_timeout(invoke(), step.timeout_ms, step.step_id)
        except asyncio.CancelledError:
            return StepResult(cancelled=True)
        except Exception as exc:
            return StepResult(error=str(exc))
        finally:
            if trace.completed_at is None:
                trace.completed_at = time.perf_counter()
            self._gate.release(step.resource_class)

    def _cancel_group(
        self,
        completed: ScheduledStep,
        pending: set[str],
        running: dict[str, asyncio.Task[StepResult]],
        remaining: dict[str, set[str]],
        results: dict[str, StepResult],
        traces: dict[str, StepTrace],
    ) -> None:
        if not completed.branch_group:
            return
        for step_id in list(pending):
            if traces[step_id].branch_group == completed.branch_group:
                pending.remove(step_id)
                results[step_id] = StepResult(cancelled=True, value=[])
                traces[step_id].cancelled = True
                traces[step_id].completed_at = time.perf_counter()
                for other_deps in remaining.values():
                    other_deps.discard(step_id)
        for step_id, task in list(running.items()):
            if traces[step_id].branch_group == completed.branch_group and step_id != completed.step_id:
                task.cancel()


def _critical_path_ms(steps: list[ScheduledStep], traces: dict[str, StepTrace]) -> float:
    by_id = {step.step_id: step for step in steps}
    memo: dict[str, float] = {}

    def walk(step_id: str) -> float:
        if step_id in memo:
            return memo[step_id]
        trace = traces[step_id]
        own = trace.duration_ms
        deps = by_id[step_id].dependencies
        best = own if not deps else own + max(walk(dep) for dep in deps if dep in traces)
        memo[step_id] = best
        return best

    if not steps:
        return 0.0
    return max(walk(step.step_id) for step in steps)
