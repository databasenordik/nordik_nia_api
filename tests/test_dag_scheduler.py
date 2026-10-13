import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.concurrency.limits import ResourceLimits
from app.data_gateway.memory import MemoryDataGateway
from app.execution.dag_scheduler import DagScheduler, ScheduledStep, StepResult
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def test_independent_steps_overlap_and_dependents_wait():
    started: dict[str, float] = {}
    finished: dict[str, float] = {}

    async def execute(step: ScheduledStep, prior: dict[str, StepResult]) -> StepResult:
        started[step.step_id] = time.perf_counter()
        await asyncio.sleep(0.08)
        finished[step.step_id] = time.perf_counter()
        return StepResult(value=step.step_id)

    steps = [
        ScheduledStep(step_id="a", operation="A", dependencies=[], resource_class="local"),
        ScheduledStep(step_id="b", operation="B", dependencies=[], resource_class="local"),
        ScheduledStep(step_id="c", operation="C", dependencies=["a", "b"], resource_class="local"),
    ]
    scheduler = DagScheduler(limits=ResourceLimits(global_concurrency=4))
    results, trace = asyncio.run(scheduler.run(steps, execute, scope_fingerprint="fp"))
    assert set(results) == {"a", "b", "c"}
    assert started["a"] < finished["b"] and started["b"] < finished["a"]
    assert started["c"] >= finished["a"] and started["c"] >= finished["b"]
    assert trace.parallel_step_peak >= 2
    assert all(item.scope_fingerprint == "fp" for item in trace.steps.values())
    assert trace.critical_path_ms >= 80


def test_singleflight_shares_identical_inflight_work():
    calls = {"n": 0}

    async def execute(step: ScheduledStep, prior: dict[str, StepResult]) -> StepResult:
        calls["n"] += 1
        await asyncio.sleep(0.05)
        return StepResult(value=1)

    steps = [
        ScheduledStep(
            step_id="one",
            operation="X",
            dependencies=[],
            idempotency_key="same",
        ),
        ScheduledStep(
            step_id="two",
            operation="X",
            dependencies=[],
            idempotency_key="same",
        ),
    ]
    scheduler = DagScheduler(limits=ResourceLimits(global_concurrency=4))
    asyncio.run(scheduler.run(steps, execute, scope_fingerprint="fp"))
    assert calls["n"] == 1


def test_mention_plan_runs_retrieval_branches_in_parallel():
    gateway = MemoryDataGateway()
    gateway.method_delays = {"fts": 0.08, "fuzzy": 0.08}
    parsed = parse_deterministic("Which records mention tuberculosis?", _scope(), static_catalog())
    assert parsed.plan is not None
    result = asyncio.run(
        AssistantTurnService(
            gateway, store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
        ).answer(_scope(), "Which records mention tuberculosis?")
    )
    assert result.status == "answered"
    assert result.parallel_step_peak >= 2
    from app.execution.executor import QueryExecutor

    execution = asyncio.run(QueryExecutor(gateway, static_catalog()).execute(parsed.plan, _scope()))
    assert execution.run_trace is not None
    fts = execution.run_trace.steps["fts"]
    fuzzy = execution.run_trace.steps["fuzzy"]
    assert fts.started_at is not None and fuzzy.started_at is not None
    assert fts.started_at < fuzzy.completed_at
    assert fuzzy.started_at < fts.completed_at
    assert fts.scope_fingerprint == fuzzy.scope_fingerprint


def test_exact_unique_cancels_sibling_fuzzy_branch():
    gateway = MemoryDataGateway()
    gateway.method_delays = {"fuzzy": 0.2}
    result = asyncio.run(
        AssistantTurnService(
            gateway, store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
        ).answer(_scope(), "Find Samuel A")
    )
    assert result.status == "answered"
    execution_plan = parse_deterministic("Find Samuel A", _scope(), static_catalog()).plan
    assert execution_plan is not None
    from app.execution.executor import QueryExecutor

    execution = asyncio.run(QueryExecutor(gateway, static_catalog()).execute(execution_plan, _scope()))
    assert "fuzzy" in execution.cancelled
    assert execution.run_trace is not None
    assert execution.run_trace.steps["fuzzy"].cancelled


def test_no_orphan_tasks_after_cancel():
    async def execute(step: ScheduledStep, prior: dict[str, StepResult]) -> StepResult:
        if step.step_id == "slow":
            await asyncio.sleep(1)
        return StepResult(value=1, early_complete=step.step_id == "fast")

    steps = [
        ScheduledStep(
            step_id="fast",
            operation="EXACT_LOOKUP",
            dependencies=[],
            branch_group="retrieval",
        ),
        ScheduledStep(
            step_id="slow",
            operation="FUZZY_SEARCH",
            dependencies=[],
            branch_group="retrieval",
        ),
    ]
    async def main() -> None:
        scheduler = DagScheduler(limits=ResourceLimits(global_concurrency=4))
        results, trace = await scheduler.run(steps, execute, scope_fingerprint="fp")
        assert results["slow"].cancelled or trace.steps["slow"].cancelled
        orphans = [task for task in asyncio.all_tasks() if not task.done() and task is not asyncio.current_task()]
        assert orphans == []

    asyncio.run(main())
