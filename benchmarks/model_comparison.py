"""Accuracy and speed for each offered model, on the same 74 questions.

The models on the bridge are not interchangeable. The default answers directly; the newer
ones reason first, which on our planner schema is the difference between 6 and 35 seconds
per call. Since roughly nine tenths of a turn's wall-clock is model time, that choice is the
single largest lever on how NIA feels -- and it is being handed to the researcher, so it
needs numbers beside it rather than a recommendation.

This runs `database_question_matrix.py` once per model rather than reimplementing grading:
the oracle, the case list and the flaky/stable classification are all there already.

Read the flaky column, not just the rate. The planner samples, so a case can pass on one run
and fail the next with nothing changed; a difference between two models is only real if it
survives that. Three runs is the minimum that says anything and still not much.

Usage::

    python -m benchmarks.model_comparison --runs 3
    python -m benchmarks.model_comparison --models grok-4.5,grok-4.6 --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.factory import allowed_models, get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402
from benchmarks.database_question_matrix import (  # noqa: E402
    _LABELS,
    CASES,
    compare,
    load_oracle,
)


class ModelReport:
    def __init__(self, model: str) -> None:
        self.model = model
        self.outcomes: dict[str, list[bool]] = {}
        self.latencies: list[float] = []
        self.errors = 0

    def record(self, case_id: str, passed: bool, latency_ms: float) -> None:
        self.outcomes.setdefault(case_id, []).append(passed)
        self.latencies.append(latency_ms)

    @property
    def observations(self) -> int:
        return sum(len(seen) for seen in self.outcomes.values())

    @property
    def passed(self) -> int:
        return sum(sum(seen) for seen in self.outcomes.values())

    @property
    def stable_pass(self) -> list[str]:
        return sorted(cid for cid, seen in self.outcomes.items() if all(seen))

    @property
    def stable_fail(self) -> list[str]:
        return sorted(cid for cid, seen in self.outcomes.items() if not any(seen))

    @property
    def flaky(self) -> list[str]:
        return sorted(
            cid for cid, seen in self.outcomes.items() if any(seen) and not all(seen)
        )

    def percentile(self, fraction: float) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return ordered[index]


async def _measure(model: str, cases, oracle, runs: int, scope: AccessScope) -> ModelReport:
    report = ModelReport(model)
    service = AssistantTurnService(
        AssistantDataGateway(get_pool()),
        store=InMemoryMemoryStore(),
        reasoner=get_reasoning_provider(model),
    )
    total = len(cases) * runs
    done = 0
    for case in cases:
        expected = oracle.expected(case)
        for _ in range(runs):
            done += 1
            started = time.perf_counter()
            try:
                result = await service.answer(
                    scope, case.question, selected_file_id=case.file_id, model=model
                )
                issues = compare(case, expected, result)
            except Exception as exc:  # a crash is a failure, not a missing observation
                report.errors += 1
                issues = [f"runner_error {type(exc).__name__}: {exc}"]
            latency_ms = (time.perf_counter() - started) * 1000
            report.record(case.id, not issues, latency_ms)
            print(
                f"  [{done:03d}/{total}] {model} {case.id} "
                f"{'PASS' if not issues else 'FAIL'} {latency_ms:.0f}ms",
                flush=True,
            )
    return report


def _print(reports: list[ModelReport], runs: int) -> None:
    print(f"\n{'=' * 78}")
    print(f"MODEL COMPARISON  --runs {runs}  {len(CASES)} cases")
    print("=" * 78)
    header = f"{'model':32} {'pass rate':>12} {'stable':>7} {'flaky':>6} {'fail':>5} {'p50':>8} {'p95':>8}"
    print(header)
    print("-" * len(header))
    for report in reports:
        rate = report.passed / report.observations if report.observations else 0.0
        print(
            f"{report.model:32} "
            f"{report.passed:>4}/{report.observations:<3} {rate:>5.1%} "
            f"{len(report.stable_pass):>7} {len(report.flaky):>6} {len(report.stable_fail):>5} "
            f"{report.percentile(0.50) / 1000:>7.1f}s {report.percentile(0.95) / 1000:>7.1f}s"
        )
    print()
    for report in reports:
        if report.stable_fail:
            print(f"  {report.model} always fails: {','.join(report.stable_fail)}")
        if report.flaky:
            print(f"  {report.model} flaky:        {','.join(report.flaky)}")
        if report.errors:
            print(f"  {report.model} runner errors: {report.errors}")
    print(
        "\nA difference between two models is only real if it survives the flaky column: "
        "the planner samples, so a single run cannot separate a model from noise."
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ids", help="Comma-separated case IDs for a focused comparison")
    parser.add_argument("--models", help="Comma-separated models; defaults to the offered list")
    args = parser.parse_args()

    selected = {item.strip().upper() for item in (args.ids or "").split(",") if item.strip()}
    cases = tuple(case for case in CASES if not selected or case.id in selected)
    models = [item.strip() for item in (args.models or "").split(",") if item.strip()]
    models = models or list(allowed_models())

    print(f"models: {', '.join(models)}")
    print(f"cases : {len(cases)}  runs: {args.runs}  "
          f"timeout: {get_settings().xai_request_timeout_ms}ms")

    await init_pool()
    oracle = await load_oracle()
    # compare() resolves a field's human label through this module-level map, which the
    # matrix's own entry point fills in. Without it the schema case cannot match a label and
    # every model appears to fail it -- which is exactly what the first run of this harness
    # reported, for all four.
    for file_id, labels in oracle.labels.items():
        for field_name, label in labels.items():
            _LABELS[(file_id, field_name)] = label
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91, 93, 94),
        can_use_private_files=True,
    )
    reports: list[ModelReport] = []
    try:
        for model in models:
            reports.append(await _measure(model, cases, oracle, args.runs, scope))
    finally:
        await close_pool()
    _print(reports, args.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
