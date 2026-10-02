"""Where a turn's wall-clock actually goes.

Runs a handful of representative questions through the real service against the real
database and prints the tracer's span breakdown per turn, so tuning targets the part that
costs seconds rather than the part that is merely easy to see.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.observability.tracing import get_tracer  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

PROBES = [
    (49, "hello"),
    (49, "List the students whose surname is Jones."),
    (49, "What is the most common community?"),
    (91, "How many students are recorded as deceased?"),
    (91, "Which students died of tuberculosis?"),
    (94, "List the names in this list."),
]


async def main() -> int:
    await init_pool()
    service = AssistantTurnService(
        AssistantDataGateway(get_pool()),
        store=InMemoryMemoryStore(),
        reasoner=get_reasoning_provider(),
    )
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91, 93, 94),
        can_use_private_files=True,
    )
    tracer = get_tracer()
    rows = []
    try:
        for file_id, question in PROBES:
            before = len(tracer.spans)
            started = time.perf_counter()
            result = await service.answer(scope, question, selected_file_id=file_id)
            wall = (time.perf_counter() - started) * 1000
            spans = tracer.spans[before:]
            buckets: dict[str, tuple[int, float]] = {}
            for span in spans:
                count, total = buckets.get(span.name, (0, 0.0))
                buckets[span.name] = (count + 1, total + span.duration_ms)
            rows.append((file_id, question, wall, result.planner_type, result.reasoning_calls, buckets))
            parts = " ".join(
                f"{name}x{count}={total:.0f}ms" for name, (count, total) in sorted(buckets.items())
            )
            print(f"[{file_id}] {wall:7.0f}ms calls={result.reasoning_calls} {result.planner_type:<12} {question}")
            print(f"          {parts}")
    finally:
        await close_pool()

    print()
    total_wall = sum(row[2] for row in rows)
    llm_total = sum(row[5].get("llm", (0, 0.0))[1] for row in rows)
    retrieval_total = sum(row[5].get("retrieval", (0, 0.0))[1] for row in rows)
    calls = sum(row[4] for row in rows)
    print(f"turns={len(rows)} reasoning_calls={calls} wall={total_wall:.0f}ms")
    print(f"  llm spans        {llm_total:8.0f}ms  ({llm_total / total_wall:5.1%})")
    print(f"  retrieval spans  {retrieval_total:8.0f}ms  ({retrieval_total / total_wall:5.1%})")
    print(f"  unaccounted      {total_wall - llm_total - retrieval_total:8.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
