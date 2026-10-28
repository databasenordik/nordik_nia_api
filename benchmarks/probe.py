"""Show what one question compiles to: TurnPlan, QueryPlan ops, and answer.

Diagnostic tool for the regression suite. Nothing here is benchmark-specific.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService, select_dataset_for_question  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.planning.router import plan_user_turn  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--file-ids", default="49,91")
    parser.add_argument("--file-id", type=int, default=None, help="already-selected dataset")
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()

    await init_pool()
    gateway = AssistantDataGateway(get_pool())
    scope = AccessScope(
        principal_id="probe",
        allowed_file_ids=tuple(int(item) for item in args.file_ids.split(",")),
    )
    catalog = await gateway.load_catalog(scope)
    chosen = args.file_id or select_dataset_for_question(
        args.question, catalog, scope, catalog.default_people_file_id
    )
    if chosen not in scope.allowed_file_ids:
        raise SystemExit(f"selected dataset {chosen} is not in --file-ids")
    print("selected dataset:", chosen)
    narrowed = scope.narrow((chosen,))
    reasoner = get_reasoning_provider()
    route = await plan_user_turn(
        args.question,
        narrowed,
        catalog.for_scope(narrowed),
        semantic_compiler=AssistantTurnService(gateway, reasoner=reasoner)._semantic_compiler(),
        selected_file_id=chosen,
        reasoner=reasoner,
        authorized_catalog=catalog,
    )
    print(
        "route:",
        route.status,
        "|",
        route.source,
        "| review:",
        route.review_status,
        "| calls:",
        route.planner_total_model_calls,
        "|",
        route.detail[:120],
    )
    if route.original_planned is not None:
        print("planned:", route.original_planned.model_dump_json()[:2400])
    if route.turn_plan is not None:
        print("turn:", json.dumps(route.turn_plan.public_dict(), ensure_ascii=False)[:1600])
    if route.plan is not None:
        print("ops:", route.plan.op_names())
        for step in route.plan.steps:
            print("   ", step.id, step.op.value, [p.payload() for p in step.where], step.fields,
                  step.value_part, step.top_n, step.interval_start, step.interval_end)
    if args.answer:
        service = AssistantTurnService(
            gateway, store=InMemoryMemoryStore(), reasoner=reasoner
        )
        result = await service.answer(scope, args.question, selected_file_id=chosen)
        print("answer:", result.answer[:1200])
    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
