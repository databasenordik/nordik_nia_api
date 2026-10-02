"""What a name lookup actually filters on, and what the model is handed back.

The goal is that a one-word name search reaches every preprocessed name column, and that
the record returned afterwards is the row as it was recorded -- not the derived columns
the search used to find it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

PROBES = [
    "Tell me about Pahpahmaush",
    "Who is Wm. Fisher?",
    "Find the student whose surname is Assiginak",
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
    try:
        for question in PROBES:
            result = await service.answer(scope, question, selected_file_id=49)
            print(f"\n=== {question}")
            if result.plan is not None:
                for step in result.plan.steps:
                    where = [
                        f"{item.field} {item.operator} {item.value!r}"
                        for item in getattr(step, "where", [])
                    ]
                    print(f"  {step.op.value:16} fields={list(step.fields)}")
                    if where:
                        print(f"  {'':16} where={where}")
            print(f"  answer: {(result.answer or '')[:150]}")
            if result.evidence:
                first = result.evidence[0]
                fields = first.get("fields", {})
                print(f"  evidence row carries {len(fields)} fields:")
                for key in sorted(fields):
                    print(f"      {key} = {str(fields[key])[:60]}")
            else:
                print("  evidence: (none)")
    finally:
        await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
