"""Sequential latency probe. Do not parallelize cases.

p50/p95 here measure the uncontended turn path. Parallel cases would measure
contention against model_max_concurrency_global instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    return [item for item in cases if isinstance(item, dict) and item.get("question")]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential per-turn latency probe")
    parser.add_argument("--suite", required=True, help="JSON suite with cases[].question")
    parser.add_argument("--label", default="run")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    cases = _load_suite(Path(args.suite))
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"LATENCY PROBE label={args.label} start={started_at} cases={len(cases)} sequential=yes")
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
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases, start=1):
            file_id = int(case.get("selected_file_id") or case.get("file_id") or 49)
            question = str(case["question"])
            result = await service.answer(scope, question, selected_file_id=file_id)
            row = {
                "id": case.get("id") or index,
                "question": question,
                "status": result.status,
                "review_status": result.review_status,
                "latency": dict(result.latency or {}),
            }
            rows.append(row)
            lat = result.latency or {}
            print(
                f"[{index:02d}/{len(cases)}] {row['id']} status={result.status} "
                f"e2e={lat.get('e2e_ms')} plan={lat.get('plan_ms')} "
                f"review={lat.get('review_ms')} persist={lat.get('persist_ms')}"
            )
    finally:
        await close_pool()
    ended_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    keys = ("e2e_ms", "ttft_ms", "plan_ms", "review_ms", "catalog_ms", "retrieval_ms", "synthesis_ms", "persist_ms")
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(row["latency"][key]) for row in rows if row["latency"].get(key) is not None]
        summary[key] = {
            "n": len(values),
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
        }
    # Share of a turn spent in a stage, as the median of the per-turn ratios rather than
    # p50(stage)/p50(e2e). The two answer different questions and diverge: on the baseline
    # persist was 0.272% per turn but 0.367% by the p50 quotient, because a turn's persist
    # cost does not track its own model latency. The per-turn median is the honest answer to
    # "is this stage worth optimizing", so it is the one reported -- named for what it is,
    # since a "_p50" suffix invited exactly the reconciliation that does not work.
    stage_share: dict[str, float] = {}
    for key in ("persist_ms", "catalog_ms", "retrieval_ms", "plan_ms", "review_ms"):
        ratios = sorted(
            float(row["latency"][key]) / float(row["latency"]["e2e_ms"])
            for row in rows
            if row["latency"].get(key) is not None and row["latency"].get("e2e_ms")
        )
        if ratios:
            stage_share[f"{key.removesuffix('_ms')}_over_e2e_median_ratio"] = _percentile(ratios, 50)
    synth = sum(1 for row in rows if row["latency"].get("final_response") == "llm_synthesis")
    unavailable = sum(1 for row in rows if row["latency"].get("planner_unavailable"))
    report = {
        "label": args.label,
        "started_at": started_at,
        "ended_at": ended_at,
        "sequential": True,
        "synthesis_rate": synth / len(rows) if rows else None,
        **stage_share,
        "planner_unavailable": unavailable,
        "summary": summary,
        "cases": rows,
    }
    print(json.dumps(
        {"summary": summary, "synthesis_rate": report["synthesis_rate"], **stage_share},
        indent=2,
    ))
    print(f"RUN end={ended_at}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
