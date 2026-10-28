"""Offline Slice-5 planner efficiency probe.

This does not pretend to measure provider latency. It measures the things we can reproduce
without credentials: model-call count for trusted control turns, prompt/schema byte proxies,
and local trusted-path overhead. Use live matrix tooling for real p50/p95 provider latency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

from app.config import get_settings
from app.llm.fake import FakeReasoningProvider
from app.llm.http_reasoner import _compact_response_schema
from app.planning.ai_planner import (
    AI_PLANNER_SYSTEM,
    AI_REVIEW_SYSTEM,
    _planner_user_prompt,
    _review_user_prompt,
    _safe_active_query_json,
    authorized_dataset_options,
    plan_user_turn_ai,
)
from app.planning.ai_turn_schema import (
    GoalRequirement,
    PlannedQueryAction,
    PlannedTurn,
    PlanReviewResponse,
)
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.turn_schema import ActiveQuery, FilterSpec
from app.security.access_scope import AccessScope


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


async def _one(question: str, context: ConversationContext):
    catalog = static_catalog()
    provider = FakeReasoningProvider()
    started = time.perf_counter()
    result = await plan_user_turn_ai(
        question,
        AccessScope(principal_id="efficiency-probe", allowed_file_ids=(49, 91)),
        catalog,
        selected_file_id=49,
        conversation=context,
        reasoner=provider,
        authorized_catalog=catalog,
    )
    return result, (time.perf_counter() - started) * 1000


async def run(iterations: int) -> dict:
    catalog = static_catalog()
    options = authorized_dataset_options(catalog)
    question = "How many students from Garden River were admitted in 1905?"
    planned = PlannedTurn(
        actions=[PlannedQueryAction(goal="count")],
        stated_requirements=[GoalRequirement(action_index=0, text="count records", goal="count")],
        final_response="deterministic",
        confidence=0.95,
    )
    primary_user = _planner_user_prompt(
        question,
        catalog=catalog,
        selected_file_id=49,
        authorized_options=options,
        context=ConversationContext(),
        input_mode="text",
        min_confidence=0.75,
    )
    review_user = _review_user_prompt(
        question,
        planned,
        catalog,
        49,
        authorized_options=options,
        context=ConversationContext(),
    )
    schema_chars = sum(
        len(json.dumps(_compact_response_schema(model), separators=(",", ":")))
        for model in (PlannedTurn, PlanReviewResponse)
    )
    active_json = _safe_active_query_json(
        ConversationContext(
            active_query=ActiveQuery(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
                requested_fields=["student_name"],
                result_set_id="not-model-visible",
            )
        )
    )
    candidate = review_user.split("CANDIDATE PLAN:\n", 1)[1]

    cases = {
        "greeting": ("hello", ConversationContext()),
        "next_page": (
            "next page",
            ConversationContext(
                active_query=ActiveQuery(datasets=[49], goal="list", limit=25, offset=0)
            ),
        ),
        "verify": (
            "Are you sure?",
            ConversationContext(active_query=ActiveQuery(datasets=[49], goal="count")),
        ),
    }
    fast: dict[str, dict] = {}
    for name, (text, context) in cases.items():
        durations: list[float] = []
        calls: list[int] = []
        status = ""
        for _ in range(iterations):
            result, duration = await _one(text, context)
            durations.append(duration)
            calls.append(result.planner_total_model_calls)
            status = result.status
        fast[name] = {
            "status": status,
            "model_calls": max(calls),
            "local_ms_p50": round(statistics.median(durations), 3),
            "local_ms_p95": round(_percentile(durations, 0.95), 3),
        }

    return {
        "iterations": iterations,
        "normal_turn": {
            "system_user_chars": len(AI_PLANNER_SYSTEM)
            + len(primary_user)
            + len(AI_REVIEW_SYSTEM)
            + len(review_user),
            "http_structured_schema_chars": schema_chars,
            "combined_input_proxy_chars": len(AI_PLANNER_SYSTEM)
            + len(primary_user)
            + len(AI_REVIEW_SYSTEM)
            + len(review_user)
            + schema_chars,
            "review_candidate_chars": len(candidate),
            "active_query_chars": len(active_json),
        },
        "trusted_fast_paths": fast,
        "note": "Local timings use FakeReasoningProvider and are not live-provider latency.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    get_settings.cache_clear()
    print(json.dumps(asyncio.run(run(args.iterations)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
