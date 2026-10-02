"""Reproducible offline release evidence for the NIA planner.

This module deliberately separates what can be proven without external infrastructure from
historical live evidence. It never treats FakeReasoningProvider as a model-accuracy oracle.
Run from ``backend`` with::

    python -m benchmarks.release_evaluation --iterations 200

For the complete qualification (pytest + compileall + this probe), use
``python -m benchmarks.release_check``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.llm.fake import FakeReasoningProvider
from app.planning.ai_planner import (
    MAX_PLANNER_MEMORY_CHARS,
    MAX_PLANNER_QUESTION_CHARS,
    plan_user_turn_ai,
)
from app.planning.catalog import static_catalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.turn_schema import (
    MAX_FILTER_VALUE_ITEMS,
    MAX_FILTER_VALUE_TEXT,
    MAX_FILTERS_PER_ACTION,
    ActiveQuery,
)
from app.security.access_scope import AccessScope
from benchmarks.planner_efficiency import run as run_efficiency

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
SCOPED_SUITE = REPO_ROOT / "docs" / "ai-planner-scoped-suite.json"
HISTORICAL_LATENCY = REPO_ROOT / "docs" / "latency-optimization-results.md"


def _suite_summary() -> dict[str, object]:
    payload = json.loads(SCOPED_SUITE.read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    selected = sorted({int(case["selected_file_id"]) for case in cases})
    return {
        "path": str(SCOPED_SUITE.relative_to(REPO_ROOT)),
        "schema_version": payload.get("schema_version"),
        "cases": len(cases),
        "all_critical": bool(cases) and all(bool(case.get("critical")) for case in cases),
        "all_have_semantic_contract": bool(cases)
        and all(bool(case.get("expected_plan")) for case in cases),
        "selected_file_ids": selected,
        "categories": dict(sorted(Counter(str(case.get("category")) for case in cases).items())),
        "expected_outcomes": dict(
            sorted(Counter(str(case.get("expected_outcome")) for case in cases).items())
        ),
        "private_cases": sum(bool(case.get("private")) for case in cases),
    }


async def _model_call_policy() -> dict[str, object]:
    catalog = static_catalog()
    scope = AccessScope(principal_id="release-eval", allowed_file_ids=(49, 91))

    async def one(question: str, context: ConversationContext) -> dict[str, object]:
        provider = FakeReasoningProvider()
        result = await plan_user_turn_ai(
            question,
            scope,
            catalog,
            selected_file_id=49,
            conversation=context,
            reasoner=provider,
            authorized_catalog=catalog,
        )
        return {
            "status": result.status,
            "model_calls": result.planner_total_model_calls,
            "review_status": result.review_status,
            "provider_structured_calls": provider.structured_calls,
        }

    return {
        "greeting": await one("hello", ConversationContext()),
        "next_page": await one(
            "next page",
            ConversationContext(active_query=ActiveQuery(datasets=[49], goal="list", limit=25)),
        ),
        "verify": await one(
            "Are you sure?",
            ConversationContext(active_query=ActiveQuery(datasets=[49], goal="count")),
        ),
        "fresh_query": await one(
            "How many students were admitted in 1905?", ConversationContext()
        ),
    }


def _gate(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


async def evaluate(iterations: int) -> dict[str, object]:
    get_settings.cache_clear()
    declared = {
        name: Settings.model_fields[name].default
        for name in (
            "planner_mode",
            "planner_review_mode",
            "planner_self_review",
            "planner_requirement_check",
            "planner_min_confidence",
        )
    }
    suite = _suite_summary()
    efficiency = await run_efficiency(iterations)
    calls = await _model_call_policy()
    datasets = static_catalog().datasets

    gates = [
        _gate("default_ai_planner", declared["planner_mode"] == "ai", f"planner_mode={declared["planner_mode"]}"),
        _gate(
            "mandatory_review_default",
            bool(declared["planner_self_review"]) and declared["planner_review_mode"] == "always",
            f"self_review={declared["planner_self_review"]}, review_mode={declared["planner_review_mode"]}",
        ),
        _gate(
            "scoped_suite_contract",
            bool(suite["all_critical"]) and bool(suite["all_have_semantic_contract"]),
            f"{suite['cases']} critical cases with semantic contracts",
        ),
        _gate(
            "all_catalog_datasets_represented",
            suite["selected_file_ids"] == sorted(dataset.file_id for dataset in datasets),
            f"selected={suite['selected_file_ids']}",
        ),
        _gate(
            "zero_call_trusted_controls",
            all(calls[key]["model_calls"] == 0 for key in ("greeting", "next_page", "verify")),
            "greeting, list pagination, and verify use trusted zero-model paths",
        ),
        _gate(
            "fresh_query_primary_plus_review",
            calls["fresh_query"]["model_calls"] == 2
            and calls["fresh_query"]["review_status"] in {"complete", "corrected"},
            f"calls={calls['fresh_query']['model_calls']}, review={calls['fresh_query']['review_status']}",
        ),
    ]

    live_configured = bool(
        (os.environ.get("REASONING_API_KEY") or os.environ.get("XAI_API_KEY"))
        and (os.environ.get("ASSISTANT_DATABASE_URL") or os.environ.get("DATABASE_URL"))
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "kind": "offline_release_evidence",
        "release_gate": "pass" if all(item["status"] == "pass" for item in gates) else "fail",
        "gates": gates,
        "settings": {
            "planner_mode": declared["planner_mode"],
            "planner_review_mode": declared["planner_review_mode"],
            "planner_self_review": declared["planner_self_review"],
            "planner_requirement_check": declared["planner_requirement_check"],
            "planner_min_confidence": declared["planner_min_confidence"],
        },
        "budgets": {
            "question_chars": MAX_PLANNER_QUESTION_CHARS,
            "memory_chars": MAX_PLANNER_MEMORY_CHARS,
            "filters_per_action": MAX_FILTERS_PER_ACTION,
            "filter_groups": 8,
            "filters_per_group": 8,
            "filter_value_items": MAX_FILTER_VALUE_ITEMS,
            "filter_value_text_chars": MAX_FILTER_VALUE_TEXT,
        },
        "catalog": {
            "dataset_count": len(datasets),
            "file_ids": [dataset.file_id for dataset in datasets],
            "field_counts": {
                str(dataset.file_id): len(static_catalog().fields_for(dataset.file_id))
                for dataset in datasets
            },
        },
        "scoped_suite": suite,
        "model_call_policy": calls,
        "efficiency": efficiency,
        "live_qualification": {
            "configured_in_current_environment": live_configured,
            "current_code_model_accuracy": "unmeasured",
            "current_code_database_truth_accuracy": "unmeasured",
            "current_code_provider_latency": "unmeasured",
            "reason": "release evaluation has no live database/provider credentials",
            "historical_evidence_file": str(HISTORICAL_LATENCY.relative_to(REPO_ROOT)),
            "historical_evidence_is_not_current_qualification": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")
    report = asyncio.run(evaluate(args.iterations))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["release_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
