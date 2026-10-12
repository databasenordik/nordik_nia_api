import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_fixtures import EVAL_CASES, UNAUTHORIZED_SOURCES, EvalCase

from app.concurrency.limits import ResourceLimits
from app.data_gateway.memory import MemoryDataGateway
from app.execution.executor import QueryExecutor
from app.execution.turn import AssistantTurnService
from app.llm.citations import validate_synthesis
from app.llm.fake import FakeReasoningProvider
from app.llm.schemas import MemoryUpdate, SynthesisAnswer
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.planning.plan_validator import PlanValidationError, validate_query_plan
from app.retrieval.types import EvidenceItem, EvidencePacket
from app.security.access_scope import AccessScope


def _scope(case: EvalCase) -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=case.allowed_file_ids)


def _service(case: EvalCase) -> tuple[AssistantTurnService, FakeReasoningProvider | None, InMemoryMemoryStore]:
    store = InMemoryMemoryStore()
    reasoner = FakeReasoningProvider()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=reasoner)
    return service, reasoner, store


def _run_case(case: EvalCase) -> dict:
    scope = _scope(case)
    catalog = static_catalog()
    parsed = parse_deterministic(case.question, scope, catalog)
    plan_valid = False
    if parsed.plan is not None:
        try:
            validate_query_plan(parsed.plan, scope, catalog)
            plan_valid = True
        except PlanValidationError:
            plan_valid = False
    service, reasoner, store = _service(case)
    first = asyncio.run(
        service.answer(scope, case.question, selected_file_id=case.selected_file_id)
    )
    second = None
    if case.follow_up:
        second = asyncio.run(
            service.answer(
                scope,
                case.follow_up,
                conversation_id=first.conversation_id,
                selected_file_id=case.selected_file_id,
            )
        )
    citations = list(first.citations)
    if second:
        citations.extend(second.citations)
    unauthorized = [
        item
        for item in citations
        if any(item.startswith(prefix) or prefix in item for prefix in UNAUTHORIZED_SOURCES)
    ]
    evidence_ids = [row.get("source_id", "") for row in first.evidence]
    unauthorized.extend(
        item
        for item in evidence_ids
        if any(str(item).startswith(prefix) for prefix in ("file:93:", "file:94:"))
    )
    return {
        "case": case,
        "plan_valid": plan_valid,
        "first": first,
        "second": second,
        "unauthorized": unauthorized,
        "reasoner": reasoner,
        "store": store,
    }


def test_eval_suite_plan_validity_aggregates_citations_and_followups():
    reports = [_run_case(case) for case in EVAL_CASES]
    unauthorized_total = sum(len(item["unauthorized"]) for item in reports)
    assert unauthorized_total == 0

    for report in reports:
        case: EvalCase = report["case"]
        first = report["first"]
        assert first.status == case.expect_status, case.id
        assert first.reasoning_calls == case.expect_reasoning_calls, (
            case.id,
            first.reasoning_calls,
        )
        if case.expect_plan_valid and parse_deterministic(
            case.question, _scope(case), static_catalog()
        ).plan:
            assert report["plan_valid"] is True, case.id
        if case.expect_count is not None:
            counts = [fact["value"] for fact in first.facts if fact["name"] == "count"]
            assert counts and counts[0] == case.expect_count, case.id
        if first.plan is not None:
            assert tuple(first.plan.scope.file_ids) == case.expect_file_ids, case.id
        if case.follow_up:
            second = report["second"]
            assert second is not None
            assert second.status == "answered", case.id
            if case.follow_up_count is not None:
                assert second.facts[0]["value"] == case.follow_up_count, case.id
            record = asyncio.run(report["store"].get(_scope(case), first.conversation_id))
            assert record is not None
            if case.follow_up_frame:
                assert record.active_frame_key == case.follow_up_frame, case.id


def test_eval_unauthorized_source_rate_is_zero_even_when_model_poisons_ids():
    packet = EvidencePacket(
        question="q",
        items=(
            EvidenceItem(
                source_id="file:49:v3:row:1102",
                file_id=49,
                version=3,
                source_row_id=1102,
                fields={"student_name": "Sarah B"},
            ),
        ),
    )
    raw = SynthesisAnswer(
        answer="poisoned",
        citations=["file:49:v3:row:1102", "file:93:v1:row:1108"],
        memory_update=MemoryUpdate(last_source_ids=["file:93:v1:row:1108"]),
    )
    cleaned = validate_synthesis(
        raw,
        packet,
        AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91)),
    )
    leaked = [item for item in cleaned.citations if item.startswith("file:93:")]
    assert leaked == []


def test_parallel_critical_path_beats_sequential_on_hybrid_retrieval():
    question = "Which records mention tuberculosis?"
    scope = AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))
    parsed = parse_deterministic(question, scope, static_catalog())
    assert parsed.plan is not None
    validate_query_plan(parsed.plan, scope, static_catalog())

    gateway = MemoryDataGateway()
    gateway.method_delays = {"fts": 0.08, "fuzzy": 0.08}

    sequential = asyncio.run(
        QueryExecutor(
            gateway, static_catalog(), limits=ResourceLimits(global_concurrency=1)
        ).execute(parsed.plan, scope)
    )
    parallel = asyncio.run(
        QueryExecutor(
            gateway, static_catalog(), limits=ResourceLimits(global_concurrency=8)
        ).execute(parsed.plan, scope)
    )
    assert sequential.run_trace is not None
    assert parallel.run_trace is not None
    assert parallel.run_trace.parallel_step_peak >= 2
    assert sequential.run_trace.parallel_step_peak == 1
    assert parallel.run_trace.latency_ms < sequential.run_trace.latency_ms
    assert parallel.run_trace.critical_path_ms <= sequential.run_trace.latency_ms
