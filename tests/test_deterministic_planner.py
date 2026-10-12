import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.planning.plan_validator import validate_query_plan
from app.security.access_scope import AccessScope

GOLDEN = "How many students have names starting with S?"


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def test_golden_question_plans_filter_count_on_file_49():
    parsed = parse_deterministic(GOLDEN, _scope(), static_catalog())
    assert parsed.plan is not None
    assert parsed.reasoning_calls == 0
    plan = validate_query_plan(parsed.plan, _scope(), static_catalog())
    assert plan.scope.file_ids == (49,)
    assert plan.op_names() == ["USE_CURRENT_VERSION", "FILTER", "COUNT"]
    filt = plan.step_map()["matches"]
    assert filt.where[0].field == "student_name"
    assert filt.where[0].operator == "STARTS_WITH"
    assert filt.where[0].value == "S"


def test_golden_question_executes_without_a_compiler_call():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = asyncio.run(service.answer(_scope(), GOLDEN))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert result.planner_type == "deterministic_fast_path"
    assert result.plan is not None
    assert result.plan.scope.file_ids == (49,)
    assert "FILTER" in result.plan.op_names()
    assert "COUNT" in result.plan.op_names()
    assert result.facts[0]["value"] == 2
    assert "2" in result.answer


def test_private_dataset_does_not_leak():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    result = asyncio.run(service.answer(_scope(), "How many records are in additional deaths?"))
    assert result.status != "access_restricted" or "Private Name" not in result.answer
    assert "Private Name" not in result.answer
    if result.plan is not None:
        assert 93 not in result.plan.scope.file_ids
    catalog = static_catalog().for_scope(_scope())
    assert catalog.dataset(93) is None
    assert catalog.dataset(94) is None


def _exec_service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


def test_community_count():
    result = asyncio.run(_exec_service().answer(_scope(), "How many students are from Garden River?"))
    assert result.status == "answered"
    assert result.facts[0]["value"] == 4


def test_deceased_and_prefix():
    result = asyncio.run(
        _exec_service().answer(_scope(), "How many students whose names start with S are deceased?")
    )
    assert result.status == "answered"
    assert result.facts[0]["value"] == 1


def test_admitted_before_year():
    result = asyncio.run(_exec_service().answer(_scope(), "How many students were admitted before 1911?"))
    assert result.status == "answered"
    assert result.facts[0]["value"] == 1


def test_synthesis_question_is_not_forced_through_count():
    parsed = parse_deterministic(
        "Summarize what the records say about causes of death",
        _scope(),
        static_catalog(),
    )
    assert parsed.plan is None
    assert parsed.status == "needs_reasoning"
    assert parsed.reasoning_calls == 0


def test_mixed_count_and_synthesis_is_not_silently_reduced():
    parsed = parse_deterministic(
        "How many students whose names start with S are deceased, and what do the records say?",
        _scope(),
        static_catalog(),
    )
    assert parsed.plan is None
    assert parsed.reasoning_calls == 0
