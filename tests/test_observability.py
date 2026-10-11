import asyncio
import sys
from datetime import datetime, timedelta, timezone

import pytest

from repo_paths import BACKEND_ROOT  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.observability.audit import get_auditor, reset_auditor
from app.observability.jobs import CLAIM_SQL, MemoryJobQueue
from app.observability.retention import RESEARCH_TABLES, RetentionPolicy, RetentionService
from app.observability.tracing import get_tracer, reset_tracer
from app.observability.worker import handle_job
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


@pytest.fixture(autouse=True)
def _reset_obs():
    reset_tracer()
    reset_auditor()
    yield
    reset_tracer()
    reset_auditor()


def test_turn_trace_covers_retrieval_and_optional_llm():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    names = [span.name for span in get_tracer().spans]
    assert "turn" in names
    assert "retrieval" in names
    turn = get_tracer().by_name("turn")[0]
    retrieval = get_tracer().by_name("retrieval")[0]
    assert retrieval.parent_id == turn.span_id
    assert "evidence" not in retrieval.attributes
    assert "rows" not in retrieval.attributes


def test_synthesis_turn_adds_llm_span():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    asyncio.run(
        service.answer(
            _scope(),
            "Summarize what the records say about causes of death.",
            selected_file_id=91,
        )
    )
    assert get_tracer().by_name("llm")
    assert get_tracer().by_name("retrieval")


def test_audit_records_query_without_evidence():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))
    events = get_auditor().events
    assert events
    last = events[-1]
    assert last.event_type == "query_completed"
    assert last.principal_id == "principal-researcher"
    assert last.authorization_outcome == "allowed"
    assert 49 in last.dataset_ids
    assert "evidence" not in last.provider_metadata


def test_turn_service_uses_injected_durable_auditor():
    class RecordingAuditor:
        def __init__(self) -> None:
            self.events = []

        async def record(self, **event):
            self.events.append(event)

    auditor = RecordingAuditor()
    service = AssistantTurnService(
        MemoryDataGateway(),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
        auditor=auditor,
    )

    asyncio.run(service.answer(_scope(), "How many students have names starting with S?"))

    assert auditor.events[-1]["event_type"] == "query_completed"
    assert auditor.events[-1]["principal_id"] == "principal-researcher"


def test_access_denied_is_audited():
    service = AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )
    asyncio.run(
        service.answer(
            AccessScope(principal_id="outsider", allowed_file_ids=()),
            "How many students have names starting with S?",
        )
    )
    last = get_auditor().events[-1]
    assert last.authorization_outcome == "denied"
    assert last.event_type in {"access_restricted", "dataset_unavailable", "unplanned", "invalid_plan"}


@pytest.mark.asyncio
async def test_job_leases_are_exclusive():
    queue = MemoryJobQueue()
    first = await queue.enqueue("fts_rebuild", idempotency_key="fts-49", target_dataset_id=49)
    duplicate = await queue.enqueue("fts_rebuild", idempotency_key="fts-49")
    assert first.id == duplicate.id
    claimed = await queue.claim("worker-a")
    other = await queue.claim("worker-b")
    assert claimed is not None
    assert claimed.worker_id == "worker-a"
    assert other is None
    await queue.complete(claimed.id)
    assert queue.jobs[claimed.id].state == "done"


def test_claim_sql_uses_skip_locked():
    assert "FOR UPDATE SKIP LOCKED" in CLAIM_SQL
    assert "lease_expires_at" in CLAIM_SQL


def test_retention_purges_assistant_state_only():
    now = datetime.now(timezone.utc)
    result_sets = [
        {"created_at": now - timedelta(hours=100), "id": "old"},
        {"created_at": now, "id": "new"},
    ]
    query_runs = [{"started_at": now - timedelta(days=30), "id": "old-run"}]
    audit = [{"created_at": now - timedelta(days=200), "id": "old-audit"}]
    report = RetentionService(RetentionPolicy(result_set_hours=72, query_trace_days=14, audit_days=90)).purge_local(
        result_sets=result_sets,
        query_runs=query_runs,
        audit_events=audit,
        now=now,
    )
    assert report.result_sets == 1
    assert result_sets == [{"created_at": now, "id": "new"}]
    assert report.query_runs == 1
    assert report.audit_events == 1
    source = (BACKEND_ROOT / "app" / "observability" / "retention.py").read_text(
        encoding="utf-8"
    )
    assert "DELETE FROM public.file_data" not in source
    assert "users" in RESEARCH_TABLES


@pytest.mark.asyncio
async def test_retention_job_completes():
    queue = MemoryJobQueue()
    job = await queue.enqueue("retention")
    claimed = await queue.claim("worker-1")
    assert claimed is not None
    await handle_job(claimed, queue)
    assert queue.jobs[job.id].state == "done"


def test_production_startup_always_runs_privilege_selftest():
    source = (BACKEND_ROOT / "app" / "main.py").read_text(
        encoding="utf-8"
    )
    assert 'settings.app_env == "production"' in source
    assert "assert_runtime_isolation" in source
    assert "if settings.app_env == \"production\":" in source
    assert "raise" in source


def test_observability_does_not_import_jwt():
    folder = BACKEND_ROOT / "app" / "observability"
    text = "\n".join(path.read_text(encoding="utf-8") for path in folder.glob("*.py"))
    assert "standalone_jwt" not in text
    assert "jwt.encode" not in text
