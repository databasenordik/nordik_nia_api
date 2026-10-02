from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

RESEARCH_TABLES = (
    "file",
    "file_data",
    "file_data_normalized",
    "users",
    "otps",
    "logs",
)


@dataclass
class RetentionPolicy:
    result_set_hours: int = 72
    query_trace_days: int = 14
    audit_days: int = 90
    conversation_days: int = 0


@dataclass
class PurgeReport:
    result_sets: int = 0
    query_runs: int = 0
    audit_events: int = 0
    conversations: int = 0
    messages: int = 0


def should_purge(created_at: datetime, *, max_age: timedelta, now: datetime | None = None) -> bool:
    if max_age.total_seconds() <= 0:
        return False
    current = now or datetime.now(UTC)
    created = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
    return created + max_age <= current


class RetentionService:
    """Deletes assistant-owned operational state only. Never touches research rows."""

    def __init__(self, policy: RetentionPolicy | None = None) -> None:
        self.policy = policy or RetentionPolicy()

    def purge_local(
        self,
        *,
        result_sets: list[dict[str, Any]] | None = None,
        query_runs: list[dict[str, Any]] | None = None,
        audit_events: list[dict[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> PurgeReport:
        current = now or datetime.now(UTC)
        report = PurgeReport()
        if result_sets is not None:
            kept = [
                item
                for item in result_sets
                if not should_purge(
                    _as_dt(item.get("expires_at") or item.get("created_at")),
                    max_age=timedelta(hours=self.policy.result_set_hours),
                    now=current,
                )
            ]
            report.result_sets = len(result_sets) - len(kept)
            result_sets[:] = kept
        if query_runs is not None:
            kept = [
                item
                for item in query_runs
                if not should_purge(
                    _as_dt(item.get("started_at") or item.get("created_at")),
                    max_age=timedelta(days=self.policy.query_trace_days),
                    now=current,
                )
            ]
            report.query_runs = len(query_runs) - len(kept)
            query_runs[:] = kept
        if audit_events is not None:
            kept = [
                item
                for item in audit_events
                if not should_purge(
                    _as_dt(item.get("created_at")),
                    max_age=timedelta(days=self.policy.audit_days),
                    now=current,
                )
            ]
            report.audit_events = len(audit_events) - len(kept)
            audit_events[:] = kept
        return report

    async def purge_postgres(self, pool) -> PurgeReport:
        report = PurgeReport()
        async with pool.acquire() as conn:
            report.result_sets = await conn.fetchval(
                """
                WITH removed AS (
                    DELETE FROM assistant.result_sets
                    WHERE expires_at IS NOT NULL AND expires_at < now()
                       OR created_at < now() - ($1::int * interval '1 hour')
                    RETURNING 1
                )
                SELECT count(*) FROM removed
                """,
                self.policy.result_set_hours,
            )
            report.query_runs = await conn.fetchval(
                """
                WITH removed AS (
                    DELETE FROM assistant.query_runs
                    WHERE started_at < now() - ($1::int * interval '1 day')
                    RETURNING 1
                )
                SELECT count(*) FROM removed
                """,
                self.policy.query_trace_days,
            )
            report.audit_events = await conn.fetchval(
                """
                WITH removed AS (
                    DELETE FROM assistant.audit_events
                    WHERE created_at < now() - ($1::int * interval '1 day')
                    RETURNING 1
                )
                SELECT count(*) FROM removed
                """,
                self.policy.audit_days,
            )
            if self.policy.conversation_days > 0:
                report.conversations = await conn.fetchval(
                    """
                    WITH removed AS (
                        DELETE FROM assistant.conversations
                        WHERE updated_at < now() - ($1::int * interval '1 day')
                        RETURNING 1
                    )
                    SELECT count(*) FROM removed
                    """,
                    self.policy.conversation_days,
                )
        return report

    async def delete_conversation(self, pool, conversation_id: str, principal_id: str) -> bool:
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM assistant.conversations
                WHERE id = $1::uuid AND principal_id = $2
                """,
                conversation_id,
                principal_id,
            )
        return result.endswith("1")


def _as_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(UTC)
