from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass
class AuditEvent:
    event_type: str
    principal_id: str
    authorization_outcome: str
    conversation_id: str | None = None
    query_run_id: str | None = None
    dataset_ids: tuple[int, ...] = ()
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(default_factory=lambda: uuid4().hex)


class MemoryAuditor:
    """Process-local audit log. Never stores evidence rows or auth secrets."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(
        self,
        *,
        event_type: str,
        principal_id: str,
        authorization_outcome: str,
        conversation_id: str | None = None,
        query_run_id: str | None = None,
        dataset_ids: tuple[int, ...] | list[int] = (),
        provider_metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_type=event_type,
            principal_id=principal_id,
            authorization_outcome=authorization_outcome,
            conversation_id=conversation_id,
            query_run_id=query_run_id,
            dataset_ids=tuple(dataset_ids),
            provider_metadata=_safe_metadata(provider_metadata or {}),
        )
        self.events.append(event)
        return event


class PostgresAuditor:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def record(
        self,
        *,
        event_type: str,
        principal_id: str,
        authorization_outcome: str,
        conversation_id: str | None = None,
        query_run_id: str | None = None,
        dataset_ids: tuple[int, ...] | list[int] = (),
        provider_metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        import json

        event = AuditEvent(
            event_type=event_type,
            principal_id=principal_id,
            authorization_outcome=authorization_outcome,
            conversation_id=conversation_id,
            query_run_id=query_run_id,
            dataset_ids=tuple(dataset_ids),
            provider_metadata=_safe_metadata(provider_metadata or {}),
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO assistant.audit_events
                    (principal_id, event_type, conversation_id, query_run_id,
                     dataset_ids, authorization_outcome, provider_metadata)
                VALUES ($1, $2, $3::uuid, $4::uuid, $5::int[], $6, $7::jsonb)
                """,
                event.principal_id,
                event.event_type,
                event.conversation_id,
                event.query_run_id,
                list(event.dataset_ids),
                event.authorization_outcome,
                json.dumps(event.provider_metadata),
            )
        return event


_AUDITOR = MemoryAuditor()


def get_auditor() -> MemoryAuditor:
    return _AUDITOR


def reset_auditor() -> None:
    _AUDITOR.events.clear()


def default_auditor():
    try:
        from app.db.pool import get_pool

        return PostgresAuditor(get_pool())
    except Exception:
        return _AUDITOR


def _safe_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    blocked = {"evidence", "rows", "sql", "password", "otp", "token", "raw_fields"}
    return {key: value for key, value in payload.items() if key.lower() not in blocked}
