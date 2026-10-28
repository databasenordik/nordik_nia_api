from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

CLAIM_SQL = """
UPDATE assistant.background_jobs AS claimed
SET
    state = 'running',
    worker_id = $1,
    attempt_count = claimed.attempt_count + 1,
    started_at = COALESCE(claimed.started_at, now()),
    heartbeat_at = now(),
    lease_expires_at = now() + ($2::int * interval '1 second')
WHERE claimed.id = (
    SELECT id
    FROM assistant.background_jobs
    WHERE state IN ('queued', 'retry')
      AND (next_retry_at IS NULL OR next_retry_at <= now())
      AND (lease_expires_at IS NULL OR lease_expires_at <= now())
    ORDER BY priority ASC, created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING claimed.*
"""


@dataclass
class Job:
    job_type: str
    id: str = field(default_factory=lambda: str(uuid4()))
    target_dataset_id: int | None = None
    resource_class: str = "local"
    priority: int = 100
    state: str = "queued"
    attempt_count: int = 0
    idempotency_key: str | None = None
    worker_id: str | None = None
    lease_expires_at: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error_summary: str | None = None


class MemoryJobQueue:
    """In-process lease queue with the same claim semantics as SKIP LOCKED."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        job_type: str,
        *,
        target_dataset_id: int | None = None,
        idempotency_key: str | None = None,
        priority: int = 100,
        payload: dict[str, Any] | None = None,
    ) -> Job:
        async with self._lock:
            if idempotency_key:
                for existing in self.jobs.values():
                    if existing.idempotency_key == idempotency_key and existing.state != "done":
                        return existing
            job = Job(
                job_type=job_type,
                target_dataset_id=target_dataset_id,
                idempotency_key=idempotency_key,
                priority=priority,
                payload=payload or {},
            )
            self.jobs[job.id] = job
            return job

    async def claim(self, worker_id: str, *, lease_seconds: int = 30) -> Job | None:
        now = datetime.now(UTC)
        async with self._lock:
            candidates = [
                job
                for job in self.jobs.values()
                if job.state in {"queued", "retry"}
                and (job.lease_expires_at is None or job.lease_expires_at <= now)
            ]
            candidates.sort(key=lambda item: (item.priority, item.id))
            if not candidates:
                return None
            job = candidates[0]
            job.state = "running"
            job.worker_id = worker_id
            job.attempt_count += 1
            job.lease_expires_at = now + timedelta(seconds=lease_seconds)
            return job

    async def complete(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.state = "done"
        job.lease_expires_at = None

    async def fail(self, job_id: str, error: str, *, retry: bool = True) -> None:
        job = self.jobs[job_id]
        job.error_summary = error
        job.state = "retry" if retry else "failed"
        job.lease_expires_at = None


class PostgresJobQueue:
    def __init__(self, pool) -> None:
        self._pool = pool

    async def enqueue(
        self,
        job_type: str,
        *,
        target_dataset_id: int | None = None,
        idempotency_key: str | None = None,
        priority: int = 100,
        payload: dict[str, Any] | None = None,
    ) -> Job:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO assistant.background_jobs
                    (job_type, target_dataset_id, resource_class, priority, state, idempotency_key)
                VALUES ($1, $2, 'local', $3, 'queued', $4)
                ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
                DO UPDATE SET job_type = EXCLUDED.job_type
                RETURNING id::text, job_type, target_dataset_id, priority, state,
                          attempt_count, idempotency_key, worker_id
                """,
                job_type,
                target_dataset_id,
                priority,
                idempotency_key,
            )
        return Job(
            id=row["id"],
            job_type=row["job_type"],
            target_dataset_id=row["target_dataset_id"],
            priority=row["priority"],
            state=row["state"],
            attempt_count=row["attempt_count"],
            idempotency_key=row["idempotency_key"],
            worker_id=row["worker_id"],
            payload=payload or {},
        )

    async def claim(self, worker_id: str, *, lease_seconds: int = 30) -> Job | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(CLAIM_SQL, worker_id, lease_seconds)
        if row is None:
            return None
        return Job(
            id=str(row["id"]),
            job_type=row["job_type"],
            target_dataset_id=row["target_dataset_id"],
            priority=row["priority"],
            state=row["state"],
            attempt_count=row["attempt_count"],
            idempotency_key=row["idempotency_key"],
            worker_id=row["worker_id"],
            lease_expires_at=row["lease_expires_at"],
        )

    async def complete(self, job_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE assistant.background_jobs
                SET state = 'done', completed_at = now(), lease_expires_at = NULL
                WHERE id = $1::uuid
                """,
                job_id,
            )

    async def fail(self, job_id: str, error: str, *, retry: bool = True) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE assistant.background_jobs
                SET state = $2, error_summary = $3, lease_expires_at = NULL,
                    next_retry_at = CASE WHEN $2 = 'retry' THEN now() + interval '30 seconds' END
                WHERE id = $1::uuid
                """,
                job_id,
                "retry" if retry else "failed",
                error,
            )
