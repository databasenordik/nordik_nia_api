from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from app.observability.jobs import MemoryJobQueue
from app.observability.retention import RetentionPolicy, RetentionService

logger = logging.getLogger("nia.jobs")


async def handle_job(job, queue) -> None:
    if job.job_type == "retention":
        settings = get_settings()
        service = RetentionService(
            RetentionPolicy(
                result_set_hours=settings.result_set_retention_hours,
                query_trace_days=settings.query_trace_retention_days,
                audit_days=settings.audit_retention_days,
                conversation_days=settings.conversation_retention_days,
            )
        )
        try:
            from app.db.pool import get_pool

            report = await service.purge_postgres(get_pool())
            logger.info("retention purged %s", report)
        except Exception:
            logger.info("retention skipped; no Postgres pool")
        await queue.complete(job.id)
        return
    if job.job_type in {"fts_rebuild", "alias_rebuild", "field_registry_validation"}:
        await queue.complete(job.id)
        return
    await queue.fail(job.id, f"unknown job type {job.job_type}", retry=False)


async def run_once(queue, worker_id: str = "nia-job-1") -> bool:
    job = await queue.claim(worker_id)
    if job is None:
        return False
    try:
        await handle_job(job, queue)
    except Exception as exc:
        await queue.fail(job.id, str(exc), retry=True)
    return True


def main() -> None:
    asyncio.run(_main())


async def _main() -> None:
    from app.db.pool import close_pool, init_pool
    from app.observability.jobs import PostgresJobQueue

    try:
        pool = await init_pool()
        queue = PostgresJobQueue(pool)
    except Exception:
        queue = MemoryJobQueue()
    claimed = await run_once(queue)
    print("claimed" if claimed else "idle")
    await close_pool()


if __name__ == "__main__":
    main()
