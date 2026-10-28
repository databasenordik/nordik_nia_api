from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.config import get_settings


@dataclass
class ResourceLimits:
    global_concurrency: int = 8
    per_class: dict[str, int] = field(
        default_factory=lambda: {"db": 8, "retrieval": 6, "local": 16, "model": 2}
    )

    @classmethod
    def from_settings(cls) -> ResourceLimits:
        settings = get_settings()
        return cls(
            global_concurrency=getattr(settings, "query_dag_max_concurrency", 8),
            per_class={
                "db": settings.db_query_concurrency,
                "retrieval": getattr(settings, "retrieval_max_concurrency", 6),
                "local": 16,
                "model": getattr(settings, "model_max_concurrency_per_turn", 2),
            },
        )


class ResourceGate:
    def __init__(self, limits: ResourceLimits | None = None) -> None:
        self.limits = limits or ResourceLimits.from_settings()
        self._global = asyncio.Semaphore(self.limits.global_concurrency)
        self._classes = {
            name: asyncio.Semaphore(max(1, size)) for name, size in self.limits.per_class.items()
        }

    def semaphore(self, resource_class: str) -> asyncio.Semaphore:
        return self._classes.get(resource_class, self._global)

    async def acquire(self, resource_class: str) -> None:
        await self._global.acquire()
        try:
            await self.semaphore(resource_class).acquire()
        except Exception:
            self._global.release()
            raise

    def release(self, resource_class: str) -> None:
        self.semaphore(resource_class).release()
        self._global.release()
