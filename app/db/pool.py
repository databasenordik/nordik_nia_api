
import asyncpg

from app.config import get_settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        from app.db.privilege_selftest import assert_runtime_isolation

        # All application entry points, including voice, share this boundary.
        # Validate once before creating the pool, never during each user turn.
        await assert_runtime_isolation(settings.assistant_database_url)
        _pool = await asyncpg.create_pool(
            dsn=settings.assistant_database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            command_timeout=max(settings.db_query_timeout_ms / 1000, 0.5),
        )
    return _pool


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("database pool is not initialized")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
