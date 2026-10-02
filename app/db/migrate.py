from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
STUB_FILES = {
    "003_app_stubs.sql",
    "007_seed.sql",
    # Grants the demo principal the private 93/94 lists; never run in production.
    "038_additional_potential_demo.sql",
    # Same, for the benchmark principal the scoped suite runs under.
    "042_regression_suite_grants.sql",
}


async def apply_migrations() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(settings.assistant_migrator_database_url)
    try:
        await conn.execute(
            "SELECT set_config('nia.assistant_runtime_password', $1, false)",
            settings.assistant_runtime_password,
        )
        await conn.execute(
            "SELECT set_config('nia.assistant_migrator_password', $1, false)",
            settings.assistant_migrator_password,
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public.assistant_schema_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        applied = {
            row["filename"]
            for row in await conn.fetch("SELECT filename FROM public.assistant_schema_migrations")
        }
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            if path.name in STUB_FILES and not settings.apply_app_stubs:
                print(f"skip {path.name} (APPLY_APP_STUBS=false)")
                continue
            sql = path.read_text(encoding="utf-8")
            print(f"apply {path.name}")
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO public.assistant_schema_migrations (filename) VALUES ($1)",
                path.name,
            )
        print("migrations complete")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(apply_migrations())


if __name__ == "__main__":
    main()
