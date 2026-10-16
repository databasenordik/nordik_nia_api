from __future__ import annotations

import asyncio

import asyncpg

from app.config import get_settings

FORBIDDEN_TABLES = (
    "users",
    "roles",
    "otps",
    "logs",
    "support_requests",
    "file_access",
    "file_edit_request",
    "form_submissions",
    "file",
    "file_data",
    "file_data_normalized",
    "data_config",
)

FORBIDDEN_VIEWS = (
    "assistant_api.v_current_records",
    "assistant_api.v_current_raw_records",
)


async def assert_runtime_isolation(dsn: str | None = None) -> None:
    settings = get_settings()
    conn = await asyncpg.connect(dsn or settings.assistant_database_url)
    failures: list[str] = []
    try:
        # Inspect permissions without attempting to read even one restricted row.
        # A misconfigured role must be detected before it can load content.
        objects = [*(f"public.{table}" for table in FORBIDDEN_TABLES), *FORBIDDEN_VIEWS]
        rows = await conn.fetch(
            "SELECT name FROM unnest($1::text[]) name "
            "WHERE has_table_privilege(current_user, to_regclass(name), 'SELECT')",
            objects,
        )
        failures.extend(f"assistant_runtime can SELECT {row['name']}" for row in rows)
        gate = await conn.fetchval(
            "SELECT to_regprocedure('assistant_api.potential_access_revision()') IS NOT NULL "
            "AND (SELECT count(*) = 2 FROM pg_class WHERE oid = ANY(ARRAY["
            "to_regclass('assistant_api.v_current_records'), "
            "to_regclass('assistant_api.v_current_raw_records')]) "
            "AND 'security_barrier=true' = ANY(reloptions))"
        )
        if not gate:
            failures.append("Potential release gate is not installed; apply migrations through 073")
        if failures:
            raise PermissionError("privilege self-test failed: " + "; ".join(failures))

        try:
            rows = await conn.fetch(
                "SELECT * FROM assistant_api.list_datasets($1, $2::int[], $3)",
                "principal-researcher",
                [49, 91, 93, 94],
                False,
            )
        except Exception as exc:
            failures.append(f"list_datasets failed for runtime: {exc}")
        else:
            private_ids = {row["file_id"] for row in rows if row.get("private")}
            if private_ids:
                failures.append(f"private datasets leaked without grant: {sorted(private_ids)}")

        if failures:
            raise PermissionError("privilege self-test failed: " + "; ".join(failures))
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(assert_runtime_isolation())
    print("privilege self-test passed")


if __name__ == "__main__":
    main()
