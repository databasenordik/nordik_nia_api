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
        for table in FORBIDDEN_TABLES:
            try:
                await conn.fetch(f"SELECT * FROM public.{table} LIMIT 1")
            except asyncpg.InsufficientPrivilegeError:
                continue
            except asyncpg.UndefinedTableError:
                continue
            else:
                failures.append(f"assistant_runtime can SELECT public.{table}")

        for view in FORBIDDEN_VIEWS:
            try:
                await conn.fetch(f"SELECT * FROM {view} LIMIT 1")
            except asyncpg.InsufficientPrivilegeError:
                continue
            except asyncpg.UndefinedTableError:
                continue
            else:
                failures.append(f"assistant_runtime can SELECT {view}")

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
