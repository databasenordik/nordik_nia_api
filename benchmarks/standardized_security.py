"""Exercise real gateway denial and revocation with synthetic rows in an isolated DB."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.data_gateway.repository import AssistantDataGateway
from app.memory.postgres import PostgresMemoryStore
from app.memory.types import ConversationRecord
from app.security.access_scope import AccessScope

TARGET = "nia_standardized_20261011"
SENTINEL = "SYNTHETIC_RESTRICTED_SENTINEL"
IDS = list(range(9900001, 9900010))


async def main():
    settings = get_settings()
    admin = await asyncpg.connect(
        settings.assistant_migrator_database_url.rsplit("/", 1)[0] + "/" + TARGET
    )
    pool = await asyncpg.create_pool(
        settings.assistant_database_url.rsplit("/", 1)[0] + "/" + TARGET
    )
    assert await admin.fetchval("SELECT current_database()") == TARGET
    scope = AccessScope(
        principal_id="principal-researcher", allowed_file_ids=(94,), can_use_private_files=True
    )
    gateway = AssistantDataGateway(pool)
    store = PostgresMemoryStore(pool)
    conversation_id = str(uuid4())
    checks = 0
    try:
        version = await admin.fetchval("SELECT version FROM public.file WHERE id=94")
        payload = json.dumps(
            {"fields": {"ACCESS": {"raw": "TRUE"}}, "canonical": {"display_name": SENTINEL}}
        )
        for row_id, flag in zip(IDS[:-1], [False, "FALSE", "yes", "1", 1, None, "", {"raw": True}], strict=True):
            await admin.execute(
                "INSERT INTO public.file_data(id,file_id,version,row_data) VALUES($1,94,$2,$3::jsonb)",
                row_id,
                version,
                json.dumps({"ACCESS": flag, "First Name": SENTINEL}),
            )
            await admin.execute(
                "INSERT INTO public.file_data_normalized(source_row_id,file_id,version,row_data_normalized,canonical_name,search_text) VALUES($1,94,$2,$3::jsonb,$4,$4)",
                row_id,
                version,
                payload,
                SENTINEL,
            )
        # A normalized orphan has a TRUE flag but no authoritative source permission.
        await admin.execute(
            "INSERT INTO public.file_data_normalized(source_row_id,file_id,version,row_data_normalized,canonical_name,search_text) VALUES($1,94,$2,$3::jsonb,$4,$4)",
            IDS[-1],
            version,
            payload,
            SENTINEL,
        )
        assert await gateway.count_records(scope, (94,), []) == 23
        assert await gateway.get_records_by_source_ids(scope, IDS) == []
        assert await gateway.get_raw_fields(scope, IDS, ["First Name"]) == []
        assert await gateway.suggest_name_matches(scope, (94,), SENTINEL) == []
        for method in ("fts", "trgm", "exact", "fuzzy"):
            assert await gateway.retrieve_candidates(scope, (94,), SENTINEL, method) == []
            checks += 1
        for rows in (
            await gateway.list_records(scope, (94,), [], limit=50),
            await gateway.sample_records(scope, (94,), [], limit=50),
        ):
            assert all(int(row["source_row_id"]) not in IDS for row in rows)
        assert (await gateway.field_stats(scope, (94,), [], "age_at_death")).record_count == 23
        assert SENTINEL not in await gateway.group_values(scope, (94,), [], "student_name")
        checks += 8
        # A mistakenly edited policy must never redirect the Potential gate to another cell.
        policy = await admin.fetchrow(
            "SELECT row_access_key FROM assistant.dataset_policy WHERE file_id=94"
        )
        try:
            await admin.execute(
                "UPDATE assistant.dataset_policy SET row_access_key='First Name' WHERE file_id=94"
            )
            assert await gateway.count_records(scope, (94,), []) == 23
        finally:
            await admin.execute(
                "UPDATE assistant.dataset_policy SET row_access_key=$1 WHERE file_id=94",
                policy["row_access_key"],
            )
        checks += 1
        # Release one synthetic row, persist context, revoke it, and read via the runtime role.
        await admin.execute(
            "UPDATE public.file_data SET row_data=jsonb_set(row_data,'{ACCESS}','true') WHERE id=$1",
            IDS[0],
        )
        assert await gateway.count_records(scope, (94,), []) == 24
        conversation = ConversationRecord(
            id=conversation_id,
            principal_id=scope.principal_id,
            selected_file_id=94,
            rolling_summary=SENTINEL,
            recent_turns=[
                {"role": "assistant", "text": SENTINEL},
                {"role": "user", "text": "My research question"},
            ],
        )
        await store.save(conversation)
        assert (await store.get(scope, conversation_id)).rolling_summary == SENTINEL
        await admin.execute(
            "UPDATE public.file_data SET row_data=jsonb_set(row_data,'{ACCESS}','false') WHERE id=$1",
            IDS[0],
        )
        safe = await store.get(scope, conversation_id)
        assert not safe.rolling_summary
        assert not any(turn.get("role") == "assistant" for turn in safe.recent_turns)
        assert all(
            not item.rolling_summary
            for item in await store.list_for_principal(scope)
            if item.id == conversation_id
        )
        assert await gateway.get_records_by_source_ids(scope, [IDS[0]]) == []
        checks += 6
        # Deliberately poison a data-derived catalog cache after revocation. The runtime
        # must remove it in SQL, before Python receives field metadata.
        try:
            await admin.execute(
                "UPDATE assistant.field_registry SET validation_rules=validation_rules || $1::jsonb WHERE file_id=94",
                json.dumps(
                    {"recorded_values": [SENTINEL], "record_count": 24, "recorded_count": 24}
                ),
            )
            fields = await gateway.get_dataset_fields(scope, 94)
            assert all(
                not any(
                    key in row["validation_rules"]
                    for key in ("recorded_values", "record_count", "recorded_count")
                )
                for row in fields
            )
            assert SENTINEL not in json.dumps(fields)
        finally:
            # Remove the fixture before leaving the isolated database.
            await admin.execute(
                "UPDATE assistant.field_registry SET validation_rules=validation_rules - ARRAY['recorded_values','record_count','recorded_count'] WHERE file_id=94"
            )
        checks += 2
        result = {
            "database": TARGET,
            "security_checks": checks,
            "denied_synthetic_rows": len(IDS),
            "result": "passed",
        }
        Path("/tmp/nia-review/security.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        await store.delete(scope, conversation_id)
        await admin.execute(
            "DELETE FROM public.file_data_normalized WHERE source_row_id=ANY($1::bigint[])", IDS
        )
        await admin.execute("DELETE FROM public.file_data WHERE id=ANY($1::bigint[])", IDS)
        await pool.close()
        await admin.close()


if __name__ == "__main__":
    asyncio.run(main())
