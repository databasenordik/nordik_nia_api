"""Real PostgreSQL/runtime-role checks and real-provider historical replay on a snapshot.

Run inside the backend container via database_review_work/load_and_qualify.sh.
Never runs migrations or writes source data in the application's configured database.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from uuid import uuid4

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from app.data_gateway.repository import AssistantDataGateway
from app.db.privilege_selftest import assert_runtime_isolation
from app.execution.derivations import numeric_value, value_part
from app.execution.filters import extract_field_value
from app.execution.turn import AssistantTurnService
from app.llm.factory import get_reasoning_provider
from app.memory.postgres import PostgresMemoryStore
from app.memory.store import InMemoryMemoryStore
from app.memory.types import ConversationRecord
from app.normalization.backfill_names import run as backfill
from app.normalization.date_expression import date_matches
from app.planning.plan_schema import Predicate
from app.security.access_scope import AccessScope

TARGET = "nia_standardized_20261011"


async def main():
    original = get_settings()
    for name, value in (
        ("ASSISTANT_DATABASE_URL", original.assistant_database_url),
        ("ASSISTANT_MIGRATOR_DATABASE_URL", original.assistant_migrator_database_url),
    ):
        os.environ[name] = value.rsplit("/", 1)[0] + "/" + TARGET
    os.environ["PLANNER_MODE"] = "ai"
    get_settings.cache_clear()
    settings = get_settings()
    conn = await asyncpg.connect(settings.assistant_migrator_database_url)
    assert await conn.fetchval("SELECT current_database()") == TARGET
    await backfill((49, 91, 93, 94), dry_run=False)
    # A second run is an idempotence check, rather than a second migration.
    before = await conn.fetchval(
        "SELECT md5(string_agg(row_data_normalized::text,',' ORDER BY id)) FROM assistant_api.v_current_records"
    )
    await backfill((49, 91, 93, 94), dry_run=False)
    after = await conn.fetchval(
        "SELECT md5(string_agg(row_data_normalized::text,',' ORDER BY id)) FROM assistant_api.v_current_records"
    )
    assert before == after
    await assert_runtime_isolation(settings.assistant_database_url)
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91, 93, 94),
        can_use_private_files=True,
    )
    report = {
        "database": TARGET,
        "source": "2026-10-11 dump and revised authorized CSV exports",
        "idempotent": True,
        "privilege_selftest": "passed",
        "counts": {},
        "date_parity_checks": 0,
        "source_column_coverage": {},
        "community_source_checks": 0,
        "exact_interval_checks": 0,
        "historical_replay": [],
        "provider_outage_retries": 0,
    }
    pool = await asyncpg.create_pool(settings.assistant_database_url, min_size=1, max_size=4)
    try:
        gateway = AssistantDataGateway(pool)
        catalog = await gateway.load_catalog(scope)
        for file_id, expected_count in ((49, 2814), (91, 84), (93, 29), (94, 23)):
            count = await gateway.count_records(scope, (file_id,), [])
            assert count == expected_count, (file_id, count)
            report["counts"][str(file_id)] = count
            records = []
            offset = 0
            while True:
                page = await gateway.list_records(scope, (file_id,), [], limit=50, offset=offset)
                if not page:
                    break
                records.extend(page)
                offset += len(page)
            assert len(records) == count
            assert all(
                (row["row_data_normalized"].get("canonical") or {}).get("name_parts")
                for row in records
            )
            source_columns = {key for row in records for key in row["row_data_normalized"]["fields"]}
            registered = {key for spec in catalog.fields_for(file_id) for key in spec.raw_json_keys}
            # ACCESS is a permission, not a person attribute presented to the AI.
            assert not (source_columns - registered - {"ACCESS"}), (file_id, source_columns - registered)
            report["source_column_coverage"][str(file_id)] = len(source_columns - {"ACCESS"})
            community = catalog.resolve_field(file_id, "community")
            for row in records:
                fields = row["row_data_normalized"]["fields"]
                recorded = next((str(fields[key].get("raw") or "").strip()
                    for key in community.raw_json_keys
                    if key in fields and str(fields[key].get("raw") or "").strip()), "")
                assert row["row_data_normalized"]["canonical"].get("community", "") == recorded
                report["community_source_checks"] += 1
            start_field = "admitted_date" if file_id == 49 else "first_admitted_date"
            endpoint_fields = (catalog.resolve_field(file_id, start_field),
                               catalog.resolve_field(file_id, "discharged_date"))
            expected_intervals = sum(all(
                re.fullmatch(r"\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])",
                             str(extract_field_value(row, spec) or ""))
                for spec in endpoint_fields) for row in records)
            intervals = await gateway.interval_stats(scope,(file_id,),[],start_field,"discharged_date")
            assert intervals.record_count == expected_intervals, (file_id,intervals.record_count,expected_intervals)
            report["exact_interval_checks"] += 1
            for spec in catalog.fields_for(file_id):
                if spec.semantic_type != "date":
                    continue
                for operator, value in (
                    ("IS_KNOWN", None),
                    ("IS_UNKNOWN", None),
                    ("YEAR_EQUALS", 1900),
                    ("BEFORE", "1900-07-01"),
                    ("AFTER", "1900-07-01"),
                    ("DATE_RANGE", ["1870-00-00", "1915-00-00"]),
                ):
                    expected = sum(
                        date_matches(extract_field_value(row, spec), operator, value)
                        for row in records
                    )
                    actual = await gateway.count_records(
                        scope,
                        (file_id,),
                        [Predicate(field=spec.semantic_field, operator=operator, value=value)],
                    )
                    assert actual == expected, (
                        file_id,
                        spec.semantic_field,
                        operator,
                        expected,
                        actual,
                    )
                    report["date_parity_checks"] += 1
        print("Runtime counts and date parity passed", flush=True)
        edge_checks = 0
        scalar_checks = 0
        for value in (
            "",
            "0",
            "0000-00-00",
            "0000-09-00",
            "0000-02-29",
            "0000-02-31",
            "1900-00-00",
            "1900-06-00",
            "1900-02-29",
            "1900-00-15",
            "1900-13-01",
            "1900-01-01 | 1901-01-01",
            "1900-01-01 & 1901-01-01",
            "1909-09-04-06",
            "1901-00-00 : 1911-00-00",
            "1911-00-00 : 1901-00-00",
            "1900-01-01 : 1901-12-31 & 1902-01-01",
            "1900-01-01 | 0000-00-00",
            "1900-00-00 : 0000-00-00",
            "1900-00-00 : 1901-13-00",
            "1900-05-01 : 1901-05-01",
            "1900-06-01 : 1900-05-01",
        ):
            try:
                exact_point = date.fromisoformat(value)
            except ValueError:
                exact_point = None
            assert await conn.fetchval("SELECT assistant_api._parse_date_value($1)",value) == exact_point
            assert await conn.fetchval("SELECT assistant_api._value_part($1,'year')",value) == value_part(value,"year")
            number = await conn.fetchval("SELECT assistant_api._numeric_value($1,NULL)",value)
            assert (float(number) if number is not None else None) == numeric_value(value), value
            scalar_checks += 3
            for operator, expected in (
                ("IS_KNOWN", None),
                ("IS_UNKNOWN", None),
                ("YEAR_EQUALS", 1900),
                ("BEFORE", "1900-07-01"),
                ("AFTER", "1900-07-01"),
                ("DATE_RANGE", [1900, 1911]),
                ("EQUALS", "1900-01-01"),
                ("NOT_EQUALS", "1900-01-01"),
                ("IN", [1900, 1901]),
                ("NOT_IN", [1900, 1901]),
            ):
                actual = await conn.fetchval(
                    "SELECT assistant_api._date_matches($1,$2,$3::jsonb)",
                    value,
                    operator,
                    json.dumps(expected),
                )
                assert actual == date_matches(value, operator, expected), (
                    value,
                    operator,
                    expected,
                    actual,
                )
                edge_checks += 1
        report["date_edge_checks"] = edge_checks
        report["date_scalar_checks"] = scalar_checks
        print(f"Date expression edge parity passed: {edge_checks}", flush=True)
        # Access revocation invalidates summaries and persisted assistant text before load.
        store = PostgresMemoryStore(pool)
        conversation = ConversationRecord(
            id=str(uuid4()),
            principal_id=scope.principal_id,
            selected_file_id=94,
            rolling_summary="SYNTHETIC_PRE_REVOCATION_TEXT",
            recent_turns=[{"role": "assistant", "text": "SYNTHETIC_PRE_REVOCATION_TEXT"}],
        )
        await store.save(conversation)
        assert (await store.get(scope, conversation.id)).rolling_summary
        async with conn.transaction():
            await conn.execute(
                "UPDATE assistant.conversation_state SET active_query_state=active_query_state - 'row_access_revision' WHERE conversation_id=$1::uuid",
                conversation.id,
            )
        safe = await store.get(scope, conversation.id)
        assert (
            safe is not None
            and not safe.rolling_summary
            and not any(turn.get("role") == "assistant" for turn in safe.recent_turns)
        )
        assert all(
            not c.rolling_summary
            for c in await store.list_for_principal(scope)
            if c.id == conversation.id
        )
        await store.delete(scope, conversation.id)
        report["memory_release_gate"] = "passed"
        provider = get_reasoning_provider()
        assert provider is not None, "Real provider is required for historical qualification"
        threads = json.loads(Path("/tmp/nia-review/historical_prompts.json").read_text())
        selected_threads = set(os.getenv("HISTORICAL_IDS", "").split(",")) - {""}
        if selected_threads:
            threads = [thread for thread in threads if thread["id"] in selected_threads]
        for thread in threads:
            memory = InMemoryMemoryStore()
            service = AssistantTurnService(gateway, catalog, store=memory, reasoner=provider)
            conversation_id = None
            for question in thread["prompts"]:
                result = await service.answer(
                    scope,
                    question,
                    selected_file_id=thread["selected_file_id"],
                    conversation_id=conversation_id,
                )
                for _retry in range(2):
                    if result.status != "planner_unavailable":
                        break
                    report["provider_outage_retries"] += 1
                    print(
                        "Provider unavailable; retrying the same prompt before continuing its thread",
                        flush=True,
                    )
                    await asyncio.sleep(1)
                    result = await service.answer(
                        scope,
                        question,
                        selected_file_id=thread["selected_file_id"],
                        conversation_id=conversation_id,
                    )
                if result.status == "planner_unavailable":
                    raise RuntimeError(
                        "Provider remained unavailable; this historical run is incomplete"
                    )
                conversation_id = result.conversation_id
                entry = {
                    "thread": thread["id"],
                    "file_id": thread["selected_file_id"],
                    "prompt": question,
                    "status": result.status,
                    "answer": result.answer,
                    "review_status": result.review_status,
                    "reasoning_calls": result.reasoning_calls,
                    "error_code": result.error_code,
                    "detail": result.detail,
                    "plan": result.plan.model_dump(mode="json") if result.plan else None,
                }
                report["historical_replay"].append(entry)
                Path("/tmp/nia-review/qualification.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            k: entry[k]
                            for k in ("thread", "file_id", "prompt", "status", "error_code")
                        }
                    ),
                    flush=True,
                )
        Path("/tmp/nia-review/qualification.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    finally:
        await pool.close()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
