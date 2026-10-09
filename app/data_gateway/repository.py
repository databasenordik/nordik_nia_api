from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import asyncpg

from app.config import get_settings
from app.data_gateway.protocol import (
    CompletenessRow,
    DuplicateGroup,
    FieldStats,
    GroupRow,
    IntervalRow,
    IntervalStats,
    RecordPage,
)
from app.planning.catalog import FieldCatalog, catalog_from_rows, static_catalog
from app.planning.plan_schema import Predicate, QueryPlan
from app.security.access_scope import AccessScope

if TYPE_CHECKING:
    from app.execution.executor import ExecutionResult


def _filters_payload(predicates: list[Predicate]) -> list[dict[str, Any]]:
    """Serialize the filter tree. A top-level list is read as an implicit AND."""
    return [item.payload() for item in predicates]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _members(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


_JSON_COLUMNS = frozenset(
    {
        "row_data_normalized",
        "selected",
        "validation_rules",
        "source_metadata",
    }
)


def _gateway_row(row: Any) -> dict[str, Any]:
    """Decode JSON columns because asyncpg returns json/jsonb as text by default."""
    result = dict(row)
    for key in _JSON_COLUMNS:
        value = result.get(key)
        if not isinstance(value, str):
            continue
        try:
            result[key] = json.loads(value)
        except (TypeError, ValueError):
            # Fail closed at projection time rather than exposing malformed JSON.
            result[key] = {}
    return result


class AssistantDataGateway:
    """The only path Assistant Core uses to read research data.

    Callers pass an immutable AccessScope. This class never opens a
    general-purpose SQL interface and never grants the LLM table names.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def list_datasets(self, scope: AccessScope) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM assistant_api.list_datasets($1, $2::int[], $3)",
                scope.principal_id,
                list(scope.allowed_file_ids),
                scope.can_use_private_files,
            )
        return [_gateway_row(row) for row in rows]

    async def get_dataset_fields(self, scope: AccessScope, file_id: int) -> list[dict[str, Any]]:
        if not scope.allows_file(file_id):
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM assistant_api.get_dataset_fields($1, $2, $3::int[], $4)",
                scope.principal_id,
                file_id,
                list(scope.allowed_file_ids),
                scope.can_use_private_files,
            )
        return [_gateway_row(row) for row in rows]

    async def load_catalog(self, scope: AccessScope) -> FieldCatalog:
        datasets = await self.list_datasets(scope)
        fields: list[dict[str, Any]] = []
        for dataset in datasets:
            fields.extend(await self.get_dataset_fields(scope, int(dataset["file_id"])))
        if not fields:
            if get_settings().app_env == "production":
                raise RuntimeError("assistant.field_registry is unavailable")
            return static_catalog().for_scope(scope)
        return catalog_from_rows(
            datasets,
            fields,
            default_people_file_id=get_settings().default_people_file_id,
        ).for_scope(scope)

    async def count_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> int:
        async with self._pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT assistant_api.structured_count($1, $2::int[], $3, $4::jsonb)",
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
            )
        return int(value or 0)

    async def count_records_by_file(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> dict[int, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM assistant_api.structured_group_count($1, $2::int[], $3, $4::jsonb)",
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
            )
        return {int(row["file_id"]): int(row["record_count"]) for row in rows}

    async def group_values(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
    ) -> dict[str, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_group_values(
                    $1, $2::int[], $3, $4::jsonb, $5
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                field,
            )
        grouped: dict[str, int] = {}
        for row in rows:
            label = str(row["value"] or "Not recorded")
            grouped[label] = grouped.get(label, 0) + int(row["record_count"])
        return grouped

    async def group_values_multi(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
        *,
        value_part: str | None = None,
        secondary_field: str | None = None,
        secondary_value_part: str | None = None,
        min_count: int = 1,
        top_n: int = 0,
        per_group_top_n: int = 0,
        include_missing: bool = False,
    ) -> list[GroupRow]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_group_values2(
                    $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8, $9, $10, $11, $12
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                field,
                value_part,
                secondary_field,
                secondary_value_part,
                int(min_count),
                int(top_n),
                int(per_group_top_n),
                bool(include_missing),
            )
        return [
            GroupRow(
                file_id=int(row["file_id"]),
                label=row["label_1"],
                secondary_label=row["label_2"],
                count=int(row["record_count"]),
            )
            for row in rows
        ]

    async def field_stats(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
        *,
        value_part: str | None = None,
    ) -> FieldStats:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM assistant_api.structured_stats(
                    $1, $2::int[], $3, $4::jsonb, $5, $6
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                field,
                value_part,
            )
        if row is None:
            return FieldStats(record_count=0, known_count=0)
        return FieldStats(
            record_count=int(row["record_count"] or 0),
            known_count=int(row["known_count"] or 0),
            minimum=_optional_float(row["min_value"]),
            maximum=_optional_float(row["max_value"]),
            average=_optional_float(row["avg_value"]),
            median=_optional_float(row["median_value"]),
            mode=_optional_float(row["mode_value"]),
            mode_count=None if row["mode_count"] is None else int(row["mode_count"]),
        )

    async def duplicate_values(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
        *,
        value_part: str | None = None,
        companion_field: str | None = None,
        min_count: int = 2,
        member_limit: int = 25,
        limit: int = 0,
    ) -> list[DuplicateGroup]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_duplicates(
                    $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8, $9, $10
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                field,
                value_part,
                companion_field,
                int(min_count),
                int(member_limit),
                int(limit),
            )
        return [
            DuplicateGroup(
                file_id=int(row["file_id"]),
                value=str(row["value"]),
                count=int(row["record_count"]),
                distinct_companions=int(row["distinct_companions"] or 0),
                members=_members(row["members"]),
            )
            for row in rows
        ]

    async def record_completeness(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        direction: str = "desc",
        limit: int = 10,
    ) -> list[CompletenessRow]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_completeness(
                    $1, $2::int[], $3, $4::jsonb, $5, $6
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                direction,
                int(limit),
            )
        return [
            CompletenessRow(
                file_id=int(row["file_id"]),
                source_row_id=int(row["source_row_id"]),
                display_name=row["display_name"],
                filled_fields=int(row["filled_fields"] or 0),
                total_fields=int(row["total_fields"] or 0),
                filled_field_labels=list(row["filled_field_labels"] or []),
            )
            for row in rows
        ]

    async def interval_rows(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        start_field: str,
        end_field: str,
        *,
        min_days: float | None = None,
        max_days: float | None = None,
        direction: str = "desc",
        limit: int = 10,
    ) -> list[IntervalRow]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_interval(
                    $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8, $9, $10
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                start_field,
                end_field,
                None if min_days is None else float(min_days),
                None if max_days is None else float(max_days),
                direction,
                int(limit),
            )
        return [
            IntervalRow(
                file_id=int(row["file_id"]),
                source_row_id=int(row["source_row_id"]),
                display_name=row["display_name"],
                start_value=row["start_value"],
                end_value=row["end_value"],
                days=float(row["interval_days"]),
                start_precision=row["start_precision"],
                end_precision=row["end_precision"],
                year_delta=None if row["year_delta"] is None else int(row["year_delta"]),
            )
            for row in rows
        ]

    async def interval_stats(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        start_field: str,
        end_field: str,
        *,
        min_days: float | None = None,
        max_days: float | None = None,
    ) -> IntervalStats:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM assistant_api.structured_interval_stats(
                    $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                start_field,
                end_field,
                None if min_days is None else float(min_days),
                None if max_days is None else float(max_days),
            )
        if row is None:
            return IntervalStats(record_count=0)
        return IntervalStats(
            record_count=int(row["record_count"] or 0),
            minimum_days=_optional_float(row["min_days"]),
            maximum_days=_optional_float(row["max_days"]),
            average_days=_optional_float(row["avg_days"]),
            median_days=_optional_float(row["median_days"]),
            negative_count=int(row["negative_count"] or 0),
            impossible_count=int(row["impossible_count"] or 0),
        )

    async def list_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 25,
        offset: int = 0,
        sort_field: str | None = None,
        sort_direction: str = "ASC",
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_list(
                    $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                sort_field,
                sort_direction,
                limit,
                offset,
            )
        return [_gateway_row(row) for row in rows]

    async def suggest_name_matches(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        *,
        max_distance: int = 2,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """People whose recorded names are within max_distance edits of `query`."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.suggest_name_matches(
                    $1, $2::int[], $3, $4, $5, $6
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                query,
                max_distance,
                limit,
            )
        return [dict(row) for row in rows]

    async def list_records_window(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        anchor: Literal["start", "end"],
        window_size: int,
        cursor: int = 0,
        page_size: int = 25,
        sort_field: str | None = None,
        sort_direction: str = "ASC",
    ) -> RecordPage:
        """Resolve a relative window and fetch its page from one database snapshot."""
        filters = json.dumps(_filters_payload(predicates))
        async with self._pool.acquire() as conn:
            async with conn.transaction(isolation="repeatable_read", readonly=True):
                value = await conn.fetchval(
                    "SELECT assistant_api.structured_count($1, $2::int[], $3, $4::jsonb)",
                    scope.principal_id,
                    list(file_ids),
                    scope.can_use_private_files,
                    filters,
                )
                total = int(value or 0)
                selected = min(max(int(window_size), 0), total)
                safe_cursor = min(max(int(cursor), 0), selected)
                base = total - selected if anchor == "end" else 0
                offset = base + safe_cursor
                effective = min(max(int(page_size), 0), selected - safe_cursor)
                rows = []
                if effective:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM assistant_api.structured_list(
                            $1, $2::int[], $3, $4::jsonb, $5, $6, $7, $8
                        )
                        """,
                        scope.principal_id,
                        list(file_ids),
                        scope.can_use_private_files,
                        filters,
                        sort_field,
                        sort_direction,
                        effective,
                        offset,
                    )
        public_rows = [_gateway_row(row) for row in rows]
        return RecordPage(
            rows=public_rows,
            total_count=total,
            offset=offset,
            page_size=page_size,
            has_more=safe_cursor + len(public_rows) < selected,
            window_anchor=anchor,
            window_size=window_size,
            window_count=selected,
            window_cursor=safe_cursor,
        )

    async def sample_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_sample(
                    $1, $2::int[], $3, $4::jsonb, $5
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                limit,
            )
        return [_gateway_row(row) for row in rows]

    async def sample_records_page(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 5,
    ) -> RecordPage:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.structured_sample_page(
                    $1, $2::int[], $3, $4::jsonb, $5
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                json.dumps(_filters_payload(predicates)),
                limit,
            )
        total = int(rows[0]["total_count"]) if rows else 0
        public_rows: list[dict[str, Any]] = []
        for row in rows:
            public_row = _gateway_row(row)
            public_row.pop("total_count", None)
            public_rows.append(public_row)
        return RecordPage(
            rows=public_rows,
            total_count=total,
            offset=0,
            page_size=limit,
            has_more=total > len(public_rows),
        )

    async def retrieve_candidates(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        method: str,
        limit: int = 20,
        *,
        predicates: list[Predicate] | None = None,
    ) -> list[dict[str, Any]]:
        if predicates:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM assistant_api.retrieve_candidates($1, $2::int[], $3, $4, $5, $6, $7::jsonb)",
                    scope.principal_id, list(file_ids), scope.can_use_private_files,
                    query, method, limit, json.dumps(_filters_payload(predicates)),
                )
            return [_gateway_row(row) for row in rows]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.retrieve_candidates(
                    $1, $2::int[], $3, $4, $5, $6
                )
                """,
                scope.principal_id,
                list(file_ids),
                scope.can_use_private_files,
                query,
                method,
                limit,
            )
        return [_gateway_row(row) for row in rows]

    async def get_raw_fields(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
        keys: list[str],
    ) -> list[dict[str, Any]]:
        if not source_row_ids or not keys:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.get_raw_fields_by_ids(
                    $1, $2::int[], $3, $4::bigint[], $5::text[]
                )
                """,
                scope.principal_id,
                list(scope.allowed_file_ids),
                scope.can_use_private_files,
                [int(item) for item in source_row_ids],
                keys,
            )
        return [_gateway_row(row) for row in rows]

    async def get_records_by_source_ids(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
    ) -> list[dict[str, Any]]:
        if not source_row_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM assistant_api.get_current_records_by_ids(
                    $1, $2::int[], $3, $4::bigint[]
                )
                """,
                scope.principal_id,
                list(scope.allowed_file_ids),
                scope.can_use_private_files,
                [int(item) for item in source_row_ids],
            )
        return [_gateway_row(row) for row in rows]

    async def save_query_run(
        self,
        *,
        turn_id: str,
        plan: QueryPlan | None,
        execution: ExecutionResult | None,
        status: str,
    ) -> None:
        """Persist DAG timing so traces survive in `assistant.query_runs` / `query_step_runs`."""
        if execution is None or execution.run_trace is None:
            return
        trace = execution.run_trace
        anchor_perf = time.perf_counter()
        anchor_wall = datetime.now(UTC)

        def to_wall(value: float | None) -> datetime | None:
            if value is None:
                return None
            return anchor_wall - timedelta(seconds=anchor_perf - value)

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                plan_id = None
                if plan is not None:
                    plan_json = plan.model_dump(mode="json")
                    plan_hash = hashlib.sha256(
                        json.dumps(plan_json, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    plan_id = await conn.fetchval(
                        """
                        INSERT INTO assistant.query_plans
                            (turn_id, planner_type, plan_json, schema_version, validation_status, plan_hash)
                        VALUES ($1, $2, $3::jsonb, $4, $5, $6)
                        RETURNING id
                        """,
                        turn_id,
                        plan.planner_type,
                        json.dumps(plan_json),
                        plan.schema_version,
                        status,
                        plan_hash,
                    )
                query_run_id = await conn.fetchval(
                    """
                    INSERT INTO assistant.query_runs
                        (plan_id, status, access_scope_fingerprint, started_at, ended_at,
                         latency_ms, critical_path_ms, parallel_step_peak, queued_ms,
                         cancel_reason, result_summary)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    RETURNING id
                    """,
                    plan_id,
                    status,
                    trace.access_scope_fingerprint,
                    to_wall(trace.started_at),
                    to_wall(trace.ended_at),
                    int(trace.latency_ms) if trace.ended_at is not None else None,
                    int(trace.critical_path_ms),
                    trace.parallel_step_peak,
                    int(trace.queued_ms),
                    trace.cancel_reason,
                    json.dumps({"methods_run": execution.methods_run, "row_count": len(execution.rows)}),
                )
                for step in trace.steps.values():
                    await conn.execute(
                        """
                        INSERT INTO assistant.query_step_runs
                            (query_run_id, step_id, operation, branch_group, ready_at, queued_at,
                             started_at, completed_at, worker_id, idempotency_key, timeout_ms,
                             cancelled, result_count, error_metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
                        """,
                        query_run_id,
                        step.step_id,
                        step.operation,
                        step.branch_group,
                        to_wall(step.ready_at),
                        to_wall(step.queued_at),
                        to_wall(step.started_at),
                        to_wall(step.completed_at),
                        step.worker_id,
                        step.idempotency_key,
                        step.timeout_ms,
                        step.cancelled,
                        step.result_count,
                        json.dumps({"error": step.error} if step.error else {}),
                    )
