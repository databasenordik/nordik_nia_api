from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any, Literal

from app.data_gateway.protocol import (
    CompletenessRow,
    DuplicateGroup,
    FieldStats,
    GroupRow,
    IntervalRow,
    IntervalStats,
    RecordPage,
)
from app.execution.derivations import date_precision, parse_date
from app.execution.derivations import numeric_value as derive_number
from app.execution.derivations import value_part as derive_part
from app.execution.filters import apply_predicates, extract_field_value
from app.planning.catalog import FieldCatalog, static_catalog
from app.planning.plan_schema import Predicate, QueryPlan
from app.security.access_scope import AccessScope
from app.value_normalization import semantic_scalar

if TYPE_CHECKING:
    from app.execution.executor import ExecutionResult


def seed_records() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "source_row_id": 1101,
            "file_id": 49,
            "version": 3,
            "canonical_name": "Samuel A",
            "canonical_community": "Garden River",
            "canonical_school": "Shingwauk",
            "search_text": "Samuel A Garden River admitted 1910",
            "raw": {"Name": "Samuel A", "Notes": "admitted 1910", "Community": "Garden River"},
            "row_data_normalized": {
                "canonical": {
                    "name": "Samuel A",
                    "community": "Garden River",
                    "dates": {"admitted": "1910"},
                    "deceased": False,
                }
            },
        },
        {
            "id": 2,
            "source_row_id": 1102,
            "file_id": 49,
            "version": 3,
            "canonical_name": "Sarah B",
            "canonical_community": "Garden River",
            "canonical_school": "Shingwauk",
            "search_text": "Sarah B Garden River tuberculosis deceased",
            "raw": {
                "Name": "Sarah B",
                "Notes": "tuberculosis mentioned",
                "Cause of Death": "tuberculosis",
            },
            "row_data_normalized": {
                "canonical": {
                    "name": "Sarah B",
                    "community": "Garden River",
                    "deceased": True,
                    "cause_of_death": "tuberculosis",
                    "notes": "tuberculosis mentioned",
                }
            },
        },
        {
            "id": 3,
            "source_row_id": 1103,
            "file_id": 49,
            "version": 3,
            "canonical_name": "Thomas C",
            "canonical_community": "Sault Ste. Marie",
            "canonical_school": "Shingwauk",
            "search_text": "Thomas C Sault Ste. Marie discharged 1912",
            "raw": {"Name": "Thomas C", "Notes": "discharged 1912"},
            "row_data_normalized": {
                "canonical": {
                    "name": "Thomas C",
                    "community": "Sault Ste. Marie",
                    "dates": {"discharged": "1912"},
                    "deceased": False,
                }
            },
        },
        {
            "id": 4,
            "source_row_id": 1104,
            "file_id": 49,
            "version": 3,
            "canonical_name": "Mary D",
            "canonical_community": "Garden River",
            "canonical_school": "Shingwauk",
            "search_text": "Mary D Garden River influenza deceased",
            "raw": {"Name": "Mary D", "Cause of Death": "influenza"},
            "row_data_normalized": {
                "canonical": {
                    "name": "Mary D",
                    "community": "Garden River",
                    "deceased": True,
                    "cause_of_death": "influenza",
                }
            },
        },
        {
            "id": 5,
            "source_row_id": 1105,
            "file_id": 49,
            "version": 3,
            "canonical_name": "Joseph E",
            "canonical_community": "Garden River",
            "canonical_school": "Shingwauk",
            "search_text": "Joseph E Garden River admitted 1911",
            "raw": {"Name": "Joseph E", "Notes": "admitted 1911"},
            "row_data_normalized": {
                "canonical": {
                    "name": "Joseph E",
                    "community": "Garden River",
                    "dates": {"admitted": "1911"},
                    "deceased": False,
                }
            },
        },
        {
            "id": 6,
            "source_row_id": 1106,
            "file_id": 91,
            "version": 1,
            "canonical_name": "Sarah B",
            "canonical_community": None,
            "canonical_school": "Shingwauk",
            "search_text": "Sarah B tuberculosis confirmed",
            "raw": {"Name": "Sarah B", "Cause of Death": "tuberculosis"},
            "row_data_normalized": {"canonical": {"name": "Sarah B", "cause_of_death": "tuberculosis"}},
        },
        {
            "id": 7,
            "source_row_id": 1108,
            "file_id": 93,
            "version": 1,
            "canonical_name": "Private Name",
            "canonical_community": None,
            "canonical_school": None,
            "search_text": "Private Name restricted tuberculosis",
            "raw": {"Name": "Private Name", "Cause of Death": "restricted"},
            "row_data_normalized": {"canonical": {"name": "Private Name"}},
        },
    ]


class MemoryDataGateway:
    """In-process gateway used by tests. Applies the same AccessScope rules."""

    def __init__(
        self,
        records: list[dict[str, Any]] | None = None,
        catalog: FieldCatalog | None = None,
    ) -> None:
        self._records = records if records is not None else seed_records()
        self._catalog = catalog or static_catalog()
        self.calls: list[str] = []
        self.method_delays: dict[str, float] = {}

    async def list_datasets(self, scope: AccessScope) -> list[dict[str, Any]]:
        rows = []
        for dataset in self._catalog.datasets:
            if not scope.allows_file(dataset.file_id, private=dataset.private):
                continue
            rows.append(
                {
                    "file_id": dataset.file_id,
                    "user_facing_label": dataset.user_facing_label,
                    "private": dataset.private,
                    "is_default_people_scope": dataset.is_default_people_scope,
                }
            )
        return rows

    async def get_dataset_fields(self, scope: AccessScope, file_id: int) -> list[dict[str, Any]]:
        if not scope.allows_file(file_id):
            return []
        return [spec.model_dump() for spec in self._catalog.fields_for(file_id)]

    async def load_catalog(self, scope: AccessScope) -> FieldCatalog:
        return self._catalog

    async def suggest_name_matches(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        *,
        max_distance: int = 2,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """The same edit-distance search the SQL gateway runs, over the fixture rows."""
        wanted = [token for token in (query or "").lower().split() if token]
        if not wanted:
            return []
        found: list[dict[str, Any]] = []
        for row in self._records:
            file_id = int(row["file_id"])
            if file_id not in file_ids or not scope.allows_file(file_id):
                continue
            normalized = row.get("row_data_normalized") or {}
            canonical = normalized.get("canonical") or {}
            recorded = normalized.get("names")
            if not isinstance(recorded, list) or not recorded:
                recorded = str(row.get("canonical_name") or "").lower().split()
            tokens = [str(item).lower() for item in recorded if str(item).strip()]
            if not tokens:
                continue
            per_token = [min(_edit_distance(word, token) for token in tokens) for word in wanted]
            if max(per_token) > max_distance:
                continue
            found.append(
                {
                    "file_id": file_id,
                    "source_row_id": int(row.get("source_row_id") or 0),
                    "display_name": str(
                        canonical.get("display_name") or row.get("canonical_name") or ""
                    ),
                    "distance": sum(per_token),
                }
            )
        found.sort(key=lambda item: (item["distance"], item["display_name"].lower()))
        return found[:limit]

    async def count_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> int:
        return len(await self.list_records(scope, file_ids, predicates, limit=10_000))

    async def count_records_by_file(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> dict[int, int]:
        return {
            file_id: await self.count_records(scope, (file_id,), predicates)
            for file_id in file_ids
            if scope.allows_file(file_id)
        }

    async def group_values(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
    ) -> dict[str, int]:
        rows = await self.list_records(scope, file_ids, predicates, limit=10_000)
        grouped: dict[str, int] = {}
        for row in rows:
            spec = self._catalog.resolve_field(int(row["file_id"]), field)
            if spec is None:
                continue
            value = semantic_scalar(extract_field_value(row, spec))
            label = str(value) if value not in (None, "") else "Not recorded"
            grouped[label] = grouped.get(label, 0) + 1
        return dict(sorted(grouped.items(), key=lambda item: (-item[1], item[0].casefold())))

    def _display(self, row: dict[str, Any], field: str) -> str | None:
        spec = self._catalog.resolve_field(int(row["file_id"]), field)
        if spec is None:
            return None
        value = semantic_scalar(extract_field_value(row, spec))
        text = "" if value is None else str(value).strip()
        return text or None

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
        rows = await self.list_records(scope, file_ids, predicates, limit=100_000)
        buckets: dict[tuple[int, str | None, str | None], int] = {}
        for row in rows:
            if self._catalog.resolve_field(int(row["file_id"]), field) is None:
                continue
            label = derive_part(self._display(row, field), value_part)
            second = (
                derive_part(self._display(row, secondary_field), secondary_value_part)
                if secondary_field
                else None
            )
            if not include_missing and (
                label is None or (secondary_field is not None and second is None)
            ):
                continue
            key = (int(row["file_id"]), label, second)
            buckets[key] = buckets.get(key, 0) + 1
        grouped = [
            GroupRow(file_id=key[0], label=key[1], secondary_label=key[2], count=count)
            for key, count in buckets.items()
            if count >= max(int(min_count), 1)
        ]
        grouped.sort(key=lambda item: (-item.count, (item.label or "").casefold()))
        if per_group_top_n and per_group_top_n > 0:
            seen: dict[tuple[int, str | None], int] = {}
            kept: list[GroupRow] = []
            for item in grouped:
                key2 = (item.file_id, item.label)
                seen[key2] = seen.get(key2, 0) + 1
                if seen[key2] <= per_group_top_n:
                    kept.append(item)
            grouped = sorted(kept, key=lambda item: ((item.label or "").casefold(), -item.count))
        if top_n and top_n > 0:
            grouped = grouped[:top_n]
        return grouped

    async def field_stats(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
        *,
        value_part: str | None = None,
    ) -> FieldStats:
        rows = await self.list_records(scope, file_ids, predicates, limit=100_000)
        values: list[float] = []
        for row in rows:
            number = derive_number(self._display(row, field), value_part)
            if number is not None:
                values.append(number)
        if not values:
            return FieldStats(record_count=len(rows), known_count=0)
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        counts: dict[float, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        mode_value, mode_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        return FieldStats(
            record_count=len(rows),
            known_count=len(values),
            minimum=ordered[0],
            maximum=ordered[-1],
            average=sum(values) / len(values),
            median=median,
            mode=mode_value,
            mode_count=mode_count,
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
        rows = await self.list_records(scope, file_ids, predicates, limit=100_000)
        buckets: dict[tuple[int, str], list[str]] = {}
        for row in rows:
            label = derive_part(self._display(row, field), value_part)
            if label is None:
                continue
            member = self._display(row, companion_field or "student_name") or str(
                row.get("canonical_name") or ""
            )
            buckets.setdefault((int(row["file_id"]), label), []).append(member)
        groups = [
            DuplicateGroup(
                file_id=key[0],
                value=key[1],
                count=len(members),
                distinct_companions=len({item.casefold() for item in members}),
                members=sorted(members, key=str.casefold)[: member_limit or len(members)],
            )
            for key, members in buckets.items()
            if len(members) >= max(int(min_count), 2)
        ]
        groups.sort(key=lambda item: (-item.count, item.value.casefold()))
        return groups[:limit] if limit and limit > 0 else groups

    async def record_completeness(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        direction: str = "desc",
        limit: int = 10,
    ) -> list[CompletenessRow]:
        rows = await self.list_records(scope, file_ids, predicates, limit=100_000)
        scored: list[CompletenessRow] = []
        for row in rows:
            specs = self._catalog.fields_for(int(row["file_id"]))
            filled = [
                spec.human_label
                for spec in specs
                if self._display(row, spec.semantic_field) is not None
            ]
            scored.append(
                CompletenessRow(
                    file_id=int(row["file_id"]),
                    source_row_id=int(row.get("source_row_id") or row.get("id") or 0),
                    display_name=self._display(row, "student_name")
                    or str(row.get("canonical_name") or ""),
                    filled_fields=len(filled),
                    total_fields=len(specs),
                    filled_field_labels=sorted(filled),
                )
            )
        ascending = direction.lower() == "asc"
        scored.sort(key=lambda item: (item.display_name or "").casefold())
        scored.sort(key=lambda item: item.filled_fields, reverse=not ascending)
        return scored[: max(int(limit), 1)]

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
        rows = await self.list_records(scope, file_ids, predicates, limit=100_000)
        measured: list[IntervalRow] = []
        for row in rows:
            start_text = self._display(row, start_field)
            end_text = self._display(row, end_field)
            start = parse_date(start_text)
            end = parse_date(end_text)
            if start is None or end is None:
                continue
            days = float((end - start).days)
            if min_days is not None and days < min_days:
                continue
            if max_days is not None and days > max_days:
                continue
            measured.append(
                IntervalRow(
                    file_id=int(row["file_id"]),
                    source_row_id=int(row.get("source_row_id") or row.get("id") or 0),
                    display_name=self._display(row, "student_name")
                    or str(row.get("canonical_name") or ""),
                    start_value=start_text,
                    end_value=end_text,
                    days=days,
                    start_precision=date_precision(start_text),
                    end_precision=date_precision(end_text),
                    year_delta=end.year - start.year,
                )
            )
        ascending = direction.lower() == "asc"
        measured.sort(key=lambda item: (item.display_name or "").casefold())
        measured.sort(key=lambda item: item.days, reverse=not ascending)
        return measured[: max(int(limit), 1)]

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
        every = await self.interval_rows(
            scope, file_ids, predicates, start_field, end_field, limit=100_000
        )
        bounded = [
            item
            for item in every
            if (min_days is None or item.days >= min_days)
            and (max_days is None or item.days <= max_days)
        ]
        negative = sum(1 for item in every if item.days < 0)
        impossible = sum(1 for item in every if item.is_impossible())
        if not bounded:
            return IntervalStats(
                record_count=0, negative_count=negative, impossible_count=impossible
            )
        ordered = sorted(item.days for item in bounded)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return IntervalStats(
            record_count=len(bounded),
            minimum_days=ordered[0],
            maximum_days=ordered[-1],
            average_days=sum(ordered) / len(ordered),
            median_days=median,
            negative_count=negative,
            impossible_count=impossible,
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
        allowed = tuple(
            file_id
            for file_id in file_ids
            if scope.allows_file(file_id, private=bool(self._catalog.dataset(file_id) and self._catalog.dataset(file_id).private))
        )
        matched = apply_predicates(self._records, predicates, self._catalog, allowed)
        if sort_field:
            key_name = {
                "student_name": "canonical_name",
                "community": "canonical_community",
                "school": "canonical_school",
            }.get(sort_field, sort_field)
            matched = sorted(
                matched,
                key=lambda row: (
                    str(row.get(key_name) or row.get("canonical_name") or "").lower(),
                    int(row.get("source_row_id") or row.get("id") or 0),
                ),
                reverse=sort_direction == "DESC",
            )
        start = max(int(offset), 0)
        return matched[start:start + max(int(limit), 0)]

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
        total = await self.count_records(scope, file_ids, predicates)
        selected = min(max(int(window_size), 0), total)
        safe_cursor = min(max(int(cursor), 0), selected)
        base = total - selected if anchor == "end" else 0
        offset = base + safe_cursor
        effective = min(max(int(page_size), 0), selected - safe_cursor)
        rows = await self.list_records(
            scope,
            file_ids,
            predicates,
            limit=effective,
            offset=offset,
            sort_field=sort_field,
            sort_direction=sort_direction,
        ) if effective else []
        return RecordPage(
            rows=rows,
            total_count=total,
            offset=offset,
            page_size=page_size,
            has_more=safe_cursor + len(rows) < selected,
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
        allowed = tuple(
            file_id
            for file_id in file_ids
            if scope.allows_file(
                file_id,
                private=bool(
                    self._catalog.dataset(file_id)
                    and self._catalog.dataset(file_id).private
                ),
            )
        )
        matched = apply_predicates(self._records, predicates, self._catalog, allowed)
        count = min(max(int(limit), 0), len(matched))
        return random.Random(0).sample(matched, count) if count else []

    async def sample_records_page(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 5,
    ) -> RecordPage:
        allowed = tuple(
            file_id
            for file_id in file_ids
            if scope.allows_file(
                file_id,
                private=bool(
                    self._catalog.dataset(file_id)
                    and self._catalog.dataset(file_id).private
                ),
            )
        )
        matched = apply_predicates(self._records, predicates, self._catalog, allowed)
        count = min(max(int(limit), 0), len(matched))
        rows = random.Random(0).sample(matched, count) if count else []
        return RecordPage(
            rows=rows,
            total_count=len(matched),
            offset=0,
            page_size=limit,
            has_more=len(matched) > len(rows),
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
        self.calls.append(method)
        delay = self.method_delays.get(method)
        if delay:
            await asyncio.sleep(delay)
        allowed = self._allowed_file_ids(scope, file_ids)
        rows = apply_predicates(self._records, predicates or [], self._catalog, tuple(allowed))
        return rows[: max(limit * 4, 20)]

    async def get_raw_fields(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
        keys: list[str],
    ) -> list[dict[str, Any]]:
        allowed = self._allowed_file_ids(scope, tuple(scope.allowed_file_ids))
        wanted = set(source_row_ids)
        rows: list[dict[str, Any]] = []
        for record in self._records:
            if int(record["file_id"]) not in allowed:
                continue
            if int(record["source_row_id"]) not in wanted:
                continue
            raw = record.get("raw") or {}
            rows.append(
                {
                    "id": record["source_row_id"],
                    "file_id": record["file_id"],
                    "version": record.get("version"),
                    "selected": {key: raw.get(key) for key in keys if key in raw},
                }
            )
        return rows

    async def get_records_by_source_ids(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
    ) -> list[dict[str, Any]]:
        allowed = self._allowed_file_ids(scope, tuple(scope.allowed_file_ids))
        wanted = set(source_row_ids)
        return [
            record
            for record in self._records
            if int(record["file_id"]) in allowed and int(record["source_row_id"]) in wanted
        ]

    def _allowed_file_ids(self, scope: AccessScope, file_ids: tuple[int, ...]) -> set[int]:
        allowed: set[int] = set()
        for file_id in file_ids:
            dataset = self._catalog.dataset(file_id)
            private = bool(dataset and dataset.private)
            if scope.allows_file(file_id, private=private):
                allowed.add(file_id)
        return allowed

    async def save_query_run(
        self,
        *,
        turn_id: str,
        plan: QueryPlan | None,
        execution: ExecutionResult | None,
        status: str,
    ) -> None:
        """No durable trace store in-memory; kept for DataGateway protocol conformance."""
        self.calls.append("save_query_run")


def _edit_distance(left: str, right: str) -> int:
    """Levenshtein, matching what the SQL gateway computes."""
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]
