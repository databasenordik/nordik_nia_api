from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import Predicate, QueryPlan
from app.security.access_scope import AccessScope

if TYPE_CHECKING:
    from app.execution.executor import ExecutionResult


@dataclass(frozen=True)
class GroupRow:
    """One grouped bucket. ``label_2`` is set only for two-field cross-tabs."""

    file_id: int
    label: str | None
    secondary_label: str | None
    count: int


@dataclass(frozen=True)
class FieldStats:
    record_count: int
    known_count: int
    minimum: float | None = None
    maximum: float | None = None
    average: float | None = None
    median: float | None = None
    mode: float | None = None
    mode_count: int | None = None


@dataclass(frozen=True)
class DuplicateGroup:
    file_id: int
    value: str
    count: int
    distinct_companions: int
    members: list[str]


@dataclass(frozen=True)
class CompletenessRow:
    file_id: int
    source_row_id: int
    display_name: str | None
    filled_fields: int
    total_fields: int
    filled_field_labels: list[str]


@dataclass(frozen=True)
class IntervalRow:
    file_id: int
    source_row_id: int
    display_name: str | None
    start_value: str | None
    end_value: str | None
    days: float
    start_precision: str | None = None
    end_precision: str | None = None
    year_delta: int | None = None

    def is_impossible(self) -> bool:
        """A negative span is only a real ordering error when the years disagree.

        Year- and month-only values parse to the first of the period, so an interval
        can be negative purely because the precision is coarse.
        """
        return self.year_delta is not None and self.year_delta < 0


@dataclass(frozen=True)
class IntervalStats:
    record_count: int
    minimum_days: float | None = None
    maximum_days: float | None = None
    average_days: float | None = None
    median_days: float | None = None
    negative_count: int = 0
    impossible_count: int = 0


@dataclass(frozen=True)
class RecordPage:
    rows: list[dict[str, Any]]
    total_count: int
    offset: int
    page_size: int
    has_more: bool
    window_anchor: Literal["start", "end"] | None = None
    window_size: int | None = None
    window_count: int | None = None
    window_cursor: int = 0


class DataGateway(Protocol):
    async def list_datasets(self, scope: AccessScope) -> list[dict[str, Any]]: ...

    async def get_dataset_fields(self, scope: AccessScope, file_id: int) -> list[dict[str, Any]]: ...

    async def load_catalog(self, scope: AccessScope) -> FieldCatalog: ...

    async def count_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> int: ...

    async def count_records_by_file(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
    ) -> dict[int, int]: ...

    async def group_values(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
    ) -> dict[str, int]: ...

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
    ) -> list[GroupRow]: ...

    async def field_stats(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        field: str,
        *,
        value_part: str | None = None,
    ) -> FieldStats: ...

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
    ) -> list[DuplicateGroup]: ...

    async def record_completeness(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        direction: str = "desc",
        limit: int = 10,
    ) -> list[CompletenessRow]: ...

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
    ) -> list[IntervalRow]: ...

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
    ) -> IntervalStats: ...

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
    ) -> list[dict[str, Any]]: ...

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
    ) -> RecordPage: ...

    async def sample_records(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]: ...

    async def sample_records_page(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        predicates: list[Predicate],
        *,
        limit: int = 5,
    ) -> RecordPage: ...

    async def retrieve_candidates(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        method: str,
        limit: int = 20,
        *,
        predicates: list[Predicate] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def get_raw_fields(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
        keys: list[str],
    ) -> list[dict[str, Any]]: ...

    async def get_records_by_source_ids(
        self,
        scope: AccessScope,
        source_row_ids: list[int],
    ) -> list[dict[str, Any]]: ...

    async def suggest_name_matches(
        self,
        scope: AccessScope,
        file_ids: tuple[int, ...],
        query: str,
        *,
        max_distance: int = 2,
        limit: int = 6,
    ) -> list[dict[str, Any]]: ...

    async def save_query_run(
        self,
        *,
        turn_id: str,
        plan: QueryPlan | None,
        execution: ExecutionResult | None,
        status: str,
    ) -> None: ...
