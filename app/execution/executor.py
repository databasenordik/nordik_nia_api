from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.concurrency.cancellation import CancellationToken
from app.config import get_settings
from app.data_gateway.protocol import (
    CompletenessRow,
    DataGateway,
    DuplicateGroup,
    FieldStats,
    GroupRow,
    IntervalRow,
    IntervalStats,
)
from app.execution.dag_scheduler import DagScheduler, RunTrace, ScheduledStep, StepResult
from app.execution.overlap import OverlapResult, overlapping_groups
from app.execution.roster import fetch_rows
from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import FilterOperator, PlanOp, PlanStep, Predicate, QueryPlan
from app.retrieval.context_builder import build_evidence_packet
from app.retrieval.deduplicator import dedupe_hits
from app.retrieval.exact_search import score_exact
from app.retrieval.fulltext_search import score_fts
from app.retrieval.fuzzy_search import score_fuzzy
from app.retrieval.types import EvidencePacket, RetrievalHit
from app.security.access_scope import AccessScope
from app.value_normalization import json_object, semantic_scalar


@dataclass
class ComputedFact:
    name: str
    value: Any
    unit: str | None = None


@dataclass
class ActionResult:
    action_id: str
    goal: str
    file_ids: tuple[int, ...] = ()
    facts: list[ComputedFact] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    hits: list[RetrievalHit] = field(default_factory=list)
    evidence: EvidencePacket | None = None
    predicates: list[Predicate] = field(default_factory=list)
    grouped_counts: dict[int, int] = field(default_factory=dict)
    distinct_field: str | None = None
    distinct_values: list[str] = field(default_factory=list)
    grouped_field: str | None = None
    value_counts: dict[str, int] = field(default_factory=dict)
    total_count: int | None = None
    offset: int = 0
    page_size: int | None = None
    has_more: bool = False
    sampled: bool = False
    projected_fields: list[str] = field(default_factory=list)
    window_anchor: str | None = None
    window_size: int | None = None
    window_count: int | None = None
    window_cursor: int = 0
    ranked_groups: list[GroupRow] = field(default_factory=list)
    group_fields: tuple[str | None, str | None] = (None, None)
    group_value_parts: tuple[str | None, str | None] = (None, None)
    stats: FieldStats | None = None
    stats_field: str | None = None
    stats_value_part: str | None = None
    duplicates: list[DuplicateGroup] = field(default_factory=list)
    duplicate_field: str | None = None
    completeness: list[CompletenessRow] = field(default_factory=list)
    completeness_direction: str | None = None
    intervals: list[IntervalRow] = field(default_factory=list)
    interval_summary: IntervalStats | None = None
    interval_fields: tuple[str, str] | None = None
    percentage: dict[str, Any] | None = None
    requested_top_n: int | None = None
    overlaps: OverlapResult | None = None


@dataclass
class ExecutionResult:
    action_results: list[ActionResult] = field(default_factory=list)
    facts: list[ComputedFact] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    reasoning_calls: int = 0
    grouped_counts: dict[int, int] = field(default_factory=dict)
    hits: list[RetrievalHit] = field(default_factory=list)
    evidence: EvidencePacket | None = None
    methods_run: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    run_trace: RunTrace | None = None
    distinct_field: str | None = None
    distinct_values: list[str] = field(default_factory=list)


class PlanExecutionError(ValueError):
    pass


_RETRIEVAL_OPS = {PlanOp.EXACT_LOOKUP, PlanOp.FULL_TEXT_SEARCH, PlanOp.FUZZY_SEARCH}
# Comparing periods reads every record that carries the shared value.
_OVERLAP_ROW_LIMIT = 10_000
_RESOURCE = {
    PlanOp.FILTER: "db",
    PlanOp.COUNT: "db",
    PlanOp.COUNT_DISTINCT: "db",
    PlanOp.SUM: "db",
    PlanOp.AVG: "db",
    PlanOp.MIN: "db",
    PlanOp.MAX: "db",
    PlanOp.PROJECT: "db",
    PlanOp.SAMPLE: "db",
    PlanOp.GET_RAW_FIELDS: "db",
    PlanOp.GET_QUOTE: "db",
    PlanOp.GET_PROVENANCE: "db",
    PlanOp.EXACT_LOOKUP: "retrieval",
    PlanOp.FULL_TEXT_SEARCH: "retrieval",
    PlanOp.FUZZY_SEARCH: "retrieval",
    PlanOp.UNION: "local",
    PlanOp.DEDUPLICATE: "local",
    PlanOp.INTERSECT: "local",
    PlanOp.COMPARE: "local",
    PlanOp.RANK: "db",
    PlanOp.STATS: "db",
    PlanOp.DUPLICATES: "db",
    PlanOp.COMPLETENESS: "db",
    PlanOp.INTERVAL: "db",
    PlanOp.INTERVAL_STATS: "db",
    PlanOp.PERCENTAGE: "local",
    PlanOp.RATIO: "local",
}


class QueryExecutor:
    """Executes a validated QueryPlan on the DAG scheduler."""

    def __init__(
        self,
        gateway: DataGateway,
        catalog: FieldCatalog,
        *,
        limits=None,
    ) -> None:
        self._gateway = gateway
        self._catalog = catalog
        self._scheduler = DagScheduler(limits=limits)

    async def execute(
        self,
        plan: QueryPlan,
        scope: AccessScope,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ExecutionResult:
        narrowed = scope.narrow(plan.scope.file_ids)
        if not narrowed.allowed_file_ids:
            raise PlanExecutionError("no authorized files remain after narrowing")
        fingerprint = narrowed.fingerprint()
        scheduled = _to_scheduled(plan, fingerprint)
        scheduler = DagScheduler(cancellation=cancellation) if cancellation else self._scheduler
        outputs, trace = await scheduler.run(
            scheduled,
            lambda step, prior: self._run_step(step, prior, plan, narrowed),
            scope_fingerprint=fingerprint,
        )
        if any(item.scope_fingerprint != fingerprint for item in trace.steps.values()):
            raise PlanExecutionError("AccessScope fingerprint drifted across DAG branches")
        return _reduce(plan, outputs, trace)

    async def _run_step(
        self,
        scheduled: ScheduledStep,
        prior: dict[str, StepResult],
        plan: QueryPlan,
        scope: AccessScope,
    ) -> StepResult:
        step: PlanStep = scheduled.payload
        settings = get_settings()
        file_ids = step.file_ids or plan.scope.file_ids
        predicates = _collect_predicates(step, prior, plan)
        hits = _collect_hits(step, prior, plan)
        limit = step.limit or _collect_limit(step, prior, plan) or settings.max_retrieval_results

        if step.op in {PlanOp.USE_CURRENT_VERSION, PlanOp.USE_DATASET}:
            return StepResult(value={"predicates": predicates})
        if step.op is PlanOp.SORT:
            return StepResult(
                value={
                    "predicates": predicates,
                    "sort_field": step.sort_field,
                    "sort_direction": step.sort_direction,
                }
            )
        if step.op is PlanOp.GROUP_BY:
            group_field = step.fields[0] if step.fields else None
            if group_field and group_field != "file_id":
                second = step.fields[1] if len(step.fields) > 1 else None
                grouped = await self._gateway.group_values_multi(
                    scope,
                    file_ids,
                    predicates,
                    group_field,
                    value_part=step.value_part,
                    secondary_field=second,
                    secondary_value_part=step.secondary_value_part,
                    min_count=step.having_min_count or 1,
                    top_n=step.top_n or 0,
                    per_group_top_n=step.per_group_top_n or 0,
                    include_missing=step.include_missing or second is None,
                )
                counts = _counts_from_groups(grouped)
                values = [item for item in counts if item != "Not recorded"]
                return StepResult(
                    value={
                        "distinct_field": group_field,
                        "distinct_values": values,
                        # GROUP_BY has already computed the distinct set, so publish its
                        # size too. "How many distinct communities are recorded?" is
                        # answerable as either COUNT_DISTINCT or GROUP_BY and the planner
                        # may legitimately pick either; without this the GROUP_BY spelling
                        # produced the right sentence and no structured count, so the
                        # number was in the prose and nowhere a caller could read it.
                        "count_distinct": len(values),
                        "grouped_field": group_field,
                        "value_counts": counts,
                        "ranked_groups": grouped,
                        "group_fields": (group_field, second),
                        "group_value_parts": (step.value_part, step.secondary_value_part),
                        "requested_top_n": step.top_n,
                        "predicates": predicates,
                    },
                    result_count=len(values),
                )
            return StepResult(value={"predicates": predicates, "group_by_file": True})
        if step.op is PlanOp.RANK:
            rank_field = step.fields[0] if step.fields else None
            if not rank_field:
                raise PlanExecutionError("RANK requires a field")
            second = step.fields[1] if len(step.fields) > 1 else None
            grouped = await self._gateway.group_values_multi(
                scope,
                file_ids,
                predicates,
                rank_field,
                value_part=step.value_part,
                secondary_field=second,
                secondary_value_part=step.secondary_value_part,
                min_count=step.having_min_count or 1,
                top_n=step.top_n or 0,
                per_group_top_n=step.per_group_top_n or 0,
                include_missing=step.include_missing,
            )
            return StepResult(
                value={
                    "ranked_groups": grouped,
                    "group_fields": (rank_field, second),
                    "group_value_parts": (step.value_part, step.secondary_value_part),
                    "grouped_field": rank_field,
                    "value_counts": _counts_from_groups(grouped),
                    "requested_top_n": step.top_n,
                    "predicates": predicates,
                },
                result_count=len(grouped),
            )
        if step.op is PlanOp.STATS:
            stats_field = step.fields[0] if step.fields else None
            if not stats_field:
                raise PlanExecutionError("STATS requires a field")
            stats = await self._gateway.field_stats(
                scope, file_ids, predicates, stats_field, value_part=step.value_part
            )
            return StepResult(
                value={
                    "stats": stats,
                    "stats_field": stats_field,
                    "stats_value_part": step.value_part,
                    "predicates": predicates,
                },
                result_count=stats.known_count,
            )
        if step.op is PlanOp.DUPLICATES:
            duplicate_field = step.fields[0] if step.fields else None
            if not duplicate_field:
                raise PlanExecutionError("DUPLICATES requires a field")
            if step.interval_start and step.interval_end:
                # Sharing a value is not the question when periods were given: which of the
                # records that share it were present at the same time is.
                known = [
                    *predicates,
                    Predicate(field=duplicate_field, operator=FilterOperator.IS_KNOWN),
                ]
                total = await self._gateway.count_records(scope, file_ids, known)
                if total > _OVERLAP_ROW_LIMIT:
                    raise PlanExecutionError("too many records to compare their periods")
                overlaps = overlapping_groups(
                    await fetch_rows(self._gateway, scope, file_ids, known, total),
                    self._catalog,
                    group_field=duplicate_field,
                    start_field=step.interval_start,
                    end_field=step.interval_end,
                    value_part=step.value_part,
                    min_count=step.having_min_count or 2,
                )
                return StepResult(
                    value={
                        "overlaps": overlaps,
                        "duplicate_field": duplicate_field,
                        "predicates": predicates,
                    },
                    result_count=len(overlaps.groups),
                )
            groups = await self._gateway.duplicate_values(
                scope,
                file_ids,
                predicates,
                duplicate_field,
                value_part=step.value_part,
                companion_field=step.companion_field,
                min_count=step.having_min_count or 2,
                member_limit=25,
                limit=step.top_n or 0,
            )
            return StepResult(
                value={
                    "duplicates": groups,
                    "duplicate_field": duplicate_field,
                    "predicates": predicates,
                },
                result_count=len(groups),
            )
        if step.op is PlanOp.COMPLETENESS:
            rows = await self._gateway.record_completeness(
                scope,
                file_ids,
                predicates,
                direction=step.direction or "desc",
                limit=step.limit or 10,
            )
            return StepResult(
                value={
                    "completeness": rows,
                    "completeness_direction": step.direction or "desc",
                    "predicates": predicates,
                },
                result_count=len(rows),
            )
        if step.op in {PlanOp.INTERVAL, PlanOp.INTERVAL_STATS}:
            if not step.interval_start or not step.interval_end:
                raise PlanExecutionError(f"{step.op.value} requires two date fields")
            payload: dict[str, Any] = {
                "interval_fields": (step.interval_start, step.interval_end),
                "predicates": predicates,
            }
            if step.op is PlanOp.INTERVAL_STATS:
                payload["interval_summary"] = await self._gateway.interval_stats(
                    scope,
                    file_ids,
                    predicates,
                    step.interval_start,
                    step.interval_end,
                    min_days=step.interval_min_days,
                    max_days=step.interval_max_days,
                )
            payload["intervals"] = await self._gateway.interval_rows(
                scope,
                file_ids,
                predicates,
                step.interval_start,
                step.interval_end,
                min_days=step.interval_min_days,
                max_days=step.interval_max_days,
                direction=step.direction or "desc",
                limit=step.limit or 10,
            )
            return StepResult(value=payload, result_count=len(payload["intervals"]))
        if step.op in {PlanOp.PERCENTAGE, PlanOp.RATIO}:
            counts = [_count_from_dep(dep, prior) for dep in _inputs(step)]
            numeric = [item for item in counts if item is not None]
            if len(numeric) < 2:
                raise PlanExecutionError(f"{step.op.value} requires two counted inputs")
            numerator, denominator = numeric[0], numeric[1]
            share = (numerator / denominator) if denominator else None
            return StepResult(
                value={
                    "percentage": {
                        "numerator": numerator,
                        "denominator": denominator,
                        "ratio": share,
                        "percent": None if share is None else round(share * 100, 2),
                        "kind": step.op.value.lower(),
                    },
                    "predicates": predicates,
                }
            )
        if step.op is PlanOp.FILTER:
            return StepResult(value={"predicates": step.where or predicates})
        if step.op is PlanOp.SAMPLE:
            page = await self._gateway.sample_records_page(
                scope, file_ids, predicates, limit=limit
            )
            rows = page.rows
            hits = [hit_from_row(row) for row in rows]
            return StepResult(
                value={
                    "rows": rows,
                    "hits": hits,
                    "predicates": predicates,
                    "limit": limit,
                    "offset": 0,
                    "page_size": limit,
                    "has_more": page.has_more,
                    "sampled": True,
                    "page_applied": True,
                    "total_count": page.total_count,
                },
                result_count=len(rows),
            )
        if step.op is PlanOp.LIMIT or step.op is PlanOp.TOP_N:
            sort_field, sort_direction = _collect_sort(step, prior)
            total = len(hits)
            offset, selected, cursor = _resolve_window(
                total,
                anchor=step.window_anchor,
                size=step.window_size,
                cursor=step.window_cursor,
                fallback_offset=step.offset or 0,
            )
            effective = min(limit, max(selected - cursor, 0)) if step.window_anchor else limit
            sliced = hits[offset:offset + effective] if hits else []
            return StepResult(
                value={
                    "hits": sliced,
                    "limit": limit,
                    "offset": offset,
                    "predicates": predicates,
                    "total_count": total if hits else None,
                    "has_more": cursor + len(sliced) < selected if step.window_anchor else offset + len(sliced) < total,
                    "window_anchor": step.window_anchor,
                    "window_size": step.window_size,
                    "window_count": selected if step.window_anchor else None,
                    "window_cursor": cursor,
                    "page_applied": bool(hits),
                    "sort_field": sort_field,
                    "sort_direction": sort_direction,
                }
            )
        if step.op in _RETRIEVAL_OPS:
            method = {
                PlanOp.EXACT_LOOKUP: "exact",
                PlanOp.FULL_TEXT_SEARCH: "fts",
                PlanOp.FUZZY_SEARCH: "fuzzy",
            }[step.op]
            query = step.query or ""
            rows = await self._gateway.retrieve_candidates(
                scope, file_ids, query, method, limit
            )
            scorer = {"exact": score_exact, "fts": score_fts, "fuzzy": score_fuzzy}[method]
            scored = [hit for row in rows if (hit := scorer(row, query)) is not None]
            unique = method == "exact" and len({hit.identity() for hit in scored if hit.score >= 0.99}) == 1
            return StepResult(
                value={"hits": scored, "method": method},
                early_complete=unique,
                result_count=len(scored),
            )
        if step.op is PlanOp.UNION:
            hits = _collect_hits(step, prior, plan, dedupe=False)
            return StepResult(
                value={"hits": hits, "predicates": predicates, "merge": "union"},
                result_count=len(hits),
            )
        if step.op is PlanOp.INTERSECT:
            groups = [_hits_from_dep(dep, prior) for dep in _inputs(step)]
            if not groups:
                return StepResult(value={"hits": [], "predicates": predicates, "merge": "intersect"})
            identities = {hit.identity() for hit in groups[0]}
            for group in groups[1:]:
                identities &= {hit.identity() for hit in group}
            first = {hit.identity(): hit for hit in groups[0]}
            fused = [first[key] for key in identities if key in first]
            return StepResult(
                value={"hits": fused, "predicates": predicates, "merge": "intersect"},
                result_count=len(fused),
            )
        if step.op is PlanOp.DEDUPLICATE:
            hits = dedupe_hits(_collect_hits(step, prior, plan, dedupe=False))
            return StepResult(
                value={"hits": hits, "predicates": predicates, "merge": "dedupe"},
                result_count=len(hits),
            )
        if step.op is PlanOp.COUNT:
            if hits:
                return StepResult(
                    value={"count": len(hits), "hits": hits, "predicates": predicates},
                    result_count=len(hits),
                )
            value_counts, grouped_field = _collect_value_counts(step, prior)
            if value_counts:
                return StepResult(
                    value={
                        "value_counts": value_counts,
                        "grouped_field": grouped_field,
                        "count": sum(value_counts.values()),
                        "predicates": predicates,
                    },
                    result_count=sum(value_counts.values()),
                )
            if step.fields == ["file_id"] or _grouped_by_file(step, prior):
                grouped = await self._gateway.count_records_by_file(
                    scope, file_ids, predicates
                )
                return StepResult(value={"grouped": grouped, "predicates": predicates})
            count = await self._gateway.count_records(scope, file_ids, predicates)
            return StepResult(value={"count": count, "predicates": predicates}, result_count=count)
        if step.op is PlanOp.PROJECT:
            offset = step.offset or _collect_offset(step, prior, plan) or 0
            if hits:
                paging = _collect_paging(step, prior)
                page_applied = bool(paging.get("page_applied"))
                page = hits if page_applied else hits[offset:offset + limit]
                rows = [hit.record for hit in page]
                total = int(paging.get("total_count") or len(hits))
                return StepResult(
                    value={
                        "hits": page,
                        "rows": rows,
                        "predicates": predicates,
                        "total_count": total,
                        "offset": int(paging.get("offset", offset)),
                        "page_size": limit,
                        "has_more": bool(paging.get("has_more", offset + len(rows) < total)),
                        "projected_fields": list(step.fields),
                        "window_anchor": paging.get("window_anchor") or step.window_anchor,
                        "window_size": paging.get("window_size") or step.window_size,
                        "window_count": paging.get("window_count"),
                        "window_cursor": int(paging.get("window_cursor") or 0),
                    }
                )
            sort_field, sort_direction = _collect_sort(step, prior)
            if step.window_anchor and step.window_size:
                page = await self._gateway.list_records_window(
                    scope,
                    file_ids,
                    predicates,
                    anchor=step.window_anchor,
                    window_size=step.window_size,
                    cursor=step.window_cursor,
                    page_size=limit,
                    sort_field=sort_field,
                    sort_direction=sort_direction or "ASC",
                )
                return StepResult(
                    value={
                        "rows": page.rows,
                        "hits": [hit_from_row(row) for row in page.rows],
                        "predicates": predicates,
                        "total_count": page.total_count,
                        "offset": page.offset,
                        "page_size": page.page_size,
                        "has_more": page.has_more,
                        "projected_fields": list(step.fields),
                        "window_anchor": page.window_anchor,
                        "window_size": page.window_size,
                        "window_count": page.window_count,
                        "window_cursor": page.window_cursor,
                    },
                    result_count=len(page.rows),
                )
            total = await self._gateway.count_records(scope, file_ids, predicates)
            rows = await self._gateway.list_records(
                scope,
                file_ids,
                predicates,
                limit=limit,
                offset=offset,
                sort_field=sort_field,
                sort_direction=sort_direction or "ASC",
            )
            return StepResult(
                value={
                    "rows": rows,
                    "hits": [hit_from_row(row) for row in rows],
                    "predicates": predicates,
                    "total_count": total,
                    "offset": offset,
                    "page_size": limit,
                    "has_more": offset + len(rows) < total,
                    "projected_fields": list(step.fields),
                },
                result_count=len(rows),
            )
        if step.op in {PlanOp.GET_EVIDENCE, PlanOp.GET_QUOTE}:
            if not hits:
                rows = await self._gateway.list_records(
                    scope, file_ids, predicates, limit=limit
                )
                hits = [hit_from_row(row) for row in rows]
            packet = build_evidence_packet(
                next((item.query for item in plan.steps if item.query), ""),
                hits[:limit],
                facts=[f"count={len(hits)}"] if hits else None,
                requested_fields=step.fields,
                catalog=self._catalog,
            )
            payload = {
                "evidence": packet,
                "hits": hits,
                "rows": [hit.record for hit in hits],
                "predicates": predicates,
            }
            if step.op is PlanOp.GET_QUOTE:
                payload["quotes"] = [
                    {"source_id": item.source_id, "fields": item.fields} for item in packet.items
                ]
            return StepResult(value=payload)
        if step.op is PlanOp.GET_PROVENANCE:
            if not hits:
                rows = await self._gateway.list_records(
                    scope, file_ids, predicates, limit=limit
                )
                hits = [hit_from_row(row) for row in rows]
            provenance = [
                {
                    "source_id": hit.source_id(),
                    "file_id": hit.file_id,
                    "version": hit.version,
                    "source_row_id": hit.source_row_id,
                }
                for hit in hits[:limit]
            ]
            return StepResult(
                value={
                    "provenance": provenance,
                    "hits": hits[:limit],
                    "rows": [hit.record for hit in hits[:limit]],
                    "predicates": predicates,
                }
            )
        if step.op is PlanOp.COMPARE:
            sides = []
            for dep in _inputs(step):
                prior_val = prior.get(dep)
                payload = prior_val.value if prior_val and isinstance(prior_val.value, dict) else {}
                sides.append(
                    {
                        "id": dep,
                        "count": payload.get("count"),
                        "value": payload.get("count"),
                    }
                )
            return StepResult(value={"compare": sides, "predicates": predicates})
        if step.op in {PlanOp.SUM, PlanOp.AVG, PlanOp.MIN, PlanOp.MAX, PlanOp.COUNT_DISTINCT}:
            field_name = step.fields[0] if step.fields else None
            if not field_name:
                raise PlanExecutionError(f"{step.op.value} requires a field")
            if step.op is PlanOp.COUNT_DISTINCT:
                grouped_values = await self._gateway.group_values(
                    scope, file_ids, predicates, field_name
                )
                recorded = [value for value in grouped_values if value != "Not recorded"]
                return StepResult(
                    value={
                        "count_distinct": len(recorded),
                        "distinct_field": field_name,
                        "predicates": predicates,
                    },
                    result_count=len(recorded),
                )
            grouped_values = await self._gateway.group_values(
                scope, file_ids, predicates, field_name
            )
            spec = next(
                (
                    self._catalog.resolve_field(file_id, field_name)
                    for file_id in file_ids
                    if self._catalog.resolve_field(file_id, field_name) is not None
                ),
                None,
            )
            weighted_values: list[tuple[int | float | str, int]] = []
            for label, count in grouped_values.items():
                if label == "Not recorded":
                    continue
                value = _typed_aggregate_value(label, getattr(spec, "semantic_type", "text"))
                if value is not None:
                    weighted_values.append((value, int(count)))
            numbers = [
                (value, count)
                for value, count in weighted_values
                if isinstance(value, int | float)
            ]
            if not numbers:
                labels = [str(item) for item, _count in weighted_values]
                if step.op is PlanOp.MIN:
                    result_value = min(labels) if labels else None
                elif step.op is PlanOp.MAX:
                    result_value = max(labels) if labels else None
                else:
                    result_value = None
            elif step.op is PlanOp.SUM:
                result_value = sum(value * count for value, count in numbers)
            elif step.op is PlanOp.AVG:
                total_count = sum(count for _value, count in numbers)
                result_value = (
                    sum(value * count for value, count in numbers) / total_count
                    if total_count
                    else None
                )
            elif step.op is PlanOp.MIN:
                result_value = min(value for value, _count in numbers)
            else:
                result_value = max(value for value, _count in numbers)
            key = step.op.value.lower()
            return StepResult(value={key: result_value, "predicates": predicates})
        if step.op is PlanOp.GET_RAW_FIELDS:
            ids = [hit.source_row_id for hit in hits]
            keys = _raw_fetch_keys(file_ids, self._catalog)
            if not keys:
                # No field on these lists is granted raw access, so there is nothing to
                # fetch. This used to request Notes, Cause of Death and Name from a literal
                # list, which bypassed the registry's answer entirely.
                return StepResult(value={"raw": [], "hits": hits})
            raw = await self._gateway.get_raw_fields(scope, ids, keys)
            return StepResult(value={"raw": raw, "hits": hits})
        raise PlanExecutionError(f"deterministic executor does not run {step.op.value}")


def _counts_from_groups(groups: list[GroupRow]) -> dict[str, int]:
    """Collapse grouped rows to label -> count, naming the missing bucket."""
    counts: dict[str, int] = {}
    for item in groups:
        label = item.label or "Not recorded"
        if item.secondary_label:
            label = f"{label} / {item.secondary_label}"
        counts[label] = counts.get(label, 0) + item.count
    return counts


def _count_from_dep(dep: str, prior: dict[str, StepResult]) -> int | None:
    result = prior.get(dep)
    if result is None or not isinstance(result.value, dict):
        return None
    value = result.value.get("count")
    if value is None:
        return None
    return int(value)


def _to_scheduled(plan: QueryPlan, fingerprint: str) -> list[ScheduledStep]:
    settings = get_settings()
    known = {step.id for step in plan.steps}
    scheduled: list[ScheduledStep] = []
    for step in plan.steps:
        raw_inputs = step.input if isinstance(step.input, list) else ([step.input] if step.input else [])
        deps = [item for item in raw_inputs if item in known]
        retrieval = step.op in _RETRIEVAL_OPS
        timeout = (
            settings.retrieval_branch_timeout_ms
            if retrieval
            else settings.db_query_timeout_ms
        )
        scheduled.append(
            ScheduledStep(
                step_id=step.id,
                operation=step.op.value,
                dependencies=deps,
                resource_class=_RESOURCE.get(step.op, "local"),
                timeout_ms=timeout,
                cancel_policy="cancel_group_on_unique" if step.op is PlanOp.EXACT_LOOKUP else "skip_optional",
                idempotency_key=f"{fingerprint}:{step.op.value}:{step.query or ''}:{step.id}",
                branch_group="retrieval" if retrieval else None,
                scope_fingerprint=fingerprint,
                payload=step,
            )
        )
    return scheduled


def _inputs(step: PlanStep) -> list[str]:
    if step.input is None:
        return []
    if isinstance(step.input, list):
        return step.input
    return [step.input]


def _collect_predicates(step: PlanStep, prior: dict[str, StepResult], plan: QueryPlan) -> list[Predicate]:
    found: list[Predicate] = []
    for dep in _inputs(step):
        result = prior.get(dep)
        if result and isinstance(result.value, dict):
            found.extend(result.value.get("predicates") or [])
    if step.where:
        found.extend(step.where)
    return found


def _hits_from_dep(dep: str, prior: dict[str, StepResult]) -> list[RetrievalHit]:
    result = prior.get(dep)
    if result and isinstance(result.value, dict):
        return list(result.value.get("hits") or [])
    return []


def _collect_hits(
    step: PlanStep,
    prior: dict[str, StepResult],
    plan: QueryPlan,
    *,
    dedupe: bool = True,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for dep in _inputs(step):
        hits.extend(_hits_from_dep(dep, prior))
    return dedupe_hits(hits) if dedupe else hits


def _collect_sort(step: PlanStep, prior: dict[str, StepResult]) -> tuple[str | None, str | None]:
    if step.sort_field:
        return step.sort_field, step.sort_direction
    for dep in _inputs(step):
        result = prior.get(dep)
        if result and isinstance(result.value, dict) and result.value.get("sort_field"):
            return str(result.value["sort_field"]), str(result.value.get("sort_direction") or "ASC")
    return None, None


def _typed_aggregate_value(value: Any, semantic_type: str) -> int | float | str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if semantic_type == "date":
        match = re.search(r"(1[6-9]\d{2}|20\d{2})", text)
        return int(match.group(1)) if match else None
    if semantic_type == "number":
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            return None
        number = float(match.group(0))
        return int(number) if number.is_integer() else number
    return text


def _collect_limit(step: PlanStep, prior: dict[str, StepResult], plan: QueryPlan) -> int | None:
    for dep in _inputs(step):
        result = prior.get(dep)
        if result and isinstance(result.value, dict) and result.value.get("limit"):
            return int(result.value["limit"])
    return None


def _collect_offset(step: PlanStep, prior: dict[str, StepResult], plan: QueryPlan) -> int | None:
    if step.offset is not None:
        return int(step.offset)
    for dep in _inputs(step):
        result = prior.get(dep)
        if result and isinstance(result.value, dict) and result.value.get("offset") is not None:
            return int(result.value["offset"])
    return None


def _collect_paging(step: PlanStep, prior: dict[str, StepResult]) -> dict[str, Any]:
    for dep in _inputs(step):
        result = prior.get(dep)
        if result and isinstance(result.value, dict) and (
            result.value.get("page_applied")
            or result.value.get("sampled")
            or result.value.get("window_anchor")
        ):
            return result.value
    return {}


def _resolve_window(
    total: int,
    *,
    anchor: str | None,
    size: int | None,
    cursor: int,
    fallback_offset: int,
) -> tuple[int, int, int]:
    if anchor is None or size is None:
        return max(fallback_offset, 0), total, max(fallback_offset, 0)
    selected = min(max(size, 0), total)
    safe_cursor = min(max(cursor, 0), selected)
    base = total - selected if anchor == "end" else 0
    return base + safe_cursor, selected, safe_cursor


def _categorical_value(row: dict[str, Any], field_name: str) -> str | None:
    direct = {
        "community": row.get("canonical_community"),
        "school": row.get("canonical_school"),
    }.get(field_name)
    if direct:
        return str(direct)
    canonical = json_object(row.get("row_data_normalized")).get("canonical")
    if isinstance(canonical, dict):
        value = canonical.get(field_name)
        if value:
            scalar = semantic_scalar(value)
            return str(scalar) if scalar is not None else None
    return None


def _collect_value_counts(
    step: PlanStep,
    prior: dict[str, StepResult],
) -> tuple[dict[str, int], str | None]:
    for dep in _inputs(step):
        result = prior.get(dep)
        payload = result.value if result and isinstance(result.value, dict) else {}
        counts = payload.get("value_counts")
        if isinstance(counts, dict):
            return ({str(key): int(value) for key, value in counts.items()}, payload.get("grouped_field"))
    return {}, None


def _grouped_by_file(step: PlanStep, prior: dict[str, StepResult]) -> bool:
    for dep in _inputs(step):
        result = prior.get(dep)
        payload = result.value if result and isinstance(result.value, dict) else {}
        if payload.get("group_by_file"):
            return True
    return False


def hit_from_row(row: dict[str, Any]) -> RetrievalHit:
    return RetrievalHit(
        file_id=int(row["file_id"]),
        version=int(row.get("version") or 1),
        source_row_id=int(row["source_row_id"]),
        canonical_name=row.get("canonical_name"),
        canonical_community=row.get("canonical_community"),
        canonical_school=row.get("canonical_school"),
        method="exact",
        score=1.0,
        record=row,
    )


def _reduce(plan: QueryPlan, outputs: dict[str, StepResult], trace: RunTrace) -> ExecutionResult:
    result = ExecutionResult(run_trace=trace)
    result.cancelled = [step_id for step_id, item in outputs.items() if item.cancelled]
    result.methods_run = [
        (item.value or {}).get("method")
        for item in outputs.values()
        if isinstance(item.value, dict) and item.value.get("method") and not item.cancelled
    ]
    grouped: dict[str, ActionResult] = {}
    order: list[str] = []
    for step in plan.steps:
        if step.op is PlanOp.USE_CURRENT_VERSION:
            continue
        action_id = step.action_id or "a0"
        if action_id not in grouped:
            grouped[action_id] = ActionResult(
                action_id=action_id,
                goal=step.action_goal or (plan.goals[0] if plan.goals else "list"),
                file_ids=step.file_ids or plan.scope.file_ids,
            )
            order.append(action_id)
        bucket = grouped[action_id]
        if step.op is PlanOp.FILTER:
            bucket.predicates.extend(step.where)
        output = outputs.get(step.id)
        if not output or not isinstance(output.value, dict):
            continue
        _accumulate(bucket, output.value)
    result.action_results = [grouped[key] for key in order]
    for bucket in result.action_results:
        result.facts.extend(bucket.facts)
        result.rows = _merge_rows(result.rows, bucket.rows)
        result.hits.extend(bucket.hits)
        if bucket.evidence is not None:
            result.evidence = bucket.evidence
        if bucket.grouped_counts:
            result.grouped_counts = bucket.grouped_counts
        if bucket.distinct_values:
            result.distinct_field = bucket.distinct_field
            result.distinct_values = bucket.distinct_values
    return result


def _accumulate(bucket: ActionResult, value: dict[str, Any]) -> None:
    if "count" in value:
        bucket.facts.append(ComputedFact(name="count", value=value["count"], unit="records"))
        if bucket.total_count is None:
            bucket.total_count = int(value["count"])
    if "compare" in value:
        bucket.facts.append(ComputedFact(name="compare", value=value["compare"]))
    for agg_name in ("sum", "avg", "min", "max", "count_distinct"):
        if agg_name in value:
            bucket.facts.append(ComputedFact(name=agg_name, value=value[agg_name]))
    if "grouped" in value:
        bucket.grouped_counts = value["grouped"]
        bucket.facts.append(ComputedFact(name="counts_by_file", value=dict(value["grouped"])))
    for key in (
        "ranked_groups",
        "overlaps",
        "duplicates",
        "completeness",
        "intervals",
        "stats",
        "interval_summary",
        "percentage",
        "stats_field",
        "stats_value_part",
        "duplicate_field",
        "completeness_direction",
        "interval_fields",
        "group_fields",
        "group_value_parts",
        "requested_top_n",
    ):
        if value.get(key) is not None:
            setattr(bucket, key, value[key])
    if value.get("stats") is not None:
        bucket.facts.append(ComputedFact(name="stats", value=value["stats"]))
    if value.get("interval_summary") is not None:
        bucket.facts.append(ComputedFact(name="interval_stats", value=value["interval_summary"]))
    if value.get("percentage") is not None:
        bucket.facts.append(ComputedFact(name="percentage", value=value["percentage"]))
    if "distinct_values" in value:
        bucket.distinct_field = value.get("distinct_field")
        bucket.distinct_values = value["distinct_values"]
    elif value.get("distinct_field") is not None:
        bucket.distinct_field = str(value["distinct_field"])
    if value.get("grouped_field") is not None:
        bucket.grouped_field = str(value["grouped_field"])
    if isinstance(value.get("value_counts"), dict):
        bucket.value_counts = {
            str(key): int(count) for key, count in value["value_counts"].items()
        }
    if "rows" in value:
        bucket.rows = _merge_rows(bucket.rows, value["rows"])
    if "hits" in value:
        bucket.hits.extend(value["hits"])
        if not bucket.rows:
            bucket.rows = [hit.record for hit in value["hits"]]
    if "evidence" in value:
        bucket.evidence = value["evidence"]
    if "provenance" in value:
        bucket.facts.append(ComputedFact(name="provenance", value=value["provenance"]))
    if value.get("total_count") is not None:
        bucket.total_count = int(value["total_count"])
    if value.get("offset") is not None:
        bucket.offset = int(value["offset"])
    if value.get("page_size") is not None:
        bucket.page_size = int(value["page_size"])
    if "has_more" in value:
        bucket.has_more = bool(value["has_more"])
    if "sampled" in value:
        bucket.sampled = bool(value["sampled"])
    if value.get("projected_fields") is not None:
        bucket.projected_fields = list(value["projected_fields"])
    if value.get("window_anchor") is not None:
        bucket.window_anchor = str(value["window_anchor"])
    if value.get("window_size") is not None:
        bucket.window_size = int(value["window_size"])
    if value.get("window_count") is not None:
        bucket.window_count = int(value["window_count"])
    if value.get("window_cursor") is not None:
        bucket.window_cursor = int(value["window_cursor"])


def _merge_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not existing:
        return list(incoming)
    seen = {(row.get("file_id"), row.get("source_row_id")) for row in existing}
    merged = list(existing)
    for row in incoming:
        key = (row.get("file_id"), row.get("source_row_id"))
        if key not in seen:
            merged.append(row)
            seen.add(key)
    return merged


def _raw_fetch_keys(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    """The source columns the registry grants raw access to, and only those.

    Raw cells are the least filtered thing this system returns, so which ones may be read
    is a registry decision with an audit trail -- not a literal in an executor branch. A
    field contributes its raw_json_keys only when raw_fetch_allowed is set on it.
    """
    keys: list[str] = []
    for file_id in file_ids:
        for spec in catalog.fields_for(file_id):
            if not spec.raw_fetch_allowed:
                continue
            for key in spec.raw_json_keys:
                if key and key not in keys:
                    keys.append(key)
    return keys
