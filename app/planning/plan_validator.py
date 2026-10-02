from __future__ import annotations

import re

from app.config import get_settings
from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import ATTACHMENT_OPS, HISTORY_OPS, PlanOp, PlanStep, QueryPlan
from app.security.access_scope import AccessScope


class PlanValidationError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_plan") -> None:
        super().__init__(message)
        self.code = code


_SQL_VERBS = re.compile(
    r"\b(select|insert|update|delete|drop|alter|truncate)\b",
    re.IGNORECASE,
)
_FORBIDDEN_TABLES = re.compile(
    r"\b(file_data_normalized|file_data|data_config|users|roles|otps|logs|support_[a-z_]+)\b",
    re.IGNORECASE,
)

_SINGLE_INPUT = {
    PlanOp.COUNT,
    PlanOp.FILTER,
    PlanOp.PROJECT,
    PlanOp.LIMIT,
    PlanOp.TOP_N,
    PlanOp.SORT,
    PlanOp.SAMPLE,
    PlanOp.DEDUPLICATE,
    PlanOp.GET_EVIDENCE,
    PlanOp.GET_RAW_FIELDS,
    PlanOp.GET_QUOTE,
    PlanOp.GET_PROVENANCE,
    PlanOp.COUNT_DISTINCT,
    PlanOp.RANK,
    PlanOp.STATS,
    PlanOp.DUPLICATES,
    PlanOp.COMPLETENESS,
    PlanOp.INTERVAL,
    PlanOp.INTERVAL_STATS,
}
_MULTI_SET = {PlanOp.UNION, PlanOp.INTERSECT, PlanOp.DIFFERENCE, PlanOp.COMPARE, PlanOp.RATIO, PlanOp.PERCENTAGE}
_NUMERIC_AGG = {PlanOp.SUM, PlanOp.AVG, PlanOp.MIN, PlanOp.MAX}


def validate_query_plan(
    plan: QueryPlan,
    scope: AccessScope,
    catalog: FieldCatalog,
    *,
    history_enabled: bool | None = None,
    attachments_enabled: bool | None = None,
) -> QueryPlan:
    settings = get_settings()
    history_ok = settings.enable_history_queries if history_enabled is None else history_enabled
    attachments_ok = (
        settings.enable_attachment_content if attachments_enabled is None else attachments_enabled
    )

    _reject_structural_injection(plan)
    _validate_budgets(plan, settings)
    _validate_scope(plan, scope)
    _validate_dag(plan)
    _validate_ops_and_fields(plan, catalog, history_ok=history_ok, attachments_ok=attachments_ok)
    return plan


def _reject_structural_injection(plan: QueryPlan) -> None:
    """Scan ids/field names only. User search values may contain words like 'select'."""
    structural: list[str] = [step.id for step in plan.steps]
    structural.extend(step.op.value for step in plan.steps)
    for step in plan.steps:
        structural.extend(step.fields)
        for name in (step.sort_field, step.companion_field, step.interval_start, step.interval_end):
            if name:
                structural.append(name)
        for predicate in step.where:
            structural.extend(leaf.field for leaf in predicate.leaves())
    for text in structural:
        if _FORBIDDEN_TABLES.search(text):
            raise PlanValidationError(
                "plans cannot name application tables",
                code="forbidden_table",
            )
        if _SQL_VERBS.search(text):
            raise PlanValidationError("plans cannot contain SQL", code="forbidden_sql")


def _validate_budgets(plan: QueryPlan, settings) -> None:
    max_steps = int(getattr(settings, "max_plan_steps", 24))
    max_files = int(getattr(settings, "max_files_per_plan", 8))
    max_inputs = int(getattr(settings, "max_inputs_per_step", 8))
    max_branches = int(getattr(settings, "max_parallel_branches", 8))
    max_query = int(getattr(settings, "max_query_text", 500))
    if len(plan.steps) > max_steps:
        raise PlanValidationError("plan exceeds MAX_PLAN_STEPS", code="budget")
    if len(plan.scope.file_ids) > max_files:
        raise PlanValidationError("plan exceeds MAX_FILES_PER_PLAN", code="budget")
    retrieval = 0
    for step in plan.steps:
        if len(_inputs(step.input)) > max_inputs:
            raise PlanValidationError("step exceeds MAX_INPUTS_PER_STEP", code="budget")
        if step.query and len(step.query) > max_query:
            raise PlanValidationError("query exceeds MAX_QUERY_TEXT", code="budget")
        if step.op in {PlanOp.EXACT_LOOKUP, PlanOp.FULL_TEXT_SEARCH, PlanOp.FUZZY_SEARCH}:
            retrieval += 1
    if retrieval > max_branches:
        raise PlanValidationError("plan exceeds MAX_PARALLEL_BRANCHES", code="budget")


def _validate_scope(plan: QueryPlan, scope: AccessScope) -> None:
    if not plan.scope.authorized_only:
        raise PlanValidationError("plans must remain authorized_only", code="scope")
    if not plan.scope.file_ids:
        raise PlanValidationError("plan scope must include at least one file", code="scope")
    extra = [file_id for file_id in plan.scope.file_ids if file_id not in scope.allowed_file_ids]
    if extra:
        raise PlanValidationError(
            "plan scope is broader than AccessScope",
            code="access_restricted",
        )
    if plan.scope.version_mode == "history" and not scope.history_allowed:
        raise PlanValidationError("history queries are not authorized", code="history_disabled")
    for step in plan.steps:
        if not step.file_ids:
            continue
        if any(file_id not in plan.scope.file_ids for file_id in step.file_ids):
            raise PlanValidationError(
                "step file_ids must be a subset of plan.scope.file_ids",
                code="access_restricted",
            )
        if any(file_id not in scope.allowed_file_ids for file_id in step.file_ids):
            raise PlanValidationError(
                "step file_ids are broader than AccessScope",
                code="access_restricted",
            )


def _validate_dag(plan: QueryPlan) -> None:
    known = {"current_records"}
    for step in plan.steps:
        inputs = _inputs(step.input)
        missing = [item for item in inputs if item not in known]
        if missing:
            raise PlanValidationError(
                f"step {step.id} depends on unknown input {missing}",
                code="dag",
            )
        if step.id in known:
            raise PlanValidationError(f"duplicate step id {step.id}", code="dag")
        known.add(step.id)


def _validate_ops_and_fields(
    plan: QueryPlan,
    catalog: FieldCatalog,
    *,
    history_ok: bool,
    attachments_ok: bool,
) -> None:
    for step in plan.steps:
        if step.op not in PlanOp:
            raise PlanValidationError(f"unsupported op {step.op}", code="operator")
        if step.op in HISTORY_OPS and not history_ok:
            raise PlanValidationError("history operations are disabled", code="history_disabled")
        if step.op in ATTACHMENT_OPS and not attachments_ok:
            raise PlanValidationError("attachment operations are disabled", code="attachments_disabled")
        _validate_op_contract(step)
        file_ids = step.file_ids or plan.scope.file_ids
        for predicate in step.where:
            for leaf in predicate.leaves():
                if leaf.operator is None:
                    raise PlanValidationError("filter leaf has no operator", code="op_contract")
                _validate_predicate(leaf.field, leaf.operator.value, file_ids, catalog)
        for name in (step.companion_field, step.interval_start, step.interval_end):
            if name:
                _validate_field(name, file_ids, catalog)
        for field_name in step.fields:
            if step.op is PlanOp.GROUP_BY and field_name == "file_id":
                continue
            if step.op is PlanOp.GET_EVIDENCE:
                specs = [catalog.resolve_field(file_id, field_name) for file_id in file_ids]
                if any(spec is not None for spec in specs):
                    if any(spec is not None and not spec.evidence_allowed for spec in specs):
                        raise PlanValidationError(
                            f"field {field_name} is not allowed in evidence",
                            code="sensitivity",
                        )
                    continue
            _validate_field(field_name, file_ids, catalog)
        if step.sort_field:
            for spec in _validate_field(step.sort_field, file_ids, catalog):
                if not spec.sortable:
                    raise PlanValidationError(
                        f"field {step.sort_field} is not sortable on file {spec.file_id}",
                        code="operator",
                    )
        if step.op in _NUMERIC_AGG or step.op is PlanOp.COUNT_DISTINCT:
            for field_name in step.fields:
                for spec in _validate_field(field_name, file_ids, catalog):
                    _validate_aggregate_capability(step.op, spec)


def _validate_op_contract(step: PlanStep) -> None:
    inputs = _inputs(step.input)
    if step.op is PlanOp.FILTER and not step.where:
        raise PlanValidationError("FILTER requires predicates", code="op_contract")
    if step.op in _SINGLE_INPUT and len(inputs) != 1:
        raise PlanValidationError(f"{step.op.value} requires exactly 1 input", code="op_contract")
    if step.op in _MULTI_SET and len(inputs) < 2:
        raise PlanValidationError(f"{step.op.value} requires at least 2 inputs", code="op_contract")
    if step.op is PlanOp.COUNT_DISTINCT and len(step.fields) != 1:
        raise PlanValidationError("COUNT_DISTINCT requires exactly 1 field", code="op_contract")
    if step.op is PlanOp.SORT and not step.sort_field:
        raise PlanValidationError("SORT requires sort_field", code="op_contract")
    if step.op in {PlanOp.LIMIT, PlanOp.TOP_N, PlanOp.SAMPLE} and not step.limit:
        raise PlanValidationError(f"{step.op.value} requires limit", code="op_contract")
    if (step.window_anchor is None) != (step.window_size is None):
        raise PlanValidationError(
            "result window requires both anchor and size",
            code="op_contract",
        )
    if step.window_anchor is not None and step.op not in {PlanOp.LIMIT, PlanOp.PROJECT}:
        raise PlanValidationError(
            "result windows are only valid on LIMIT and PROJECT",
            code="op_contract",
        )
    if step.op in _NUMERIC_AGG and len(step.fields) != 1:
        raise PlanValidationError(f"{step.op.value} requires one numeric field", code="op_contract")
    if step.op is PlanOp.RANK and not step.fields:
        raise PlanValidationError("RANK requires at least one group field", code="op_contract")
    if step.op is PlanOp.RANK and len(step.fields) > 2:
        raise PlanValidationError("RANK supports at most two group fields", code="op_contract")
    if step.op in {PlanOp.STATS, PlanOp.DUPLICATES} and len(step.fields) != 1:
        raise PlanValidationError(f"{step.op.value} requires exactly 1 field", code="op_contract")
    if step.op in {PlanOp.INTERVAL, PlanOp.INTERVAL_STATS} and not (
        step.interval_start and step.interval_end
    ):
        raise PlanValidationError(
            f"{step.op.value} requires interval_start and interval_end", code="op_contract"
        )


_NUMERIC_TYPES = frozenset({"number", "numeric", "integer", "float", "int"})
_ORDERED_TYPES = _NUMERIC_TYPES | {"date"}


def _validate_aggregate_capability(op: PlanOp, spec) -> None:
    kind = (spec.semantic_type or "text").lower()
    if op in {PlanOp.SUM, PlanOp.AVG}:
        if not spec.aggregatable or kind not in _NUMERIC_TYPES:
            raise PlanValidationError(
                f"{op.value} is not allowed on {spec.semantic_field}",
                code="operator",
            )
    if op in {PlanOp.MIN, PlanOp.MAX}:
        if kind not in _ORDERED_TYPES:
            raise PlanValidationError(
                f"{op.value} is not allowed on {spec.semantic_field}",
                code="operator",
            )
    if op is PlanOp.COUNT_DISTINCT and not spec.aggregatable and kind not in {"entity", "text", "date"}:
        raise PlanValidationError(
            f"COUNT_DISTINCT is not allowed on {spec.semantic_field}",
            code="operator",
        )


def _validate_predicate(field: str, operator: str, file_ids: tuple[int, ...], catalog: FieldCatalog) -> None:
    for spec in _validate_field(field, file_ids, catalog):
        if operator not in spec.allowed_operators:
            raise PlanValidationError(
                f"operator {operator} is not allowed on {field} for file {spec.file_id}",
                code="operator",
            )


def _validate_field(field: str, file_ids: tuple[int, ...], catalog: FieldCatalog):
    resolved = [catalog.resolve_field(file_id, field) for file_id in file_ids]
    missing = [file_id for file_id, spec in zip(file_ids, resolved, strict=False) if spec is None]
    if missing:
        raise PlanValidationError(
            f"unknown semantic field {field} on files {missing}",
            code="unknown_field",
        )
    return [item for item in resolved if item is not None]


def _inputs(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)
