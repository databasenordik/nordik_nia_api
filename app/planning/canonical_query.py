from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import (
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)
from app.planning.plan_validator import PlanValidationError
from app.security.access_scope import AccessScope


class TurnType(StrEnum):
    CONVERSATION = "conversation"
    RESEARCH = "research"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"


class CanonicalGoal(StrEnum):
    COUNT = "count"
    LIST = "list"
    DISTINCT = "distinct"
    SYNTHESIS = "synthesis"
    DESCRIBE_AVAILABLE_DATA = "describe_available_data"


class RetrievalMode(StrEnum):
    AUTO = "auto"
    EXACT = "exact"
    FTS = "fts"
    FUZZY = "fuzzy"


class CanonicalFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    operator: FilterOperator
    value: Any = None


class CanonicalSort(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: Literal["ASC", "DESC"] = "ASC"


class CanonicalQuery(BaseModel):
    """One structured interpretation of a user turn (the CanonicalTurn IR).

    The LLM Turn Interpreter emits this. It is intentionally less expressive
    than QueryPlan. Conversation and catalog questions are answered from this
    object; only research/follow-up turns become a QueryPlan.
    """

    model_config = ConfigDict(extra="forbid")

    turn_type: TurnType = Field(validation_alias=AliasChoices("turn_type", "type"))
    intent: str | None = Field(default=None, max_length=80)
    canonical_text: str = Field(default="", max_length=1000)
    file_ids: tuple[int, ...] = Field(
        default=(),
        validation_alias=AliasChoices("file_ids", "datasets"),
    )
    goal: CanonicalGoal | None = None
    filters: tuple[CanonicalFilter, ...] = ()
    search_text: str | None = Field(default=None, max_length=500)
    retrieval_mode: RetrievalMode = RetrievalMode.AUTO
    projection_fields: tuple[str, ...] = ()
    group_by: tuple[str, ...] = ()
    sort: CanonicalSort | None = None
    limit: int | None = Field(default=None, ge=1, le=50)
    needs_synthesis: bool = False
    requires_full_planner: bool = False
    unresolved: tuple[str, ...] = ()
    clarification_question: str | None = Field(
        default=None,
        max_length=500,
        validation_alias=AliasChoices("clarification_question", "clarification"),
    )
    answer_directly: bool = False
    direct_answer: str | None = Field(default=None, max_length=1000)
    response_text: str | None = Field(default=None, max_length=1000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("file_ids", mode="before")
    @classmethod
    def _ids(cls, value: Any) -> tuple[int, ...]:
        if value is None:
            return ()
        return tuple(int(item) for item in value)

    @property
    def spoken_or_direct_text(self) -> str | None:
        text = (self.direct_answer or self.response_text or "").strip()
        return text or None


# Public name used by the turn interpreter. CanonicalQuery remains the model.
CanonicalTurn = CanonicalQuery


class CanonicalQueryError(ValueError):
    pass


def canonical_query_to_plan(
    query: CanonicalQuery,
    scope: AccessScope,
    catalog: FieldCatalog,
    *,
    min_confidence: float = 0.80,
) -> QueryPlan:
    if query.turn_type not in {TurnType.RESEARCH, TurnType.FOLLOW_UP}:
        raise CanonicalQueryError("canonical query is not a research turn")
    if query.goal is CanonicalGoal.DESCRIBE_AVAILABLE_DATA:
        raise CanonicalQueryError("describe_available_data is answered from the catalog")
    if query.requires_full_planner:
        raise CanonicalQueryError("canonical query requires the full planner")
    if query.unresolved:
        raise CanonicalQueryError("canonical query contains unresolved meaning")
    if query.confidence < min_confidence:
        raise CanonicalQueryError("canonical query confidence is below threshold")
    if query.goal is None:
        raise CanonicalQueryError("canonical research query has no goal")

    scoped = catalog.for_scope(scope)
    file_ids = tuple(query.file_ids)
    if not file_ids:
        raise CanonicalQueryError("canonical research query did not select a dataset")
    if any(file_id not in scope.allowed_file_ids for file_id in file_ids):
        raise PlanValidationError("canonical query exceeded AccessScope", code="access_restricted")
    if any(scoped.dataset(file_id) is None for file_id in file_ids):
        raise PlanValidationError("canonical query selected a hidden dataset", code="access_restricted")

    steps: list[PlanStep] = [
        PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records")
    ]
    current = "scope_current"

    if query.filters:
        steps.append(
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input=current,
                where=[
                    Predicate(field=item.field, operator=item.operator, value=item.value)
                    for item in query.filters
                ],
            )
        )
        current = "matches"

    if query.search_text:
        current = _append_retrieval(
            steps,
            current=current,
            query_text=query.search_text,
            mode=query.retrieval_mode,
        )

    if query.group_by:
        for field in query.group_by:
            _require_field_on_all(file_ids, field, scoped)
        steps.append(PlanStep(id="grouped", op=PlanOp.GROUP_BY, input=current, fields=list(query.group_by)))
        current = "grouped"

    if query.sort is not None:
        _require_field_on_all(file_ids, query.sort.field, scoped)
        steps.append(
            PlanStep(
                id="sorted",
                op=PlanOp.SORT,
                input=current,
                sort_field=query.sort.field,
                sort_direction=query.sort.direction,
            )
        )
        current = "sorted"

    if query.limit is not None:
        steps.append(PlanStep(id="limited", op=PlanOp.LIMIT, input=current, limit=query.limit))
        current = "limited"

    goals: list[str] = [query.goal.value]
    if query.goal is CanonicalGoal.COUNT:
        steps.append(PlanStep(id="result", op=PlanOp.COUNT, input=current))
    elif query.goal is CanonicalGoal.DISTINCT:
        if len(query.group_by) != 1:
            raise CanonicalQueryError("distinct requires exactly one group_by field")
        # GROUP_BY is the result itself; rename through a projection-free terminal.
        # Existing executors treat the final named step as the result.
        steps[-1] = steps[-1].model_copy(update={"id": "result"})
    else:
        projection = list(query.projection_fields) or _default_projection(file_ids, scoped)
        if not projection:
            raise CanonicalQueryError("no common projection field exists for selected datasets")
        for field in projection:
            _require_field_on_all(file_ids, field, scoped)
        steps.append(PlanStep(id="result", op=PlanOp.PROJECT, input=current, fields=projection))
        if query.goal is CanonicalGoal.SYNTHESIS or query.needs_synthesis:
            evidence = [field for field in projection if _evidence_allowed_on_any(file_ids, field, scoped)]
            for candidate in ("notes", "cause_of_death"):
                if candidate not in evidence and _field_exists_on_all(file_ids, candidate, scoped):
                    if _evidence_allowed_on_any(file_ids, candidate, scoped):
                        evidence.append(candidate)
            if evidence:
                steps.append(
                    PlanStep(
                        id="evidence",
                        op=PlanOp.GET_EVIDENCE,
                        input="result",
                        fields=evidence,
                        limit=min(query.limit or 8, 8),
                    )
                )
            goals.append("synthesis")

    return QueryPlan(
        scope=PlanScope(file_ids=file_ids, version_mode="current", authorized_only=True),
        goals=list(dict.fromkeys(goals)),
        steps=steps,
        planner_type="semantic_interpreter",
    )


def _append_retrieval(
    steps: list[PlanStep],
    *,
    current: str,
    query_text: str,
    mode: RetrievalMode,
) -> str:
    if mode is RetrievalMode.EXACT:
        steps.append(PlanStep(id="exact", op=PlanOp.EXACT_LOOKUP, input=current, query=query_text))
        return "exact"
    if mode is RetrievalMode.FTS:
        steps.append(PlanStep(id="fts", op=PlanOp.FULL_TEXT_SEARCH, input=current, query=query_text))
        return "fts"
    if mode is RetrievalMode.FUZZY:
        steps.append(PlanStep(id="fuzzy", op=PlanOp.FUZZY_SEARCH, input=current, query=query_text))
        return "fuzzy"

    # AUTO: lexical recall with explicit set semantics. The executor may cancel
    # fuzzy work if a unique high-confidence exact/FTS result is already enough.
    steps.append(PlanStep(id="fts", op=PlanOp.FULL_TEXT_SEARCH, input=current, query=query_text))
    steps.append(PlanStep(id="fuzzy", op=PlanOp.FUZZY_SEARCH, input=current, query=query_text))
    steps.append(PlanStep(id="fused", op=PlanOp.UNION, input=["fts", "fuzzy"]))
    steps.append(PlanStep(id="unique", op=PlanOp.DEDUPLICATE, input="fused"))
    return "unique"


def _default_projection(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    result: list[str] = []
    for candidate in ("student_name", "community"):
        if _field_exists_on_all(file_ids, candidate, catalog):
            result.append(candidate)
    return result


def _field_exists_on_all(file_ids: tuple[int, ...], field: str, catalog: FieldCatalog) -> bool:
    return bool(file_ids) and all(catalog.resolve_field(file_id, field) is not None for file_id in file_ids)


def _require_field_on_all(file_ids: tuple[int, ...], field: str, catalog: FieldCatalog) -> None:
    missing = [file_id for file_id in file_ids if catalog.resolve_field(file_id, field) is None]
    if missing:
        raise CanonicalQueryError(f"field {field} is unavailable on datasets {missing}")


def _evidence_allowed_on_any(file_ids: tuple[int, ...], field: str, catalog: FieldCatalog) -> bool:
    specs = [catalog.resolve_field(file_id, field) for file_id in file_ids]
    return any(spec is not None and spec.evidence_allowed for spec in specs)
