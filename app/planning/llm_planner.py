from __future__ import annotations

from app.llm.base import ReasoningProvider
from app.llm.prompts import PLANNER_SYSTEM, planner_user_prompt
from app.llm.schemas import PlannedQuery
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


class ConstrainedPlanner:
    def __init__(self, provider: ReasoningProvider, catalog: FieldCatalog) -> None:
        self._provider = provider
        self._catalog = catalog

    async def plan(
        self,
        question: str,
        scope: AccessScope,
        memory_text: str = "",
    ) -> QueryPlan:
        scoped_catalog = self._catalog.for_scope(scope)
        drafted = await self._provider.plan_structured(
            system=PLANNER_SYSTEM,
            user=planner_user_prompt(question, scope, scoped_catalog, memory_text),
        )
        return planned_query_to_plan(drafted, scope)


def planned_query_to_plan(drafted: PlannedQuery, scope: AccessScope) -> QueryPlan:
    requested = tuple(int(file_id) for file_id in drafted.file_ids)
    unauthorized = tuple(file_id for file_id in requested if file_id not in scope.allowed_file_ids)
    if unauthorized:
        raise PlanValidationError(
            "requested datasets are not in the current access scope",
            code="access_restricted",
        )
    file_ids = requested
    if not file_ids:
        raise PlanValidationError(
            "planner did not name an authorized dataset",
            code="access_restricted",
        )
    steps: list[PlanStep] = []
    for item in drafted.steps:
        op = PlanOp(item.op)
        where = [
            Predicate(field=pred.field, operator=FilterOperator(pred.operator), value=pred.value)
            for pred in item.where
        ]
        step_files = None
        extra = getattr(item, "file_ids", None)
        if extra:
            step_files = tuple(int(file_id) for file_id in extra)
        steps.append(
            PlanStep(
                id=item.id,
                op=op,
                input=item.input,
                where=where,
                fields=item.fields,
                query=item.query,
                limit=item.limit,
                file_ids=step_files,
            )
        )
    goals = list(drafted.goals)
    if drafted.needs_synthesis and "cause_summary" not in goals:
        goals.append("synthesis")
    return QueryPlan(
        scope=PlanScope(file_ids=file_ids, version_mode="current", authorized_only=True),
        goals=goals,
        steps=steps,
        planner_type="constrained_llm",
    )
