"""Contradiction checks on an unexpanded QueryPlan predicate tree."""

from __future__ import annotations

from app.planning.plan_schema import FilterOperator, Predicate, QueryPlan
from app.planning.plan_validator import PlanValidationError

_SINGLE_VALUED = frozenset(
    {
        FilterOperator.EQUALS,
        FilterOperator.YEAR_EQUALS,
        FilterOperator.IS_TRUE,
        FilterOperator.IS_FALSE,
    }
)


def validate_predicate_contract(plan: QueryPlan) -> QueryPlan:
    for step in plan.steps:
        if step.where:
            # A top-level list is an implicit AND in the gateway.
            _reject_and_branch(list(step.where))
        for predicate in step.where:
            _reject_contradiction(predicate)
    return plan


def _reject_contradiction(predicate: Predicate) -> None:
    if predicate.is_group():
        if (predicate.op or "and").lower() == "and":
            _reject_and_branch(list(predicate.items))
        for item in predicate.items:
            _reject_contradiction(item)
        return
    _reject_and_branch([predicate])


def _reject_and_branch(items: list[Predicate]) -> None:
    leaves = _conjunctive_leaves(items)
    by_field: dict[str, list[Predicate]] = {}
    for leaf in leaves:
        if not leaf.field or leaf.operator is None:
            continue
        by_field.setdefault(leaf.field, []).append(leaf)
    for field, group in by_field.items():
        equals = [
            item
            for item in group
            if item.operator in _SINGLE_VALUED and item.value not in (None, "", [])
        ]
        values = {_norm(item.value) for item in equals}
        if len(values) > 1:
            raise PlanValidationError(
                f"contradictory equalities on {field}",
                code="op_contract",
            )
        required = {_norm(item.value) for item in group if item.operator in {FilterOperator.EQUALS, FilterOperator.CONTAINS, FilterOperator.IN}}
        excluded = {
            _norm(item.value)
            for item in group
            if item.operator in {FilterOperator.NOT_EQUALS, FilterOperator.NOT_CONTAINS, FilterOperator.NOT_IN}
        }
        if required & excluded:
            raise PlanValidationError(
                f"value both required and excluded on {field}",
                code="op_contract",
            )
        operators = {item.operator for item in group}
        if FilterOperator.IS_KNOWN in operators and FilterOperator.IS_UNKNOWN in operators:
            raise PlanValidationError(
                f"IS_KNOWN and IS_UNKNOWN on {field}",
                code="op_contract",
            )


def _conjunctive_leaves(items: list[Predicate]) -> list[Predicate]:
    leaves: list[Predicate] = []
    for item in items:
        if item.is_group() and (item.op or "and").lower() == "and":
            leaves.extend(_conjunctive_leaves(list(item.items)))
        elif not item.is_group():
            leaves.append(item)
    return leaves


def _norm(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(_norm(item) for item in value)
    return " ".join(str(value).casefold().split())
