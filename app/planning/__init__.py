"""Safe natural-language query planning.

Preferred entry point: ``plan_user_turn``. Natural language is compiled to a
TurnPlan; trusted code builds the executable QueryPlan.
"""

from app.planning.catalog import FieldCatalog, static_catalog
from app.planning.plan_schema import QueryPlan
from app.planning.plan_validator import PlanValidationError, validate_query_plan
from app.planning.router import PlanRouteResult, plan_user_turn
from app.planning.semantic_compiler import SemanticCompiler
from app.planning.timing import PlanningTimings
from app.planning.turn_schema import TurnPlan

__all__ = [
    "FieldCatalog",
    "PlanRouteResult",
    "PlanValidationError",
    "PlanningTimings",
    "QueryPlan",
    "SemanticCompiler",
    "TurnPlan",
    "plan_user_turn",
    "static_catalog",
    "validate_query_plan",
]
