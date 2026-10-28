from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlanOp(StrEnum):
    USE_DATASET = "USE_DATASET"
    USE_CURRENT_VERSION = "USE_CURRENT_VERSION"
    USE_HISTORY_VERSION = "USE_HISTORY_VERSION"
    USE_PREVIOUS_RESULT = "USE_PREVIOUS_RESULT"
    USE_TOPIC_FRAME = "USE_TOPIC_FRAME"
    FILTER = "FILTER"
    PROJECT = "PROJECT"
    COUNT = "COUNT"
    COUNT_DISTINCT = "COUNT_DISTINCT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    SORT = "SORT"
    SAMPLE = "SAMPLE"
    LIMIT = "LIMIT"
    GROUP_BY = "GROUP_BY"
    EXISTS = "EXISTS"
    DATE_RANGE = "DATE_RANGE"
    TOP_N = "TOP_N"
    EXACT_LOOKUP = "EXACT_LOOKUP"
    ALIAS_LOOKUP = "ALIAS_LOOKUP"
    REFERENCE_LOOKUP = "REFERENCE_LOOKUP"
    FULL_TEXT_SEARCH = "FULL_TEXT_SEARCH"
    FUZZY_SEARCH = "FUZZY_SEARCH"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    SEMANTIC_CLASSIFY = "SEMANTIC_CLASSIFY"
    RESOLVE_ENTITY = "RESOLVE_ENTITY"
    RESOLVE_PRONOUN = "RESOLVE_PRONOUN"
    RESOLVE_ORDINAL = "RESOLVE_ORDINAL"
    USE_ACTIVE_QUERY = "USE_ACTIVE_QUERY"
    APPLY_CONSTRAINT_EDIT = "APPLY_CONSTRAINT_EDIT"
    INTERSECT = "INTERSECT"
    UNION = "UNION"
    DIFFERENCE = "DIFFERENCE"
    EXCLUDE = "EXCLUDE"
    RATIO = "RATIO"
    PERCENTAGE = "PERCENTAGE"
    COMPARE = "COMPARE"
    RANK = "RANK"
    NORMALIZE = "NORMALIZE"
    GET_EVIDENCE = "GET_EVIDENCE"
    GET_RAW_FIELDS = "GET_RAW_FIELDS"
    GET_QUOTE = "GET_QUOTE"
    GET_PROVENANCE = "GET_PROVENANCE"
    CHECK_CONFLICTS = "CHECK_CONFLICTS"
    DEDUPLICATE = "DEDUPLICATE"
    GET_APPROVED_ATTACHMENTS = "GET_APPROVED_ATTACHMENTS"
    STATS = "STATS"
    INTERVAL = "INTERVAL"
    INTERVAL_STATS = "INTERVAL_STATS"
    COMPLETENESS = "COMPLETENESS"
    DUPLICATES = "DUPLICATES"


class FilterOperator(StrEnum):
    EQUALS = "EQUALS"
    STARTS_WITH = "STARTS_WITH"
    CONTAINS = "CONTAINS"
    IN = "IN"
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"
    IS_UNKNOWN = "IS_UNKNOWN"
    YEAR_EQUALS = "YEAR_EQUALS"
    BEFORE = "BEFORE"
    AFTER = "AFTER"
    DATE_RANGE = "DATE_RANGE"
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    NUMBER_RANGE = "NUMBER_RANGE"
    IS_KNOWN = "IS_KNOWN"
    FULL_TEXT_SEARCH = "FULL_TEXT_SEARCH"
    FUZZY_SEARCH = "FUZZY_SEARCH"
    GET_QUOTE = "GET_QUOTE"
    NOT_EQUALS = "NOT_EQUALS"
    NOT_CONTAINS = "NOT_CONTAINS"
    CONTAINS_ANY = "CONTAINS_ANY"
    NOT_CONTAINS_ANY = "NOT_CONTAINS_ANY"
    STARTS_WITH_ANY = "STARTS_WITH_ANY"
    NOT_IN = "NOT_IN"


DETERMINISTIC_OPS = frozenset(
    {
        PlanOp.USE_DATASET,
        PlanOp.USE_CURRENT_VERSION,
        PlanOp.FILTER,
        PlanOp.PROJECT,
        PlanOp.COUNT,
        PlanOp.COUNT_DISTINCT,
        PlanOp.SORT,
        PlanOp.SAMPLE,
        PlanOp.LIMIT,
        PlanOp.GROUP_BY,
        PlanOp.EXISTS,
        PlanOp.TOP_N,
        PlanOp.EXACT_LOOKUP,
        PlanOp.FULL_TEXT_SEARCH,
        PlanOp.FUZZY_SEARCH,
        PlanOp.UNION,
        PlanOp.DEDUPLICATE,
        PlanOp.GET_EVIDENCE,
        PlanOp.GET_QUOTE,
        PlanOp.GET_PROVENANCE,
        PlanOp.COMPARE,
        PlanOp.COUNT_DISTINCT,
        PlanOp.SUM,
        PlanOp.AVG,
        PlanOp.MIN,
        PlanOp.MAX,
        PlanOp.STATS,
        PlanOp.INTERVAL,
        PlanOp.INTERVAL_STATS,
        PlanOp.COMPLETENESS,
        PlanOp.DUPLICATES,
        PlanOp.RANK,
        PlanOp.PERCENTAGE,
        PlanOp.RATIO,
    }
)

HISTORY_OPS = frozenset({PlanOp.USE_HISTORY_VERSION})
ATTACHMENT_OPS = frozenset({PlanOp.GET_APPROVED_ATTACHMENTS})
FORBIDDEN_PLAN_TOKENS = (
    "select ",
    "insert ",
    "update ",
    "delete ",
    "drop ",
    "alter ",
    "truncate ",
    "file_data",
    "file_data_normalized",
    "data_config",
    " from ",
    "users",
    "otps",
    "support_",
)


class PlanScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_ids: tuple[int, ...]
    version_mode: Literal["current", "history"] = "current"
    authorized_only: bool = True

    @field_validator("file_ids", mode="before")
    @classmethod
    def _ids(cls, value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in value)


class Predicate(BaseModel):
    """One filter condition, or a boolean group of them.

    A leaf carries ``field`` + ``operator``. A group carries ``op`` ("and"/"or"/"not")
    and ``items``. Keeping both shapes in one model means every ``where`` list stays a
    ``list[Predicate]`` and callers that only understand leaves keep working.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = ""
    operator: FilterOperator | None = None
    value: Any = None
    op: Literal["and", "or", "not"] | None = None
    items: list["Predicate"] = Field(default_factory=list)

    @model_validator(mode="after")
    def _leaf_or_group(self) -> "Predicate":
        if self.op is not None:
            if not self.items:
                raise ValueError("filter group requires at least one item")
            return self
        if not self.field or self.operator is None:
            raise ValueError("filter leaf requires a field and an operator")
        return self

    def is_group(self) -> bool:
        return self.op is not None

    def leaves(self) -> list["Predicate"]:
        """Flatten to leaf conditions, for validation and human-readable summaries."""
        if not self.is_group():
            return [self]
        found: list[Predicate] = []
        for item in self.items:
            found.extend(item.leaves())
        return found

    def payload(self) -> dict[str, Any]:
        """Serialize to the boolean-tree shape the data gateway sends to Postgres."""
        if self.is_group():
            return {"op": self.op, "items": [item.payload() for item in self.items]}
        return {
            "field": self.field,
            "operator": self.operator.value if self.operator is not None else None,
            "value": self.value,
        }


def all_of(*items: Predicate) -> Predicate:
    return Predicate(op="and", items=list(items))


def any_of(*items: Predicate) -> Predicate:
    return Predicate(op="or", items=list(items))


def not_(item: Predicate) -> Predicate:
    return Predicate(op="not", items=[item])


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    op: PlanOp
    input: str | list[str] | None = None
    where: list[Predicate] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)
    value_part: Literal["raw", "year", "decade", "first_token", "last_token", "number", "base_name"] | None = None
    secondary_value_part: Literal[
        "raw", "year", "decade", "first_token", "last_token", "number", "base_name"
    ] | None = None
    having_min_count: int | None = Field(default=None, ge=1)
    top_n: int | None = Field(default=None, ge=1, le=500)
    per_group_top_n: int | None = Field(default=None, ge=1, le=50)
    include_missing: bool = False
    companion_field: str | None = Field(default=None, max_length=64)
    interval_start: str | None = Field(default=None, max_length=64)
    interval_end: str | None = Field(default=None, max_length=64)
    interval_min_days: float | None = None
    interval_max_days: float | None = None
    direction: Literal["asc", "desc"] | None = None
    sort_field: str | None = None
    sort_direction: Literal["ASC", "DESC"] = "ASC"
    limit: int | None = Field(default=None, ge=1, le=500)
    offset: int | None = Field(default=None, ge=0)
    window_anchor: Literal["start", "end"] | None = None
    window_size: int | None = Field(default=None, ge=1, le=5000)
    window_cursor: int = Field(default=0, ge=0)
    file_ids: tuple[int, ...] | None = None
    query: str | None = Field(default=None, max_length=500)
    action_id: str | None = Field(default=None, max_length=16)
    action_goal: str | None = Field(default=None, max_length=32)

    @field_validator("file_ids", mode="before")
    @classmethod
    def _ids(cls, value: Any) -> tuple[int, ...] | None:
        if value is None:
            return None
        return tuple(int(item) for item in value)

    @field_validator("input", mode="before")
    @classmethod
    def _input(cls, value: Any) -> str | list[str] | None:
        if value is None or isinstance(value, str | list):
            return value
        raise ValueError("input must be a step id or list of step ids")


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: PlanScope
    goals: list[str] = Field(default_factory=list)
    steps: list[PlanStep]
    planner_type: Literal[
        "deterministic",
        "semantic_normalizer",
        "semantic_interpreter",
        "semantic_compiler",
        "constrained_llm",
        "ai_planner",
    ] = "deterministic"
    schema_version: int = 1

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "QueryPlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("QueryPlan step ids must be unique")
        if not self.steps:
            raise ValueError("QueryPlan must contain at least one step")
        return self

    def op_names(self) -> list[str]:
        return [step.op.value for step in self.steps]

    def step_map(self) -> dict[str, PlanStep]:
        return {step.id: step for step in self.steps}
