from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ALLOWED_OPERATORS = frozenset(
    {
        "EQUALS",
        "STARTS_WITH",
        "CONTAINS",
        "IN",
        "IS_TRUE",
        "IS_FALSE",
        "IS_UNKNOWN",
        "IS_KNOWN",
        "YEAR_EQUALS",
        "BEFORE",
        "AFTER",
        "DATE_RANGE",
        "GREATER_THAN",
        "LESS_THAN",
        "NUMBER_RANGE",
        "FULL_TEXT_SEARCH",
        "FUZZY_SEARCH",
        "GET_QUOTE",
        "NOT_EQUALS",
        "NOT_CONTAINS",
        "CONTAINS_ANY",
        "NOT_CONTAINS_ANY",
        "STARTS_WITH_ANY",
        "NOT_IN",
    }
)
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 50
MAX_WINDOW_SIZE = 5000
MAX_FILTERS_PER_ACTION = 16
MAX_COMPARE_BRANCHES = 8
MAX_COMPARE_FILTERS = 8
MAX_FILTER_VALUE_ITEMS = 32
MAX_FILTER_VALUE_TEXT = 500
MAX_OFFSET = 1_000_000
MAX_INTERVAL_DAYS = 1_000_000
MAX_UNRESOLVED_ITEMS = 16
PRESENTATION_FORMATS = frozenset({"natural", "numbered_list", "bullets", "table"})
TRUSTED_CONVERSATIONAL_INTENTS = frozenset(
    {
        "assistant_identity",
        "audio_check",
        "greeting",
        "thanks",
        "farewell",
        "capability",
        "acknowledgement",
        "user_identity",
        "user_introduction",
        "correction",
        "general_conversation",
        "dataset_not_selected",
    }
)
ALLOWED_GOALS = frozenset(
    {
        "count",
        "list",
        "distinct",
        "aggregate",
        "compare",
        "search",
        "quote",
        "provenance",
        "rank",
        "stats",
        "duplicates",
        "percentage",
        "interval",
        "completeness",
        "dossier",
    }
)
GOAL_LITERAL = Literal[
    "count",
    "list",
    "distinct",
    "aggregate",
    "compare",
    "search",
    "quote",
    "provenance",
    "rank",
    "stats",
    "duplicates",
    "percentage",
    "interval",
    "completeness",
    "dossier",
]
VALUE_PART_LITERAL = Literal[
    "raw", "year", "decade", "first_token", "last_token", "number", "base_name"
]
# Goals whose answer is a computed value. A paginated page of rows is never an
# acceptable substitute for one of these.
CALCULATION_GOALS = frozenset(
    {"count", "rank", "stats", "percentage", "duplicates", "interval", "completeness", "distinct"}
)
# Goals that are inherently a whole-dataset computation, so carrying no filters is
# normal rather than a sign the compiler produced a conversational dump.
ANALYTIC_GOALS = frozenset(
    {"rank", "stats", "percentage", "duplicates", "interval", "completeness", "distinct", "aggregate"}
)
CONVERSATIONAL_INTENTS = frozenset(
    {
        "assistant_identity",
        "audio_check",
        "greeting",
        "thanks",
        "farewell",
        "capability",
        "describe_available_data",
        "acknowledgement",
        "general_conversation",
        "user_identity",
        "user_introduction",
        "correction",
        "dataset_not_selected",
    }
)
# Structural identifiers only. Never apply this to user/free-form text.
FORBIDDEN_IDENTIFIER_TOKENS = (
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
    "users",
    "otps",
    "support_",
)


def _reject_forbidden_identifier(value: str, *, label: str) -> str:
    lowered = value.lower()
    for token in FORBIDDEN_IDENTIFIER_TOKENS:
        if token in lowered:
            raise ValueError(f"{label} contains a forbidden token")
    return value


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1, max_length=64)
    operator: str
    value: str | int | float | bool | list | None = None

    @field_validator("field")
    @classmethod
    def _field(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or (" " in cleaned and "." in cleaned):
            raise ValueError("field must be a semantic field name")
        return _reject_forbidden_identifier(cleaned, label="field")

    @field_validator("operator")
    @classmethod
    def _operator(cls, value: str) -> str:
        operator = value.strip().upper()
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"unsupported operator {value}")
        return operator

    @field_validator("value")
    @classmethod
    def _bounded_value(cls, value):
        """Keep model-controlled filter payloads finite and reasonably small.

        Filter values are data, not executable structure, but an unbounded list/string can
        still turn one malformed structured response into a large allocation or an enormous
        database parameter. Nested containers are not part of the filter contract.
        """

        def _scalar(item):
            if isinstance(item, str):
                if len(item) > MAX_FILTER_VALUE_TEXT:
                    raise ValueError("filter value text is too long")
                return item
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("filter numeric values must be finite")
            if item is None or isinstance(item, int | float | bool):
                return item
            raise ValueError("filter list values must be scalar")

        if isinstance(value, list):
            if len(value) > MAX_FILTER_VALUE_ITEMS:
                raise ValueError("filter value list is too long")
            return [_scalar(item) for item in value]
        return _scalar(value)


class FilterGroupSpec(BaseModel):
    """One conjunction in a bounded disjunctive-normal-form filter expression.

    Every filter inside a group is ANDed. Multiple groups on a query are ORed.
    This represents mixed Boolean expressions without exposing a recursive schema to
    the structured-output provider.
    """

    model_config = ConfigDict(extra="forbid")

    filters: list[FilterSpec] = Field(min_length=1, max_length=8)


class AggregateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Literal[
        "count", "count_distinct", "sum", "avg", "min", "max", "median", "mode", "stats"
    ]
    field: str | None = Field(default=None, max_length=64)

    @field_validator("field")
    @classmethod
    def _field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_forbidden_identifier(value.strip(), label="aggregate.field")


class CompareBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, max_length=80)
    filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_COMPARE_FILTERS)
    search_text: str | None = Field(default=None, max_length=500)


class ResultWindow(BaseModel):
    """A bounded selection within the full, deterministically sorted result set."""

    model_config = ConfigDict(extra="forbid")

    anchor: Literal["start", "end"] = "start"
    size: int = Field(ge=1, le=MAX_WINDOW_SIZE)
    cursor: int = Field(default=0, ge=0, le=MAX_OFFSET)


class QueryAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["query"] = "query"
    datasets: list[int] = Field(default_factory=list)
    goal: GOAL_LITERAL
    filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    # Mixed AND/OR expressions use bounded disjunctive normal form: every
    # FilterGroupSpec is a conjunction and the groups are ORed. Keep this separate
    # from the flat filters surface so simple turns stay compact and cache-friendly.
    filter_groups: list[FilterGroupSpec] = Field(default_factory=list, max_length=8)
    # "or" turns the flat filter list into a disjunction. Multi-term matching within one
    # field is expressed with CONTAINS_ANY instead, so a family of historical spellings
    # stays a single condition.
    filter_logic: Literal["and", "or"] = "and"
    denominator_filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    search_text: str | None = Field(default=None, max_length=500)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    group_value_part: VALUE_PART_LITERAL | None = None
    secondary_group_value_part: VALUE_PART_LITERAL | None = None
    having_min_count: int | None = Field(default=None, ge=1, le=MAX_OFFSET)
    top_n: int | None = Field(default=None, ge=1, le=500)
    per_group_top_n: int | None = Field(default=None, ge=1, le=50)
    include_missing: bool = False
    companion_field: str | None = Field(default=None, max_length=64)
    stats_value_part: VALUE_PART_LITERAL | None = None
    interval_start: str | None = Field(default=None, max_length=64)
    interval_end: str | None = Field(default=None, max_length=64)
    interval_min_days: float | None = Field(default=None, ge=0, le=MAX_INTERVAL_DAYS, allow_inf_nan=False)
    interval_max_days: float | None = Field(default=None, ge=0, le=MAX_INTERVAL_DAYS, allow_inf_nan=False)
    sort_by: str | None = Field(default=None, max_length=64)
    sort_direction: Literal["asc", "desc"] | None = None
    sample: bool = False
    requested_fields: list[str] = Field(default_factory=list, max_length=64)
    window: ResultWindow | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    offset: int | None = Field(default=None, ge=0, le=MAX_OFFSET)
    exhaustive: bool = False
    presentation: Literal["natural", "numbered_list", "bullets", "table"] | None = None
    verify_previous: bool = False
    aggregate: AggregateSpec | None = None
    compare: list[CompareBranch] = Field(default_factory=list, max_length=MAX_COMPARE_BRANCHES)

    @model_validator(mode="after")
    def _one_filter_surface(self) -> QueryAction:
        if self.filters and self.filter_groups:
            raise ValueError("query cannot mix flat filters with filter_groups")
        if self.filter_groups and self.filter_logic != "and":
            raise ValueError("filter_logic is only valid for flat filters; filter_groups are OR-of-AND")
        return self

    @field_validator("group_by")
    @classmethod
    def _group_by(cls, value: list[str]) -> list[str]:
        return [_reject_forbidden_identifier(item, label="group_by") for item in value]

    @field_validator("sort_by")
    @classmethod
    def _sort_by(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_forbidden_identifier(value, label="sort_by")

    @field_validator("requested_fields")
    @classmethod
    def _requested_fields(cls, value: list[str]) -> list[str]:
        return [
            _reject_forbidden_identifier(item.strip(), label="requested_fields")
            for item in value
            if item.strip()
        ]


class ModifyPreviousAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["modify_previous"] = "modify_previous"
    changes: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    # replace keeps the historical follow-up behavior ("instead use 1910").
    # refine means the new conditions narrow the inherited population ("which of those...").
    change_mode: Literal["replace", "refine"] = "replace"
    goal: GOAL_LITERAL | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    offset: int | None = Field(default=None, ge=0, le=MAX_OFFSET)
    sort_by: str | None = Field(default=None, max_length=64)
    sort_direction: Literal["asc", "desc"] | None = None
    sample: bool | None = None
    requested_fields: list[str] | None = Field(default=None, max_length=64)
    window: ResultWindow | None = None
    exhaustive: bool | None = None
    presentation: Literal["natural", "numbered_list", "bullets", "table"] | None = None
    page: Literal["next", "previous", "first"] | None = None
    restore_frame: str | None = Field(default=None, max_length=80)
    # Regroup the previous answer by a derived form of the same field,
    # e.g. base_name to merge "Chisasibi #66" into "Chisasibi".
    group_value_part: VALUE_PART_LITERAL | None = None

    @model_validator(mode="after")
    def _has_an_edit(self) -> ModifyPreviousAction:
        if (
            self.changes
            or self.goal
            or self.limit is not None
            or self.offset is not None
            or self.sort_by
            or self.sort_direction
            or self.sample is not None
            or self.requested_fields is not None
            or self.window is not None
            or self.exhaustive is not None
            or self.presentation
            or self.page
            or self.restore_frame
            or self.group_value_part
        ):
            return self
        raise ValueError("modify_previous must change the previous action")

    @field_validator("requested_fields")
    @classmethod
    def _requested_fields(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [
            _reject_forbidden_identifier(item.strip(), label="requested_fields")
            for item in value
            if item.strip()
        ]


class VerifyPreviousAction(BaseModel):
    """Challenge or re-check the previous result. Trusted code re-runs it."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["verify_previous"] = "verify_previous"
    target_action_id: str | None = Field(default=None, max_length=16)


class ClarifyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["clarify"] = "clarify"
    question: str = Field(min_length=1, max_length=500)


class ConversationAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["respond"] = "respond"
    intent: Literal[
        "assistant_identity",
        "audio_check",
        "greeting",
        "thanks",
        "farewell",
        "capability",
        "describe_available_data",
        "acknowledgement",
        "general_conversation",
        "user_identity",
        "user_introduction",
        "correction",
        "dataset_not_selected",
    ]
    response: str = Field(default="", max_length=1000)


class ReferenceAction(BaseModel):
    """Point at a previously listed result. Trusted code resolves the row."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["reference"] = "reference"
    selector: Literal["first", "second", "third", "fourth", "fifth", "last"] | None = None
    index: int | None = Field(default=None, ge=0, le=49)
    target: Literal["row", "action"] | None = None


Action = Annotated[
    QueryAction
    | ModifyPreviousAction
    | VerifyPreviousAction
    | ClarifyAction
    | ConversationAction
    | ReferenceAction,
    Field(discriminator="type"),
]


class ActiveQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets: list[int] = Field(max_length=8)
    goal: str
    filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    filter_groups: list[FilterGroupSpec] = Field(default_factory=list, max_length=8)
    denominator_filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    group_by: list[str] = Field(default_factory=list, max_length=2)
    sort_by: str | None = None
    sort_direction: Literal["asc", "desc"] | None = None
    requested_fields: list[str] = Field(default_factory=list, max_length=64)
    window: ResultWindow | None = None
    result_set_id: str | None = None
    search_text: str | None = None
    filter_logic: Literal["and", "or"] = "and"
    group_value_part: str | None = None
    secondary_group_value_part: str | None = None
    having_min_count: int | None = None
    top_n: int | None = None
    per_group_top_n: int | None = None
    include_missing: bool = False
    companion_field: str | None = None
    stats_value_part: str | None = None
    interval_start: str | None = None
    interval_end: str | None = None
    sample: bool = False
    limit: int | None = None
    offset: int | None = None
    exhaustive: bool = False
    presentation: str | None = None
    aggregate: AggregateSpec | None = None
    compare: list[CompareBranch] = Field(default_factory=list, max_length=MAX_COMPARE_BRANCHES)
    interval_min_days: float | None = None
    interval_max_days: float | None = None
    total_count: int | None = None
    returned_count: int | None = None
    has_more: bool = False
    frame_key: str | None = None
    topic: str | None = None
    action_id: str | None = None


class TurnPlan(BaseModel):
    """Semantic meaning of one user turn. Never executable SQL or table names."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    normalized_request: str = Field(default="", max_length=1000)
    actions: list[Action] = Field(min_length=1, max_length=8)
    final_response: Literal["compiler_response", "deterministic", "llm_synthesis"]
    needs_evidence: bool = False
    needs_explanation: bool = False
    needs_inference: bool = False
    unresolved: list[str] = Field(default_factory=list, max_length=MAX_UNRESOLVED_ITEMS)
    confidence: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _consistent_actions(self) -> TurnPlan:
        if not self.actions:
            raise ValueError("TurnPlan must contain at least one action")
        return self

    def query_actions(self) -> list[QueryAction]:
        return [item for item in self.actions if isinstance(item, QueryAction)]

    def modify_actions(self) -> list[ModifyPreviousAction]:
        return [item for item in self.actions if isinstance(item, ModifyPreviousAction)]

    def clarify_actions(self) -> list[ClarifyAction]:
        return [item for item in self.actions if isinstance(item, ClarifyAction)]

    def respond_actions(self) -> list[ConversationAction]:
        return [item for item in self.actions if isinstance(item, ConversationAction)]

    def reference_actions(self) -> list[ReferenceAction]:
        return [item for item in self.actions if isinstance(item, ReferenceAction)]

    def verify_actions(self) -> list[VerifyPreviousAction]:
        return [item for item in self.actions if isinstance(item, VerifyPreviousAction)]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalized_request": self.normalized_request,
            "actions": [item.model_dump(mode="json") for item in self.actions],
            "final_response": self.final_response,
            "needs_evidence": self.needs_evidence,
            "needs_explanation": self.needs_explanation,
            "needs_inference": self.needs_inference,
            "unresolved": list(self.unresolved),
            "confidence": self.confidence,
        }
