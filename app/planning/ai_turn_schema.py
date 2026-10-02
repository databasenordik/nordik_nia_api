"""AI-facing planner envelopes. No dataset identifiers are allowed on this surface."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from app.planning.turn_schema import (
    ALLOWED_GOALS,
    ALLOWED_OPERATORS,
    GOAL_LITERAL,
    MAX_COMPARE_BRANCHES,
    MAX_FILTERS_PER_ACTION,
    MAX_INTERVAL_DAYS,
    MAX_OFFSET,
    MAX_UNRESOLVED_ITEMS,
    VALUE_PART_LITERAL,
    AggregateSpec,
    ClarifyAction,
    CompareBranch,
    ConversationAction,
    FilterGroupSpec,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    ReferenceAction,
    ResultWindow,
    TurnPlan,
    VerifyPreviousAction,
    _reject_forbidden_identifier,
)


def _require_discriminator_in_schema(schema: dict[str, Any]) -> None:
    """Make provider-facing JSON Schema match discriminated-union validation."""
    properties = schema.get("properties", {})
    # "confidence" is required of the model, not of Python: code that builds a turn may leave
    # it at the default, but a planner that omits it -- as it often did once the schema stopped
    # showing a default -- was scored 0.0 and every such answer came back as "I want to be sure
    # I understood you". The threshold applied to the stated value is unchanged.
    for discriminator in ("type", "kind", "verdict", "confidence"):
        if discriminator not in properties:
            continue
        required = list(schema.get("required", []))
        if discriminator not in required:
            required.append(discriminator)
            schema["required"] = required


def clip_free_text(value: Any, limit: int) -> Any:
    """Shorten a free-text field the model overfilled instead of rejecting the whole turn.

    The planner restates the request in its own words, and handed a long paste -- a tester's
    table of corrected figures -- it restated all of it. The length limit then failed the
    turn at validation, twice, and the researcher read "couldn't produce a usable plan". These
    fields describe the turn; nothing is computed from their full text, so a restatement cut at
    a sentence boundary serves exactly as well as the whole one.
    """
    if not isinstance(value, str) or len(value) <= limit:
        return value
    cut = value[:limit]
    stop = max(cut.rfind(". "), cut.rfind("\n"))
    return (cut[: stop + 1] if stop >= limit // 2 else cut).rstrip()


# Descriptive text the model writes about a requirement or a review finding. Nothing is
# computed from it, so an overlong note is shortened rather than failing the whole review: a
# 420-character "missing" explanation turned a correct correction into "couldn't produce a
# usable plan".
NoteText = Annotated[
    str,
    BeforeValidator(lambda value: clip_free_text(value, 400)),
    Field(min_length=1, max_length=400),
]


class AIPlannerModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_require_discriminator_in_schema
    )


class PlannedQueryAction(AIPlannerModel):
    """QueryAction without datasets. Trusted code stamps the selected file."""

    type: Literal["query"] = "query"
    goal: GOAL_LITERAL
    filters: list[FilterSpec] = Field(default_factory=list, max_length=MAX_FILTERS_PER_ACTION)
    filter_groups: list[FilterGroupSpec] = Field(default_factory=list, max_length=8)
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
    def _one_filter_surface(self) -> PlannedQueryAction:
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


class PlannedConversationAction(AIPlannerModel):
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

    @field_validator("response", mode="before")
    @classmethod
    def _clip_response(cls, value: Any) -> Any:
        return clip_free_text(value, 1000)

    @model_validator(mode="after")
    def _response_required(self) -> PlannedConversationAction:
        if self.intent == "general_conversation" and not self.response.strip():
            raise ValueError("general_conversation requires a non-empty response")
        return self


class PlannedModifyPreviousAction(ModifyPreviousAction):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_require_discriminator_in_schema
    )


class PlannedVerifyPreviousAction(VerifyPreviousAction):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_require_discriminator_in_schema
    )


class PlannedClarifyAction(ClarifyAction):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_require_discriminator_in_schema
    )


class PlannedReferenceAction(ReferenceAction):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra=_require_discriminator_in_schema
    )


PlannedAction = Annotated[
    PlannedQueryAction
    | PlannedModifyPreviousAction
    | PlannedVerifyPreviousAction
    | PlannedClarifyAction
    | PlannedConversationAction
    | PlannedReferenceAction,
    Field(discriminator="type"),
]


class GoalRequirement(AIPlannerModel):
    kind: Literal["goal"] = "goal"
    action_index: int = Field(ge=0)
    text: NoteText
    goal: GOAL_LITERAL


class FilterRequirement(AIPlannerModel):
    kind: Literal["filter"] = "filter"
    action_index: int = Field(ge=0)
    text: NoteText
    collection: Literal["filters", "filter_groups", "denominator_filters", "compare"] = "filters"
    field: str = Field(min_length=1, max_length=64)
    operator: str
    value: str | int | float | bool | list | None = None
    branch_index: int | None = Field(default=None, ge=0)
    branch_label: str | None = Field(default=None, max_length=80)
    group_index: int | None = Field(default=None, ge=0, le=7)

    @field_validator("operator")
    @classmethod
    def _operator(cls, value: str) -> str:
        operator = value.strip().upper()
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"unsupported operator {value}")
        return operator

    @model_validator(mode="after")
    def _compare_coordinates(self) -> FilterRequirement:
        if self.collection == "compare" and self.branch_index is None and not self.branch_label:
            raise ValueError("compare filter requirements need a branch index or label")
        if self.collection == "filter_groups" and self.group_index is None:
            raise ValueError("filter_groups requirements need a group_index")
        if self.collection != "filter_groups" and self.group_index is not None:
            raise ValueError("group_index is only valid for filter_groups requirements")
        return self


class FilterLogicRequirement(AIPlannerModel):
    kind: Literal["filter_logic"] = "filter_logic"
    action_index: int = Field(ge=0)
    text: NoteText
    filter_logic: Literal["and", "or"]


class SearchTextRequirement(AIPlannerModel):
    kind: Literal["search_text"] = "search_text"
    action_index: int = Field(ge=0)
    text: NoteText
    search_text: str = Field(min_length=1, max_length=500)
    branch_index: int | None = Field(default=None, ge=0)
    branch_label: str | None = Field(default=None, max_length=80)


class ProjectionRequirement(AIPlannerModel):
    kind: Literal["projection"] = "projection"
    action_index: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=2000)
    requested_fields: list[str] = Field(min_length=1, max_length=64)

    @field_validator("text")
    @classmethod
    def _trim_text(cls, value: str) -> str:
        return value[:2000]


class GroupingRequirement(AIPlannerModel):
    kind: Literal["grouping"] = "grouping"
    action_index: int = Field(ge=0)
    text: NoteText
    field: str = Field(min_length=1, max_length=64)
    position: Literal[0, 1] = 0
    value_part: VALUE_PART_LITERAL | None = None


class SortingRequirement(AIPlannerModel):
    kind: Literal["sorting"] = "sorting"
    action_index: int = Field(ge=0)
    text: NoteText
    field: str = Field(min_length=1, max_length=64)
    direction: Literal["asc", "desc"] = "asc"


class IntervalRequirement(AIPlannerModel):
    kind: Literal["interval"] = "interval"
    action_index: int = Field(ge=0)
    text: NoteText
    interval_start: str | None = Field(default=None, max_length=64)
    interval_end: str | None = Field(default=None, max_length=64)
    interval_min_days: float | None = None
    interval_max_days: float | None = None


class AggregateRequirement(AIPlannerModel):
    kind: Literal["aggregate"] = "aggregate"
    action_index: int = Field(ge=0)
    text: NoteText
    function: str
    field: str | None = Field(default=None, max_length=64)
    stats_value_part: VALUE_PART_LITERAL | None = None


class CompanionRequirement(AIPlannerModel):
    kind: Literal["companion_field"] = "companion_field"
    action_index: int = Field(ge=0)
    text: NoteText
    companion_field: str = Field(min_length=1, max_length=64)


class HavingRequirement(AIPlannerModel):
    kind: Literal["having_min_count"] = "having_min_count"
    action_index: int = Field(ge=0)
    text: NoteText
    having_min_count: int = Field(ge=1)


class NumericSlotRequirement(AIPlannerModel):
    kind: Literal["numeric_slot"] = "numeric_slot"
    action_index: int = Field(ge=0)
    text: NoteText
    slot: Literal["top_n", "per_group_top_n", "limit", "offset"]
    value: int


class WindowRequirement(AIPlannerModel):
    kind: Literal["window"] = "window"
    action_index: int = Field(ge=0)
    text: NoteText
    anchor: Literal["start", "end"] = "start"
    size: int = Field(ge=1)
    cursor: int = Field(default=0, ge=0)


class OptionRequirement(AIPlannerModel):
    kind: Literal["option"] = "option"
    action_index: int = Field(ge=0)
    text: NoteText
    option: Literal["include_missing", "sample", "exhaustive", "presentation", "verify_previous"]
    value: bool | str


class ModifyEditRequirement(AIPlannerModel):
    kind: Literal["modify_edit"] = "modify_edit"
    action_index: int = Field(ge=0)
    text: NoteText
    edit: str
    value: Any = None


StatedRequirement = Annotated[
    GoalRequirement
    | FilterRequirement
    | FilterLogicRequirement
    | SearchTextRequirement
    | ProjectionRequirement
    | GroupingRequirement
    | SortingRequirement
    | IntervalRequirement
    | AggregateRequirement
    | CompanionRequirement
    | HavingRequirement
    | NumericSlotRequirement
    | WindowRequirement
    | OptionRequirement
    | ModifyEditRequirement,
    Field(discriminator="kind"),
]


_QUERY_REQUIREMENT_KINDS = frozenset(
    {
        "goal",
        "filter",
        "filter_logic",
        "search_text",
        "projection",
        "grouping",
        "sorting",
        "interval",
        "aggregate",
        "companion_field",
        "having_min_count",
        "numeric_slot",
        "window",
        "option",
    }
)
_ACTIONABLE_TYPES = frozenset({"query", "modify_previous", "verify_previous"})


class PlannedTurn(AIPlannerModel):
    schema_version: Literal["1"] = "1"
    normalized_request: str = Field(default="", max_length=1000)
    actions: list[PlannedAction] = Field(min_length=1, max_length=8)
    stated_requirements: list[StatedRequirement] = Field(default_factory=list, max_length=128)
    final_response: Literal["compiler_response", "deterministic", "llm_synthesis"]
    needs_evidence: bool = False
    needs_explanation: bool = False
    needs_inference: bool = False
    unresolved: list[str] = Field(default_factory=list, max_length=MAX_UNRESOLVED_ITEMS)
    confidence: float = Field(default=0.0, ge=0, le=1)

    @field_validator("normalized_request", mode="before")
    @classmethod
    def _clip_normalized_request(cls, value: Any) -> Any:
        return clip_free_text(value, 1000)

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_small_talk_beside_real_work(cls, value: Any) -> Any:
        """Drop an empty general_conversation respond that rides along with real actions.

        The planner sometimes appends a conversational flourish to a perfectly good query and
        leaves its text blank. An empty general_conversation is genuinely unusable on its own,
        so the action validator rejects it -- and takes the query down with it, costing a
        schema-validation retry and sometimes the turn.

        Nothing is lost by dropping it: the wording of an answer that has query results comes
        from synthesis, not from this field. Guarded on there being another actionable action,
        because a turn whose only action is blank small talk really does have nothing to say
        and should still fail.
        """
        if not isinstance(value, dict):
            return value
        actions = value.get("actions") or []
        if len(actions) < 2:
            return value

        def _get(action: Any, key: str) -> Any:
            return action.get(key) if isinstance(action, dict) else getattr(action, key, None)

        if not any(_get(action, "type") in _ACTIONABLE_TYPES for action in actions):
            return value
        kept = [
            action
            for action in actions
            if not (
                _get(action, "type") == "respond"
                and _get(action, "intent") == "general_conversation"
                and not str(_get(action, "response") or "").strip()
            )
        ]
        if len(kept) == len(actions):
            return value
        # Requirements name their action by position, so the indexes have to move with it.
        moved = {}
        offset = 0
        for index, action in enumerate(actions):
            if action in kept:
                moved[index] = index - offset
            else:
                offset += 1
        requirements = []
        for item in value.get("stated_requirements") or []:
            is_dict = isinstance(item, dict)
            index = item.get("action_index") if is_dict else getattr(item, "action_index", None)
            if index in moved:
                requirements.append(
                    {**item, "action_index": moved[index]}
                    if is_dict
                    else item.model_copy(update={"action_index": moved[index]})
                )
            elif not isinstance(index, int):
                requirements.append(item)
        updated = dict(value)
        updated["actions"] = kept
        updated["stated_requirements"] = requirements
        return updated

    @model_validator(mode="before")
    @classmethod
    def _drop_query_requirements_without_query(cls, value: Any) -> Any:
        """Drop a query requirement that has no query action to describe.

        This is not condition loss: there is nothing for such a requirement to be checked
        against. Schema questions often attach a leftover goal requirement to
        describe_available_data.

        Dropped per requirement, by the action it names, rather than per turn. The two rules
        have to agree on what "no query action" means and they did not: this guard used to
        stand down whenever the turn held any actionable action, while _requirement_targets
        below rejects unless the named action is specifically a query. A follow-up planned as
        modify_previous alone fell in that gap -- the planner stated the filter it was
        carrying, which is the honest thing to state, and the turn was thrown away for it.
        Live, that turned "what are their names" into "planning service unavailable".
        """
        if not isinstance(value, dict):
            return value
        actions = value.get("actions") or []
        if not actions:
            return value
        types: list[str | None] = []
        for action in actions:
            if isinstance(action, dict):
                types.append(action.get("type"))
            else:
                types.append(getattr(action, "type", None))
        kept = []
        for item in value.get("stated_requirements") or []:
            if isinstance(item, dict):
                kind, index = item.get("kind"), item.get("action_index")
            else:
                kind, index = getattr(item, "kind", None), getattr(item, "action_index", None)
            # An out-of-range index is left alone for _requirement_indexes to reject. That is
            # a planner that has lost track of its own actions, which is worth failing on,
            # and not a requirement that merely has nowhere to sit.
            in_range = isinstance(index, int) and 0 <= index < len(types)
            if kind in _QUERY_REQUIREMENT_KINDS and in_range and types[index] != "query":
                continue
            kept.append(item)
        updated = dict(value)
        updated["stated_requirements"] = kept
        return updated

    @model_validator(mode="after")
    def _requirement_indexes(self) -> PlannedTurn:
        count = len(self.actions)
        for item in self.stated_requirements:
            if item.action_index >= count:
                raise ValueError("stated requirement action_index is out of range")
            action = self.actions[item.action_index]
            if item.kind == "modify_edit" and not isinstance(action, ModifyPreviousAction):
                raise ValueError("modify_edit requirements must target a modify_previous action")
            if item.kind in {
                "goal",
                "filter",
                "filter_logic",
                "search_text",
                "projection",
                "grouping",
                "sorting",
                "interval",
                "aggregate",
                "companion_field",
                "having_min_count",
                "numeric_slot",
                "window",
                "option",
            } and not isinstance(action, PlannedQueryAction):
                raise ValueError(f"{item.kind} requirements must target a query action")
            if isinstance(item, FilterRequirement) and item.collection == "compare":
                if not isinstance(action, PlannedQueryAction):
                    raise ValueError("compare filter requirements must target a query action")
                if item.branch_index is not None and item.branch_index >= len(action.compare):
                    raise ValueError("compare branch_index is out of range")
            if isinstance(item, FilterRequirement) and item.collection == "filter_groups":
                if not isinstance(action, PlannedQueryAction):
                    raise ValueError("filter_groups requirements must target a query action")
                if item.group_index is None or item.group_index >= len(action.filter_groups):
                    raise ValueError("filter group_index is out of range")
        return self

    def query_actions(self) -> list[PlannedQueryAction]:
        return [item for item in self.actions if isinstance(item, PlannedQueryAction)]

    def modify_actions(self) -> list[ModifyPreviousAction]:
        return [item for item in self.actions if isinstance(item, ModifyPreviousAction)]

    def clarify_actions(self) -> list[ClarifyAction]:
        return [item for item in self.actions if isinstance(item, ClarifyAction)]

    def respond_actions(self) -> list[PlannedConversationAction]:
        return [item for item in self.actions if isinstance(item, PlannedConversationAction)]

    def reference_actions(self) -> list[ReferenceAction]:
        return [item for item in self.actions if isinstance(item, ReferenceAction)]

    def verify_actions(self) -> list[VerifyPreviousAction]:
        return [item for item in self.actions if isinstance(item, VerifyPreviousAction)]


REVIEW_CLARIFICATION_REASON = Literal[
    "dataset_identity",
    "missing_field",
    "ambiguous_value",
    "ambiguous_range",
    "ambiguous_reference",
    "unsupported_logic",
    "other",
]


ShortReviewText = NoteText


class ReviewComplete(AIPlannerModel):
    verdict: Literal["complete"] = "complete"
    # A complete data review is only meaningful if the reviewer first states what it
    # independently reconstructed from the question. Trusted code compares this list
    # with the candidate in both directions for narrowing constraints.
    reconstructed_requirements: list[StatedRequirement] = Field(min_length=1, max_length=128)


class ReviewCorrected(AIPlannerModel):
    verdict: Literal["corrected"] = "corrected"
    missing: list[ShortReviewText] = Field(min_length=1, max_length=32)
    reconstructed_requirements: list[StatedRequirement] = Field(min_length=1, max_length=128)
    plan: PlannedTurn


class ReviewNeedsClarification(AIPlannerModel):
    verdict: Literal["needs_clarification"] = "needs_clarification"
    missing: list[ShortReviewText] = Field(min_length=1, max_length=32)
    reason: REVIEW_CLARIFICATION_REASON
    reconstructed_requirements: list[StatedRequirement] = Field(default_factory=list, max_length=128)
    question: str = Field(default="", max_length=500)


ReviewedVerdict = Annotated[
    ReviewComplete | ReviewCorrected | ReviewNeedsClarification,
    Field(discriminator="verdict"),
]


class PlanReviewResponse(RootModel[ReviewedVerdict]):
    """Named wrapper so xAI SDK 1.19 chat.parse can take a BaseModel subclass."""

    @model_validator(mode="before")
    @classmethod
    def _strip_complete_extras(cls, value: Any) -> Any:
        """Trust verdict=complete and ignore leftover missing/plan keys.

        ReviewComplete remains extra=forbid, and a complete verdict still requires the
        independent `reconstructed_requirements` audit. Live models sometimes dump correction
        fields onto a complete verdict; those leftovers are slop, not a correction.
        """
        payload = value
        if isinstance(value, dict) and "root" in value and isinstance(value["root"], dict):
            payload = value["root"]
        if isinstance(payload, dict) and payload.get("verdict") == "complete":
            # Providers sometimes materialize every union field with harmless empty/null
            # values. Strip those. But a non-empty `missing` list or a replacement `plan`
            # contradicts verdict=complete and is semantic evidence we must not discard.
            if payload.get("missing") or payload.get("plan") not in (None, {}):
                return value
            cleaned = {"verdict": "complete"}
            if "reconstructed_requirements" in payload:
                cleaned["reconstructed_requirements"] = payload["reconstructed_requirements"]
            return cleaned
        return value


def bind_scope(planned: PlannedTurn, file_id: int) -> TurnPlan:
    """Stamp the selected dataset onto every query action. Never trust the model."""
    actions = []
    for action in planned.actions:
        if isinstance(action, PlannedQueryAction):
            payload = action.model_dump(mode="python")
            payload["datasets"] = [file_id]
            actions.append(QueryAction.model_validate(payload))
        elif isinstance(action, PlannedConversationAction):
            actions.append(
                ConversationAction(intent=action.intent, response=action.response)
            )
        else:
            actions.append(action)
    return TurnPlan(
        schema_version=planned.schema_version,
        normalized_request=planned.normalized_request,
        actions=actions,
        final_response=planned.final_response,
        needs_evidence=planned.needs_evidence,
        needs_explanation=planned.needs_explanation,
        needs_inference=planned.needs_inference,
        unresolved=list(planned.unresolved),
        confidence=planned.confidence,
    )


def planned_turn_contains_dataset_id(payload: Any) -> bool:
    """True when a raw model payload tried to name a dataset identifier."""
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"datasets", "dataset_id", "file_id", "file_ids", "selected_file_id"}:
                return True
            if planned_turn_contains_dataset_id(value):
                return True
    elif isinstance(payload, list):
        return any(planned_turn_contains_dataset_id(item) for item in payload)
    return False


__all__ = [
    "AIPlannerModel",
    "ALLOWED_GOALS",
    "PlanReviewResponse",
    "PlannedConversationAction",
    "PlannedQueryAction",
    "PlannedTurn",
    "ReviewComplete",
    "ReviewCorrected",
    "ReviewNeedsClarification",
    "StatedRequirement",
    "bind_scope",
    "planned_turn_contains_dataset_id",
]
