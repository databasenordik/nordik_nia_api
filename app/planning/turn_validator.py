from __future__ import annotations

import re

from app.planning.catalog import FieldCatalog
from app.planning.plan_validator import PlanValidationError
from app.planning.turn_schema import (
    ALLOWED_OPERATORS,
    CONVERSATIONAL_INTENTS,
    TRUSTED_CONVERSATIONAL_INTENTS,
    ClarifyAction,
    ConversationAction,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    ReferenceAction,
    TurnPlan,
    VerifyPreviousAction,
)
from app.security.access_scope import AccessScope

MAX_TURN_ACTIONS = 8
MIN_TURN_CONFIDENCE = 0.75
_SQL = re.compile(r"\b(?:select|insert|update|delete|drop|alter|truncate)\b", re.IGNORECASE)
_FORBIDDEN_TABLES = re.compile(
    r"\b(?:file_data_normalized|file_data|data_config|users|roles|otps|logs|support_[a-z_]+)\b",
    re.IGNORECASE,
)
# Best-effort hallucination check only. Not a security guarantee.
# Research facts must come from QueryAction execution, never from this intent.
_RESEARCH_CLAIM = re.compile(
    r"\b(?:\d+\s+(?:students?|records?|people|persons?)|"
    r"(?:there\s+(?:are|were|is)|count(?:ed)?\s+of)\s+\d+|"
    r"from\s+garden river|"
    r"admitted\s+(?:before|after|in)\s+\d{4}|"
    r"file\s+\d+)\b",
    re.IGNORECASE,
)
_NUMERIC_TYPES = frozenset({"number", "numeric", "integer", "float", "int"})
_ORDERED_TYPES = _NUMERIC_TYPES | {"date"}


def validate_turn_plan(
    turn: TurnPlan,
    *,
    scope: AccessScope,
    catalog: FieldCatalog,
    has_active_query: bool = False,
    strict: bool = False,
) -> TurnPlan:
    """Reject unsafe or unexecutable semantic output before QueryPlan construction."""

    from app.config import get_settings

    min_confidence = get_settings().planner_min_confidence if strict else MIN_TURN_CONFIDENCE
    if strict:
        turn = _drop_orphan_followups_strict(turn, has_active_query=has_active_query)
    else:
        turn = _drop_orphan_followups(
            turn,
            has_active_query=has_active_query,
            default_file_id=catalog.default_people_file_id,
        )
    if len(turn.actions) > MAX_TURN_ACTIONS:
        raise PlanValidationError("turn exceeds action budget", code="budget")

    # What a low self-reported confidence costs the turn, which differs by mode:
    #
    #   strict (the AI path): the turn is replaced by a canned clarification even when it
    #     is perfectly executable. This is the gate that makes a "review only when
    #     confidence is low" rule dead on arrival -- nothing below the threshold survives
    #     to reach review.
    #   non-strict (legacy): only a turn with nothing executable is clarified, because
    #     discarding a runnable query for a hard question answers nobody.
    #
    # The two are deliberately different and the earlier comment here described only the
    # second, which read as though the strict branch were a bug. Whether strict should keep
    # discarding executable plans is a policy question that wants measurement, not an
    # edit: it is the confidence gate the latency work is scheduled to shadow.
    executable = bool(
        turn.query_actions()
        or turn.modify_actions()
        or turn.verify_actions()
        or turn.reference_actions()
    )
    if turn.confidence < min_confidence and (
        strict or (not executable and not _trusted_conversation(turn))
    ):
        if strict and executable:
            return _clarification_turn(
                turn,
                "I want to be sure I understood you. Could you say that another way?",
                unresolved=["low_confidence"],
            )
        if not executable and not _trusted_conversation(turn):
            return _clarification_turn(
                turn,
                "I want to be sure I understood you. Could you say that another way?",
                unresolved=["low_confidence"],
            )
    if turn.unresolved and not turn.clarify_actions() and (strict or not executable):
        return _clarification_turn(
            turn,
            "Could you be more specific about what you'd like to know?",
            unresolved=list(turn.unresolved),
        )

    scoped = catalog.for_scope(scope)
    authorized_ids = {item.file_id for item in scoped.datasets}
    if _mixes_general_conversation_with_research(turn) and not strict:
        # The research actions are the answer; a general-conversation aside alongside
        # them is noise. Dropping it beats failing the whole turn.
        turn = turn.model_copy(
            update={
                "actions": [
                    action
                    for action in turn.actions
                    if not (
                        isinstance(action, ConversationAction)
                        and action.intent == "general_conversation"
                    )
                ]
            }
        )

    for action in turn.actions:
        if isinstance(action, QueryAction):
            _validate_query_action(action, scope, scoped, authorized_ids)
        elif isinstance(action, ModifyPreviousAction):
            if not has_active_query:
                raise PlanValidationError("follow-up has no active query", code="invalid_followup")
            datasets = action_datasets(action, scoped)
            for change in action.changes:
                _validate_filter(change, datasets, scoped)
            if action.sort_by:
                _require_field_on_every_dataset(action.sort_by, datasets, scoped)
        elif isinstance(action, VerifyPreviousAction):
            if not has_active_query:
                raise PlanValidationError("follow-up has no active query", code="invalid_followup")
        elif isinstance(action, ConversationAction):
            if action.intent not in CONVERSATIONAL_INTENTS:
                raise PlanValidationError(
                    "respond actions require a conversational intent",
                    code="invalid_intent",
                )
            if action.intent == "general_conversation" and _RESEARCH_CLAIM.search(action.response or ""):
                raise PlanValidationError(
                    "general conversation cannot contain research-database facts",
                    code="research_fact_in_respond",
                )
        elif isinstance(action, ReferenceAction):
            if not has_active_query:
                raise PlanValidationError("follow-up has no active query", code="invalid_followup")
            if action.selector is None and action.index is None:
                raise PlanValidationError("reference action needs a selector or index", code="op_contract")

    _reject_structural_injection(turn)
    if turn.needs_evidence:
        _validate_evidence_request(turn, scoped)
    return turn


def _drop_orphan_followups(
    turn: TurnPlan,
    *,
    has_active_query: bool,
    default_file_id: int,
) -> TurnPlan:
    """Remove follow-up actions that have nothing to follow.

    The compiler sometimes renders the second clause of a compound question as
    modify_previous. On the first turn of a conversation there is no previous result,
    and failing the whole turn threw away the query action that was already correct.
    """
    if has_active_query:
        return turn
    changed = False
    keep: list = []
    for action in turn.actions:
        if not isinstance(action, ModifyPreviousAction | VerifyPreviousAction | ReferenceAction):
            keep.append(action)
            continue
        changed = True
        if isinstance(action, ModifyPreviousAction):
            # A refinement is a perfectly good first query when there is nothing to
            # refine: its conditions, goal, and ordering stand on their own.
            keep.append(
                QueryAction(
                    datasets=[default_file_id] if default_file_id else [],
                    goal=action.goal or "list",
                    filters=list(action.changes),
                    sort_by=action.sort_by,
                    sort_direction=action.sort_direction,
                    requested_fields=list(action.requested_fields or []),
                    limit=action.limit,
                    presentation=action.presentation,
                )
            )
    if not changed:
        return turn
    if not keep:
        return _clarification_turn(
            turn,
            "Which records would you like me to look at?",
            unresolved=["no_previous_result"],
        )
    return turn.model_copy(update={"actions": keep})


def _drop_orphan_followups_strict(turn: TurnPlan, *, has_active_query: bool) -> TurnPlan:
    """AI mode: orphan follow-ups become a clarification, never a promoted query."""
    if has_active_query:
        return turn
    if any(
        isinstance(action, ModifyPreviousAction | VerifyPreviousAction | ReferenceAction)
        for action in turn.actions
    ):
        return _clarification_turn(
            turn,
            "Which records would you like me to look at?",
            unresolved=["no_previous_result"],
        )
    return turn


def _mixes_general_conversation_with_research(turn: TurnPlan) -> bool:
    has_general = any(item.intent == "general_conversation" for item in turn.respond_actions())
    if not has_general:
        return False
    return bool(
        turn.query_actions()
        or turn.modify_actions()
        or turn.reference_actions()
        or turn.verify_actions()
    )


def _trusted_conversation(turn: TurnPlan) -> bool:
    if not turn.actions or not all(isinstance(item, ConversationAction) for item in turn.actions):
        return False
    return all(item.intent in TRUSTED_CONVERSATIONAL_INTENTS for item in turn.respond_actions())


def action_datasets(action: QueryAction | ModifyPreviousAction, catalog: FieldCatalog) -> list[int]:
    if isinstance(action, QueryAction) and action.datasets:
        return list(action.datasets)
    default = catalog.default_people_file_id
    if default:
        return [default]
    return [item.file_id for item in catalog.datasets[:1]]


def _clarification_turn(turn: TurnPlan, question: str, *, unresolved: list[str]) -> TurnPlan:
    if turn.clarify_actions():
        question = turn.clarify_actions()[0].question
    return TurnPlan(
        normalized_request=turn.normalized_request,
        actions=[ClarifyAction(question=question)],
        final_response="compiler_response",
        unresolved=unresolved,
        confidence=turn.confidence,
    )


def _validate_query_action(
    action: QueryAction,
    scope: AccessScope,
    catalog: FieldCatalog,
    authorized_ids: set[int],
) -> None:
    if not action.datasets:
        raise PlanValidationError("query action did not name a dataset", code="unknown_dataset")
    unknown = [file_id for file_id in action.datasets if file_id not in authorized_ids]
    if unknown:
        raise PlanValidationError(
            "requested datasets are not in the current access scope",
            code="access_restricted",
        )
    unauthorized = [file_id for file_id in action.datasets if file_id not in scope.allowed_file_ids]
    if unauthorized:
        raise PlanValidationError(
            "requested datasets are not in the current access scope",
            code="access_restricted",
        )
    for spec in action.filters:
        _validate_filter(spec, action.datasets, catalog)
    for group in action.filter_groups:
        for spec in group.filters:
            _validate_filter(spec, action.datasets, catalog)
    for spec in action.denominator_filters:
        _validate_filter(spec, action.datasets, catalog)
    for field in action.group_by:
        _require_field_on_every_dataset(field, action.datasets, catalog)
    for field in (action.sort_by, action.companion_field, action.interval_start, action.interval_end):
        if field:
            _require_field_on_every_dataset(field, action.datasets, catalog)
    if action.compare and action.goal != "compare":
        # Branches compile only for goal=compare; anywhere else their conditions would vanish
        # and the answer would cover every record.
        raise PlanValidationError(
            f"compare branches only apply to goal=compare, not {action.goal}", code="op_contract"
        )
    if action.goal in {"rank", "duplicates", "distinct"} and not action.group_by:
        raise PlanValidationError(f"{action.goal} requires a group_by field", code="op_contract")
    if action.goal == "interval" and not (action.interval_start and action.interval_end):
        raise PlanValidationError(
            "interval requires interval_start and interval_end", code="op_contract"
        )
    if action.goal == "stats" and not (
        (action.aggregate is not None and action.aggregate.field)
        or action.sort_by
        or action.requested_fields
    ):
        raise PlanValidationError("stats requires a field to measure", code="op_contract")
    if action.goal in {"aggregate", "stats"} or action.aggregate is not None:
        _validate_aggregate(action, catalog)
    if action.goal == "compare":
        if len(action.compare) < 2:
            raise PlanValidationError("compare requires at least two branches", code="op_contract")
        for branch in action.compare:
            for spec in branch.filters:
                _validate_filter(spec, action.datasets, catalog)
def _validate_filter(spec: FilterSpec, datasets: list[int], catalog: FieldCatalog) -> None:
    if spec.operator not in ALLOWED_OPERATORS:
        raise PlanValidationError(f"unsupported operator {spec.operator}", code="operator")
    for file_id in datasets:
        field = catalog.resolve_field(file_id, spec.field)
        if field is None:
            raise PlanValidationError(
                f"unknown semantic field {spec.field} on file {file_id}",
                code="unknown_field",
            )
        if spec.operator not in field.allowed_operators:
            raise PlanValidationError(
                f"operator {spec.operator} is not allowed on {spec.field} for file {file_id}",
                code="operator",
            )


def _validate_aggregate(action: QueryAction, catalog: FieldCatalog) -> None:
    spec = action.aggregate
    if spec is None:
        raise PlanValidationError("aggregate requires an aggregate spec", code="op_contract")
    if spec.function == "count" and not spec.field:
        return
    if not spec.field:
        raise PlanValidationError(f"{spec.function} requires a field", code="op_contract")
    for file_id in action.datasets:
        field = catalog.resolve_field(file_id, spec.field)
        if field is None:
            raise PlanValidationError(
                f"unknown semantic field {spec.field} on file {file_id}",
                code="unknown_field",
            )
        kind = (field.semantic_type or "text").lower()
        if spec.function in {"sum", "avg"}:
            if not field.aggregatable or kind not in _NUMERIC_TYPES:
                raise PlanValidationError(
                    f"{spec.function} is not allowed on {spec.field}",
                    code="operator",
                )
        elif spec.function in {"min", "max"}:
            if kind not in _ORDERED_TYPES:
                raise PlanValidationError(
                    f"{spec.function} is not allowed on {spec.field}",
                    code="operator",
                )
        elif spec.function in {"median", "mode", "stats"}:
            if kind not in _ORDERED_TYPES and kind not in {"entity", "text"}:
                raise PlanValidationError(
                    f"{spec.function} is not allowed on {spec.field}",
                    code="operator",
                )
        elif spec.function == "count_distinct":
            if not field.aggregatable and kind not in {"entity", "text", "date"}:
                raise PlanValidationError(
                    f"count_distinct is not allowed on {spec.field}",
                    code="operator",
                )


def _require_field_on_every_dataset(name: str, datasets: list[int], catalog: FieldCatalog) -> None:
    for file_id in datasets:
        if catalog.resolve_field(file_id, name) is None:
            raise PlanValidationError(
                f"unknown semantic field {name} on file {file_id}",
                code="unknown_field",
            )


def _validate_evidence_request(turn: TurnPlan, catalog: FieldCatalog) -> None:
    datasets: list[int] = []
    for action in turn.query_actions():
        datasets.extend(action.datasets)
    if not datasets:
        return
    allowed = False
    for file_id in datasets:
        for field in catalog.fields_for(file_id):
            if field.evidence_allowed:
                allowed = True
                break
    if not allowed:
        raise PlanValidationError("evidence is not allowed for the selected fields", code="sensitivity")


def _reject_structural_injection(turn: TurnPlan) -> None:
    """Scan identifiers only. Free-form search/response/normalized text is allowed."""
    structural: list[str] = []
    for action in turn.actions:
        structural.append(action.type)
        if isinstance(action, QueryAction):
            structural.extend(item.field for item in action.filters)
            for group in action.filter_groups:
                structural.extend(item.field for item in group.filters)
            structural.extend(action.group_by)
            if action.sort_by:
                structural.append(action.sort_by)
            if action.aggregate is not None and action.aggregate.field:
                structural.append(action.aggregate.field)
            for branch in action.compare:
                structural.extend(item.field for item in branch.filters)
        elif isinstance(action, ModifyPreviousAction):
            structural.extend(item.field for item in action.changes)
        elif isinstance(action, ConversationAction):
            structural.append(action.intent)
    for blob in structural:
        if _FORBIDDEN_TABLES.search(blob):
            raise PlanValidationError("plans cannot name application tables", code="forbidden_table")
        if _SQL.search(blob):
            raise PlanValidationError("plans cannot contain SQL", code="forbidden_sql")
