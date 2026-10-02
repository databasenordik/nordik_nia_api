from __future__ import annotations

from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import (
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
    all_of,
    any_of,
)
from app.planning.plan_validator import PlanValidationError
from app.planning.turn_schema import (
    ALLOWED_GOALS,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    ActiveQuery,
    AggregateSpec,
    FilterGroupSpec,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    ResultWindow,
    TurnPlan,
    VerifyPreviousAction,
)
from app.retrieval.entity_resolver import resolve_entity
from app.security.access_scope import AccessScope

_GOAL_ALIASES = {
    "exact_count": "count",
    "count": "count",
    "list": "list",
    "distinct": "distinct",
    "distinct_values": "distinct",
    "aggregate": "aggregate",
    "compare": "compare",
    "search": "search",
    "quote": "quote",
    "provenance": "provenance",
    "rank": "rank",
    "top": "rank",
    "stats": "stats",
    "duplicates": "duplicates",
    "percentage": "percentage",
    "ratio": "percentage",
    "interval": "interval",
    "completeness": "completeness",
    "dossier": "dossier",
}

# One page stays 25/50 rows. An explicit exhaustive enumeration ("name every student
# who ...") may return the whole matching set up to this ceiling, because answering it
# with an arbitrary first page is the defect this replaces. The ceiling keeps a
# whole-dataset "list everything" from turning into thousands of lines.
MAX_ENUMERATION_SIZE = 200

_AGG_OP = {
    "count": PlanOp.COUNT,
    "count_distinct": PlanOp.COUNT_DISTINCT,
    "sum": PlanOp.SUM,
    "avg": PlanOp.AVG,
    "min": PlanOp.MIN,
    "max": PlanOp.MAX,
}
# avg/min/max/median/mode over one field are all one descriptive-statistics pass.
_STATS_FUNCTIONS = frozenset({"avg", "min", "max", "median", "mode", "stats"})


class QueryPlanBuildError(ValueError):
    pass


class UnsupportedFieldsError(QueryPlanBuildError):
    def __init__(
        self,
        requested: list[str],
        *,
        file_ids: tuple[int, ...],
        catalog: FieldCatalog,
    ) -> None:
        self.requested = requested
        self.file_ids = file_ids
        self.dataset_labels = [
            catalog.dataset(file_id).user_facing_label
            for file_id in file_ids
            if catalog.dataset(file_id) is not None
        ]
        self.available_fields = _common_field_labels(file_ids, catalog)
        super().__init__(f"unsupported requested fields: {', '.join(requested)}")


def compiled_query_actions(
    turn: TurnPlan,
    *,
    scope: AccessScope,
    catalog: FieldCatalog,
    active_query: ActiveQuery | None = None,
    frames: dict[str, ActiveQuery] | None = None,
) -> list[QueryAction]:
    """Effective QueryActions after follow-up edits. Same list the builder compiles."""
    scoped = catalog.for_scope(scope)
    return [
        _effective_query(action, active_query, frames, scoped)
        for action in turn.actions
        if _is_data_action(action)
    ]


def query_action_from_state(state) -> QueryAction:
    """Replay a stored action. Prefer the original QueryAction when persisted."""
    source = getattr(state, "source_action", None)
    if source is not None:
        return source.model_copy(deep=True) if hasattr(source, "model_copy") else QueryAction.model_validate(source)
    return _query_action_from_fields(state)


def _query_action_from_fields(state) -> QueryAction:
    goal = _normalized_goal(state.goal)
    filters, filter_groups, filter_logic = _filter_surfaces_from_predicates(list(state.filters))
    return QueryAction(
        datasets=list(state.file_ids),
        goal=goal,
        filters=filters,
        filter_groups=filter_groups,
        filter_logic=filter_logic,
        search_text=state.retrieval_query,
        sort_by=getattr(state, "sort_by", None),
        sort_direction=getattr(state, "sort_direction", None),
        sample=bool(getattr(state, "sample", False)),
        requested_fields=list(getattr(state, "requested_fields", None) or []),
        window=getattr(state, "window", None),
        limit=state.limit,
        offset=getattr(state, "offset", None),
        exhaustive=bool(getattr(state, "exhaustive", False)),
        presentation=getattr(state, "presentation", None),
    )


def _normalized_goal(goal: str | None) -> str:
    normalized = _GOAL_ALIASES.get(str(goal or ""), str(goal or ""))
    return normalized if normalized in ALLOWED_GOALS else "count"


def _leaf_filter(predicate: Predicate) -> FilterSpec:
    if predicate.is_group() or not predicate.field or predicate.operator is None:
        raise QueryPlanBuildError("stored predicate cannot be represented as a filter leaf")
    return FilterSpec(field=predicate.field, operator=str(predicate.operator), value=predicate.value)


def _only_leaf_items(predicate: Predicate) -> list[Predicate] | None:
    if not predicate.is_group():
        return [predicate]
    if predicate.op != "and":
        return None
    if any(item.is_group() for item in predicate.items):
        return None
    return list(predicate.items)


def _filter_surfaces_from_predicates(
    predicates: list[Predicate],
) -> tuple[list[FilterSpec], list[FilterGroupSpec], str]:
    """Losslessly recover flat filters or bounded DNF from a stored predicate tree.

    Frames normally persist the original QueryAction. This fallback is deliberately
    strict: if an old frame contains a tree outside the AI planner's bounded DNF
    surface, failing is safer than flattening it into a different population.
    """
    if not predicates:
        return [], [], "and"
    if all(not item.is_group() for item in predicates):
        return [_leaf_filter(item) for item in predicates], [], "and"
    if len(predicates) != 1:
        # Top-level lists are implicit AND. They can be represented as one conjunction
        # only when every child is a leaf or an AND-of-leaves.
        leaves: list[Predicate] = []
        for item in predicates:
            part = _only_leaf_items(item)
            if part is None:
                raise QueryPlanBuildError("stored predicate tree is outside bounded DNF")
            leaves.extend(part)
        return [_leaf_filter(item) for item in leaves], [], "and"

    root = predicates[0]
    if not root.is_group():
        return [_leaf_filter(root)], [], "and"
    if root.op == "and":
        part = _only_leaf_items(root)
        if part is None:
            raise QueryPlanBuildError("stored predicate tree is outside bounded DNF")
        return [_leaf_filter(item) for item in part], [], "and"
    if root.op != "or":
        raise QueryPlanBuildError("stored NOT predicate is outside AI follow-up grammar")

    groups: list[FilterGroupSpec] = []
    for child in root.items:
        part = _only_leaf_items(child)
        if part is None:
            raise QueryPlanBuildError("stored predicate tree is outside bounded DNF")
        groups.append(FilterGroupSpec(filters=[_leaf_filter(item) for item in part]))
    if len(groups) == 1:
        return list(groups[0].filters), [], "and"
    return [], groups, "and"


def reconstruct_query_action(
    plan: QueryPlan,
    *,
    action_id: str,
    goal: str,
    predicates,
) -> QueryAction:
    """Fallback when the original QueryAction was not persisted."""
    from app.planning.turn_schema import AggregateSpec, CompareBranch, FilterSpec

    steps = [step for step in plan.steps if (step.action_id or "a0") == action_id]
    file_ids = list(plan.scope.file_ids)
    search_text = None
    limit = None
    group_by: list[str] = []
    sort_by = None
    sort_direction = None
    sample = False
    requested_fields: list[str] = []
    window = None
    aggregate = None
    compare_wheres: list[list] = []
    for step in steps:
        if step.file_ids:
            file_ids = list(step.file_ids)
        if step.query:
            search_text = step.query
        if step.limit:
            limit = step.limit
        if step.op is PlanOp.GROUP_BY and step.fields and step.fields != ["file_id"]:
            group_by = list(step.fields)
        if step.op is PlanOp.SORT and step.sort_field:
            sort_by = step.sort_field
            sort_direction = "desc" if step.sort_direction == "DESC" else "asc"
        if step.op is PlanOp.SAMPLE:
            sample = True
        if step.op is PlanOp.PROJECT:
            requested_fields = list(step.fields)
            if step.window_anchor and step.window_size:
                window = ResultWindow(
                    anchor=step.window_anchor,
                    size=step.window_size,
                    cursor=step.window_cursor,
                )
        if step.op in {PlanOp.SUM, PlanOp.AVG, PlanOp.MIN, PlanOp.MAX, PlanOp.COUNT_DISTINCT}:
            function = {
                PlanOp.SUM: "sum",
                PlanOp.AVG: "avg",
                PlanOp.MIN: "min",
                PlanOp.MAX: "max",
                PlanOp.COUNT_DISTINCT: "count_distinct",
            }[step.op]
            aggregate = AggregateSpec(
                function=function,
                field=step.fields[0] if step.fields else None,
            )
        if step.op is PlanOp.FILTER and step.where:
            compare_wheres.append(list(step.where))
    goal = _normalized_goal(goal)
    compare = []
    filters, filter_groups, filter_logic = _filter_surfaces_from_predicates(list(predicates))
    if goal == "compare":
        compare = [
            CompareBranch(
                filters=[
                    FilterSpec(field=item.field, operator=str(item.operator), value=item.value)
                    for item in wheres
                ]
            )
            for wheres in compare_wheres
        ]
        filters = []
    return QueryAction(
        datasets=file_ids,
        goal=goal,
        filters=filters,
        filter_groups=filter_groups,
        filter_logic=filter_logic,
        search_text=search_text,
        group_by=group_by,
        sort_by=sort_by,
        sort_direction=sort_direction,
        sample=sample,
        requested_fields=requested_fields,
        window=window,
        limit=limit,
        aggregate=aggregate,
        compare=compare,
    )


def build_query_plan(
    turn: TurnPlan,
    *,
    scope: AccessScope,
    catalog: FieldCatalog,
    active_query: ActiveQuery | None = None,
    frames: dict[str, ActiveQuery] | None = None,
) -> QueryPlan:
    """Deterministic TurnPlan → QueryPlan. Independent actions stay independent branches."""

    scoped = catalog.for_scope(scope)
    queries = [
        _effective_query(action, active_query, frames, scoped)
        for action in turn.actions
        if _is_data_action(action)
    ]
    if not queries:
        raise QueryPlanBuildError("turn has no executable query action")

    needs_evidence = turn.needs_evidence or turn.final_response == "llm_synthesis"
    wants_synthesis = (
        turn.final_response == "llm_synthesis"
        or turn.needs_explanation
        or turn.needs_inference
    )

    if len(queries) == 1:
        return _compile_single(
            queries[0],
            scope=scope,
            catalog=scoped,
            needs_evidence=needs_evidence,
            wants_synthesis=wants_synthesis,
        )

    return _compile_parallel(
        queries,
        scope=scope,
        catalog=scoped,
        needs_evidence=needs_evidence,
        wants_synthesis=wants_synthesis,
    )


def _stamp(steps: list[PlanStep], *, action_id: str, action_goal: str) -> list[PlanStep]:
    stamped: list[PlanStep] = []
    for step in steps:
        if step.op is PlanOp.USE_CURRENT_VERSION:
            stamped.append(step)
            continue
        stamped.append(step.model_copy(update={"action_id": action_id, "action_goal": action_goal}))
    return stamped


def _is_data_action(action) -> bool:
    return isinstance(action, QueryAction | ModifyPreviousAction | VerifyPreviousAction)


def _effective_query(
    action: QueryAction | ModifyPreviousAction | VerifyPreviousAction,
    active_query: ActiveQuery | None,
    frames: dict[str, ActiveQuery] | None,
    catalog: FieldCatalog,
) -> QueryAction:
    if isinstance(action, QueryAction):
        datasets = list(action.datasets) or (
            [catalog.default_people_file_id] if catalog.default_people_file_id else []
        )
        return action.model_copy(update={"datasets": datasets})

    source = _source_query(action, active_query, frames)
    if source is None:
        raise PlanValidationError("follow-up target is unavailable or stale", code="invalid_followup")
    if isinstance(action, VerifyPreviousAction):
        # Verification means re-run the same semantic query. Turning a count/rank/stats
        # result into an arbitrary list page answers a different question.
        return _query_from_active(source, verify_previous=True)

    goal = _normalized_goal(action.goal or source.goal)

    source_groups = [group.model_copy(deep=True) for group in source.filter_groups]
    filters: list[FilterSpec] = list(source.filters)
    filter_groups: list[FilterGroupSpec] = source_groups
    filter_logic = source.filter_logic
    if action.changes:
        if filter_groups:
            if action.change_mode == "refine":
                updated_groups = [
                    FilterGroupSpec(
                        filters=_append_refinements(list(group.filters), action.changes)
                    )
                    for group in filter_groups
                ]
                filter_groups = _dedupe_filter_groups(updated_groups)
            else:
                filter_groups = _replace_group_changes(filter_groups, action.changes)
        elif filter_logic == "or" and len(filters) > 1 and action.change_mode == "refine":
            # (A OR B) refined by C becomes (A AND C) OR (B AND C), never A OR B OR C.
            filter_groups = _dedupe_filter_groups(
                [
                    FilterGroupSpec(filters=_append_refinements([item], action.changes))
                    for item in filters
                ]
            )
            filters = []
            filter_logic = "and"
        elif action.change_mode == "refine":
            filters = _append_refinements(filters, action.changes)
        else:
            filters = _apply_changes(filters, action.changes)

    # "list them" / "show them in a table" after a distinct answer means show
    # those distinct values, not the underlying rows.
    if (
        source.goal == "distinct"
        and action.goal == "list"
        and not action.changes
        and action.requested_fields is None
    ):
        goal = "distinct"

    page_size = action.limit or source.limit or DEFAULT_PAGE_SIZE
    offset = source.offset or 0
    window = action.window.model_copy(deep=True) if action.window is not None else (
        source.window.model_copy(deep=True) if source.window is not None else None
    )
    if action.offset is not None:
        offset = action.offset
    elif action.page == "next":
        if window is not None:
            window = window.model_copy(update={"cursor": window.cursor + page_size})
        else:
            offset = offset + page_size
    elif action.page == "previous":
        if window is not None:
            window = window.model_copy(update={"cursor": max(0, window.cursor - page_size)})
        else:
            offset = max(0, offset - page_size)
    elif action.page == "first" or (action.goal and action.goal != source.goal):
        offset = 0
        if window is not None:
            window = window.model_copy(update={"cursor": 0})
    elif action.presentation and action.page is None:
        offset = 0

    exhaustive = source.exhaustive if action.exhaustive is None else action.exhaustive
    if action.page or (action.goal == "list"):
        exhaustive = True if action.goal == "list" and action.exhaustive is None else exhaustive
    sort_by = action.sort_by or source.sort_by
    sort_direction = action.sort_direction or source.sort_direction
    sample = source.sample if action.sample is None else action.sample
    if sample:
        offset = 0
        window = None
        sort_by = None
        sort_direction = None
    group_value_part = action.group_value_part or source.group_value_part
    presentation = action.presentation or source.presentation
    requested_fields = (
        list(action.requested_fields)
        if action.requested_fields is not None
        else list(source.requested_fields)
    )
    if goal == "list" and not sample and (exhaustive or presentation == "numbered_list") and not sort_by:
        sort_by = "student_name"
        sort_direction = sort_direction or "asc"

    return QueryAction(
        datasets=list(source.datasets),
        goal=goal,
        filters=filters,
        filter_groups=filter_groups,
        filter_logic=filter_logic,
        denominator_filters=list(source.denominator_filters),
        search_text=source.search_text,
        group_by=list(source.group_by),
        group_value_part=group_value_part,
        secondary_group_value_part=source.secondary_group_value_part,
        having_min_count=source.having_min_count,
        top_n=source.top_n,
        per_group_top_n=source.per_group_top_n,
        include_missing=bool(source.include_missing),
        companion_field=source.companion_field,
        stats_value_part=source.stats_value_part,
        interval_start=source.interval_start,
        interval_end=source.interval_end,
        interval_min_days=source.interval_min_days,
        interval_max_days=source.interval_max_days,
        sort_by=sort_by,
        sort_direction=sort_direction,
        sample=sample,
        requested_fields=requested_fields,
        window=window if goal == "list" else None,
        limit=page_size if goal == "list" else action.limit or source.limit,
        offset=offset if goal == "list" else 0,
        exhaustive=bool(exhaustive) if goal == "list" else False,
        presentation=presentation if goal == "list" else None,
        aggregate=source.aggregate.model_copy(deep=True) if source.aggregate is not None else None,
        compare=[branch.model_copy(deep=True) for branch in source.compare],
        verify_previous=False,
    )


def _query_from_active(source: ActiveQuery, *, verify_previous: bool = False) -> QueryAction:
    return QueryAction(
        datasets=list(source.datasets),
        goal=_normalized_goal(source.goal),
        filters=list(source.filters),
        filter_groups=[group.model_copy(deep=True) for group in source.filter_groups],
        filter_logic=source.filter_logic,
        denominator_filters=list(source.denominator_filters),
        search_text=source.search_text,
        group_by=list(source.group_by),
        group_value_part=source.group_value_part,
        secondary_group_value_part=source.secondary_group_value_part,
        having_min_count=source.having_min_count,
        top_n=source.top_n,
        per_group_top_n=source.per_group_top_n,
        include_missing=bool(source.include_missing),
        companion_field=source.companion_field,
        stats_value_part=source.stats_value_part,
        interval_start=source.interval_start,
        interval_end=source.interval_end,
        interval_min_days=source.interval_min_days,
        interval_max_days=source.interval_max_days,
        sort_by=source.sort_by,
        sort_direction=source.sort_direction,
        sample=source.sample,
        requested_fields=list(source.requested_fields),
        window=source.window.model_copy(deep=True) if source.window is not None else None,
        limit=source.limit,
        offset=source.offset,
        exhaustive=source.exhaustive,
        presentation=source.presentation,
        aggregate=source.aggregate.model_copy(deep=True) if source.aggregate is not None else None,
        compare=[branch.model_copy(deep=True) for branch in source.compare],
        verify_previous=verify_previous,
    )


def _append_refinements(existing: list[FilterSpec], changes: list[FilterSpec]) -> list[FilterSpec]:
    merged = list(existing)
    for change in changes:
        if change not in merged:
            merged.append(change)
    return merged


def _dedupe_filter_groups(groups: list[FilterGroupSpec]) -> list[FilterGroupSpec]:
    seen: set[tuple[tuple[str, str, str], ...]] = set()
    result: list[FilterGroupSpec] = []
    for group in groups:
        signature = tuple(
            sorted(
                (item.field.casefold(), item.operator, repr(item.value))
                for item in group.filters
            )
        )
        if signature in seen:
            continue
        seen.add(signature)
        result.append(group)
    return result


def _source_query(
    action: ModifyPreviousAction | VerifyPreviousAction,
    active_query: ActiveQuery | None,
    frames: dict[str, ActiveQuery] | None,
) -> ActiveQuery | None:
    target = getattr(action, "target_action_id", None)
    if target:
        if frames:
            for frame in frames.values():
                if frame.action_id == target or frame.frame_key == target:
                    return frame
        # An explicit target that is gone/stale must not silently fall back to the
        # current topic, which may be a different research population.
        return None
    restore = getattr(action, "restore_frame", None)
    if restore:
        if frames:
            key = str(restore).lower()
            for frame_key, frame in frames.items():
                haystack = " ".join(
                    part
                    for part in (frame_key, frame.topic, frame.frame_key, frame.action_id)
                    if part
                ).lower()
                if key in haystack or frame_key.lower() in key:
                    return frame
        return None
    return active_query


def _apply_changes(existing: list[FilterSpec], changes: list[FilterSpec]) -> list[FilterSpec]:
    """Replace a field constraint as one semantic slot; add it only when absent.

    A flat OR can carry several alternatives for the same field. Replacing that field
    should not leave stale alternatives behind (``A OR B`` -> ``C OR B`` was an
    accidental partial edit). All inherited constraints on the edited field are
    therefore replaced by the one new constraint.
    """
    merged = list(existing)
    for change in changes:
        positions = [index for index, item in enumerate(merged) if item.field == change.field]
        if not positions:
            merged.append(change)
            continue
        insert_at = positions[0]
        merged = [item for item in merged if item.field != change.field]
        merged.insert(insert_at, change)
    return merged


def _replace_group_changes(
    groups: list[FilterGroupSpec], changes: list[FilterSpec]
) -> list[FilterGroupSpec]:
    """Apply explicit substitutions to only the DNF branches that contain that field.

    For ``(A AND B) OR C``, changing A to D means ``(D AND B) OR C``. Adding D to
    the C branch would silently narrow an unrelated alternative. If the supposedly
    replaced field exists in no branch, fail closed; a new condition must be emitted
    as ``change_mode=refine`` instead.
    """
    result = [group.model_copy(deep=True) for group in groups]
    for change in changes:
        matched = False
        updated: list[FilterGroupSpec] = []
        for group in result:
            members = list(group.filters)
            if any(item.field == change.field for item in members):
                matched = True
                positions = [
                    index for index, item in enumerate(members) if item.field == change.field
                ]
                insert_at = positions[0]
                members = [item for item in members if item.field != change.field]
                members.insert(insert_at, change)
            updated.append(FilterGroupSpec(filters=members))
        if not matched:
            raise PlanValidationError(
                f"cannot replace absent inherited field {change.field}",
                code="invalid_followup",
            )
        result = updated
    return _dedupe_filter_groups(result)


def _compile_single(
    query: QueryAction,
    *,
    scope: AccessScope,
    catalog: FieldCatalog,
    needs_evidence: bool,
    wants_synthesis: bool,
) -> QueryPlan:
    file_ids = _resolve_datasets(query.datasets, scope, catalog)
    steps = [PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records")]
    branch_steps, goals, _terminal = _compile_branch(
        query,
        prefix="",
        source="scope_current",
        file_ids=file_ids,
        catalog=catalog,
        needs_evidence=needs_evidence,
        wants_synthesis=wants_synthesis,
    )
    steps.extend(_stamp(branch_steps, action_id="a0", action_goal=query.goal))
    return QueryPlan(
        scope=PlanScope(file_ids=file_ids, version_mode="current", authorized_only=True),
        goals=list(dict.fromkeys(goals)),
        steps=steps,
        planner_type="semantic_compiler",
    )


def _compile_parallel(
    queries: list[QueryAction],
    *,
    scope: AccessScope,
    catalog: FieldCatalog,
    needs_evidence: bool,
    wants_synthesis: bool,
) -> QueryPlan:
    resolved = [_resolve_datasets(query.datasets, scope, catalog) for query in queries]
    scope_ids = tuple(dict.fromkeys(file_id for group in resolved for file_id in group))
    steps = [PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records")]
    goals: list[str] = []
    for index, (query, file_ids) in enumerate(zip(queries, resolved, strict=True)):
        branch_steps, branch_goals, _terminal = _compile_branch(
            query,
            prefix=f"a{index}_",
            source="scope_current",
            file_ids=file_ids,
            catalog=catalog,
            needs_evidence=needs_evidence,
            wants_synthesis=wants_synthesis,
        )
        steps.extend(_stamp(branch_steps, action_id=f"a{index}", action_goal=query.goal))
        goals.extend(branch_goals)
    return QueryPlan(
        scope=PlanScope(file_ids=scope_ids, version_mode="current", authorized_only=True),
        goals=list(dict.fromkeys(goals)),
        steps=steps,
        planner_type="semantic_compiler",
    )


def _compile_branch(
    query: QueryAction,
    *,
    prefix: str,
    source: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
    needs_evidence: bool,
    wants_synthesis: bool,
) -> tuple[list[PlanStep], list[str], str]:
    goal = _GOAL_ALIASES.get(query.goal, query.goal)
    if goal == "compare":
        return _compile_compare(
            query,
            prefix=prefix,
            source=source,
            file_ids=file_ids,
            catalog=catalog,
        )

    steps: list[PlanStep] = []
    current = source
    predicates = _predicates_for(query, file_ids, catalog)
    if predicates:
        step_id = f"{prefix}matches" if prefix else "matches"
        steps.append(
            PlanStep(
                id=step_id,
                op=PlanOp.FILTER,
                input=current,
                where=predicates,
                file_ids=file_ids,
            )
        )
        current = step_id

    if query.search_text or goal == "search":
        current = _append_retrieval(
            steps,
            prefix=prefix,
            current=current,
            query_text=(query.search_text or "").strip() or " ",
            file_ids=file_ids,
        )

    if goal == "rank":
        group_fields = [_canonical_field(name, file_ids, catalog) for name in query.group_by]
        if not group_fields:
            raise QueryPlanBuildError("rank requires a group_by field")
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.RANK,
                input=current,
                fields=group_fields,
                value_part=query.group_value_part,
                secondary_value_part=query.secondary_group_value_part,
                having_min_count=query.having_min_count,
                top_n=query.top_n,
                per_group_top_n=query.per_group_top_n,
                include_missing=query.include_missing,
                file_ids=file_ids,
            )
        )
        return steps, ["rank"], result_id

    if goal == "stats":
        stats_field = _stats_field(query, file_ids, catalog)
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.STATS,
                input=current,
                fields=[stats_field],
                value_part=query.stats_value_part,
                file_ids=file_ids,
            )
        )
        goals = ["stats"]
        if query.group_by:
            grouped_id = f"{prefix}by_group" if prefix else "by_group"
            steps.append(
                PlanStep(
                    id=grouped_id,
                    op=PlanOp.RANK,
                    input=current,
                    fields=[_canonical_field(name, file_ids, catalog) for name in query.group_by],
                    value_part=query.group_value_part,
                    top_n=query.top_n,
                    file_ids=file_ids,
                )
            )
            goals.append("rank")
        return steps, goals, result_id

    if goal == "duplicates":
        duplicate_field = _canonical_field(
            query.group_by[0] if query.group_by else (query.sort_by or "student_name"),
            file_ids,
            catalog,
        )
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.DUPLICATES,
                input=current,
                fields=[duplicate_field],
                value_part=query.group_value_part,
                companion_field=(
                    _canonical_field(query.companion_field, file_ids, catalog)
                    if query.companion_field
                    else None
                ),
                having_min_count=query.having_min_count or 2,
                top_n=query.top_n,
                # With two dates bounding each record's period, only members present at the
                # same time count as sharing it.
                interval_start=(
                    _canonical_field(query.interval_start, file_ids, catalog)
                    if query.interval_start and query.interval_end
                    else None
                ),
                interval_end=(
                    _canonical_field(query.interval_end, file_ids, catalog)
                    if query.interval_start and query.interval_end
                    else None
                ),
                file_ids=file_ids,
            )
        )
        return steps, ["duplicates"], result_id

    if goal == "completeness":
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.COMPLETENESS,
                input=current,
                direction=query.sort_direction or "desc",
                limit=min(query.limit or 10, 50),
                file_ids=file_ids,
            )
        )
        return steps, ["completeness"], result_id

    if goal == "interval":
        start_field, end_field = _interval_fields(query, file_ids, catalog)
        summary_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=summary_id,
                op=PlanOp.INTERVAL_STATS,
                input=current,
                interval_start=start_field,
                interval_end=end_field,
                interval_min_days=query.interval_min_days,
                interval_max_days=query.interval_max_days,
                direction=query.sort_direction or "desc",
                limit=min(query.limit or 10, 50),
                file_ids=file_ids,
            )
        )
        return steps, ["interval"], summary_id

    if goal == "percentage":
        numerator_id = f"{prefix}numerator" if prefix else "numerator"
        denominator_id = f"{prefix}denominator" if prefix else "denominator"
        result_id = f"{prefix}result" if prefix else "result"
        denominator_predicates = [
            _to_predicate(item, file_ids, catalog) for item in query.denominator_filters
        ]
        denominator_source = source
        if denominator_predicates:
            scoped_id = f"{prefix}scope_filter" if prefix else "scope_filter"
            steps.append(
                PlanStep(
                    id=scoped_id,
                    op=PlanOp.FILTER,
                    input=source,
                    where=denominator_predicates,
                    file_ids=file_ids,
                )
            )
            denominator_source = scoped_id
        steps.append(
            PlanStep(id=numerator_id, op=PlanOp.COUNT, input=current, file_ids=file_ids)
        )
        steps.append(
            PlanStep(
                id=denominator_id, op=PlanOp.COUNT, input=denominator_source, file_ids=file_ids
            )
        )
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.PERCENTAGE,
                input=[numerator_id, denominator_id],
                file_ids=file_ids,
            )
        )
        return steps, ["percentage"], result_id

    if goal == "distinct":
        group_fields = [_canonical_field(name, file_ids, catalog) for name in query.group_by]
        if not group_fields:
            raise QueryPlanBuildError("distinct requires a group_by field")
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.GROUP_BY,
                input=current,
                fields=group_fields,
                value_part=query.group_value_part,
                secondary_value_part=query.secondary_group_value_part,
                having_min_count=query.having_min_count,
                top_n=query.top_n,
                per_group_top_n=query.per_group_top_n,
                include_missing=query.include_missing,
                file_ids=file_ids,
            )
        )
        return steps, ["distinct_values"], result_id

    if goal == "aggregate":
        aggregate_steps, aggregate_goals, terminal = _compile_aggregate(
            query,
            steps=steps,
            prefix=prefix,
            current=current,
            file_ids=file_ids,
            catalog=catalog,
        )
        if needs_evidence:
            _append_evidence(
                aggregate_steps,
                prefix,
                current,
                file_ids,
                catalog,
                query.limit,
                PlanOp.GET_EVIDENCE,
            )
        return aggregate_steps, aggregate_goals, terminal

    if goal == "count":
        result_id = f"{prefix}result" if prefix else "result"
        if len(file_ids) > 1 and not predicates and not query.search_text:
            grouped_id = f"{prefix}by_dataset" if prefix else "by_dataset"
            steps.append(
                PlanStep(id=grouped_id, op=PlanOp.GROUP_BY, input=current, fields=["file_id"], file_ids=file_ids)
            )
            current = grouped_id
        steps.append(PlanStep(id=result_id, op=PlanOp.COUNT, input=current, file_ids=file_ids))
        if needs_evidence:
            _append_evidence(steps, prefix, current if predicates else source, file_ids, catalog, query.limit, PlanOp.GET_EVIDENCE)
        return steps, ["count"], result_id

    if goal == "quote":
        result_id = f"{prefix}result" if prefix else "result"
        quote_fields = [
            name
            for name in ("notes", "cause_of_death", "student_name")
            if _quoteable(file_ids, name, catalog)
        ]
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.GET_QUOTE,
                input=current,
                fields=quote_fields or _default_projection(file_ids, catalog),
                limit=min(query.limit or 8, 8),
                file_ids=file_ids,
            )
        )
        goals = ["quote"]
        if wants_synthesis:
            goals.append("synthesis")
        return steps, goals, result_id

    if goal == "provenance":
        result_id = f"{prefix}result" if prefix else "result"
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.GET_PROVENANCE,
                input=current,
                limit=min(query.limit or 20, 50),
                file_ids=file_ids,
            )
        )
        return steps, ["provenance"], result_id

    if goal == "dossier":
        # Everything the authorized catalog holds about the named person, so the
        # synthesis step has the family, admission, death, and burial fields instead
        # of a four-field preview.
        result_id = f"{prefix}result" if prefix else "result"
        every_field = _all_projectable_fields(file_ids, catalog)
        steps.append(
            PlanStep(
                id=result_id,
                op=PlanOp.PROJECT,
                input=current,
                fields=every_field,
                limit=min(query.limit or 10, 25),
                file_ids=file_ids,
            )
        )
        evidence_id = f"{prefix}evidence" if prefix else "evidence"
        steps.append(
            PlanStep(
                id=evidence_id,
                op=PlanOp.GET_EVIDENCE,
                input=result_id,
                fields=_all_evidence_fields(file_ids, catalog),
                limit=min(query.limit or 10, 25),
                file_ids=file_ids,
            )
        )
        return steps, ["dossier", "synthesis"], result_id

    # list / search: presentation grouping is SORT, never SQL GROUP_BY
    sort_field = None
    if query.group_by and goal == "list":
        sort_field = _canonical_field(query.group_by[0], file_ids, catalog)
        sort_direction = "ASC"
    elif query.sort_by and not query.sample:
        sort_field = _canonical_field(query.sort_by, file_ids, catalog)
        sort_direction = "DESC" if (query.sort_direction or "").lower() == "desc" else "ASC"
    elif goal == "list" and not query.sample:
        sort_field = _default_list_sort(file_ids, catalog)
        sort_direction = "ASC"
    if sort_field:
        sort_id = f"{prefix}sorted" if prefix else "sorted"
        steps.append(
            PlanStep(
                id=sort_id,
                op=PlanOp.SORT,
                input=current,
                sort_field=sort_field,
                sort_direction=sort_direction,
                file_ids=file_ids,
            )
        )
        current = sort_id

    if goal == "list" and not query.sample:
        count_id = f"{prefix}total" if prefix else "total"
        steps.append(
            PlanStep(id=count_id, op=PlanOp.COUNT, input=current, file_ids=file_ids)
        )

    window = query.window
    exhaustive_list = goal == "list" and query.exhaustive and not query.sample
    ceiling = MAX_ENUMERATION_SIZE if exhaustive_list else MAX_PAGE_SIZE
    default_page = MAX_ENUMERATION_SIZE if exhaustive_list else DEFAULT_PAGE_SIZE
    page = query.limit or (
        min(window.size, ceiling)
        if window is not None and goal == "list"
        else default_page if goal in {"list", "search"} else None
    )
    offset = query.offset or 0
    if page and query.sample:
        sampled_id = f"{prefix}sampled" if prefix else "sampled"
        steps.append(
            PlanStep(
                id=sampled_id,
                op=PlanOp.SAMPLE,
                input=current,
                limit=min(page, MAX_PAGE_SIZE),
                file_ids=file_ids,
            )
        )
        current = sampled_id
    elif page:
        limited_id = f"{prefix}limited" if prefix else "limited"
        steps.append(
            PlanStep(
                id=limited_id,
                op=PlanOp.LIMIT,
                input=current,
                limit=min(page, ceiling),
                offset=offset,
                window_anchor=window.anchor if window else None,
                window_size=window.size if window else None,
                window_cursor=window.cursor if window else 0,
                file_ids=file_ids,
            )
        )
        current = limited_id

    projection = _requested_projection(query.requested_fields, file_ids, catalog)
    if query.group_by:
        for name in query.group_by:
            field = _canonical_field(name, file_ids, catalog)
            if field not in projection:
                projection.append(field)
    if not projection:
        raise QueryPlanBuildError("no common projection field exists for selected datasets")
    result_id = f"{prefix}result" if prefix else "result"
    steps.append(
        PlanStep(
            id=result_id,
            op=PlanOp.PROJECT,
            input=current,
            fields=projection,
            limit=min(page, ceiling) if page else None,
            offset=offset if goal == "list" else None,
            window_anchor=window.anchor if window else None,
            window_size=window.size if window else None,
            window_cursor=window.cursor if window else 0,
            file_ids=file_ids,
        )
    )
    goals = [goal]
    if query.verify_previous:
        goals.append("verify")
    if needs_evidence or goal == "search":
        _append_evidence(steps, prefix, result_id, file_ids, catalog, query.limit, PlanOp.GET_EVIDENCE)
    if wants_synthesis:
        goals.append("synthesis")
    return steps, goals, result_id


def _all_projectable_fields(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    """Every semantic field common to the selected datasets, identity first."""
    common = [
        spec.semantic_field
        for spec in catalog.fields_for(file_ids[0])
        if all(catalog.resolve_field(file_id, spec.semantic_field) is not None for file_id in file_ids)
    ] if file_ids else []
    identity = "student_name"
    ordered = [identity] if identity in common else []
    ordered.extend(name for name in common if name != identity)
    return ordered or _default_projection(file_ids, catalog)


def _all_evidence_fields(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    return [
        name
        for name in _all_projectable_fields(file_ids, catalog)
        if _evidence_allowed(file_ids, name, catalog)
    ]


def _stats_field(query: QueryAction, file_ids: tuple[int, ...], catalog: FieldCatalog) -> str:
    candidate = (
        (query.aggregate.field if query.aggregate is not None else None)
        or query.sort_by
        or (query.requested_fields[0] if query.requested_fields else None)
    )
    if not candidate:
        raise QueryPlanBuildError("stats requires a field to measure")
    return _canonical_field(candidate, file_ids, catalog)


def _interval_fields(
    query: QueryAction,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> tuple[str, str]:
    if not query.interval_start or not query.interval_end:
        raise QueryPlanBuildError("interval requires a start and an end date field")
    return (
        _canonical_field(query.interval_start, file_ids, catalog),
        _canonical_field(query.interval_end, file_ids, catalog),
    )


def _compile_compare(
    query: QueryAction,
    *,
    prefix: str,
    source: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> tuple[list[PlanStep], list[str], str]:
    if len(query.compare) < 2:
        raise QueryPlanBuildError("compare requires at least two branches")
    steps: list[PlanStep] = []
    terminals: list[str] = []
    for index, branch in enumerate(query.compare):
        current = source
        predicates = [_to_predicate(item, file_ids, catalog) for item in branch.filters]
        if predicates:
            match_id = f"{prefix}cmp{index}_matches"
            steps.append(
                PlanStep(
                    id=match_id,
                    op=PlanOp.FILTER,
                    input=current,
                    where=predicates,
                    file_ids=file_ids,
                )
            )
            current = match_id
        if branch.search_text:
            current = _append_retrieval(
                steps,
                prefix=f"{prefix}cmp{index}_",
                current=current,
                query_text=branch.search_text,
                file_ids=file_ids,
            )
        count_id = f"{prefix}cmp{index}_count"
        steps.append(PlanStep(id=count_id, op=PlanOp.COUNT, input=current, file_ids=file_ids))
        terminals.append(count_id)
    result_id = f"{prefix}result" if prefix else "result"
    steps.append(PlanStep(id=result_id, op=PlanOp.COMPARE, input=terminals, file_ids=file_ids))
    return steps, ["compare"], result_id


def _compile_aggregate(
    query: QueryAction,
    *,
    steps: list[PlanStep],
    prefix: str,
    current: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> tuple[list[PlanStep], list[str], str]:
    spec = query.aggregate or AggregateSpec(function="count")
    if query.group_by:
        grouped_id = f"{prefix}grouped" if prefix else "grouped"
        steps.append(
            PlanStep(
                id=grouped_id,
                op=PlanOp.GROUP_BY,
                input=current,
                fields=[_canonical_field(name, file_ids, catalog) for name in query.group_by],
                file_ids=file_ids,
            )
        )
        current = grouped_id
    result_id = f"{prefix}result" if prefix else "result"
    if spec.function in {"median", "mode", "stats"}:
        op = PlanOp.STATS
    else:
        try:
            op = _AGG_OP[spec.function]
        except KeyError as exc:  # defensive if a future schema expands before this compiler
            raise QueryPlanBuildError(
                f"unsupported aggregate function {spec.function}"
            ) from exc
    fields: list[str] = []
    if spec.function != "count":
        if not spec.field:
            raise QueryPlanBuildError(f"{spec.function} requires a field")
        fields = [_canonical_field(spec.field, file_ids, catalog)]
    steps.append(
        PlanStep(id=result_id, op=op, input=current, fields=fields, file_ids=file_ids)
    )
    return steps, ["aggregate", spec.function], result_id


def _append_retrieval(
    steps: list[PlanStep],
    *,
    prefix: str,
    current: str,
    query_text: str,
    file_ids: tuple[int, ...],
) -> str:
    fts_id = f"{prefix}fts" if prefix else "fts"
    fuzzy_id = f"{prefix}fuzzy" if prefix else "fuzzy"
    fused_id = f"{prefix}fused" if prefix else "fused"
    unique_id = f"{prefix}unique" if prefix else "unique"
    steps.append(
        PlanStep(id=fts_id, op=PlanOp.FULL_TEXT_SEARCH, input=current, query=query_text, file_ids=file_ids)
    )
    steps.append(
        PlanStep(id=fuzzy_id, op=PlanOp.FUZZY_SEARCH, input=current, query=query_text, file_ids=file_ids)
    )
    steps.append(PlanStep(id=fused_id, op=PlanOp.UNION, input=[fts_id, fuzzy_id], file_ids=file_ids))
    steps.append(PlanStep(id=unique_id, op=PlanOp.DEDUPLICATE, input=fused_id, file_ids=file_ids))
    return unique_id


def _append_evidence(
    steps: list[PlanStep],
    prefix: str,
    source: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
    limit: int | None,
    op: PlanOp,
) -> None:
    # Once a record is located, hand back the record. The four fixed fields this used to
    # pick -- name, community, cause of death, notes -- meant a question about a birth
    # date, a parent or an admission reached the model with none of them in the row, and
    # the answer came back as "not recorded" over a row that recorded it.
    #
    # _all_evidence_fields is already filtered by evidence_allowed, which is how the
    # derived name and date columns stay out: they are how the row was found, not what it
    # says.
    evidence_fields = _all_evidence_fields(file_ids, catalog)
    if not evidence_fields:
        return
    steps.append(
        PlanStep(
            id=f"{prefix}evidence" if prefix else "evidence",
            op=op,
            input=source,
            fields=evidence_fields,
            limit=min(limit or 8, 8),
            file_ids=file_ids,
        )
    )


def _resolve_datasets(datasets: list[int], scope: AccessScope, catalog: FieldCatalog) -> tuple[int, ...]:
    requested = tuple(int(item) for item in datasets)
    if not requested:
        default = catalog.default_people_file_id
        if default in scope.allowed_file_ids and catalog.dataset(default) is not None:
            return (default,)
        visible = [item.file_id for item in catalog.datasets if item.file_id in scope.allowed_file_ids]
        if not visible:
            raise PlanValidationError("no authorized dataset could be selected", code="access_restricted")
        return (visible[0],)
    extra = [file_id for file_id in requested if file_id not in scope.allowed_file_ids]
    if extra:
        raise PlanValidationError("requested datasets are not in the current access scope", code="access_restricted")
    hidden = [file_id for file_id in requested if catalog.dataset(file_id) is None]
    if hidden:
        raise PlanValidationError("requested datasets are not in the current access scope", code="access_restricted")
    return requested


def _predicates_for(
    query: QueryAction,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> list[Predicate]:
    """Compile flat filters or bounded DNF into the trusted predicate tree."""
    if query.filter_groups:
        conjunctions: list[Predicate] = []
        for group in query.filter_groups:
            leaves = [_to_predicate(item, file_ids, catalog) for item in group.filters]
            conjunctions.append(leaves[0] if len(leaves) == 1 else all_of(*leaves))
        if len(conjunctions) == 1:
            return [conjunctions[0]]
        return [any_of(*conjunctions)]

    leaves = [_to_predicate(item, file_ids, catalog) for item in query.filters]
    if len(leaves) > 1 and query.filter_logic == "or":
        return [any_of(*leaves)]
    return leaves


def _to_predicate(spec: FilterSpec, file_ids: tuple[int, ...], catalog: FieldCatalog) -> Predicate:
    resolved = None
    for file_id in file_ids:
        resolved = catalog.resolve_field(file_id, spec.field) or resolved
    field_name = resolved.semantic_field if resolved else spec.field
    value = spec.value
    if field_name == "community" and isinstance(value, str):
        entity = resolve_entity(value)
        if entity and not entity.ambiguous:
            value = entity.canonical
    if spec.operator in {"YEAR_EQUALS", "BEFORE", "AFTER"} and isinstance(value, str) and value.isdigit():
        value = int(value)
    return Predicate(field=field_name, operator=FilterOperator(spec.operator), value=value)


def _canonical_field(name: str, file_ids: tuple[int, ...], catalog: FieldCatalog) -> str:
    for file_id in file_ids:
        spec = catalog.resolve_field(file_id, name)
        if spec is not None:
            return spec.semantic_field
    return name


def _default_list_sort(file_ids: tuple[int, ...], catalog: FieldCatalog) -> str | None:
    for candidate in ("student_name", "community"):
        if all(catalog.resolve_field(file_id, candidate) is not None for file_id in file_ids):
            return candidate
    return None


def _default_projection(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    for candidate in ("student_name", "community"):
        if all(catalog.resolve_field(file_id, candidate) is not None for file_id in file_ids):
            return [candidate]
    return []


def _requested_projection(
    requested: list[str],
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> list[str]:
    if not requested:
        return _default_projection(file_ids, catalog)
    projection: list[str] = []
    identity = _canonical_field("student_name", file_ids, catalog)
    if all(catalog.resolve_field(file_id, identity) is not None for file_id in file_ids):
        projection.append(identity)
    unsupported: list[str] = []
    for raw in requested:
        specs = [catalog.resolve_field(file_id, raw) for file_id in file_ids]
        if not specs or any(spec is None for spec in specs):
            unsupported.append(raw)
            continue
        semantic = specs[0].semantic_field
        if any(spec.semantic_field != semantic for spec in specs if spec is not None):
            unsupported.append(raw)
            continue
        if semantic not in projection:
            projection.append(semantic)
    if unsupported:
        raise UnsupportedFieldsError(unsupported, file_ids=file_ids, catalog=catalog)
    return projection


def _common_field_labels(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[str]:
    if not file_ids:
        return []
    first = catalog.fields_for(file_ids[0])
    labels: list[str] = []
    for spec in first:
        if all(catalog.resolve_field(file_id, spec.semantic_field) is not None for file_id in file_ids):
            labels.append(spec.human_label)
    return labels


def _evidence_allowed(file_ids: tuple[int, ...], field: str, catalog: FieldCatalog) -> bool:
    specs = [catalog.resolve_field(file_id, field) for file_id in file_ids]
    return any(spec is not None and spec.evidence_allowed for spec in specs)


def _quoteable(file_ids: tuple[int, ...], field: str, catalog: FieldCatalog) -> bool:
    specs = [catalog.resolve_field(file_id, field) for file_id in file_ids]
    return any(spec is not None and spec.quoteable for spec in specs)
