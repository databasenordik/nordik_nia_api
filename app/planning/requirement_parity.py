"""One-to-one comparison of stated planner requirements against bound actions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.planning.ai_turn_schema import (
    AggregateRequirement,
    CompanionRequirement,
    FilterLogicRequirement,
    FilterRequirement,
    GoalRequirement,
    GroupingRequirement,
    HavingRequirement,
    IntervalRequirement,
    ModifyEditRequirement,
    NumericSlotRequirement,
    OptionRequirement,
    PlannedTurn,
    ProjectionRequirement,
    SearchTextRequirement,
    SortingRequirement,
    StatedRequirement,
    WindowRequirement,
)
from app.planning.plan_validator import PlanValidationError
from app.planning.turn_schema import (
    ModifyPreviousAction,
    QueryAction,
    TurnPlan,
    VerifyPreviousAction,
)

logger = logging.getLogger("nia.planning.parity")

@dataclass(frozen=True)
class RequirementMatch:
    stated: StatedRequirement
    matched: bool


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        # A value list is a set: IN [F, female] and IN [female, F, female] select the same rows.
        return "|".join(sorted({_norm(item) for item in value}))
    return " ".join(str(value).casefold().split())


def extract_expressed(turn: TurnPlan) -> list[StatedRequirement]:
    expressed: list[StatedRequirement] = []
    for index, action in enumerate(turn.actions):
        if isinstance(action, QueryAction):
            expressed.extend(_from_query(index, action))
        elif isinstance(action, ModifyPreviousAction):
            expressed.extend(_from_modify(index, action))
        elif isinstance(action, VerifyPreviousAction):
            expressed.append(
                OptionRequirement(
                    action_index=index,
                    text="verify previous",
                    option="verify_previous",
                    value=True,
                )
            )
    return expressed


def _from_query(index: int, action: QueryAction) -> list[StatedRequirement]:
    items: list[StatedRequirement] = [
        GoalRequirement(action_index=index, text=f"goal {action.goal}", goal=action.goal)
    ]
    if action.filter_logic != "and" and not action.filter_groups:
        items.append(
            FilterLogicRequirement(
                action_index=index,
                text=f"filter logic {action.filter_logic}",
                filter_logic=action.filter_logic,
            )
        )
    for spec in action.filters:
        items.append(
            FilterRequirement(
                action_index=index,
                text=f"{spec.field} {spec.operator} {spec.value}",
                collection="filters",
                field=spec.field,
                operator=spec.operator,
                value=spec.value,
            )
        )
    for group_index, group in enumerate(action.filter_groups):
        for spec in group.filters:
            items.append(
                FilterRequirement(
                    action_index=index,
                    text=f"boolean group {group_index + 1}: {spec.field} {spec.operator} {spec.value}",
                    collection="filter_groups",
                    field=spec.field,
                    operator=spec.operator,
                    value=spec.value,
                    group_index=group_index,
                )
            )
    for spec in action.denominator_filters:
        items.append(
            FilterRequirement(
                action_index=index,
                text=f"denominator {spec.field} {spec.operator} {spec.value}",
                collection="denominator_filters",
                field=spec.field,
                operator=spec.operator,
                value=spec.value,
            )
        )
    if action.search_text:
        items.append(
            SearchTextRequirement(
                action_index=index,
                text=action.search_text,
                search_text=action.search_text,
            )
        )
    if action.requested_fields:
        items.append(
            ProjectionRequirement(
                action_index=index,
                text=("project " + ", ".join(action.requested_fields))[:2000],
                requested_fields=list(action.requested_fields),
            )
        )
    for position, field in enumerate(action.group_by[:2]):
        part = action.group_value_part if position == 0 else action.secondary_group_value_part
        items.append(
            GroupingRequirement(
                action_index=index,
                text=f"group by {field}",
                field=field,
                position=0 if position == 0 else 1,
                value_part=part,
            )
        )
    if action.sort_by:
        items.append(
            SortingRequirement(
                action_index=index,
                text=f"sort {action.sort_by} {action.sort_direction or 'asc'}",
                field=action.sort_by,
                direction=action.sort_direction or "asc",
            )
        )
    if (
        action.interval_start
        or action.interval_end
        or action.interval_min_days is not None
        or action.interval_max_days is not None
    ):
        items.append(
            IntervalRequirement(
                action_index=index,
                text="interval",
                interval_start=action.interval_start,
                interval_end=action.interval_end,
                interval_min_days=action.interval_min_days,
                interval_max_days=action.interval_max_days,
            )
        )
    if action.aggregate is not None:
        items.append(
            AggregateRequirement(
                action_index=index,
                text=f"{action.aggregate.function} {action.aggregate.field or ''}".strip(),
                function=action.aggregate.function,
                field=action.aggregate.field,
                stats_value_part=action.stats_value_part,
            )
        )
    if action.companion_field:
        items.append(
            CompanionRequirement(
                action_index=index,
                text=action.companion_field,
                companion_field=action.companion_field,
            )
        )
    if action.having_min_count is not None:
        items.append(
            HavingRequirement(
                action_index=index,
                text=f"having {action.having_min_count}",
                having_min_count=action.having_min_count,
            )
        )
    for slot, value in (
        ("top_n", action.top_n),
        ("per_group_top_n", action.per_group_top_n),
        ("limit", action.limit),
        ("offset", action.offset),
    ):
        if value is not None:
            items.append(
                NumericSlotRequirement(
                    action_index=index,
                    text=f"{slot} {value}",
                    slot=slot,  # type: ignore[arg-type]
                    value=value,
                )
            )
    if action.window is not None:
        items.append(
            WindowRequirement(
                action_index=index,
                text="window",
                anchor=action.window.anchor,
                size=action.window.size,
                cursor=action.window.cursor,
            )
        )
    if action.include_missing:
        items.append(
            OptionRequirement(
                action_index=index, text="include missing", option="include_missing", value=True
            )
        )
    if action.sample:
        items.append(OptionRequirement(action_index=index, text="sample", option="sample", value=True))
    if action.exhaustive:
        items.append(
            OptionRequirement(action_index=index, text="exhaustive", option="exhaustive", value=True)
        )
    if action.presentation:
        items.append(
            OptionRequirement(
                action_index=index,
                text=f"presentation {action.presentation}",
                option="presentation",
                value=action.presentation,
            )
        )
    if action.verify_previous:
        items.append(
            OptionRequirement(
                action_index=index, text="verify previous", option="verify_previous", value=True
            )
        )
    for branch_index, branch in enumerate(action.compare):
        for spec in branch.filters:
            items.append(
                FilterRequirement(
                    action_index=index,
                    text=f"compare {branch.label or branch_index} {spec.field}",
                    collection="compare",
                    field=spec.field,
                    operator=spec.operator,
                    value=spec.value,
                    branch_index=branch_index,
                    branch_label=branch.label,
                )
            )
        if branch.search_text:
            items.append(
                SearchTextRequirement(
                    action_index=index,
                    text=branch.search_text,
                    search_text=branch.search_text,
                    branch_index=branch_index,
                    branch_label=branch.label,
                )
            )
    return items


def _from_modify(index: int, action: ModifyPreviousAction) -> list[StatedRequirement]:
    items: list[StatedRequirement] = []
    for spec in action.changes:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text=f"{spec.field} {spec.operator} {spec.value}",
                edit="filter",
                value=spec.model_dump(mode="json"),
            )
        )
    if action.changes and action.change_mode == "refine":
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text="refine previous population",
                edit="change_mode",
                value="refine",
            )
        )
    if action.goal:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"goal {action.goal}", edit="goal", value=action.goal)
        )
    if action.page:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"page {action.page}", edit="page", value=action.page)
        )
    if action.restore_frame:
        items.append(
            ModifyEditRequirement(
                action_index=index, text="restore frame", edit="restore_frame", value=action.restore_frame
            )
        )
    if action.sort_by:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"sort {action.sort_by}", edit="sort_by", value=action.sort_by)
        )
    if action.sort_direction:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text=f"sort {action.sort_direction}",
                edit="sort_direction",
                value=action.sort_direction,
            )
        )
    if action.limit is not None:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"limit {action.limit}", edit="limit", value=action.limit)
        )
    if action.offset is not None:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"offset {action.offset}", edit="offset", value=action.offset)
        )
    if action.sample is not None:
        items.append(
            ModifyEditRequirement(action_index=index, text=f"sample {action.sample}", edit="sample", value=action.sample)
        )
    if action.requested_fields is not None:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text="project " + ", ".join(action.requested_fields),
                edit="requested_fields",
                value=list(action.requested_fields),
            )
        )
    if action.window is not None:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text="window",
                edit="window",
                value=action.window.model_dump(mode="json"),
            )
        )
    if action.exhaustive is not None:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text=f"exhaustive {action.exhaustive}",
                edit="exhaustive",
                value=action.exhaustive,
            )
        )
    if action.presentation:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text=f"presentation {action.presentation}",
                edit="presentation",
                value=action.presentation,
            )
        )
    if action.group_value_part:
        items.append(
            ModifyEditRequirement(
                action_index=index,
                text=f"group value part {action.group_value_part}",
                edit="group_value_part",
                value=action.group_value_part,
            )
        )
    return items


def _same(stated: StatedRequirement, expressed: StatedRequirement) -> bool:
    if stated.kind != expressed.kind or stated.action_index != expressed.action_index:
        return False
    if isinstance(stated, GoalRequirement) and isinstance(expressed, GoalRequirement):
        return stated.goal == expressed.goal
    if isinstance(stated, FilterRequirement) and isinstance(expressed, FilterRequirement):
        return (
            stated.collection == expressed.collection
            and _norm(stated.field) == _norm(expressed.field)
            and stated.operator == expressed.operator
            and _norm(stated.value) == _norm(expressed.value)
            and stated.branch_index == expressed.branch_index
            and _norm(stated.branch_label or "") == _norm(expressed.branch_label or "")
            and stated.group_index == expressed.group_index
        )
    if isinstance(stated, FilterLogicRequirement) and isinstance(expressed, FilterLogicRequirement):
        return stated.filter_logic == expressed.filter_logic
    if isinstance(stated, SearchTextRequirement) and isinstance(expressed, SearchTextRequirement):
        return (
            _norm(stated.search_text) == _norm(expressed.search_text)
            and stated.branch_index == expressed.branch_index
            and _norm(stated.branch_label or "") == _norm(expressed.branch_label or "")
        )
    if isinstance(stated, ProjectionRequirement) and isinstance(expressed, ProjectionRequirement):
        return [_norm(item) for item in stated.requested_fields] == [
            _norm(item) for item in expressed.requested_fields
        ]
    if isinstance(stated, GroupingRequirement) and isinstance(expressed, GroupingRequirement):
        return (
            _norm(stated.field) == _norm(expressed.field)
            and stated.position == expressed.position
            and stated.value_part == expressed.value_part
        )
    if isinstance(stated, SortingRequirement) and isinstance(expressed, SortingRequirement):
        return _norm(stated.field) == _norm(expressed.field) and stated.direction == expressed.direction
    if isinstance(stated, IntervalRequirement) and isinstance(expressed, IntervalRequirement):
        return (
            _norm(stated.interval_start) == _norm(expressed.interval_start)
            and _norm(stated.interval_end) == _norm(expressed.interval_end)
            and stated.interval_min_days == expressed.interval_min_days
            and stated.interval_max_days == expressed.interval_max_days
        )
    if isinstance(stated, AggregateRequirement) and isinstance(expressed, AggregateRequirement):
        return (
            _norm(stated.function) == _norm(expressed.function)
            and _norm(stated.field) == _norm(expressed.field)
            and stated.stats_value_part == expressed.stats_value_part
        )
    if isinstance(stated, CompanionRequirement) and isinstance(expressed, CompanionRequirement):
        return _norm(stated.companion_field) == _norm(expressed.companion_field)
    if isinstance(stated, HavingRequirement) and isinstance(expressed, HavingRequirement):
        return stated.having_min_count == expressed.having_min_count
    if isinstance(stated, NumericSlotRequirement) and isinstance(expressed, NumericSlotRequirement):
        return stated.slot == expressed.slot and stated.value == expressed.value
    if isinstance(stated, WindowRequirement) and isinstance(expressed, WindowRequirement):
        return stated.anchor == expressed.anchor and stated.size == expressed.size and stated.cursor == expressed.cursor
    if isinstance(stated, OptionRequirement) and isinstance(expressed, OptionRequirement):
        return stated.option == expressed.option and _norm(stated.value) == _norm(expressed.value)
    if isinstance(stated, ModifyEditRequirement) and isinstance(expressed, ModifyEditRequirement):
        return _norm(stated.edit) == _norm(expressed.edit) and _norm(stated.value) == _norm(expressed.value)
    return False


def _canonicalize_dnf_groups(
    requirements: list[StatedRequirement],
) -> list[StatedRequirement]:
    """Make OR-of-AND group numbering semantic rather than positional.

    DNF group order does not change meaning. A primary plan may emit ``[A,B] OR [C]``
    while an independent reviewer reconstructs ``[C] OR [B,A]``. Comparing raw
    group_index values would reject those equivalent expressions. Canonical ranks are
    derived from each group's normalized filter content, while membership inside each
    conjunction remains intact.
    """
    by_action: dict[int, dict[int, list[FilterRequirement]]] = {}
    for item in requirements:
        if (
            isinstance(item, FilterRequirement)
            and item.collection == "filter_groups"
            and item.group_index is not None
        ):
            by_action.setdefault(item.action_index, {}).setdefault(item.group_index, []).append(item)

    mappings: dict[tuple[int, int], int] = {}
    for action_index, groups in by_action.items():
        signatures: list[tuple[tuple[tuple[str, str, str], ...], int]] = []
        for old_index, members in groups.items():
            signature = tuple(
                sorted(
                    (
                        _norm(member.field),
                        str(member.operator),
                        _norm(member.value),
                    )
                    for member in members
                )
            )
            signatures.append((signature, old_index))
        for canonical_index, (_, old_index) in enumerate(sorted(signatures)):
            mappings[(action_index, old_index)] = canonical_index

    if not mappings:
        return list(requirements)

    normalized: list[StatedRequirement] = []
    for item in requirements:
        if (
            isinstance(item, FilterRequirement)
            and item.collection == "filter_groups"
            and item.group_index is not None
        ):
            normalized.append(
                item.model_copy(
                    update={
                        "group_index": mappings.get(
                            (item.action_index, item.group_index), item.group_index
                        )
                    }
                )
            )
        else:
            normalized.append(item)
    return normalized


def _explode(requirements: list[StatedRequirement]) -> list[StatedRequirement]:
    """Split a multi-field projection into one requirement per field.

    A projection is a set of fields, not a sequence. Comparing the lists whole made
    parity order-sensitive, so a plan that projected exactly the requested fields in a
    different order was reported as having dropped them. Per-field entries make the
    comparison order-insensitive and strictly sharper: a genuinely missing field still
    fails to match, and the error names that field instead of the whole projection.
    """
    exploded: list[StatedRequirement] = []
    for item in _canonicalize_dnf_groups(requirements):
        if isinstance(item, ProjectionRequirement) and len(item.requested_fields) != 1:
            for name in item.requested_fields:
                exploded.append(
                    item.model_copy(update={"text": f"project {name}", "requested_fields": [name]})
                )
            continue
        exploded.append(item)
    return exploded


def unaccounted_conditions(planned: PlannedTurn, bound: TurnPlan) -> list[str]:
    """Conditions the plan carries that no stated requirement accounts for.

    The reverse direction -- a stated requirement with nothing to match -- is a hard error in
    check_requirement_parity, because dropping a condition answers a question nobody asked.
    An invented one is just as wrong and reads exactly as confident: "records missing a cause
    of death" came back filtered to those with a known death date too, 2 instead of 6, and
    "missing a community" picked up a known cause of death, 0 instead of 1.

    Nothing is deleted on this signal. The planner
    under-declaring a filter the researcher did ask for is indistinguishable at this layer
    from inventing one, and silently dropping the first case would be the very condition loss
    the check above exists to prevent. The list is handed to the reviewer, which can see
    the question and decide. A separate narrowing-only provenance gate fails closed later if
    a filter/search/OR condition remains unexplained after review.
    """
    expressed = _explode(extract_expressed(bound))
    unused = list(expressed)
    for stated in _explode(list(planned.stated_requirements)):
        match_index = next((index for index, item in enumerate(unused) if _same(stated, item)), None)
        if match_index is None:
            continue
        unused.pop(match_index)
    return sorted(item.text for item in unused if item.text)



def requirements_not_reconstructed(
    candidate_requirements: list[StatedRequirement],
    reconstructed_requirements: list[StatedRequirement],
) -> list[str]:
    """Candidate-declared requirements absent from independent review reconstruction."""
    unused_reconstructed = list(_explode(list(reconstructed_requirements)))
    missing: list[str] = []
    for candidate in _explode(list(candidate_requirements)):
        match_index = next(
            (index for index, item in enumerate(unused_reconstructed) if _same(candidate, item)),
            None,
        )
        if match_index is None:
            missing.append(candidate.text)
            continue
        unused_reconstructed.pop(match_index)
    return sorted(item for item in missing if item)


def requirements_not_expressed(
    requirements: list[StatedRequirement], bound: TurnPlan
) -> list[str]:
    """Reviewer-reconstructed requirements that the candidate does not express."""
    unused_expressed = list(_explode(extract_expressed(bound)))
    missing: list[str] = []
    for expected in _explode(list(requirements)):
        match_index = next(
            (index for index, item in enumerate(unused_expressed) if _same(expected, item)),
            None,
        )
        if match_index is None:
            missing.append(expected.text)
            continue
        unused_expressed.pop(match_index)
    return sorted(item for item in missing if item)


def narrowing_not_reconstructed(
    requirements: list[StatedRequirement], bound: TurnPlan
) -> list[str]:
    """Candidate narrowing constraints absent from the reviewer's reconstruction."""
    unused_expressed = list(_explode(extract_expressed(bound)))
    for expected in _explode(list(requirements)):
        match_index = next(
            (index for index, item in enumerate(unused_expressed) if _same(expected, item)),
            None,
        )
        if match_index is not None:
            unused_expressed.pop(match_index)
    narrowing_types = (FilterRequirement, SearchTextRequirement, FilterLogicRequirement)
    return sorted(
        item.text for item in unused_expressed
        if isinstance(item, narrowing_types) and item.text
    )


def correction_semantics_not_reconstructed(
    reconstructed_requirements: list[StatedRequirement],
    original: TurnPlan,
    corrected: TurnPlan,
) -> list[str]:
    """Semantic changes introduced by review that its reconstruction does not justify.

    A corrected review returns a *full replacement* ``PlannedTurn``. Live providers often
    preserve an unchanged action field (for example an explicit ``limit 5``) while omitting
    the duplicate stated-requirement entry in the replacement. Treating every such omission
    as a new semantic invention caused correct repairs to fail with
    ``review_correction_unverified``.

    The safety property we actually need is narrower and stronger: anything the reviewer
    *changes or adds* relative to the already-validated primary action must be present in the
    independent reconstruction. Unchanged semantics keep the provenance they already had.
    Presentation-only formatting and an explicit zero offset are execution/display defaults,
    not changes to the selected population or requested statistic, so they are ignored here.
    Narrowing predicates remain protected separately by ``narrowing_not_reconstructed``.
    """
    original_unused = list(_explode(extract_expressed(original)))
    introduced: list[StatedRequirement] = []
    for item in _explode(extract_expressed(corrected)):
        match_index = next(
            (index for index, prior in enumerate(original_unused) if _same(item, prior)),
            None,
        )
        if match_index is None:
            introduced.append(item)
        else:
            original_unused.pop(match_index)

    reconstructed_unused = list(_explode(list(reconstructed_requirements)))
    missing: list[str] = []
    for item in introduced:
        # These fields do not alter which rows/statistic the query computes. Review models
        # frequently materialize them while emitting a full replacement object.
        if isinstance(item, OptionRequirement) and item.option == "presentation":
            continue
        if isinstance(item, NumericSlotRequirement) and item.slot == "offset" and item.value == 0:
            continue
        match_index = next(
            (
                index
                for index, expected in enumerate(reconstructed_unused)
                if _same(item, expected)
            ),
            None,
        )
        if match_index is None:
            if item.text:
                missing.append(item.text)
            continue
        reconstructed_unused.pop(match_index)
    return sorted(missing)


def unaccounted_narrowing_conditions(planned: PlannedTurn, bound: TurnPlan) -> list[str]:
    """Unstated constraints that can change which records are eligible.

    Grouping, projection, presentation, and other output-shape differences are still
    reported by :func:`unaccounted_conditions` for review, but they do not justify
    failing closed when the reviewer is unavailable. Filters, search text and OR
    logic do because they can silently change the selected population.
    """
    expressed = _explode(extract_expressed(bound))
    unused = list(expressed)
    for stated in _explode(list(planned.stated_requirements)):
        match_index = next((index for index, item in enumerate(unused) if _same(stated, item)), None)
        if match_index is None:
            continue
        unused.pop(match_index)
    narrowing_types = (FilterRequirement, SearchTextRequirement, FilterLogicRequirement)
    return sorted(item.text for item in unused if isinstance(item, narrowing_types) and item.text)


def check_requirement_parity(planned: PlannedTurn, bound: TurnPlan) -> None:
    # Compared in canonical form: a requirement that states only a default ("combine the
    # conditions with AND", include_missing=false, offset 0) is satisfied by a plan that
    # leaves the default in place, and "first 5" as a window is the same as limit 5.
    expressed = [item for item, _ in canonical_requirements(extract_expressed(bound), dedupe=False)]
    unused = list(expressed)
    for stated, original in canonical_requirements(
        list(planned.stated_requirements), dedupe=False
    ):
        match_index = next((index for index, item in enumerate(unused) if _same(stated, item)), None)
        if match_index is None:
            raise PlanValidationError(
                f"dropped_requirement: {original.text}",
                code="dropped_requirement",
            )
        unused.pop(match_index)
    if unused:
        logger.info(
            "plan carries %d condition(s) no stated requirement accounts for: %s",
            len(unused),
            "; ".join(sorted(item.text for item in unused))[:300],
        )


# --- Reviewer reconstruction reconciliation -------------------------------------------------
#
# The reviewer's reconstruction and the candidate plan are two independent model outputs, and
# the same meaning reaches them in different shapes: "first 5" as a window in one and a limit
# in the other, "do not include missing values" as an explicit include_missing=false that the
# candidate leaves at its default, "combine with AND" as filter_logic=and that flat filters
# already imply. Comparing those slot by slot turned agreement into "I couldn't safely
# reconcile the planned query" on questions both readings answered identically.
#
# The comparison below keeps the property the gate exists for -- no condition that changes
# which records are counted may be dropped or invented -- and stops treating representation as
# disagreement. Requirements fall into three classes:
#
#   narrowing  filters, search text, OR logic. Must agree in both directions, exactly as before.
#   statistic  goal, grouping, aggregate, interval, companion, having threshold, follow-up edits.
#              What is computed. A conflicting value is a real disagreement.
#   shape      projection, sort, limit/window/top_n/offset, options. How the answer is laid out.
#
# A reconstructed statistic or shape requirement the candidate simply lacks is not evidence the
# candidate is wrong, only that it is less specific; it is filled in from the reconstruction,
# which is independent provenance for it. A conflicting value is never guessed between.

_NARROWING_REQUIREMENTS = (FilterRequirement, SearchTextRequirement, FilterLogicRequirement)
_SHAPE_REQUIREMENTS = (
    ProjectionRequirement,
    SortingRequirement,
    WindowRequirement,
    NumericSlotRequirement,
    OptionRequirement,
)
# Options whose false value is simply the default, so stating it says nothing.
_DEFAULT_FALSE_OPTIONS = frozenset({"include_missing", "sample", "exhaustive", "verify_previous"})


def requirement_class(item: StatedRequirement) -> str:
    if isinstance(item, _NARROWING_REQUIREMENTS):
        return "narrowing"
    if isinstance(item, _SHAPE_REQUIREMENTS):
        return "shape"
    return "statistic"


def _is_false(value: Any) -> bool:
    return value is False or _norm(value) in {"false", "0", "no", ""}


def _canonical_one(item: StatedRequirement) -> StatedRequirement | None:
    """The comparison form of one requirement, or None when it states only a default."""
    if isinstance(item, FilterRequirement) and item.collection != "compare" and (
        item.branch_index is not None or item.branch_label
    ):
        # Branch coordinates mean nothing outside a comparison.
        item = item.model_copy(update={"branch_index": None, "branch_label": None})
    if isinstance(item, OptionRequirement):
        if item.option == "presentation":
            return None
        if item.option in _DEFAULT_FALSE_OPTIONS and _is_false(item.value):
            return None
    if isinstance(item, FilterLogicRequirement) and item.filter_logic == "and":
        return None
    if isinstance(item, NumericSlotRequirement) and item.slot == "offset" and item.value == 0:
        return None
    if isinstance(item, HavingRequirement) and item.having_min_count <= 1:
        # A group exists because a record fell into it, so "keep groups of at least one"
        # keeps every group. Materializing that threshold states the default and narrows
        # nothing, which is the same reason offset 0 and filter_logic=and are dropped here.
        return None
    if (
        isinstance(item, AggregateRequirement)
        and _norm(item.function) == "count"
        and not item.field
        and not item.stats_value_part
    ):
        # Counting rows is what every action does unless told otherwise.
        return None
    if isinstance(item, WindowRequirement) and item.anchor == "start" and not item.cursor:
        # A window over the first N rows is the same selection as limit N.
        return NumericSlotRequirement(
            action_index=item.action_index, text=item.text, slot="limit", value=item.size
        )
    return item


def _signature(item: StatedRequirement) -> str:
    data = item.model_dump(mode="json", exclude={"text"})
    return repr(sorted((key, _norm(value)) for key, value in data.items()))


def canonical_requirements(
    requirements: list[StatedRequirement], *, dedupe: bool = True
) -> list[tuple[StatedRequirement, StatedRequirement]]:
    """(comparison form, original) pairs with defaults removed.

    Duplicates are collapsed unless `dedupe` is false. Two model outputs are compared as sets
    of meaning; a plan's own stated requirements stay one-to-one with what it expresses.
    """
    seen: set[str] = set()
    out: list[tuple[StatedRequirement, StatedRequirement]] = []
    for original in _explode(list(requirements)):
        canonical = _canonical_one(original)
        if canonical is None:
            continue
        key = _signature(canonical)
        if dedupe and key in seen:
            continue
        seen.add(key)
        out.append((canonical, original))
    return out


def _slot(item: StatedRequirement) -> tuple[Any, ...] | None:
    """The action slot a requirement occupies; two different values in one slot conflict.

    Projection has no slot: fields accumulate, so a further field never conflicts.
    """
    index = item.action_index
    if isinstance(item, GoalRequirement):
        return ("goal", index)
    if isinstance(item, SortingRequirement):
        return ("sort", index)
    if isinstance(item, WindowRequirement) or (
        isinstance(item, NumericSlotRequirement) and item.slot == "limit"
    ):
        return ("rows", index)
    if isinstance(item, NumericSlotRequirement):
        return (item.slot, index)
    if isinstance(item, GroupingRequirement):
        return ("group", index, item.position)
    if isinstance(item, AggregateRequirement):
        return ("aggregate", index)
    if isinstance(item, IntervalRequirement):
        return ("interval", index)
    if isinstance(item, CompanionRequirement):
        return ("companion", index)
    if isinstance(item, HavingRequirement):
        return ("having", index)
    if isinstance(item, OptionRequirement):
        return ("option", index, item.option)
    if isinstance(item, ModifyEditRequirement):
        return ("modify", index, _norm(item.edit))
    return None


@dataclass
class ReconstructionCheck:
    """How a candidate plan stands against the reviewer's independent reconstruction."""

    missing_narrowing: list[str]
    extra_narrowing: list[str]
    conflicts: list[str]
    fills: list[StatedRequirement]

    @property
    def consistent(self) -> bool:
        return not (self.missing_narrowing or self.extra_narrowing or self.conflicts)

    def detail(self) -> str:
        parts = []
        if self.missing_narrowing:
            parts.append("missing from candidate: " + "; ".join(self.missing_narrowing))
        if self.conflicts:
            parts.append("conflicts with candidate: " + "; ".join(self.conflicts))
        if self.extra_narrowing:
            parts.append(
                "candidate narrowing not reconstructed: " + "; ".join(self.extra_narrowing)
            )
        return " | ".join(parts)


def reconcile_reconstruction(
    reconstructed: list[StatedRequirement], bound: TurnPlan
) -> ReconstructionCheck:
    expressed = canonical_requirements(_without_redundant_presence(extract_expressed(bound), bound))
    unused = list(expressed)
    unmatched: list[tuple[StatedRequirement, StatedRequirement]] = []
    for canonical, original in canonical_requirements(
        _without_redundant_presence(_outside_comparisons(reconstructed, bound), bound)
    ):
        match_index = next(
            (index for index, (item, _) in enumerate(unused) if _same(canonical, item)), None
        )
        if match_index is None:
            unmatched.append((canonical, original))
        else:
            unused.pop(match_index)
    missing_narrowing: list[str] = []
    conflicts: list[str] = []
    fills: list[StatedRequirement] = []
    for canonical, original in unmatched:
        if requirement_class(canonical) == "narrowing":
            missing_narrowing.append(original.text)
            continue
        slot = _slot(canonical)
        occupied = slot is not None and any(_slot(item) == slot for item, _ in expressed)
        if occupied or isinstance(canonical, GoalRequirement | ModifyEditRequirement):
            conflicts.append(original.text)
            continue
        fills.append(original)
    extra_narrowing = sorted(
        original.text
        for item, original in unused
        if requirement_class(item) == "narrowing" and original.text
    )
    return ReconstructionCheck(
        missing_narrowing=sorted(missing_narrowing),
        extra_narrowing=extra_narrowing,
        conflicts=sorted(conflicts),
        fills=fills,
    )


def unverified_correction_changes(
    reconstructed: list[StatedRequirement],
    original: TurnPlan,
    corrected: TurnPlan,
) -> list[StatedRequirement]:
    """What a correction changed or added that the reconstruction does not account for.

    Compared in canonical form, so a correction that restates a default or turns a start window
    into the equivalent limit has changed nothing. Projection is never reported: adding a
    column to an answer changes no record and no statistic.
    """
    original_unused = [
        item
        for item, _ in canonical_requirements(
            _without_redundant_presence(extract_expressed(original), original)
        )
    ]
    introduced: list[tuple[StatedRequirement, StatedRequirement]] = []
    for canonical, raw in canonical_requirements(
        _without_redundant_presence(extract_expressed(corrected), corrected)
    ):
        match_index = next(
            (index for index, prior in enumerate(original_unused) if _same(canonical, prior)),
            None,
        )
        if match_index is None:
            introduced.append((canonical, raw))
        else:
            original_unused.pop(match_index)
    reconstructed_unused = [item for item, _ in canonical_requirements(reconstructed)]
    unverified: list[StatedRequirement] = []
    for canonical, raw in introduced:
        if isinstance(canonical, ProjectionRequirement):
            continue
        match_index = next(
            (index for index, item in enumerate(reconstructed_unused) if _same(canonical, item)),
            None,
        )
        if match_index is None:
            unverified.append(raw)
        else:
            reconstructed_unused.pop(match_index)
    return unverified


_PRESENCE_OPERATORS = frozenset({"IS_KNOWN"})


# Aggregates that read only the records holding a value for the field they measure, so
# that restricting the query to those records cannot change what they report. "count" is
# absent on purpose: counting records is a question about the population itself.
_VALUE_ONLY_AGGREGATES = frozenset(
    {"count_distinct", "sum", "avg", "min", "max", "median", "mode", "stats"}
)


def _already_excludes_missing(action: QueryAction, field: str) -> bool:
    """Whether the query reads only records holding a value for this field anyway.

    Two shapes qualify. A grouped query with include_missing=false keeps only records that
    have a value for every grouped field -- the gateway and structured_group_values2 both
    read a missing label as null-or-blank, which is the test IS_KNOWN makes. And an
    aggregate measuring one field reads only the records that hold a value for it: the
    oldest age, the number of distinct causes, and their like are the same number whether
    or not the empty records are filtered out first.

    In both shapes a presence filter on that field selects no record the query was not
    going to read, so writing it and not writing it are the same request.
    """
    if action.compare:
        return False
    grouped_count = (
        action.aggregate is not None
        and action.aggregate.function in {"count", "mode"}
        and (not action.aggregate.field or action.aggregate.field == field)
    )
    grouped = (
        bool(action.group_by)
        and field in action.group_by
        and not action.include_missing
        # A rank already ignores a blank group label. An aggregate does too when it is
        # only counting or finding the mode of that same label. An average of a different
        # field still changes if blank-label rows are removed, so that case stays narrowing.
        and (action.goal != "aggregate" or grouped_count)
    )
    measured = (
        action.aggregate is not None
        and action.aggregate.field == field
        and action.aggregate.function in _VALUE_ONLY_AGGREGATES
        and not action.group_by
    )
    return grouped or measured


# Operators that can only match a record that holds a value: each compares the recorded
# value with a stated one, and a record with nothing recorded matches none of them. The
# negative operators are deliberately absent -- "not equal to X" is true of a record that
# recorded nothing -- and so are the search operators, which match through a different
# path than this comparison.
_VALUE_MATCH_OPERATORS = frozenset(
    {
        "EQUALS",
        "IN",
        "CONTAINS",
        "CONTAINS_ANY",
        "STARTS_WITH",
        "YEAR_EQUALS",
        "BEFORE",
        "AFTER",
        "DATE_RANGE",
        "GREATER_THAN",
        "LESS_THAN",
        "NUMBER_RANGE",
    }
)


def _value_condition_implies_presence(action: QueryAction, field: str) -> bool:
    """Whether the action already requires a recorded value for this field.

    "Admitted before 1900" cannot be true of a record with no admission date, so adding
    "admission date is recorded" beside it asks for the same records. One call writes the
    pair and the other writes the comparison alone; reading that as a disagreement refused
    the question. Only a conjunction qualifies: under OR the presence condition widens the
    population instead of restating it, and mixed AND/OR lives in filter_groups, where the
    conjunction a condition belongs to is not this simple.
    """
    if action.filter_logic != "and" or action.filter_groups:
        return False
    return any(
        item.field == field
        and item.operator in _VALUE_MATCH_OPERATORS
        and item.value is not None
        and str(item.value).strip() != ""
        for item in action.filters
    )


def _without_redundant_presence(
    requirements: list[StatedRequirement], plan: TurnPlan
) -> list[StatedRequirement]:
    """Drop presence conditions that the action's own grouping already applies.

    Two model calls write the superlative rule "missing values do not compete" two ways:
    one sets include_missing=false, the other adds IS_KNOWN on the grouped field. They
    select the same records, so one side writing it and the other not is not a
    disagreement about the question -- and reading it as one refused every "most common"
    turn whose reviewer chose the other wording.
    """
    kept: list[StatedRequirement] = []
    for item in requirements:
        action = (
            plan.actions[item.action_index]
            if 0 <= item.action_index < len(plan.actions)
            else None
        )
        if (
            isinstance(item, FilterRequirement)
            # Only the population the grouping itself reads. A denominator or a
            # comparison branch counts a different set of records, where the same
            # condition does narrow something.
            and item.collection == "filters"
            and item.operator in _PRESENCE_OPERATORS
            and isinstance(action, QueryAction)
            and (
                _already_excludes_missing(action, item.field)
                or _value_condition_implies_presence(action, item.field)
            )
        ):
            continue
        kept.append(item)
    return kept


def _outside_comparisons(
    requirements: list[StatedRequirement], bound: TurnPlan
) -> list[StatedRequirement]:
    """Reconstructed branch conditions for an action that compares nothing are plain filters."""
    comparing = {
        index
        for index, action in enumerate(bound.actions)
        if isinstance(action, QueryAction) and action.compare
    }
    out: list[StatedRequirement] = []
    for item in requirements:
        if item.action_index not in comparing:
            if isinstance(item, FilterRequirement) and item.collection == "compare":
                item = item.model_copy(
                    update={"collection": "filters", "branch_index": None, "branch_label": None}
                )
            elif isinstance(item, SearchTextRequirement) and (
                item.branch_index is not None or item.branch_label
            ):
                item = item.model_copy(update={"branch_index": None, "branch_label": None})
        out.append(item)
    return out
