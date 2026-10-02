from __future__ import annotations

from typing import Any

from app.execution.executor import ActionResult, ExecutionResult
from app.execution.name_correction import is_name_field
from app.execution.roster import Roster, roster_sentence
from app.execution.row_projection import display_value, field_descriptors, semantic_value
from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import FilterOperator, Predicate, QueryPlan


def _row_name(row: dict, catalog: FieldCatalog) -> str:
    """Prefer the canonical display name over the folded `canonical_name` column."""
    value = semantic_value(row, "student_name", catalog)
    text = "" if value is None else str(value).strip()
    return text or str(row.get("canonical_name") or "unnamed record")


def format_spoken_answer(
    plan: QueryPlan,
    result: ExecutionResult,
    catalog: FieldCatalog,
) -> str | None:
    """Terse spoken variant, or None when the written answer is already short."""
    if result.action_results:
        parts = [_spoken_action(item, plan, catalog) for item in result.action_results]
        spoken = " ".join(part for part in parts if part)
        return spoken or None
    if result.distinct_field:
        values = result.distinct_values
        plural = {"community": "communities", "school": "schools"}.get(
            result.distinct_field, f"{result.distinct_field} values"
        )
        singular = {"community": "community", "school": "school"}.get(
            result.distinct_field, result.distinct_field
        )
        if not values:
            return f"No {plural} recorded."
        noun = singular if len(values) == 1 else plural
        return f"{len(values)} {noun}: {_spoken_list(values)}."
    count_fact = next((fact for fact in result.facts if fact.name == "count"), None)
    if count_fact is not None:
        value = int(count_fact.value)
        noun = "student" if plan.scope.file_ids == (49,) else "record"
        return f"{value} {noun}." if value == 1 else f"{value} {noun}s."
    return None


def _spoken_action(action: ActionResult, plan: QueryPlan, catalog: FieldCatalog) -> str | None:
    file_ids = action.file_ids or plan.scope.file_ids
    if (
        action.value_counts
        and action.grouped_field
        and any(fact.name == "count" for fact in action.facts)
    ):
        recorded = sum(
            count for label, count in action.value_counts.items() if label != "Not recorded"
        )
        return f"{recorded} records have a recorded {action.grouped_field.replace('_', ' ')}."
    # A GROUP_BY that carries the values themselves was asked to show them. It also
    # publishes count_distinct so a caller can read the number, but that fact is not a
    # signal to replace the list with a tally -- COUNT_DISTINCT carries no values and
    # still falls through to the count below.
    if action.distinct_field and action.distinct_values:
        return _format_distinct(action.distinct_field, action.distinct_values, "authorized records")
    distinct_count = next((fact for fact in action.facts if fact.name == "count_distinct"), None)
    if distinct_count is not None:
        return _format_distinct_count(
            action.distinct_field or "value", int(distinct_count.value), "authorized records"
        )
    count_fact = next((fact for fact in action.facts if fact.name == "count"), None)
    if count_fact is not None:
        value = int(count_fact.value)
        noun = "student" if file_ids == (49,) else "record"
        return f"{value} {noun}." if value == 1 else f"{value} {noun}s."
    if action.rows:
        total = action.total_count
        names = [_row_name(row, catalog) for row in action.rows]
        preview = _spoken_list(names[:3])
        if action.sampled:
            noun = "students" if file_ids == (49,) else "records"
            return f"Here are {len(names)} random {noun}: {_spoken_list(names)}."
        if total is not None and (action.has_more or total > len(names)):
            noun = "student" if file_ids == (49,) else "record"
            plural = noun if total == 1 else f"{noun}s"
            more = " Say next to continue." if action.has_more or (total > (action.offset or 0) + len(names)) else ""
            return f"{total} {plural}. Starting with {preview}.{more}"
        if len(names) > 5:
            return f"{len(names)} records. Starting with {preview}."
        return _spoken_list(names) + "."
    return None


def _spoken_list(values: list[str]) -> str:
    items = [str(value) for value in values]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def format_deterministic_answer(
    question: str,
    plan: QueryPlan,
    result: ExecutionResult,
    catalog: FieldCatalog,
    *,
    roster: Roster | None = None,
) -> str:
    if result.action_results:
        merged = _format_count_with_distribution(
            result.action_results, plan, catalog, roster=roster
        )
        if merged is not None:
            return merged
        verbose = len(result.action_results) == 1
        # A plan that asks the same thing twice -- a planner and a reviewer each adding the
        # action -- should not print the same answer twice.
        parts = list(
            dict.fromkeys(
                part
                for part in (
                    _format_action(action, plan, catalog, verbose=verbose)
                    for action in result.action_results
                )
                if part
            )
        )
        if (
            roster is not None
            and parts
            and len(result.action_results) == 1
            and is_plain_count(result.action_results[0])
        ):
            sentence = roster_sentence(roster)
            if sentence:
                parts[0] = f"{parts[0]} {sentence}"
        # A table or a numbered list only renders as one when it starts on a line of its own.
        joiner = "\n\n" if any("\n" in part for part in parts) else " "
        return joiner.join(parts)

    labels = _scope_text(plan, catalog)
    if result.distinct_field:
        return _format_distinct(result.distinct_field, result.distinct_values, labels)
    if result.grouped_counts:
        return _format_grouped(result.grouped_counts, catalog)
    return "No matching authorized records were found."


def _format_count_with_distribution(
    actions: list[ActionResult],
    plan: QueryPlan,
    catalog: FieldCatalog,
    *,
    roster: Roster | None = None,
) -> str | None:
    """One answer for "how many are there, and how are they spread across X".

    The planner asks that as two actions over the same records -- a count and a breakdown --
    and they were printed one after the other: "Students: 2781 students. By community in the
    Student master list:" followed by every one of 370 labels. The count is the breakdown's
    own total, so it is said once, at the head of a summary a researcher can read.
    """
    if len(actions) != 2:
        return None
    counts = [item for item in actions if is_plain_count(item)]
    spreads = [
        item
        for item in actions
        if item.ranked_groups
        and not item.group_fields[1]
        and item.requested_top_n is None
        and item.stats is None
    ]
    if len(counts) != 1 or len(spreads) != 1:
        return None
    count, spread = counts[0], spreads[0]
    file_ids = tuple(count.file_ids or plan.scope.file_ids)
    if tuple(spread.file_ids or plan.scope.file_ids) != file_ids:
        return None
    if [item.payload() for item in count.predicates] != [
        item.payload() for item in spread.predicates
    ]:
        return None
    total = int(next(fact.value for fact in count.facts if fact.name == "count"))
    return _format_ranked(
        spread,
        _scope_text(plan, catalog, file_ids),
        catalog=catalog,
        file_ids=file_ids,
        total=total,
        constraint=_constraint_text(count.predicates, catalog, file_ids),
        roster=roster,
    )


def is_plain_count(action: ActionResult) -> bool:
    return (
        any(fact.name == "count" for fact in action.facts)
        and not action.ranked_groups
        and not action.rows
        and not action.value_counts
        and not action.distinct_values
        and action.percentage is None
    )


def _format_action(
    action: ActionResult,
    plan: QueryPlan,
    catalog: FieldCatalog,
    *,
    verbose: bool,
) -> str:
    file_ids = action.file_ids or plan.scope.file_ids
    labels = _scope_text(plan, catalog, file_ids)
    constraint = _constraint_text(action.predicates, catalog, file_ids)
    # Computed answers are rendered before any row page, so a calculation is never
    # reported as "showing 1-25 of N".
    if action.percentage is not None:
        return _format_percentage(action, labels, constraint)
    if action.stats is not None:
        return _format_stats(action, labels)
    if action.overlaps is not None:
        return _format_overlaps(action, labels, catalog=catalog, file_ids=file_ids)
    if action.duplicates:
        return _format_duplicates(action, labels)
    if action.completeness:
        return _format_completeness(action, labels)
    if action.interval_summary is not None or action.intervals:
        return _format_intervals(action, labels)
    if action.ranked_groups and action.goal == "rank":
        return _format_ranked(
            action, labels, catalog=catalog, file_ids=file_ids, constraint=constraint
        )
    if (
        action.value_counts
        and action.grouped_field
        and any(fact.name == "count" for fact in action.facts)
    ):
        return _format_value_counts(
            action.grouped_field,
            action.value_counts,
            labels,
            catalog=catalog,
            file_ids=file_ids,
        )
    # A GROUP_BY that carries the values themselves was asked to show them. It also
    # publishes count_distinct so a caller can read the number, but that fact is not a
    # signal to replace the list with a tally -- COUNT_DISTINCT carries no values and
    # still falls through to the count below.
    if action.distinct_field and action.distinct_values:
        return _format_distinct(action.distinct_field, action.distinct_values, labels)
    distinct_count = next((fact for fact in action.facts if fact.name == "count_distinct"), None)
    if distinct_count is not None:
        return _format_distinct_count(
            action.distinct_field or "value", int(distinct_count.value), labels
        )
    if action.distinct_field:
        return _format_distinct(action.distinct_field, action.distinct_values, labels)
    if action.grouped_counts:
        return _format_grouped(action.grouped_counts, catalog)
    compare_fact = next((fact for fact in action.facts if fact.name == "compare"), None)
    if compare_fact is not None:
        sides = compare_fact.value or []
        parts = []
        for side in sides:
            count = side.get("count")
            label = side.get("id") or "branch"
            if count is not None:
                parts.append(f"{label}: {count}")
        if parts:
            return "Comparison: " + "; ".join(parts) + "."
    aggregate = _format_numeric_aggregates(action, labels, constraint)
    if aggregate is not None:
        return aggregate
    if action.rows or (action.goal == "list" and action.total_count is not None):
        return _format_page(action, plan, catalog, labels, constraint, verbose=verbose)
    count_fact = next((fact for fact in action.facts if fact.name == "count"), None)
    if count_fact is not None:
        noun = "student" if file_ids == (49,) else "record"
        value = int(count_fact.value)
        if verbose:
            plural = noun if value == 1 else f"{noun}s"
            verb = "is" if value == 1 else "are"
            if constraint:
                return f"There {verb} {value} {plural} in the {labels} {constraint}."
            return f"There {verb} {value} {plural} in the {labels}."
        subject = _compact_subject(action.predicates, noun, plural=value != 1)
        return f"{subject}: {value} {noun}." if value == 1 else f"{subject}: {value} {noun}s."
    if action.evidence and action.evidence.items:
        shown = ", ".join(
            str(item.fields.get("student_name") or "unnamed record") for item in action.evidence.items
        )
        if verbose:
            return f"Matching records in the {labels}{(' ' + constraint) if constraint else ''}: {shown}."
        return f"{_compact_list_heading(action.predicates)} {shown}."
    return "No matching authorized records were found."


def _format_page(
    action: ActionResult,
    plan: QueryPlan,
    catalog: FieldCatalog,
    labels: str,
    constraint: str,
    *,
    verbose: bool,
) -> str:
    file_ids = action.file_ids or plan.scope.file_ids
    names = [_row_name(row, catalog) for row in action.rows]
    total = action.total_count
    if total is None:
        count_fact = next((fact for fact in action.facts if fact.name == "count"), None)
        if count_fact is not None:
            total = int(count_fact.value)
    total = int(total) if total is not None else len(names)
    offset = action.offset or 0
    page_size = action.page_size or len(names) or 25
    start = offset + 1 if names else 0
    end = offset + len(names)
    noun = "student" if file_ids == (49,) else "record"
    plural = noun if total == 1 else f"{noun}s"
    scope = f"in the {labels}{(' ' + constraint) if constraint else ''}"
    if total == 0 or not names:
        return f"There are no matching {plural} {scope}."
    header = f"There {'is' if total == 1 else 'are'} {total} {plural} {scope}."
    fields = list(action.projected_fields) or ["student_name"]
    descriptors = {
        item["name"]: item["label"]
        for item in field_descriptors(fields, file_ids, catalog)
    }
    lines: list[str] = []
    for index, row in enumerate(action.rows):
        name = _row_name(row, catalog)
        extras = [
            f"{descriptors[field]}: {display_value(semantic_value(row, field, catalog))}"
            for field in fields
            if field != "student_name"
        ]
        suffix = f" | {'; '.join(extras)}" if extras else ""
        lines.append(f"{offset + index + 1}. {name}{suffix}")
    numbered = "\n".join(lines)
    if action.sampled:
        return (
            f"Random sample of {len(names)} from {total} matching {plural} {scope}:\n"
            f"{numbered}"
        )
    if total <= len(names) and offset == 0 and not action.window_anchor:
        return f"{header}\n\n{numbered}"
    page = f"Showing {start}-{end} of {total}"
    if action.window_anchor and action.window_size:
        selected = action.window_count if action.window_count is not None else min(action.window_size, total)
        within_start = action.window_cursor + 1
        within_end = action.window_cursor + len(names)
        position = "last" if action.window_anchor == "end" else "first"
        page += f" (items {within_start}-{within_end} of the requested {position} {selected})"
    page += f":\n{numbered}"
    more = '\n\nSay "next" to continue.' if action.has_more else ""
    if verbose or action.has_more or total > page_size or action.window_anchor:
        return f"{header}\n{page}{more}"
    heading = _compact_list_heading(action.predicates)
    return f"{heading}\n{numbered}"


def _humanize(field: str | None) -> str:
    return (field or "value").replace("_", " ")


def _format_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "not available"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


_AGGREGATE_WORDS = {
    "min": "lowest",
    "max": "highest",
    "sum": "total",
    "avg": "average",
    "count_distinct": "distinct values",
}


def _format_numeric_aggregates(
    action: ActionResult,
    scope_text: str,
    constraint: str,
) -> str | None:
    """Render MIN / MAX / SUM / AVG results.

    Without this, a plan whose only output was a numeric aggregate fell through to
    "No matching authorized records were found" even though it had computed a value.
    """
    parts: list[str] = []
    for fact in action.facts:
        if fact.name not in _AGGREGATE_WORDS or fact.value is None:
            continue
        parts.append(f"{_AGGREGATE_WORDS[fact.name]} {_format_number(fact.value)}")
    if not parts:
        return None
    subject = f" {constraint}" if constraint else ""
    return f"In the {scope_text}{subject}: " + "; ".join(parts) + "."


# A distribution this short is shown whole. A longer one leads with its most common values,
# which is what a researcher reads first, and says how many more there are.
_DISTRIBUTION_FULL_MAX = 30
_DISTRIBUTION_TOP = 15


def _format_ranked(
    action: ActionResult,
    scope_text: str,
    *,
    catalog: FieldCatalog | None = None,
    file_ids: tuple[int, ...] = (),
    total: int | None = None,
    constraint: str = "",
    roster: Roster | None = None,
) -> str:
    groups = action.ranked_groups
    primary, secondary = action.group_fields
    label = _field_label(primary, file_ids, catalog)
    if not groups:
        return f"No {label} values are recorded in the {scope_text}."
    requested = action.requested_top_n
    shown = groups if requested is None else groups[:requested]
    head = shown[0]
    if len(shown) == 1:
        # The groups only add up to the whole when all of them came back. A ranking cut to its
        # top group at the query returns that one group, and summing it said "1882 has the
        # most, with 5 of 5 records" of a list holding 82.
        whole = requested is None or len(groups) > len(shown)
        everything = total if total is not None else (
            sum(item.count for item in groups) if whole else None
        )
        share = f" of {everything:,}" if everything is not None else ""
        return (
            f"{head.label or 'Not recorded'} has the most, with {head.count:,}"
            f"{share} records in the {scope_text}."
        )
    noun = _record_noun(file_ids)
    suffix = (
        f" (showing {len(shown)} of {len(groups)} groups)" if len(groups) > len(shown) else ""
    )
    if secondary:
        second = _field_label(secondary, file_ids, catalog)
        pairs = [
            (str(item.label or "Not recorded"), str(item.secondary_label or "Not recorded"), item.count)
            for item in shown
        ]
        return f"By {label} and {second} in the {scope_text}{suffix}:\n\n" + _markdown_table(
            [label, second, _count_heading(noun)], pairs
        )
    if requested is None:
        counted = roster if roster is not None and roster.differs else None
        return _distribution_summary(
            [(item.label, item.count) for item in groups],
            field_label=label,
            scope_text=scope_text,
            noun=noun,
            total=total,
            constraint=constraint,
            roster=counted,
            people_recorded=(
                counted.recorded_for(primary, catalog)
                if counted is not None and catalog is not None and primary
                else None
            ),
        )
    heading = f"Top {len(shown)} by {label}" if len(groups) > len(shown) else f"By {label}"
    rows = [(str(item.label or "Not recorded"), item.count) for item in shown]
    return f"{heading} in the {scope_text}{suffix}:\n\n" + _markdown_table(
        [label, _count_heading(noun)], rows
    )


def _distribution_summary(
    pairs: list[tuple[str | None, int]],
    *,
    field_label: str,
    scope_text: str,
    noun: str,
    total: int | None,
    constraint: str = "",
    roster: Roster | None = None,
    people_recorded: int | None = None,
) -> str:
    """How many records there are, how many carry the field, and the values it holds.

    Totals come first because they are what a researcher checks an answer against: the rows
    in the list, how many have the field filled in and how many are blank, and how many
    different values it takes. Then the values themselves, most common first, as a table.
    """
    recorded = sorted(
        (
            (str(label), count)
            for label, count in pairs
            if label not in (None, "", "Not recorded")
        ),
        key=lambda item: (-item[1], item[0].casefold()),
    )
    blank = [count for label, count in pairs if label in (None, "", "Not recorded")]
    recorded_total = sum(count for _label, count in recorded)
    if blank:
        missing: int | None = sum(blank)
    elif total is not None:
        missing = max(total - recorded_total, 0)
    else:
        missing = None
    sentences: list[str] = []
    if total is not None:
        subject = f" {constraint}" if constraint else ""
        sentences.append(
            f"There {'is' if total == 1 else 'are'} **{_records_phrase(total, noun)}** "
            f"in the {scope_text}{subject}."
        )
    if roster is not None:
        sentences.append(roster_sentence(roster))
    usage = (
        f"Using the **{field_label}** field, **{recorded_total:,} "
        f"{'has' if recorded_total == 1 else 'have'} a value recorded**"
    )
    if missing is not None:
        usage += f" and **{missing:,} {'is' if missing == 1 else 'are'} blank**"
    sentences.append(usage + ".")
    if roster is not None and people_recorded is not None:
        sentences.append(
            f"Counting each person once, **{people_recorded:,} of the {roster.unique:,}** "
            f"have a value recorded and **{roster.unique - people_recorded:,}** do not."
        )
    distinct = len(recorded)
    if not distinct:
        return " ".join(sentences)
    sentences.append(
        f"There {'is' if distinct == 1 else 'are'} **{distinct:,} distinct "
        f"value{'' if distinct == 1 else 's'}**."
    )
    shown = recorded if distinct <= _DISTRIBUTION_FULL_MAX else recorded[:_DISTRIBUTION_TOP]
    rows: list[tuple[str, int]] = list(shown)
    if blank:
        rows.append(("Not recorded", sum(blank)))
    lead = "The most common:" if len(shown) < distinct else "All of them:"
    text = (
        " ".join(sentences)
        + f" {lead}\n\n"
        + _markdown_table([field_label, _count_heading(noun)], rows)
    )
    if len(shown) < distinct:
        text += (
            f"\n\n…and {distinct - len(shown):,} more. Ask for every {field_label} value "
            "to see them all."
        )
    return text


def _markdown_table(header: list[str], rows: list[tuple[Any, ...]]) -> str:
    """A table whose last column is a count, right-aligned."""
    rule = ["---"] * (len(header) - 1) + ["---:"]
    lines = [
        "| " + " | ".join(_table_cell(item) for item in header) + " |",
        "| " + " | ".join(rule) + " |",
    ]
    lines.extend("| " + " | ".join(_table_cell(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def _table_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    text = " ".join(str(value if value is not None else "Not recorded").split())
    return text.replace("|", "\\|")


def _record_noun(file_ids: tuple[int, ...]) -> str:
    return "student" if tuple(file_ids) == (49,) else "record"


def _records_phrase(count: int, noun: str) -> str:
    plural = "record" if count == 1 else "records"
    return f"{count:,} {plural}" if noun == "record" else f"{count:,} {noun} {plural}"


def _count_heading(noun: str) -> str:
    return "Students" if noun == "student" else "Records"


def _field_label(
    field_name: str | None, file_ids: tuple[int, ...], catalog: FieldCatalog | None
) -> str:
    if catalog is not None and field_name:
        for file_id in file_ids:
            spec = catalog.resolve_field(file_id, field_name)
            if spec is not None:
                return spec.human_label
    return _humanize(field_name)


def _format_stats(action: ActionResult, scope_text: str) -> str:
    stats = action.stats
    if stats is None or not stats.known_count:
        return (
            f"No numeric {_humanize(action.stats_field)} values are recorded "
            f"in the {scope_text}."
        )
    unit = "" if action.stats_value_part not in {"year", "decade"} else " (year)"
    parts = [
        f"lowest {_format_number(stats.minimum)}",
        f"highest {_format_number(stats.maximum)}",
        f"average {_format_number(stats.average)}",
        f"median {_format_number(stats.median)}",
    ]
    if stats.mode is not None:
        parts.append(f"most frequent {_format_number(stats.mode)} ({stats.mode_count} records)")
    return (
        f"{_humanize(action.stats_field).capitalize()}{unit} across "
        f"{stats.known_count:,} of {stats.record_count:,} records in the {scope_text}: "
        + "; ".join(parts)
        + "."
    )


def _format_percentage(action: ActionResult, scope_text: str, constraint: str) -> str:
    payload = action.percentage or {}
    numerator = int(payload.get("numerator") or 0)
    denominator = int(payload.get("denominator") or 0)
    percent = payload.get("percent")
    if not denominator:
        return f"There are no records to measure against in the {scope_text}."
    subject = f" {constraint}" if constraint else ""
    return (
        f"{percent:.2f}% ({numerator:,} of {denominator:,}) of the {scope_text}"
        f"{subject}."
    )


def _format_overlaps(
    action: ActionResult,
    scope_text: str,
    *,
    catalog: FieldCatalog,
    file_ids: tuple[int, ...],
) -> str:
    """Groups sharing a value whose members were present at the same time, with what could be checked."""
    result = action.overlaps
    assert result is not None
    need = result.min_count
    noun = _record_noun(file_ids)
    nouns = f"{noun}s"
    shared_by = _field_label(result.group_field, file_ids, catalog)
    start = _sentence_label(_field_label(result.start_field, file_ids, catalog))
    end = _sentence_label(_field_label(result.end_field, file_ids, catalog))
    if not result.shared:
        return (
            f"Grouping the {nouns} in the {scope_text} by the exact text of **{shared_by}**, no "
            f"value is shared by {need} or more of them, so there are no groups to compare."
        )
    sentences = [
        f"**{result.shared:,}** {'value' if result.shared == 1 else 'values'} of **{shared_by}** "
        f"{'is' if result.shared == 1 else 'are each'} shared by {need} or more {nouns} in the "
        f"{scope_text}.",
        f"For **{result.checkable:,}** of them, at least {need} of those {nouns} have "
        f"{_article(start)} {start} and {_article(end)} {end} that place them in time, and in "
        f"**{len(result.groups):,}** of those, {need} or more were there during the same period.",
    ]
    unchecked = result.shared - result.checkable
    if unchecked:
        sentences.append(
            f"The other **{unchecked:,}** cannot be checked: fewer than {need} of their {nouns} "
            f"have both dates in a form that places them in time."
        )
    if not result.groups:
        return " ".join(sentences)
    rows: list[tuple[Any, ...]] = []
    for group in result.groups[:_DISTRIBUTION_TOP]:
        shown = "; ".join(
            f"{stay.name} ({_years(stay.start, stay.end)})" for stay in group.together[:6]
        )
        if len(group.together) > 6:
            shown += f"; and {len(group.together) - 6} more"
        rows.append(
            (
                group.value,
                f"{len(group.together)} of {group.members}",
                shown,
                _episode_years(group.episodes),
            )
        )
    text = " ".join(sentences) + "\n\n" + _markdown_table(
        [shared_by, "Together", f"Who ({start} to {end})", "When together"], rows
    )
    if len(result.groups) > _DISTRIBUTION_TOP:
        text += f"\n\n…and {len(result.groups) - _DISTRIBUTION_TOP:,} more."
    return text


def _years(first: Any, last: Any) -> str:
    return str(first.year) if first.year == last.year else f"{first.year}–{last.year}"


def _episode_years(episodes: list[tuple[Any, Any]]) -> str:
    """Separate stretches of togetherness, merged only where their years touch."""
    merged: list[list[int]] = []
    for began, ended in episodes:
        if merged and began.year <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], ended.year)
        else:
            merged.append([began.year, ended.year])
    return "; ".join(str(first) if first == last else f"{first}–{last}" for first, last in merged)


def _article(phrase: str) -> str:
    return "an" if phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _format_duplicates(action: ActionResult, scope_text: str) -> str:
    groups = action.duplicates
    field = _humanize(action.duplicate_field)
    if not groups:
        return f"No {field} value appears on more than one record in the {scope_text}."
    rows = sum(item.count for item in groups)
    extra = rows - len(groups)
    shared = sum(1 for item in groups if item.distinct_companions > 1)
    shared_note = (
        f" {shared} of them are shared by more than one distinct record."
        if shared
        else ""
    )
    lines = [
        f"{item.value}: {item.count}"
        + (f" ({', '.join(item.members)})" if item.members and item.count <= 6 else "")
        for item in groups
    ]
    return (
        f"{len(groups):,} {field} values appear on more than one record in the "
        f"{scope_text}, covering {rows:,} records ({extra:,} beyond the first in each "
        f"group).{shared_note}\n" + "\n".join(lines)
    )


def _format_completeness(action: ActionResult, scope_text: str) -> str:
    rows = action.completeness
    if not rows:
        return f"No records were available to measure in the {scope_text}."
    least = (action.completeness_direction or "desc").lower() == "asc"
    edge = rows[0].filled_fields
    tied = [item for item in rows if item.filled_fields == edge]
    names = "; ".join(f"{item.display_name} ({item.filled_fields}/{item.total_fields} fields)" for item in tied)
    wording = "least" if least else "most"
    return (
        f"Measured by how many catalog fields carry a value, the {wording} complete "
        f"records in the {scope_text}: {names}."
    )


def _format_intervals(action: ActionResult, scope_text: str) -> str:
    summary = action.interval_summary
    rows = action.intervals
    start, end = action.interval_fields or ("start", "end")
    span = f"{_humanize(start)} to {_humanize(end)}"
    if summary is None and not rows:
        return f"No record in the {scope_text} has both a {span} date."
    lines: list[str] = []
    if summary is not None and summary.record_count:
        lines.append(
            f"Across {summary.record_count:,} records in the {scope_text} with both dates, "
            f"{span} spans: shortest {_format_number(summary.minimum_days)} days, "
            f"longest {_format_number(summary.maximum_days)} days, average "
            f"{_format_number(summary.average_days)} days "
            f"({_format_number((summary.average_days or 0) / 365.25)} years), median "
            f"{_format_number(summary.median_days)} days."
        )
        if summary.impossible_count:
            lines.append(
                f"{summary.impossible_count} record(s) have an end date in an earlier year "
                f"than the start date, which is not possible."
            )
    for item in rows[:10]:
        note = " — impossible ordering" if item.is_impossible() else ""
        lines.append(
            f"{item.display_name}: {item.start_value} to {item.end_value} = "
            f"{_format_number(item.days)} days{note}"
        )
    return "\n".join(lines)


def _scope_text(
    plan: QueryPlan,
    catalog: FieldCatalog,
    file_ids: tuple[int, ...] | None = None,
) -> str:
    labels = [
        catalog.dataset(file_id).user_facing_label
        if catalog.dataset(file_id)
        else f"dataset {file_id}"
        for file_id in (file_ids or plan.scope.file_ids)
    ]
    return labels[0] if len(labels) == 1 else ", ".join(labels)


def _format_distinct(field: str, values: list[str], scope_text: str) -> str:
    singular = {"community": "community", "school": "school"}.get(field, field)
    plural = {"community": "communities", "school": "schools"}.get(field, f"{field} values")
    if not values:
        return f"No {plural} are recorded in the {scope_text}."
    noun = singular if len(values) == 1 else plural
    # Semicolons keep multi-word labels such as "Sault Ste. Marie, Michigan" readable
    # in a long enumeration.
    return f"The {scope_text} has {len(values)} {noun}: {'; '.join(values)}."


def _format_distinct_count(field: str, count: int, scope_text: str) -> str:
    singular = {"community": "community", "school": "school"}.get(field, field)
    plural = {"community": "communities", "school": "schools"}.get(
        field, f"{field.replace('_', ' ')} values"
    )
    noun = singular if count == 1 else plural
    return f"There {'is' if count == 1 else 'are'} {count} {noun} in the {scope_text}."


def _format_value_counts(
    field: str,
    counts: dict[str, int],
    scope_text: str,
    *,
    catalog: FieldCatalog | None = None,
    file_ids: tuple[int, ...] = (),
) -> str:
    return _distribution_summary(
        list(counts.items()),
        field_label=_field_label(field, file_ids, catalog),
        scope_text=scope_text,
        noun=_record_noun(file_ids),
        total=sum(counts.values()),
    )


def _format_grouped(grouped: dict[int, int], catalog: FieldCatalog) -> str:
    parts = [
        f"{catalog.dataset(file_id).user_facing_label if catalog.dataset(file_id) else file_id}: {count}"
        for file_id, count in grouped.items()
    ]
    return "Current authorized record counts: " + "; ".join(parts) + "."


def _compact_subject(predicates: list[Predicate], noun: str, *, plural: bool) -> str:
    community = _community(predicates)
    if community:
        return community
    prefix = _starts_with(predicates)
    if prefix:
        return f"Names starting with {prefix}"
    return noun.capitalize() + ("s" if plural else "")


def _compact_list_heading(predicates: list[Predicate]) -> str:
    prefix = _starts_with(predicates)
    if prefix:
        return f"Students whose names start with {prefix}:"
    community = _community(predicates)
    if community:
        return f"Students from {community}:"
    return "Matching records:"


def _community(predicates: list[Predicate]) -> str | None:
    for predicate in predicates:
        if predicate.operator is FilterOperator.EQUALS and predicate.field == "community":
            return str(predicate.value)
    return None


def _starts_with(predicates: list[Predicate]) -> str | None:
    for predicate in predicates:
        if predicate.operator is FilterOperator.STARTS_WITH:
            return str(predicate.value)
    return None


def format_access_denied() -> str:
    return "That request is outside the datasets available in this session."


def format_unplanned(detail: str) -> str:
    return (
        "I wasn't able to answer that one accurately. "
        "Could you try asking it a different way?"
    )


_TEXT_MATCH = frozenset(
    {FilterOperator.CONTAINS, FilterOperator.CONTAINS_ANY, FilterOperator.FULL_TEXT_SEARCH}
)
_TEXT_EXCLUDE = frozenset({FilterOperator.NOT_CONTAINS, FilterOperator.NOT_CONTAINS_ANY})
_VALUE_MATCH = frozenset({FilterOperator.EQUALS, FilterOperator.IN})
_VALUE_EXCLUDE = frozenset({FilterOperator.NOT_EQUALS, FilterOperator.NOT_IN})


def _constraint_text(
    predicates: list[Predicate],
    catalog: FieldCatalog | None = None,
    file_ids: tuple[int, ...] = (),
) -> str:
    """The conditions an answer was computed under, in words.

    A count that leaves out what it counted is unverifiable. "There are 0 students in the
    Student master list" was the whole answer to a question about transfers, and nothing in
    it said the zero came from searching Notes for a word that is written elsewhere. Naming
    the field and the words lets the researcher see a wrong search for what it is.
    """
    parts: list[str] = []
    for predicate in predicates:
        if predicate.is_group():
            mention = _group_mention(predicate, catalog, file_ids)
            if mention:
                parts.append(mention)
            continue
        if predicate.operator is FilterOperator.STARTS_WITH:
            parts.append(f"whose names start with {predicate.value}")
        elif predicate.operator is FilterOperator.EQUALS and predicate.field == "community":
            parts.append(f"from {predicate.value}")
        elif predicate.operator is FilterOperator.IS_TRUE and predicate.field == "deceased_status":
            parts.append("recorded as deceased")
        elif predicate.operator is FilterOperator.YEAR_EQUALS:
            parts.append(f"with {predicate.field.replace('_', ' ')} in {predicate.value}")
        elif predicate.operator is FilterOperator.BEFORE:
            parts.append(f"with {predicate.field.replace('_', ' ')} before {predicate.value}")
        elif predicate.operator is FilterOperator.AFTER:
            parts.append(f"with {predicate.field.replace('_', ' ')} after {predicate.value}")
        elif catalog is None or is_name_field(predicate.field, file_ids, catalog):
            # A searched name is shown in the rows it found; repeating it adds nothing.
            continue
        elif predicate.operator in _TEXT_MATCH:
            parts.append(
                f"with {_quoted_terms(predicate.value)} in "
                f"{_sentence_label(_field_label(predicate.field, file_ids, catalog))}"
            )
        elif predicate.operator in _TEXT_EXCLUDE:
            parts.append(
                f"without {_quoted_terms(predicate.value)} in "
                f"{_sentence_label(_field_label(predicate.field, file_ids, catalog))}"
            )
        elif predicate.operator in _VALUE_MATCH or predicate.operator in _VALUE_EXCLUDE:
            # "deceased students with no community" came back as "194 students ... with no
            # recorded First Nation / Community": correct, and unverifiable, because the
            # deceased condition it was computed under was not said anywhere.
            negated = "not " if predicate.operator in _VALUE_EXCLUDE else ""
            parts.append(
                f"whose {_sentence_label(_field_label(predicate.field, file_ids, catalog))} "
                f"is {negated}{_quoted_terms(predicate.value)}"
            )
        elif predicate.operator is FilterOperator.IS_KNOWN:
            parts.append(
                f"with a recorded {_sentence_label(_field_label(predicate.field, file_ids, catalog))}"
            )
        elif predicate.operator is FilterOperator.IS_UNKNOWN:
            parts.append(
                f"with no recorded {_sentence_label(_field_label(predicate.field, file_ids, catalog))}"
            )
    return " ".join(parts)


def _group_mention(
    predicate: Predicate, catalog: FieldCatalog | None, file_ids: tuple[int, ...]
) -> str:
    """'with "..." in additional information or death details' for a widened search."""
    if catalog is None or predicate.op != "or":
        return ""
    leaves = predicate.leaves()
    if not leaves or any(
        leaf.operator not in _TEXT_MATCH or is_name_field(leaf.field, file_ids, catalog)
        for leaf in leaves
    ):
        return ""
    labels = list(
        dict.fromkeys(
            _sentence_label(_field_label(leaf.field, file_ids, catalog)) for leaf in leaves
        )
    )
    terms: list[str] = []
    for leaf in leaves:
        for term in _terms_of(leaf.value):
            if term.casefold() not in {item.casefold() for item in terms}:
                terms.append(term)
    return f"with {_quoted_terms(terms)} in {' or '.join(labels)}"


def _sentence_label(label: str) -> str:
    """A label as it reads mid-sentence: "Additional information" -> "additional information".

    Only a label with no other capitals is lowered, so "First Nation / Community" keeps its
    proper nouns.
    """
    if len(label) > 1 and label[0].isupper() and not any(ch.isupper() for ch in label[1:]):
        return label[0].lower() + label[1:]
    return label


def _terms_of(value: Any) -> list[str]:
    items = value if isinstance(value, list | tuple) else [value]
    return [" ".join(str(item).split()) for item in items if str(item or "").strip()]


def _quoted_terms(value: Any) -> str:
    return " or ".join(f'"{term}"' for term in _terms_of(value))
