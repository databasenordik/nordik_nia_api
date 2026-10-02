"""Records that share a value and were present at the same time.

Asked "are there families with several children attending during the same period?", NIA ranked
parents' names by how many records carry them and stopped there. That answers "which families
had several children", not "at the same time": two siblings a decade apart were counted exactly
like two who overlapped.

The question is about periods, so this compares them. Records are grouped by the shared value;
each record's period runs from its start date to its end date; a sweep over those periods finds
the times when enough members of a group were present together. A date written as a year alone
covers that whole year, and a cell that holds two admissions or cannot be read is not placed at
all -- such records are counted, so the answer says how much of a group could be checked rather
than quietly judging a family by the children whose dates happen to parse.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from app.execution.derivations import date_precision, parse_date
from app.execution.derivations import value_part as derive_part
from app.execution.row_projection import semantic_value
from app.planning.catalog import FieldCatalog
from app.value_normalization import json_object


@dataclass(frozen=True)
class Stay:
    name: str
    start: date
    end: date


@dataclass
class OverlapGroup:
    value: str
    members: int
    dated: int
    together: list[Stay] = field(default_factory=list)
    first: date | None = None
    last: date | None = None
    # Each stretch during which enough members were present at once. Two siblings together in
    # the 1930s and two others in the 1950s are two episodes, not one span across both.
    episodes: list[tuple[date, date]] = field(default_factory=list)


@dataclass
class OverlapResult:
    group_field: str
    start_field: str
    end_field: str
    min_count: int
    shared: int = 0
    checkable: int = 0
    groups: list[OverlapGroup] = field(default_factory=list)


def overlapping_groups(
    rows: list[dict[str, Any]],
    catalog: FieldCatalog,
    *,
    group_field: str,
    start_field: str,
    end_field: str,
    value_part: str | None = None,
    min_count: int = 2,
) -> OverlapResult:
    """Groups where at least ``min_count`` members' periods overlap.

    ``shared`` counts groups with at least ``min_count`` members; ``checkable`` those with at
    least ``min_count`` members whose period could be placed; ``groups`` the ones where that
    many were present together, largest simultaneous set first.
    """
    need = max(int(min_count or 2), 2)
    result = OverlapResult(
        group_field=group_field, start_field=start_field, end_field=end_field, min_count=need
    )
    by_value: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for row in rows:
        raw = semantic_value(row, group_field, catalog)
        label = derive_part(raw, value_part) if raw is not None else None
        label = " ".join(str(label).split()) if label else ""
        if not label:
            continue
        by_value.setdefault(label.casefold(), (label, []))[1].append(row)

    for label, members in by_value.values():
        if len(members) < need:
            continue
        result.shared += 1
        stays = [
            stay
            for row in members
            if (stay := _stay(row, catalog, start_field=start_field, end_field=end_field))
            is not None
        ]
        if len(stays) < need:
            continue
        result.checkable += 1
        together, episodes = _together(stays, need)
        if together:
            result.groups.append(
                OverlapGroup(
                    value=label,
                    members=len(members),
                    dated=len(stays),
                    together=sorted(together, key=lambda stay: (stay.start, stay.name.casefold())),
                    first=episodes[0][0] if episodes else None,
                    last=episodes[-1][1] if episodes else None,
                    episodes=episodes,
                )
            )
    result.groups.sort(key=lambda group: (-len(group.together), group.value.casefold()))
    return result


def _stay(row: dict[str, Any], catalog: FieldCatalog, *, start_field: str, end_field: str) -> Stay | None:
    start = _bound(_stored(row, start_field, catalog), end=False)
    end = _bound(_stored(row, end_field, catalog), end=True)
    if start is None or end is None or end < start:
        return None
    name = semantic_value(row, "student_name", catalog) or row.get("canonical_name") or "Unnamed record"
    return Stay(name=" ".join(str(name).split()), start=start, end=end)


def _stored(row: dict[str, Any], field_name: str, catalog: FieldCatalog) -> Any:
    """The stored date node itself, so its recorded precision is not lost to a scalar."""
    spec = catalog.resolve_field(int(row.get("file_id") or 0), field_name)
    if spec is None or not spec.canonical_json_path:
        return None
    node: Any = json_object(row.get("row_data_normalized"))
    for part in spec.canonical_json_path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _bound(node: Any, *, end: bool) -> date | None:
    """Where a period starts or ends. A year alone covers the whole year; two dates, none."""
    if isinstance(node, dict):
        kind = str(node.get("kind") or "")
        if kind in {"range", "unparsed"}:
            return None
        if kind == "year" and str(node.get("year") or "").isdigit():
            year = int(node["year"])
            return date(year, 12, 31) if end else date(year, 1, 1)
        text = node.get("iso") or node.get("raw")
    else:
        text = node
    parsed = parse_date(text)
    if parsed is None or not end:
        return parsed
    precision = date_precision(text)
    if precision == "year":
        return date(parsed.year, 12, 31)
    if precision == "month":
        return date(parsed.year, parsed.month, calendar.monthrange(parsed.year, parsed.month)[1])
    return parsed


def _together(stays: list[Stay], need: int) -> tuple[list[Stay], list[tuple[date, date]]]:
    """Members present while at least ``need`` were, and each stretch during which they were."""
    events: list[tuple[date, int, int]] = []
    for index, stay in enumerate(stays):
        events.append((stay.start, 1, index))
        # A period includes its last day, so it leaves the day after -- and a departure is
        # processed before an arrival on the same day, so back-to-back stays do not overlap.
        events.append((stay.end + timedelta(days=1), -1, index))
    events.sort(key=lambda item: (item[0], item[1]))
    active: set[int] = set()
    present: set[int] = set()
    episodes: list[tuple[date, date]] = []
    began: date | None = None
    for day, change, index in events:
        enough_before = len(active) >= need
        if change > 0:
            active.add(index)
        else:
            active.discard(index)
        enough_now = len(active) >= need
        if enough_now:
            present.update(active)
            if not enough_before:
                began = day
        if enough_before and not enough_now and began is not None:
            episodes.append((began, day - timedelta(days=1)))
            began = None
    return [stays[index] for index in present], episodes
