"""Counting the people a list holds, not only its rows.

"How many children are on this list?" was answered with the row count, 56, on a list where a
tester counted 49 children: three rows are blank, one is a "TO BE REVIEWED" placeholder, two
are sentences about a family rather than one child, and one child is entered twice. The row
count is still a true fact and stays the computed answer; what was missing is the count a
researcher means, and the reasons the two differ.

The name parser has already said which rows hold no person -- it flags placeholders and family
narratives -- so this only reads those flags. A repeated entry is the same parsed name with no
recorded identifier that tells the two rows apart: two rows called John Smith with different
student numbers are two people, and are counted as two.

Reading every matching row is only cheap on a short list, so the turn service asks for a
roster only when the count is within ``roster_max_rows``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.execution.row_projection import semantic_value
from app.planning.catalog import FieldCatalog
from app.planning.plan_schema import Predicate
from app.security.access_scope import AccessScope
from app.value_normalization import json_object

# Recorded facts that distinguish two people who share a name. A value on one row and a blank
# on the other is not a disagreement; two different values are.
_IDENTITY_FIELDS = ("student_number", "birth_date", "death_date", "community")
# The gateway returns at most this many rows per call.
_PAGE = 500


@dataclass
class Person:
    key: str
    name: str
    identity: dict[str, str]
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Roster:
    rows: int
    people: list[Person]
    blank: int = 0
    placeholders: int = 0
    notes: int = 0
    repeats: list[str] = field(default_factory=list)

    @property
    def unique(self) -> int:
        return len(self.people)

    @property
    def differs(self) -> bool:
        return self.unique != self.rows

    def recorded_for(self, field_name: str, catalog: FieldCatalog) -> int:
        """People with a value for the field on any of their rows."""
        return sum(
            1
            for person in self.people
            if any(_plain(semantic_value(row, field_name, catalog)) for row in person.rows)
        )


async def fetch_rows(
    gateway: Any,
    scope: AccessScope,
    file_ids: tuple[int, ...],
    predicates: list[Predicate],
    total: int,
) -> list[dict[str, Any]]:
    """Every matching row, a page at a time."""
    rows: list[dict[str, Any]] = []
    while len(rows) < total:
        page = await gateway.list_records(
            scope, file_ids, predicates, limit=min(_PAGE, total - len(rows)), offset=len(rows)
        )
        if not page:
            break
        rows.extend(page)
    return rows


def roster_from_rows(rows: list[dict[str, Any]], catalog: FieldCatalog) -> Roster:
    roster = Roster(rows=len(rows), people=[])
    for row in rows:
        file_id = int(row.get("file_id") or 0)
        canonical = json_object(row.get("row_data_normalized")).get("canonical")
        parts = canonical.get("name_parts") if isinstance(canonical, dict) else None
        parts = parts if isinstance(parts, dict) else {}
        flags = {item.strip() for item in str(parts.get("flags") or "").split(";") if item.strip()}
        name = _plain(semantic_value(row, "student_name", catalog)) or _plain(row.get("canonical_name"))
        if "placeholder" in flags:
            roster.placeholders += 1
            continue
        if "narrative" in flags:
            roster.notes += 1
            continue
        key = _person_key(parts, name)
        if not key:
            roster.blank += 1
            continue
        identity = {
            field_name: value
            for field_name in _IDENTITY_FIELDS
            if catalog.resolve_field(file_id, field_name) is not None
            and (value := _fold(semantic_value(row, field_name, catalog)))
        }
        same = next(
            (
                person
                for person in roster.people
                if person.key == key and not _conflicts(person.identity, identity)
            ),
            None,
        )
        if same is None:
            roster.people.append(Person(key=key, name=name, identity=identity, rows=[row]))
            continue
        same.rows.append(row)
        for field_name, value in identity.items():
            same.identity.setdefault(field_name, value)
        roster.repeats.append(same.name)
    return roster


def roster_sentence(roster: Roster) -> str:
    """Why the number of people differs from the number of rows, or "" when it does not."""
    if not roster.differs:
        return ""
    others = roster.rows - roster.unique
    pieces: list[str] = []
    if roster.blank:
        pieces.append(_plural(roster.blank, "blank row", "blank rows"))
    if roster.placeholders:
        pieces.append(_plural(roster.placeholders, "placeholder row", "placeholder rows"))
    if roster.notes:
        pieces.append(_plural(roster.notes, "note about a family", "notes about a family"))
    if roster.repeats:
        names = "; ".join(dict.fromkeys(roster.repeats))
        pieces.append(
            f"{_plural(len(roster.repeats), 'repeated entry', 'repeated entries')} ({names})"
        )
    people = "person" if roster.unique == 1 else "people"
    verb = "is" if others == 1 else "are"
    return (
        f"Counting people rather than rows, that is **{roster.unique:,} unique named {people}**; "
        f"the other {_plural(others, 'row', 'rows')} {verb} {_join(pieces)}."
    )


def _person_key(parts: dict[str, Any], name: str) -> str:
    first = _fold(parts.get("first"))
    last = _fold(parts.get("last"))
    if first or last:
        return f"{first}|{last}"
    return _fold(name)


def _conflicts(left: dict[str, str], right: dict[str, str]) -> bool:
    return any(left[key] != right[key] for key in left.keys() & right.keys())


def _fold(value: Any) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(value or "").casefold()).split())


def _plain(value: Any) -> str:
    return " ".join(str(value).split()) if value not in (None, "", [], {}) else ""


def _plural(count: int, one: str, many: str) -> str:
    return f"{count:,} {one if count == 1 else many}"


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"
