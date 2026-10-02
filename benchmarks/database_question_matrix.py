"""Database-grounded end-to-end benchmark for NIA's semantic query pipeline.

The oracle reads current records directly from PostgreSQL and computes expected
results without calling planner, retrieval, formatting, or assistant API code.
The same natural-language questions are then sent through AssistantTurnService
with the real catalog, gateway, and configured reasoning provider.

Run from the repository root:

    python backend/benchmarks/database_question_matrix.py

Set BENCHMARK_DATABASE_URL when the migrator URL cannot read the source tables.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import asyncpg

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings  # noqa: E402
from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService, TurnResult  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402

Kind = Literal[
    "schema",
    "count",
    "distinct_count",
    "distinct_values",
    "distribution",
    "mode",
    "min",
    "max",
    "list",
    "lookup",
    "unsupported",
]


@dataclass(frozen=True)
class Condition:
    field: str
    operator: str
    value: Any = None


@dataclass(frozen=True)
class Case:
    id: str
    file_id: int
    question: str
    kind: Kind
    field: str | None = None
    conditions: tuple[Condition, ...] = ()
    direction: Literal["asc", "desc"] = "asc"
    limit: int = 5
    projection: tuple[str, ...] = ()
    search: str | None = None
    include_missing: bool = False
    value_part: Literal["year"] | None = None
    window_anchor: Literal["start", "end"] | None = None


CASES: tuple[Case, ...] = (
    Case("M01", 49, "How many records are in the master list?", "count"),
    Case("M02", 49, "What fields are available in the master list?", "schema"),
    Case("M03", 49, "How many students are recorded as deceased?", "count", conditions=(Condition("deceased_status", "true"),)),
    Case("M04", 49, "How many students are recorded as not deceased?", "count", conditions=(Condition("deceased_status", "false"),)),
    Case("M05", 49, "How many students have an unknown deceased status?", "count", conditions=(Condition("deceased_status", "equals", "unknown"),)),
    Case("M06", 49, "How many students have a recorded community?", "count", conditions=(Condition("community", "known"),)),
    Case("M07", 49, "How many student records are missing a community?", "count", conditions=(Condition("community", "missing"),)),
    Case("M08", 49, "How many distinct communities are recorded?", "distinct_count", field="community"),
    Case("M09", 49, "Which community appears most often?", "mode", field="community", direction="desc"),
    Case("M11", 49, "How many students were admitted in 1964?", "count", conditions=(Condition("admitted_date", "year", 1964),)),
    Case("M12", 49, "How many students were admitted before 1900?", "count", conditions=(Condition("admitted_date", "before", 1900),)),
    Case("M13", 49, "How many students were admitted between 1940 and 1945?", "count", conditions=(Condition("admitted_date", "range", (1940, 1945)),)),
    Case("M14", 49, "How many students were discharged after 1965?", "count", conditions=(Condition("discharged_date", "after", 1965),)),
    Case("M15", 49, "How many students have a recorded date of birth?", "count", conditions=(Condition("birth_date", "known"),)),
    Case("M16", 49, "How many student records are missing the date of birth?", "count", conditions=(Condition("birth_date", "missing"),)),
    Case("M17", 49, "How many students have a recorded age greater than 15?", "count", conditions=(Condition("age", "gt", 15),)),
    Case("M18", 49, "How many students have an age between 10 and 12?", "count", conditions=(Condition("age", "number_range", (10, 12)),)),
    Case("M19", 49, "What is the highest recorded student age?", "max", field="age"),
    Case("M20", 49, "What is the lowest recorded student age?", "min", field="age"),
    Case("M21", 49, "List students whose names start with Z.", "list", conditions=(Condition("student_name", "starts", "z"),), projection=("student_name",), limit=20),
    Case("M22", 49, "Show the student number and community for Albert Penance.", "lookup", search="albert penance", projection=("student_name", "student_number", "community")),
    Case("M23", 49, "Show all information for Albert Penance.", "lookup", search="albert penance", projection=("*",)),
    Case("M24", 49, "List the last 5 students alphabetically by full student name.", "list", field="student_name", direction="asc", limit=5, projection=("student_name",), window_anchor="end"),
    Case("M25", 49, "List the first 5 students with their names and communities.", "list", field="student_name", direction="asc", limit=5, projection=("student_name", "community")),
    Case("M26", 49, "How many records have a student number?", "count", conditions=(Condition("student_number", "known"),)),
    Case("M27", 49, "What is the most common mapping location?", "mode", field="mapping_location", direction="desc"),
    Case("M28", 49, "How many notes contain student register?", "count", conditions=(Condition("notes", "contains", "student register"),)),
    Case("D01", 91, "How many records are in the confirmed deaths list?", "count"),
    Case("D02", 91, "What fields are available in confirmed deaths?", "schema"),
    Case("D03", 91, "How many confirmed death records have a recorded cause of death?", "count", conditions=(Condition("cause_of_death", "known"),)),
    Case("D04", 91, "How many confirmed death records are missing a cause of death?", "count", conditions=(Condition("cause_of_death", "missing"),)),
    Case("D05", 91, "What different causes of death are recorded?", "distinct_values", field="cause_of_death"),
    Case("D06", 91, "What is the most common cause of death?", "mode", field="cause_of_death", direction="desc"),
    Case("D07", 91, "How many causes of death contain tuberculosis?", "count", conditions=(Condition("cause_of_death", "contains", "tuberculosis"),)),
    Case("D08", 91, "How many confirmed deaths mention drowning as the cause?", "count", conditions=(Condition("cause_of_death", "contains", "drown"),)),
    Case("D09", 91, "How many students died in 1900?", "count", conditions=(Condition("death_date", "year", 1900),)),
    Case("D10", 91, "Which year has the most confirmed deaths?", "mode", field="death_date", direction="desc", value_part="year"),
    Case("D11", 91, "What is the earliest recorded death year?", "min", field="death_date"),
    Case("D12", 91, "What is the latest recorded death year?", "max", field="death_date"),
    Case("D13", 91, "How many students were older than 15 when they died?", "count", conditions=(Condition("age_at_death", "gt", 15),)),
    Case("D14", 91, "How many students were between 10 and 14 years old at death?", "count", conditions=(Condition("age_at_death", "number_range", (10, 14)),)),
    Case("D15", 91, "What is the maximum recorded age at death?", "max", field="age_at_death"),
    Case("D16", 91, "What is the minimum recorded age at death?", "min", field="age_at_death"),
    Case("D17", 91, "Show the distribution of recorded genders.", "distribution", field="gender", include_missing=True),
    Case("D18", 91, "How many confirmed death records are female?", "count", conditions=(Condition("gender", "equals", "F"),)),
    Case("D19", 91, "How many confirmed death records have no recorded gender?", "count", conditions=(Condition("gender", "missing"),)),
    Case("D20", 91, "How many confirmed death records have a recorded community or reserve?", "count", conditions=(Condition("community", "known"),)),
    Case("D21", 91, "How many distinct communities or reserves are recorded?", "distinct_count", field="community"),
    Case("D22", 91, "How many places of death contain Shingwauk?", "count", conditions=(Condition("place_of_death", "contains", "shingwauk"),)),
    Case("D23", 91, "How many children died at school?", "count", conditions=(Condition("location_of_death", "contains", "school"),)),
    Case("D24", 91, "Show all information for Albert Penance.", "lookup", search="albert penance", projection=("*",)),
    Case("D25", 91, "List the first 5 deaths chronologically with name and death date.", "list", field="death_date", direction="asc", limit=5, projection=("student_name", "death_date")),
    Case("D26", 91, "List the latest 5 deaths with name, date of death, and cause.", "list", field="death_date", direction="desc", limit=5, projection=("student_name", "death_date", "cause_of_death")),
    Case("D27", 91, "How many records have a burial date?", "count", conditions=(Condition("burial_date", "known"),)),
    Case("D28", 91, "What is the most common recorded place of burial?", "mode", field="place_of_burial", direction="desc"),
    Case("D29", 91, "How many records indicate that census documents were used?", "count", conditions=(Condition("census_documents_used", "true"),)),
    Case("D30", 91, "How many records mention another school or institution?", "count", conditions=(Condition("other_schools", "known"),)),
    Case("A01", 93, "How many records are in additional deaths?", "count"),
    Case("A02", 93, "How many additional death records have a recorded community?", "count", conditions=(Condition("community", "known"),)),
    Case("A03", 93, "How many additional death records are missing a community?", "count", conditions=(Condition("community", "missing"),)),
    Case("A04", 93, "How many distinct communities are recorded in additional deaths?", "distinct_count", field="community"),
    Case("A05", 93, "List the first 5 additional death names.", "list", field="student_name", direction="asc", limit=5, projection=("student_name",)),
    Case("A06", 93, "How many additional death records have a student number?", "count", conditions=(Condition("student_number", "known"),)),
    Case("P01", 94, "How many records are in the potential list?", "count"),
    Case("P02", 94, "How many potential records have a recorded cause of death?", "count", conditions=(Condition("cause_of_death", "known"),)),
    Case("P03", 94, "How many potential records are missing a cause of death?", "count", conditions=(Condition("cause_of_death", "missing"),)),
    Case("P04", 94, "What is the most common cause of death in potential records?", "mode", field="cause_of_death", direction="desc"),
    Case("P05", 94, "How many potential records have a death date after 1900?", "count", conditions=(Condition("death_date", "after", 1900),)),
    Case("P06", 94, "What is the maximum recorded age at death in potential records?", "max", field="age_at_death"),
    Case("P07", 94, "List the first 5 potential records by name.", "list", field="student_name", direction="asc", limit=5, projection=("student_name",)),
    Case("P08", 94, "How many potential records mention another school?", "count", conditions=(Condition("other_schools", "known"),)),
    # Was "unsupported" while file 94 had no community field registered. The list does record
    # origin -- in COMMUNITY/RESERVE and in pasted name cells -- so this is an ordinary count.
    Case("P09", 94, "How many potential records are from Garden River?", "count", conditions=(Condition("community", "contains", "garden river"),)),
    Case("P10", 94, "How many potential records have a student number?", "unsupported"),
)


@dataclass
class Oracle:
    records: dict[int, list[dict[str, Any]]]
    paths: dict[tuple[int, str], str]
    labels: dict[int, dict[str, str]]
    rules: dict[tuple[int, str], dict[str, Any]]
    types: dict[tuple[int, str], str] = dataclasses.field(default_factory=dict)
    # Text ordering is a property of the database, not of Python. structured_list sorts on
    # lower(value) under this database's en_US.utf8 collation, which folds punctuation --
    # "[Blarbie] Dunn" files under B. str.casefold() compares codepoints instead, so "[" (91)
    # sorts before "a" (97) and the two disagree on every bracketed or accented name.
    # Rather than reimplement ICU, the ordering is asked of the database once at load and
    # looked up here; which rows and which values are still decided independently.
    collation_rank: dict[tuple[int, str], dict[str, int]] = dataclasses.field(default_factory=dict)

    def value(self, file_id: int, row: dict[str, Any], field_name: str) -> Any:
        path = self.paths.get((file_id, field_name))
        current: Any = row.get("row_data_normalized") or {}
        walked = None
        if path:
            walked = current
            for part in path.split("."):
                if not isinstance(walked, dict):
                    walked = None
                    break
                walked = walked.get(part)
            scalar = _scalar(walked)
            if scalar not in (None, ""):
                return scalar
        direct_column = {
            "canonical.name": "canonical_name",
            "canonical.display_name": "canonical_name",
            "canonical.community": "canonical_community",
            "canonical.school": "canonical_school",
        }.get(path or "")
        if direct_column:
            direct = row.get(direct_column)
            if direct not in (None, ""):
                return direct
        return None

    def expected(self, case: Case) -> dict[str, Any]:
        if case.kind == "unsupported":
            return {"value": None}
        rows = [row for row in self.records.get(case.file_id, []) if self.matches(case.file_id, row, case.conditions)]
        if case.kind == "schema":
            return {"fields": list(self.labels[case.file_id]), "field_count": len(self.labels[case.file_id])}
        if case.kind == "count":
            return {"value": len(rows)}
        if case.kind in {"distinct_count", "distinct_values", "distribution", "mode"}:
            values = [self.value(case.file_id, row, case.field or "") for row in rows]
            if case.value_part == "year":
                values = [_year(value) for value in values]
            normalized = [str(value) for value in values if value not in (None, "")]
            counts = Counter(normalized)
            if case.include_missing:
                counts["Not recorded"] += len(values) - len(normalized)
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))
            if case.kind == "distinct_count":
                return {"value": len(counts), "values": sorted(counts, key=str.casefold)}
            if case.kind == "distinct_values":
                return {"value": len(counts), "values": sorted(counts, key=str.casefold)}
            if case.kind == "distribution":
                return {"counts": dict(ordered)}
            return {"value": ordered[0][0] if ordered else None, "count": ordered[0][1] if ordered else 0}
        if case.kind in {"min", "max"}:
            values = [self.value(case.file_id, row, case.field or "") for row in rows]
            typed = [_comparable(value, date_field=(case.field or "").endswith("_date")) for value in values]
            typed = [value for value in typed if value is not None]
            selected = (min(typed) if case.kind == "min" else max(typed)) if typed else None
            return {"value": selected}
        if case.kind == "lookup":
            needle = (case.search or "").casefold()
            rows = [row for row in rows if needle in str(self.value(case.file_id, row, "student_name") or "").casefold()]
            fields = self._projection(case)
            return {"rows": [self._project(case.file_id, row, fields) for row in rows], "fields": fields}
        if case.kind == "list":
            sort_field = case.field or "student_name"
            if sort_field.endswith("_date"):
                present = [
                    row
                    for row in rows
                    if _date_key(self.value(case.file_id, row, sort_field)) is not None
                ]
                missing = [
                    row
                    for row in rows
                    if _date_key(self.value(case.file_id, row, sort_field)) is None
                ]
            else:
                present = [row for row in rows if self.value(case.file_id, row, sort_field) not in (None, "")]
                missing = [row for row in rows if self.value(case.file_id, row, sort_field) in (None, "")]
            if sort_field.endswith("_date"):
                # Direction applies to the requested date only. Identity and
                # source-row tie breakers remain ascending for stable pages.
                present.sort(
                    key=lambda row: (
                        str(row.get("canonical_name") or "").casefold(),
                        int(row.get("source_row_id") or 0),
                    )
                )
                present.sort(
                    key=lambda row: _date_key(self.value(case.file_id, row, sort_field)) or (0, 0, 0),
                    reverse=case.direction == "desc",
                )
            else:
                present.sort(
                    key=lambda row: self._text_sort_key(case.file_id, sort_field, row),
                    reverse=case.direction == "desc",
                )
            missing.sort(key=lambda row: (str(row.get("canonical_name") or "").casefold(), int(row.get("source_row_id") or 0)))
            rows = present + missing
            fields = self._projection(case)
            selected = rows[-case.limit :] if case.window_anchor == "end" else rows[: case.limit]
            return {"rows": [self._project(case.file_id, row, fields) for row in selected], "fields": fields, "total": len(rows)}
        raise ValueError(f"unsupported case kind: {case.kind}")

    def matches(self, file_id: int, row: dict[str, Any], conditions: tuple[Condition, ...]) -> bool:
        return all(
            _matches(
                self.value(file_id, row, condition.field),
                condition,
                rules=self.rules.get((file_id, condition.field), {}),
            )
            for condition in conditions
        )

    def _text_sort_key(self, file_id: int, field_name: str, row: dict[str, Any]) -> tuple[Any, ...]:
        """Order one row the way structured_list would.

        The database compares a numeric key only when the field's semantic_type is
        'number'; every other field sorts on lower(text) under the database collation. The
        old key extracted a number from any value that contained one, so a name like
        "Emma IsaacsMay 1, 1912Walpole Island" sorted as 1912 -- ahead of every name that
        held no digits.
        """
        value = self.value(file_id, row, field_name)
        text = str(value or "").strip()
        if self.types.get((file_id, field_name)) == "number":
            number = _number(value)
            comparable: tuple[int, Any] = (1, number) if number is not None else (2, text.casefold())
        else:
            ranks = self.collation_rank.get((file_id, field_name), {})
            comparable = (1, ranks.get(text.lower(), len(ranks)))
        return (
            *comparable,
            str(row.get("canonical_name") or "").casefold(),
            int(row.get("source_row_id") or 0),
        )

    def _projection(self, case: Case) -> list[str]:
        if case.projection == ("*",):
            # "All information" means the record as it was recorded. The derived name and
            # date columns are how a row is found, not what it says: they repeat the source
            # cell in the pipeline's own vocabulary, and NIA deliberately leaves them out of
            # the row it returns. Expecting them here would have held the product to the
            # opposite of its rule -- M23 failed on a missing discharged_date_iso, D24 on a
            # missing name_residue, both of which are ours and neither of which is a fact
            # about the person.
            return [
                name
                for name in self.labels[case.file_id]
                if not (self.paths.get((case.file_id, name)) or "").startswith(
                    ("canonical.name_parts.", "canonical.date_parts.")
                )
                and (self.paths.get((case.file_id, name)) or "") != "names"
            ]
        return list(case.projection or ("student_name",))

    def _project(self, file_id: int, row: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        return {field_name: self.value(file_id, row, field_name) for field_name in fields}


def _scalar(value: Any) -> Any:
    if not isinstance(value, dict):
        return value if value not in ("", []) else None
    for key in ("iso", "raw", "display", "value", "year", "normalized"):
        candidate = value.get(key)
        if candidate not in (None, "", []):
            return candidate
    return None


def _year(value: Any) -> int | None:
    if isinstance(value, dict):
        value = _scalar(value)
    match = re.search(r"(1[6-9]\d{2}|20\d{2})", str(value or ""))
    return int(match.group(1)) if match else None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group(0)) if match else None


def _bool(value: Any) -> bool | None:
    text = str(value or "").strip().casefold()
    if text in {"yes", "true", "t", "1", "y", "deceased"}:
        return True
    if text in {"no", "false", "f", "0", "n"}:
        return False
    return None


def _unknown_terms(rules: dict[str, Any] | None) -> list[str]:
    families = (rules or {}).get("value_synonyms")
    if not isinstance(families, dict):
        return []
    for key, members in families.items():
        if str(key).casefold() == "unknown" and isinstance(members, list):
            return [str(item) for item in members]
    return []


def _synonym_terms(rules: dict[str, Any] | None, value: Any) -> list[str]:
    families = (rules or {}).get("value_synonyms")
    if not isinstance(families, dict):
        return [str(value)]
    needle = str(value or "").strip().casefold()
    if not needle:
        return [str(value)]
    for key, members in families.items():
        if not isinstance(members, list):
            continue
        names = [str(key), *[str(item) for item in members]]
        if any(item.casefold() == needle for item in names):
            return names
    return [str(value)]


def _contains_any(value: Any, terms: list[str]) -> bool:
    haystack = str(value).casefold()
    return any(term.casefold() in haystack for term in terms if term)


def _matches(value: Any, condition: Condition, *, rules: dict[str, Any] | None = None) -> bool:
    op = condition.operator
    if op == "known":
        if value in (None, "", []):
            return False
        return not _contains_any(value, _unknown_terms(rules))
    if op == "missing":
        return value in (None, "", [])
    if op == "true":
        return _bool(value) is True
    if op == "false":
        return _bool(value) is False
    if value in (None, ""):
        return False
    if op == "equals":
        return str(value).strip().casefold() == str(condition.value).strip().casefold()
    if op == "contains":
        return _contains_any(value, _synonym_terms(rules, condition.value))
    if op == "starts":
        return str(value).casefold().startswith(str(condition.value).casefold())
    if op in {"year", "before", "after", "range"}:
        actual = _year(value)
        if op == "year":
            return actual == int(condition.value)
        if op == "before":
            return actual is not None and actual < int(condition.value)
        if op == "after":
            return actual is not None and actual > int(condition.value)
        return actual is not None and int(condition.value[0]) <= actual <= int(condition.value[1])
    actual_number = _number(value)
    if op == "gt":
        return actual_number is not None and actual_number > float(condition.value)
    if op == "number_range":
        return actual_number is not None and float(condition.value[0]) <= actual_number <= float(condition.value[1])
    raise ValueError(f"unknown condition operator {op}")


def _comparable(value: Any, *, date_field: bool) -> int | float | str | None:
    if value in (None, ""):
        return None
    if date_field:
        return _year(value)
    number = _number(value)
    return number if number is not None else str(value)


def _sort_key(value: Any, row: dict[str, Any], *, date_field: bool = False) -> tuple[Any, ...]:
    text = str(value or "").strip()
    if date_field:
        comparable: tuple[int, Any] = (0, _date_key(value) or (0, 0, 0))
    else:
        number = _number(value)
        comparable = (1, number) if number is not None else (2, text.casefold())
    return (*comparable, str(row.get("canonical_name") or "").casefold(), int(row.get("source_row_id") or 0))


def _date_key(value: Any) -> tuple[int, int, int] | None:
    """Parse a conservative chronological key without trusting malformed years."""

    text = str(_scalar(value) if isinstance(value, dict) else value or "").strip()
    year_match = re.search(r"(?<!\d)(1[6-9]\d{2}|20\d{2})(?!\d)", text)
    if not year_match:
        return None
    year = int(year_match.group(1))
    month = day = 0
    leading = re.search(r"(?<!\d)(1[6-9]\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text)
    trailing = re.search(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](1[6-9]\d{2}|20\d{2})(?!\d)", text)
    if leading:
        month, day = int(leading.group(2)), int(leading.group(3))
    elif trailing:
        day, month = int(trailing.group(1)), int(trailing.group(2))
    else:
        month_names = {
            name: index
            for index, name in enumerate(
                ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"),
                start=1,
            )
        }
        named = re.search(
            r"(?<!\d)(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(1[6-9]\d{2}|20\d{2})(?!\d)",
            text,
            re.I,
        )
        if named:
            day, month = int(named.group(1)), month_names[named.group(2).casefold()]
    if not 0 <= month <= 12 or not 0 <= day <= 31:
        return None
    return year, month, day


async def load_oracle() -> Oracle:
    settings = get_settings()
    dsn = os.getenv("BENCHMARK_DATABASE_URL") or settings.assistant_migrator_database_url
    connection = await asyncpg.connect(dsn)
    try:
        record_rows = await connection.fetch(
            """
            SELECT file_id, source_row_id, canonical_name, canonical_community,
                   canonical_school, row_data_normalized
            FROM assistant_api.v_current_records
            WHERE file_id = ANY($1::int[])
            ORDER BY file_id, source_row_id
            """,
            [49, 91, 93, 94],
        )
        field_rows = await connection.fetch(
            """
            SELECT file_id, semantic_field, human_label, canonical_json_path,
                   validation_rules, semantic_type
            FROM assistant.field_registry
            WHERE file_id = ANY($1::int[])
            ORDER BY file_id, id
            """,
            [49, 91, 93, 94],
        )
        collation_rank = await _collation_ranks(connection)
    finally:
        await connection.close()
    records: dict[int, list[dict[str, Any]]] = {49: [], 91: [], 93: [], 94: []}
    for item in record_rows:
        row = dict(item)
        if isinstance(row["row_data_normalized"], str):
            row["row_data_normalized"] = json.loads(row["row_data_normalized"])
        records.setdefault(int(row["file_id"]), []).append(row)
    paths = {(int(row["file_id"]), str(row["semantic_field"])): str(row["canonical_json_path"] or "") for row in field_rows}
    labels: dict[int, dict[str, str]] = {49: {}, 91: {}, 93: {}, 94: {}}
    rules: dict[tuple[int, str], dict[str, Any]] = {}
    for row in field_rows:
        labels[int(row["file_id"])][str(row["semantic_field"])] = str(row["human_label"])
        raw_rules = row["validation_rules"]
        if isinstance(raw_rules, str):
            raw_rules = json.loads(raw_rules)
        rules[(int(row["file_id"]), str(row["semantic_field"]))] = dict(raw_rules or {})
    types = {
        (int(row["file_id"]), str(row["semantic_field"])): str(row["semantic_type"] or "")
        for row in field_rows
    }
    return Oracle(
        records=records,
        paths=paths,
        labels=labels,
        rules=rules,
        types=types,
        collation_rank=collation_rank,
    )


async def _collation_ranks(connection: Any) -> dict[tuple[int, str], dict[str, int]]:
    """Ask the database to order the values every sorted list case will sort on.

    One query per (list, field) actually used by a case -- a handful in total. The ORDER BY
    mirrors structured_list's: lower(value) under the database's own collation.
    """
    wanted = {
        (case.file_id, case.field)
        for case in CASES
        if case.kind in {"list", "lookup"} and case.field and not case.field.endswith("_date")
    }
    ranks: dict[tuple[int, str], dict[str, int]] = {}
    for file_id, field_name in sorted(wanted):
        rows = await connection.fetch(
            """
            SELECT lower(v.value) AS value,
                   row_number() OVER (ORDER BY lower(v.value) ASC NULLS LAST) - 1 AS position
            FROM assistant_api.v_current_records r
            CROSS JOIN LATERAL (
                SELECT NULLIF(btrim(assistant_api._field_display_at_path(
                    r.row_data_normalized, r.canonical_name,
                    r.canonical_community, r.canonical_school,
                    (SELECT canonical_json_path FROM assistant.field_registry f
                     WHERE f.file_id = r.file_id AND f.semantic_field = $2 LIMIT 1)
                )), '') AS value
            ) v
            WHERE r.file_id = $1 AND v.value IS NOT NULL
            """,
            file_id,
            field_name,
        )
        ranks[(file_id, field_name)] = {
            str(row["value"]): int(row["position"]) for row in rows
        }
    return ranks


def _fact(result: TurnResult, name: str) -> Any:
    for item in result.facts:
        if item.get("name") == name:
            return item.get("value")
    return None


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _normalize(value)).strip()


def _mentions_number(text: str, value: Any) -> bool:
    """True when `value` appears in text, ignoring thousands separators."""
    raw = str(value)
    if raw in text:
        return True
    compact = re.sub(r"[,\s]", "", raw)
    haystack = re.sub(r",", "", text)
    return bool(compact) and compact in haystack


def compare(case: Case, expected: dict[str, Any], result: TurnResult) -> list[str]:
    issues: list[str] = []
    if case.kind == "unsupported":
        if result.plan is not None:
            return ["unsupported field produced a QueryPlan"]
        if result.status in {"unsupported_fields", "clarification", "dataset_not_selected"}:
            return []
        return [f"status={result.status}, expected unsupported-field refusal"]
    if result.status != "answered":
        return [f"status={result.status} error={result.error_code or result.detail}"]
    if result.selected_file_id != case.file_id:
        issues.append(f"selected_file_id={result.selected_file_id}, expected {case.file_id}")
    if result.plan is not None and tuple(result.plan.scope.file_ids) != (case.file_id,):
        issues.append("query plan did not remain exclusively dataset-scoped")
    answer = _normalize(result.answer)
    if case.kind == "schema":
        missing = [
            field_name
            for field_name in expected["fields"]
            if _normalize(field_name.replace("_", " ")) not in answer
            and _normalize(oracle_label(case.file_id, field_name)) not in answer
        ]
        if missing:
            issues.append(f"schema response omitted fields: {', '.join(missing)}")
    elif case.kind == "count":
        actual = _fact(result, "count")
        if actual != expected["value"]:
            issues.append(f"count={actual!r}, expected {expected['value']!r}")
    elif case.kind == "distinct_count":
        actual = _fact(result, "count_distinct")
        if actual != expected["value"]:
            issues.append(f"count_distinct={actual!r}, expected {expected['value']!r}")
    elif case.kind == "distinct_values":
        missing = [value for value in expected["values"] if _normalize(value) not in answer]
        if missing:
            issues.append(f"distinct response omitted {len(missing)} values; first={missing[:3]}")
    elif case.kind == "distribution":
        missing = [
            f"{label}={count}"
            for label, count in expected["counts"].items()
            if _normalize(label) not in answer or not _mentions_number(result.answer, count)
        ]
        if missing:
            issues.append(f"distribution incomplete: {missing[:4]}")
    elif case.kind == "mode":
        if _normalize(expected["value"]) not in answer or not _mentions_number(result.answer, expected["count"]):
            issues.append(f"mode answer did not report {expected['value']!r} ({expected['count']})")
    elif case.kind in {"min", "max"}:
        actual = _fact(result, case.kind)
        if isinstance(expected["value"], float) and expected["value"].is_integer():
            expected_value: Any = int(expected["value"])
        else:
            expected_value = expected["value"]
        if actual != expected_value:
            issues.append(f"{case.kind}={actual!r}, expected {expected_value!r}")
    elif case.kind in {"list", "lookup"}:
        if not result.list_results:
            issues.append("no structured list result")
        else:
            actual_rows = result.list_results[0].get("rows") or []
            actual_values = [row.get("values") or {} for row in actual_rows]
            expected_rows = expected["rows"]
            if len(actual_values) != len(expected_rows):
                issues.append(f"row_count={len(actual_values)}, expected {len(expected_rows)}")
            for index, expected_row in enumerate(expected_rows[: len(actual_values)]):
                for field_name, expected_value in expected_row.items():
                    actual_value = actual_values[index].get(field_name)
                    same = (
                        _normalize_name(actual_value) == _normalize_name(expected_value)
                        if field_name.endswith("_name") or field_name == "student_name"
                        else _normalize(actual_value) == _normalize(expected_value)
                    )
                    if not same:
                        issues.append(f"row {index + 1} {field_name}={actual_value!r}, expected {expected_value!r}")
                        break
                if len(issues) >= 5:
                    break
    return issues


_LABELS: dict[tuple[int, str], str] = {}


def oracle_label(file_id: int, field_name: str) -> str:
    return _LABELS.get((file_id, field_name), field_name.replace("_", " "))


def expected_summary(case: Case, expected: dict[str, Any]) -> str:
    if case.kind == "unsupported":
        return "unsupported-field refusal"
    if case.kind == "schema":
        return f"{expected['field_count']} fields: {', '.join(expected['fields'])}"
    if case.kind in {"count", "distinct_count", "min", "max"}:
        return str(expected["value"])
    if case.kind == "mode":
        return f"{expected['value']} ({expected['count']} records)"
    if case.kind == "distribution":
        return ", ".join(f"{key}: {value}" for key, value in expected["counts"].items())
    if case.kind == "distinct_values":
        return f"{expected['value']} values: {', '.join(expected['values'])}"
    rows = expected.get("rows") or []
    return json.dumps(rows, ensure_ascii=False, default=str)


def _connection_lost(exc: BaseException) -> bool:
    """True when the database (or its proxy) went away mid-suite.

    Those runs are void, not flaky cases. A refused port or dropped connection
    after a stack restart must abort rather than grind through remaining cases.
    """
    name = type(exc).__name__
    if name in {
        "ConnectionRefusedError",
        "ConnectionResetError",
        "InterfaceError",
        "PostgresConnectionError",
        "OSError",
    }:
        if name == "OSError" and getattr(exc, "errno", None) not in {111, 10061, 104}:
            return False
        return True
    text = str(exc).casefold()
    needles = (
        "connection refused",
        "connection was closed",
        "connection does not exist",
        "server closed the connection",
        "could not connect to server",
        "the database system is shutting down",
    )
    return any(item in text for item in needles)


def _repeat(cases: tuple[Case, ...], runs: int) -> list[Case]:
    """Every case, `runs` times, grouped so one case's runs sit together in the log."""
    return [case for case in cases for _ in range(max(1, runs))]


def _report_repeated(report: list[dict[str, Any]], runs: int, as_json: bool) -> int:
    """Score repeated runs, separating real defects from sampling noise.

    A case that fails every run is a defect. A case that only sometimes passes is not a
    softer kind of pass -- one user in N still gets the wrong answer -- so flaky cases
    count against the suite too. They are reported apart because they need a different
    fix: a stable failure is a bug in the plan, a flaky one is the planner sampling.
    """
    outcomes: dict[str, list[bool]] = {}
    for item in report:
        outcomes.setdefault(str(item["id"]), []).append(item["status"] == "PASS")
    stable_pass = sorted(cid for cid, seen in outcomes.items() if all(seen))
    stable_fail = sorted(cid for cid, seen in outcomes.items() if not any(seen))
    flaky = sorted(cid for cid, seen in outcomes.items() if any(seen) and not all(seen))
    observations = sum(len(seen) for seen in outcomes.values())
    passed = sum(sum(seen) for seen in outcomes.values())

    print(f"\nRUNS {runs} over {len(outcomes)} cases = {observations} observations")
    print(f"  stable pass  {len(stable_pass):3d}")
    print(f"  flaky        {len(flaky):3d}   {','.join(flaky)}")
    print(f"  stable fail  {len(stable_fail):3d}   {','.join(stable_fail)}")
    print(f"  observation pass rate {passed}/{observations} = {passed / observations:.1%}")
    for cid in flaky:
        print(f"    {cid}: {''.join('P' if ok else 'F' for ok in outcomes[cid])}")
    print(f"\nSUMMARY total={len(outcomes)} passed={len(stable_pass)} failed={len(flaky) + len(stable_fail)}")
    print("FAILED_CASES " + ",".join(sorted(flaky + stable_fail)))
    if as_json:
        print(json.dumps(
            {
                "runs": runs,
                "stable_pass": stable_pass,
                "flaky": flaky,
                "stable_fail": stable_fail,
                "observations": observations,
                "passed": passed,
                "cases": report,
            },
            ensure_ascii=False,
            indent=2,
        ))
    return 1 if (flaky or stable_fail) else 0


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable report")
    parser.add_argument("--expected-only", action="store_true", help="Print database-derived expected answers without running NIA")
    parser.add_argument("--ids", help="Comma-separated case IDs for a focused run")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "Repeat every case N times and classify each as stable pass, flaky, or stable "
            "fail. The planner samples, so a single run cannot tell a real defect from "
            "noise: on a 10-case probe, 4 cases flipped across three identical runs."
        ),
    )
    args = parser.parse_args()
    selected_ids = {item.strip().upper() for item in (args.ids or "").split(",") if item.strip()}
    cases = tuple(case for case in CASES if not selected_ids or case.id in selected_ids)
    oracle = await load_oracle()
    for file_id, labels in oracle.labels.items():
        for field_name, label in labels.items():
            _LABELS[(file_id, field_name)] = label
    if args.expected_only:
        for case in cases:
            print(f"{case.id}\t{case.file_id}\t{case.question}\t{expected_summary(case, oracle.expected(case))}")
        return 0
    await init_pool()
    service = AssistantTurnService(
        AssistantDataGateway(get_pool()),
        store=InMemoryMemoryStore(),
        reasoner=get_reasoning_provider(),
    )
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91, 93, 94),
        can_use_private_files=True,
    )
    failures = 0
    report: list[dict[str, Any]] = []
    run_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"RUN start={run_started} cases={len(cases)} runs={args.runs} sequential=yes")
    try:
        for index, case in enumerate(_repeat(cases, args.runs), start=1):
            expected = oracle.expected(case)
            try:
                result = await service.answer(
                    scope,
                    case.question,
                    selected_file_id=case.file_id,
                )
            except Exception as exc:
                if _connection_lost(exc):
                    print(
                        f"VOID run: database connection lost at {case.id} "
                        f"({type(exc).__name__}: {exc}). "
                        "This run is not data — do not fold it into pass rates."
                    )
                    raise SystemExit(2) from exc
                failures += 1
                issue = f"runner_error {type(exc).__name__}: {exc}"
                report.append(
                    {
                        "id": case.id,
                        "kind": case.kind,
                        "dataset": case.file_id,
                        "question": case.question,
                        "expected": expected_summary(case, expected),
                        "actual": "",
                        "status": "FAIL",
                        "issues": [issue],
                        "planner": "error",
                        "ops": [],
                    }
                )
                print(f"[{index:02d}/{len(cases) * args.runs}] {case.id} FAIL: {case.question}")
                print(f"  issue:    {issue}")
                continue
            issues = compare(case, expected, result)
            failures += bool(issues)
            report.append(
                {
                    "id": case.id,
                    "kind": case.kind,
                    "dataset": case.file_id,
                    "question": case.question,
                    "expected": expected_summary(case, expected),
                    "actual": result.answer,
                    "status": "PASS" if not issues else "FAIL",
                    "issues": issues,
                    "planner": result.planner_type,
                    "ops": result.plan.op_names() if result.plan else [],
                    "latency": dict(result.latency or {}),
                    "review_status": result.review_status,
                    "final_response": (result.latency or {}).get("final_response"),
                }
            )
            print(f"[{index:02d}/{len(cases) * args.runs}] {case.id} {'PASS' if not issues else 'FAIL'}: {case.question}")
            if issues:
                print(f"  planner:  {result.planner_type}; ops={result.plan.op_names() if result.plan else []}")
                if result.plan:
                    planned_filters = [
                        item.model_dump(mode="json")
                        for step in result.plan.steps
                        for item in step.where
                    ]
                    print(f"  filters:  {planned_filters}")
                print(f"  expected: {expected_summary(case, expected)}")
                print(f"  actual:   {result.answer[:500]}")
                for issue in issues:
                    print(f"  issue:    {issue}")
    finally:
        await close_pool()
    if args.runs > 1:
        return _report_repeated(report, args.runs, args.json)
    print(f"\nSUMMARY total={len(cases)} passed={len(cases) - failures} failed={failures}")
    print("FAILED_CASES " + ",".join(item["id"] for item in report if item["status"] == "FAIL"))
    if args.json:
        print(json.dumps({"total": len(cases), "passed": len(cases) - failures, "failed": failures, "cases": report}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
