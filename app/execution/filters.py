from __future__ import annotations

import re
from typing import Any

from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.plan_schema import FilterOperator, Predicate
from app.value_normalization import json_object


def extract_field_value(record: dict[str, Any], spec: FieldSpec) -> Any:
    """Read one semantic field from a record.

    The canonical JSON wins over the denormalized `canonical_*` columns, which are
    lowercased and punctuation-stripped for lookup. Reading them back turned every
    grouped label and listed name into its folded form.
    """
    path = spec.canonical_json_path or ""
    if path in {"canonical.name", "canonical.display_name"}:
        return _walk(record.get("row_data_normalized"), path) or record.get("canonical_name")
    if path == "canonical.community":
        return (
            _walk(record.get("row_data_normalized"), path)
            or record.get("canonical_community")
        )
    if path == "canonical.school":
        return _walk(record.get("row_data_normalized"), path) or record.get("canonical_school")
    if path == "canonical.deceased_status":
        row = json_object(record.get("row_data_normalized"))
        canonical = row.get("canonical") if isinstance(row, dict) else {}
        if isinstance(canonical, dict):
            if "deceased_status" in canonical:
                return canonical.get("deceased_status")
            if "deceased" in canonical:
                return canonical.get("deceased")
    if path == "chat.narrative_bundle.notes":
        notes = _walk(record.get("row_data_normalized"), path)
        if notes is not None:
            return notes
        row = json_object(record.get("row_data_normalized"))
        canonical = row.get("canonical") if isinstance(row, dict) else {}
        if isinstance(canonical, dict):
            return canonical.get("notes")
    return _walk(record.get("row_data_normalized"), path)


def _walk(row: Any, path: str) -> Any:
    current: Any = row or {}
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def predicate_matches(record: dict[str, Any], predicate: Predicate, spec: FieldSpec) -> bool:
    value = extract_field_value(record, spec)
    operator = predicate.operator
    if operator in _NEGATIVE_OPERATORS:
        return _negative_match(value, predicate)
    if operator is FilterOperator.IS_UNKNOWN:
        return value in (None, "", [])
    if operator is FilterOperator.IS_KNOWN:
        return value not in (None, "", [])
    if operator is FilterOperator.IS_TRUE:
        return _as_bool(value) is True
    if operator is FilterOperator.IS_FALSE:
        return _as_bool(value) is False
    if value is None:
        return False
    text = str(value)
    if operator is FilterOperator.EQUALS:
        return text.casefold() == str(predicate.value).casefold()
    if operator is FilterOperator.STARTS_WITH:
        return text.casefold().startswith(str(predicate.value).casefold())
    if operator is FilterOperator.CONTAINS:
        return str(predicate.value).casefold() in text.casefold()
    if operator is FilterOperator.IN:
        return text.casefold() in {item.casefold() for item in _terms(predicate.value)}
    if operator is FilterOperator.CONTAINS_ANY:
        return any(term.casefold() in text.casefold() for term in _terms(predicate.value) if term)
    if operator is FilterOperator.STARTS_WITH_ANY:
        return any(
            text.casefold().startswith(term.casefold()) for term in _terms(predicate.value) if term
        )
    year = _year(value)
    target = _year(predicate.value)
    if operator is FilterOperator.YEAR_EQUALS:
        return year is not None and year == target
    if operator is FilterOperator.BEFORE:
        return year is not None and target is not None and year < target
    if operator is FilterOperator.AFTER:
        return year is not None and target is not None and year > target
    if operator is FilterOperator.DATE_RANGE:
        bounds = predicate.value if isinstance(predicate.value, list | tuple) else []
        if len(bounds) < 2 or year is None:
            return False
        start, end = _year(bounds[0]), _year(bounds[1])
        return start is not None and end is not None and start <= year <= end
    if operator in {
        FilterOperator.GREATER_THAN,
        FilterOperator.LESS_THAN,
        FilterOperator.NUMBER_RANGE,
    }:
        number = _number(value)
        if number is None:
            return False
        if operator is FilterOperator.NUMBER_RANGE:
            bounds = predicate.value if isinstance(predicate.value, list | tuple) else []
            if len(bounds) < 2:
                return False
            start, end = _number(bounds[0]), _number(bounds[1])
            return start is not None and end is not None and start <= number <= end
        target_number = _number(predicate.value)
        if target_number is None:
            return False
        if operator is FilterOperator.GREATER_THAN:
            return number > target_number
        return number < target_number
    return False


_NEGATIVE_OPERATORS = frozenset(
    {
        FilterOperator.NOT_EQUALS,
        FilterOperator.NOT_CONTAINS,
        FilterOperator.NOT_CONTAINS_ANY,
        FilterOperator.NOT_IN,
    }
)


def _terms(value: Any) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)] if value is not None else []


def _negative_match(value: Any, predicate: Predicate) -> bool:
    """A record with no recorded value genuinely does not carry the excluded value."""
    text = "" if value in (None, "", []) else str(value).casefold()
    operator = predicate.operator
    if operator is FilterOperator.NOT_EQUALS:
        return text != str(predicate.value).casefold()
    if operator is FilterOperator.NOT_CONTAINS:
        return str(predicate.value).casefold() not in text
    if operator is FilterOperator.NOT_CONTAINS_ANY:
        return not any(term.casefold() in text for term in _terms(predicate.value) if term)
    return text not in {item.casefold() for item in _terms(predicate.value)}


def matches_predicate_tree(
    record: dict[str, Any],
    predicate: Predicate,
    catalog: FieldCatalog,
    file_id: int,
) -> bool:
    """Evaluate a leaf or a boolean group against one record. Unknown fields fail closed."""
    if predicate.is_group():
        results = (
            matches_predicate_tree(record, item, catalog, file_id) for item in predicate.items
        )
        if predicate.op == "or":
            return any(results)
        if predicate.op == "not":
            return not matches_predicate_tree(record, predicate.items[0], catalog, file_id)
        return all(results)
    spec = catalog.resolve_field(file_id, predicate.field)
    if spec is None:
        return False
    return predicate_matches(record, predicate, spec)


def apply_predicates(
    records: list[dict[str, Any]],
    predicates: list[Predicate],
    catalog: FieldCatalog,
    file_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        file_id = int(record["file_id"])
        if file_id not in file_ids:
            continue
        if all(
            matches_predicate_tree(record, predicate, catalog, file_id)
            for predicate in predicates
        ):
            matched.append(record)
    return matched


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "deceased"}:
        return True
    if text in {"false", "f", "0", "no"}:
        return False
    return None


def _year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    match = re.search(r"(1[6-9]\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None
