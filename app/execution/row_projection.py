from __future__ import annotations

from typing import Any

from app.planning.catalog import FieldCatalog, FieldSpec
from app.value_normalization import json_object, semantic_scalar

_DIRECT_PATHS = {
    "canonical.name": "canonical_name",
    "canonical.display_name": "canonical_name",
    "canonical.community": "canonical_community",
    "canonical.school": "canonical_school",
}


def semantic_value(
    row: dict[str, Any],
    field: str,
    catalog: FieldCatalog,
) -> Any:
    """Read a semantic field for display.

    The canonical JSON is preferred over the denormalized `canonical_*` columns:
    those are lowercased and punctuation-stripped for matching, so reading them back
    turns "Albert PENANCE (Manitoba Vital Stats)" into
    "albert penance manitoba vital stats".
    """
    file_id = int(row.get("file_id") or 0)
    spec = catalog.resolve_field(file_id, field)
    normalized = json_object(row.get("row_data_normalized"))
    direct = _DIRECT_PATHS.get(spec.canonical_json_path or "") if spec is not None else None
    if direct:
        stored = _read_path(normalized, spec.canonical_json_path or "")
        scalar = semantic_scalar(stored) if stored is not None else None
        if scalar not in (None, ""):
            return scalar
        return row.get(direct)
    if spec is not None and spec.canonical_json_path:
        value = _read_path(normalized, spec.canonical_json_path)
        if value is not None:
            return semantic_scalar(value)
    canonical = normalized.get("canonical") if isinstance(normalized, dict) else None
    if isinstance(canonical, dict):
        if field.endswith("_date"):
            dates = canonical.get("dates")
            if isinstance(dates, dict):
                value = dates.get(field.removesuffix("_date")) or dates.get(field)
                if value is not None:
                    return semantic_scalar(value)
        value = canonical.get(field)
        if value is None and field == "deceased_status":
            value = canonical.get("deceased")
        if value is not None:
            return semantic_scalar(value)
    return None


def public_projected_row(
    row: dict[str, Any],
    fields: list[str],
    catalog: FieldCatalog,
) -> dict[str, Any]:
    return {
        "file_id": row.get("file_id"),
        "source_row_id": row.get("source_row_id"),
        "canonical_name": row.get("canonical_name"),
        "canonical_community": row.get("canonical_community"),
        "values": {field: semantic_value(row, field, catalog) for field in fields},
    }


def field_descriptors(
    fields: list[str],
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for field in fields:
        spec = _first_spec(field, file_ids, catalog)
        result.append(
            {
                "name": field,
                "label": spec.human_label if spec is not None else field.replace("_", " ").title(),
            }
        )
    return result


def display_value(value: Any) -> str:
    if value is None or value == "":
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list | tuple):
        return ", ".join(display_value(item) for item in value)
    return str(value)


def _first_spec(
    field: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> FieldSpec | None:
    for file_id in file_ids:
        spec = catalog.resolve_field(file_id, field)
        if spec is not None:
            return spec
    return None


def _read_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current
