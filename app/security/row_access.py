"""Release gate applied before a Potential record enters Python processing."""

from typing import Any

DATA_CACHE_KEYS = frozenset({"recorded_values", "recorded_count", "record_count"})


def released_validation_rules(file_id: int, rules: dict[str, Any]) -> dict[str, Any]:
    return {key: rules[key] for key in rules if file_id != 94 or key not in DATA_CACHE_KEYS}


def access_is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.strip().casefold() == "true")


def row_is_released(record: dict[str, Any]) -> bool:
    if int(record.get("file_id", record.get("__file_id", 0))) != 94:
        return True
    # Source permission wins; normalized payloads may be stale after a flag change.
    raw = record.get("raw")
    if isinstance(raw, dict):
        return access_is_true(raw.get("ACCESS"))
    return False
