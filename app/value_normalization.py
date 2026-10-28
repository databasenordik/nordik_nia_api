from __future__ import annotations

import json
from typing import Any


def json_object(value: Any) -> dict[str, Any]:
    """Return a decoded JSON object from asyncpg or in-memory gateway values."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def semantic_scalar(value: Any) -> Any:
    """Collapse normalized typed values into stable user-facing scalars."""
    if not isinstance(value, dict):
        return value
    for key in ("iso", "raw", "display", "value", "year"):
        candidate = value.get(key)
        if candidate not in (None, ""):
            return candidate
    return None
