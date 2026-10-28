from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.retrieval.types import EvidenceItem, EvidencePacket, RetrievalHit
from app.value_normalization import json_object, semantic_scalar

MAX_NARRATIVE_CHARS = 240


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_evidence_packet(
    question: str,
    hits: list[RetrievalHit],
    *,
    facts: list[str] | None = None,
    requested_fields: list[str] | None = None,
    raw_by_id: dict[tuple[int, int], dict[str, Any]] | None = None,
    include_raw: bool = False,
    catalog: Any | None = None,
) -> EvidencePacket:
    settings = get_settings()
    max_rows = settings.max_retrieval_results
    max_tokens = settings.max_evidence_tokens
    items: list[EvidenceItem] = []
    used = estimate_tokens(question)
    truncated = False
    for hit in hits[:max_rows]:
        fields = requested_fields or _record_fields(hit.file_id, catalog)
        compact = _compact_fields(hit.record, fields, catalog)
        raw = {}
        if include_raw and raw_by_id:
            raw = raw_by_id.get(hit.identity(), {})
        item = EvidenceItem(
            source_id=hit.source_id(),
            file_id=hit.file_id,
            version=hit.version,
            source_row_id=hit.source_row_id,
            fields=compact,
            raw_fields=raw,
        )
        cost = estimate_tokens(str(compact) + str(raw))
        if used + cost > max_tokens:
            truncated = True
            break
        items.append(item)
        used += cost
    return EvidencePacket(
        question=question,
        facts=tuple(facts or ()),
        items=tuple(items),
        token_estimate=used,
        truncated=truncated,
    )


# Columns this pipeline derived from the record rather than read out of it.
_DERIVED_PREFIXES = ("canonical.name_parts.", "canonical.date_parts.")

# Used only when there is no catalog to ask. With one, the record's own columns are used.
_FALLBACK_FIELDS = ["student_name", "community", "cause_of_death", "notes"]


def _record_fields(file_id: int, catalog: Any | None) -> list[str]:
    """The record as it was recorded: every original column of one list.

    A name lookup is found through the derived columns -- first_name, last_name, the
    spellings and the search index -- but those are how the row was located, not what it
    says. Handing them back describes the person in the pipeline's vocabulary and repeats
    what the source cell already contains, so they are left out and the original columns
    go in their place.

    The previous default was four fixed fields, which meant a question about a birth date
    or a parent's name reached the model with neither in the row.
    """
    if catalog is None:
        return list(_FALLBACK_FIELDS)
    try:
        specs = catalog.fields_for(file_id)
    except Exception:
        return list(_FALLBACK_FIELDS)
    fields = [
        spec.semantic_field
        for spec in specs
        if getattr(spec, "evidence_allowed", True)
        and not (getattr(spec, "canonical_json_path", "") or "").startswith(_DERIVED_PREFIXES)
    ]
    return fields or list(_FALLBACK_FIELDS)


def _compact_fields(
    record: dict[str, Any],
    requested: list[str],
    catalog: Any | None = None,
) -> dict[str, Any]:
    """Project the requested semantic fields of one record.

    With a catalog, every field in the registry can be projected. Without one, a small
    built-in map covers the common fields. The catalog path matters for record
    summaries: a hardcoded map silently dropped family, admission, death, and burial
    values, so a summary of a person read as "the database has no records".
    """
    if catalog is not None:
        from app.execution.row_projection import semantic_value

        compact: dict[str, Any] = {}
        for field_name in requested:
            value = semantic_value(record, field_name, catalog)
            if value is None or value == "":
                continue
            if isinstance(value, str) and len(value) > MAX_NARRATIVE_CHARS:
                value = value[:MAX_NARRATIVE_CHARS].rstrip() + "…"
            compact[field_name] = value
        return compact
    row = json_object(record.get("row_data_normalized"))
    canonical = row.get("canonical") if isinstance(row, dict) else {}
    if not isinstance(canonical, dict):
        canonical = {}
    chat = row.get("chat") if isinstance(row, dict) else {}
    bundle = chat.get("narrative_bundle") if isinstance(chat, dict) else {}
    mapping = {
        "student_name": record.get("canonical_name")
        or canonical.get("display_name")
        or canonical.get("name"),
        "community": record.get("canonical_community") or canonical.get("community"),
        "school": record.get("canonical_school") or canonical.get("school"),
        "cause_of_death": canonical.get("cause_of_death"),
        "notes": (bundle.get("notes") if isinstance(bundle, dict) else None) or canonical.get("notes"),
        "deceased_status": canonical.get("deceased_status")
        if "deceased_status" in canonical
        else canonical.get("deceased"),
    }
    compact: dict[str, Any] = {}
    for field_name in requested:
        value = semantic_scalar(mapping.get(field_name))
        if value is None:
            continue
        if isinstance(value, str) and len(value) > MAX_NARRATIVE_CHARS:
            value = value[:MAX_NARRATIVE_CHARS].rstrip() + "…"
        compact[field_name] = value
    return compact


# A whole-list packet carries the complete list, so a long note is worth keeping where an
# evidence sample would trim it. Still bounded, so one pathological cell cannot dominate.
MAX_WHOLE_LIST_VALUE_CHARS = 2000


def build_whole_list_packet(
    question: str,
    records: list[dict[str, Any]],
    *,
    catalog: Any,
    fields: list[str],
    max_chars: int,
) -> EvidencePacket | None:
    """Package an entire list for the model to read.

    Unlike :func:`build_evidence_packet` this does not sample. ``max_retrieval_results``
    and ``max_evidence_tokens`` bound how much *evidence* accompanies a computed answer;
    here the rows are the answer's whole basis, so trimming them would silently turn "how
    many" into a guess over a subset.

    Returns ``None`` when the rendered payload would exceed ``max_chars``. The caller then
    falls back to the query path, which is why the threshold is a real safety net rather
    than an assumption about list size.
    """
    from app.execution.row_projection import semantic_value

    items: list[EvidenceItem] = []
    budget = len(question)
    for record in records:
        projected: dict[str, Any] = {}
        for field_name in fields:
            value = semantic_value(record, field_name, catalog)
            if value is None or value == "":
                continue
            if isinstance(value, str) and len(value) > MAX_WHOLE_LIST_VALUE_CHARS:
                value = value[:MAX_WHOLE_LIST_VALUE_CHARS].rstrip() + "…"
            projected[field_name] = value
            budget += len(field_name) + len(str(value)) + 2
        if budget > max_chars:
            return None
        file_id = int(record.get("file_id") or 0)
        version = int(record.get("version") or 0)
        source_row_id = int(record.get("source_row_id") or 0)
        items.append(
            EvidenceItem(
                source_id=f"file:{file_id}:v{version}:row:{source_row_id}",
                file_id=file_id,
                version=version,
                source_row_id=source_row_id,
                fields=projected,
            )
        )
    return EvidencePacket(
        question=question,
        items=tuple(items),
        token_estimate=estimate_tokens(" " * budget),
        truncated=False,
    )


# Beyond this many distinct values a frequency table stops being a useful fact and starts
# being a second copy of the list.
MAX_TABULATED_VALUES = 40


def whole_list_facts(packet: EvidencePacket, *, total_rows: int) -> list[str]:
    """Deterministic tallies over the rows, so the model never counts by hand.

    Counting 82 rows reliably is something SQL does perfectly and a language model does
    not: asked how many records carried a cause of death, the model answered 77 where the
    true figure is 72, and 3 where it was 6. The rows are already in memory, so every count
    it might need is computed here exactly and stated as fact.

    This leaves the model doing what it is actually good at -- reading a fused, misspelt,
    annotated cell and understanding what it says.
    """
    facts = [f"Total records in this list: {total_rows}."]
    present: dict[str, int] = {}
    values: dict[str, dict[str, int]] = {}
    for item in packet.items:
        for field_name, value in item.fields.items():
            text = str(value).strip()
            if not text:
                continue
            present[field_name] = present.get(field_name, 0) + 1
            values.setdefault(field_name, {})
            values[field_name][text] = values[field_name].get(text, 0) + 1

    for field_name in sorted(present):
        filled = present[field_name]
        facts.append(
            f"{field_name}: {filled} of {total_rows} records have a value, "
            f"{total_rows - filled} are empty."
        )
    for field_name in sorted(values):
        table = values[field_name]
        if len(table) > MAX_TABULATED_VALUES:
            continue
        ranked = sorted(table.items(), key=lambda pair: (-pair[1], pair[0]))
        rendered = "; ".join(f"{name} = {count}" for name, count in ranked)
        facts.append(f"{field_name} exact value counts ({len(table)} distinct): {rendered}")
    return facts
