"""Write parsed name parts into canonical.name_parts for every authorized list.

Reads the as-recorded name cells, runs ``name_parser`` over them, and stores the result
under ``row_data_normalized.canonical.name_parts``. Originals are never touched -- this
only adds a sibling object -- so every displayed name and citation still shows exactly
what the source recorded.

Idempotent: re-running recomputes from the source cells and replaces the block. The
Master List is regenerated from raw source cells and parser rules by default; stored
name overrides are not used for it unless explicitly requested. Other lists retain
their existing override behavior.

Usage (from the backend container)::

    python -m app.normalization.backfill_names            # all four lists
    python -m app.normalization.backfill_names --file-id 49
    python -m app.normalization.backfill_names --dry-run  # report coverage only
    python -m app.normalization.backfill_names --use-master-name-overrides  # opt-in
    python -m app.normalization.backfill_names --file-id 49 --only-name-issues"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Any

import asyncpg

from app.config import get_settings
from app.normalization.date_parser import parse_date
from app.normalization.name_parser import (
    NameParts,
    master_parts_match_clean_source,
    parse_full_name,
    parse_master,
    place_from_residue,
)

# Lists whose given/middle/surname already sit in their own source cells. Everything else
# packs the whole name into one cell.
SPLIT_NAME_FILES = frozenset({49})
DEFAULT_FILES = (49, 91, 93, 94)

# canonical.name_parts key -> the NameParts attribute it stores.
SCALAR_KEYS = {"first": "first", "last": "last", "indigenous": "indigenous"}
JOINED_KEYS = {
    "middle": "middle",
    "extra_first": "extra_first",
    "extra_last": "extra_last",
    "first_spelling": "first_spelling",
    "last_spelling": "last_spelling",
    "first_meaning": "first_meaning",
    "last_meaning": "last_meaning",
    "indigenous_spelling": "indigenous_spelling",
    "indigenous_meaning": "indigenous_meaning",
    "alias": "alias",
    "residue": "residue",
    "flags": "flags",
}
# assistant.name_overrides column -> canonical.name_parts key.
OVERRIDE_KEYS = {
    "first_name": "first",
    "middle_names": "middle",
    "last_name": "last",
    "other_first_names": "extra_first",
    "other_last_names": "extra_last",
    "first_name_spellings": "first_spelling",
    "last_name_spellings": "last_spelling",
    "first_name_meaning": "first_meaning",
    "last_name_meaning": "last_meaning",
    "indigenous_name": "indigenous",
    "indigenous_name_spellings": "indigenous_spelling",
    "indigenous_name_meaning": "indigenous_meaning",
    "name_alias": "alias",
    "name_residue": "residue",
    "name_flags": "flags",
}


def _at(blob: Any, path: str) -> Any:
    node: Any = blob
    for step in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(step)
    return node


def _source_cell(
    row: dict[str, Any],
    semantic_field: str,
    sources: dict[str, list[str]],
    *,
    raw_only: bool = False,
) -> str:
    """Read the cell exactly as a person typed it.

    The canonical values are already cleaned by ingest -- canonical.first_name holds
    "Willie Pashegezhik" where the source cell reads "Willie Pashegezhik (Cloud Running in
    a Line)". Parsing the cleaned value would discard the very parentheticals this module
    exists to interpret, so the raw source column wins and the canonical path is only a
    fallback for fields that have no raw key.
    """
    blob = row.get("row_data_normalized") or {}
    for path in sources.get(semantic_field, []):
        if raw_only and not path.startswith("fields."):
            continue
        text = _as_text(_at(blob, path))
        if text:
            return text
    return ""


def _as_text(node: Any) -> str:
    """Render a canonical node as the value a person recorded.

    A registry path may point at a plain string or at a normalized field object shaped
    ``{"raw": ..., "normalized": ..., "tokens": [...]}``. Falling back to str() on the
    object leaks Python's dict repr into the data -- the same defect
    docs/pipeline-fixes.md records as its first root cause -- so the object's recorded
    value is read explicitly, preferring the as-written form over the folded one.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        for key in ("raw", "display", "value", "normalized"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(node, list):
        return "; ".join(_as_text(item) for item in node if _as_text(item))
    return str(node).strip()


def parts_for_row(row: dict[str, Any], file_id: int, sources: dict[str, list[str]]) -> NameParts:
    if file_id in SPLIT_NAME_FILES:
        return parse_master(
            # Master regeneration is intentionally raw-only. Canonical values are parser
            # outputs and must never become inputs to a later normalization run.
            first=_source_cell(row, "first_name_source", sources, raw_only=True),
            middle=_source_cell(row, "middle_names_source", sources, raw_only=True),
            last=_source_cell(row, "last_name_source", sources, raw_only=True),
            indigenous=_source_cell(row, "indigenous_name", sources, raw_only=True),
            # The standardized Master list keeps name variants, Indigenous names and
            # their meanings in a comments column instead of inside the name cells.
            comments=_source_cell(row, "name_comments", sources, raw_only=True),
        )
    return parse_full_name(
        full=_source_cell(row, "student_name", sources),
        indigenous=_source_cell(row, "indigenous_name", sources),
    )


def block_for(parts: NameParts, override: dict[str, Any] | None) -> dict[str, str]:
    """Build the canonical.name_parts object, letting curation win over the parser."""
    block: dict[str, str] = {}
    for key, attribute in SCALAR_KEYS.items():
        value = getattr(parts, attribute)
        if value:
            block[key] = value
    for key, attribute in JOINED_KEYS.items():
        values = getattr(parts, attribute)
        if values:
            block[key] = "; ".join(values)
    block["source"] = "parsed"
    if override:
        applied = False
        for column, key in OVERRIDE_KEYS.items():
            value = override.get(column)
            if value is not None and str(value).strip():
                block[key] = str(value).strip()
                applied = True
        if applied:
            block["source"] = "curated"
    return block


def fold_label(text: str) -> str:
    """The folded form ingest stores in canonical_community, which is what filters match on."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").casefold()).split())


def apply_community(
    canonical: dict[str, Any],
    recorded: str,
    parts: NameParts,
    *,
    derive: bool,
) -> str | None:
    """Keep canonical.community in step with the row; return the folded column value to write.

    A community the list records in its own column always wins. Only when that cell is blank
    is one read from the name cell, and it is marked as derived so a later run -- with a
    better parser, or after the column is filled -- can tell it from a recorded value.

    Returns None to leave canonical_community as ingest wrote it, "" to clear a value an
    earlier run derived, or the folded place when one is derived now.
    """
    if recorded:
        if canonical.get("community_source") == "name_cell":
            canonical.pop("community_source", None)
            canonical["community"] = recorded
            return fold_label(recorded)
        if not canonical.get("community"):
            # Display reads the canonical JSON before the folded column; without this the
            # list's own values read back as "sault ste marie".
            canonical["community"] = recorded
        return None
    place = place_from_residue(parts) if derive else ""
    if place:
        canonical["community"] = place
        canonical["community_source"] = "name_cell"
        return fold_label(place)
    if canonical.get("community_source") == "name_cell":
        canonical.pop("community", None)
        canonical.pop("community_source", None)
        return ""
    return None


def date_block(row: dict[str, Any], date_fields: list[str], sources: dict[str, list[str]]) -> dict:
    """Parse every date cell on the row into its recorded components.

    Only the parts the source actually recorded are stored. A cell that says "1874" gets a
    year and nothing else, rather than being padded to 1 January as the production SQL
    parser does.
    """
    block: dict[str, dict[str, Any]] = {}
    for semantic_field in date_fields:
        raw = _source_cell(row, semantic_field, sources)
        if not raw:
            continue
        parts = parse_date(raw)
        entry: dict[str, Any] = {}
        if parts.year is not None:
            entry["year"] = parts.year
        if parts.month is not None:
            entry["month"] = parts.month
        if parts.day is not None:
            entry["day"] = parts.day
        if parts.iso:
            entry["iso"] = parts.iso
        if parts.precision:
            entry["precision"] = parts.precision
        if parts.extra:
            entry["extra"] = "; ".join(parts.extra)
        if parts.flags:
            entry["flags"] = "; ".join(parts.flags)
        if entry:
            block[semantic_field] = entry
    return block


async def run(
    file_ids: tuple[int, ...],
    *,
    dry_run: bool,
    use_master_name_overrides: bool = False,
    only_name_issues: bool = False,
) -> None:
    """Regenerate normalization generically and update only rows whose payload changes.

    There is deliberately no reference-row list, expected row count, or name-specific scope.
    Every row is parsed from its raw source cells.  Idempotence is enforced by comparing the
    newly computed payload with the stored payload and skipping the UPDATE when they are equal.
    """
    settings = get_settings()
    conn = await asyncpg.connect(settings.assistant_migrator_database_url)
    try:
        for file_id in file_ids:
            registry = await conn.fetch(
                "SELECT semantic_field, canonical_json_path, raw_json_keys, semantic_type "
                "FROM assistant.field_registry WHERE file_id = $1",
                file_id,
            )
            # Only the original date columns; the components this backfill writes are
            # themselves registered and must not be re-parsed.
            date_fields = [
                r["semantic_field"]
                for r in registry
                if r["semantic_type"] == "date"
                and not r["semantic_field"].endswith(
                    ("_year", "_month", "_day", "_iso", "_precision", "_extra", "_flags")
                )
            ]
            community_path = next(
                (r["canonical_json_path"] for r in registry if r["semantic_field"] == "community"),
                None,
            )
            # Raw source columns first, canonical path last. The registry already records
            # which spreadsheet columns a field came from, so nothing is hardcoded here.
            sources: dict[str, list[str]] = {}
            for record in registry:
                paths = [f"fields.{key}" for key in (record["raw_json_keys"] or [])]
                if record["canonical_json_path"]:
                    paths.append(record["canonical_json_path"])
                sources[record["semantic_field"]] = paths

            # Migration 043 repoints the Master list's first/middle/last at the parsed
            # values, so the as-recorded cells are read under their own aliases. The
            # spreadsheet columns behind them stay whatever the registry says they are:
            # naming them here would strand the backfill the next time a column is
            # renamed, which is what happened when the Master list was standardized.
            for semantic_field in ("first_name", "middle_names", "last_name"):
                sources.setdefault(f"{semantic_field}_source", sources.get(semantic_field, []))

            rows = await conn.fetch(
                """
                SELECT n.id, n.source_row_id, n.row_data_normalized
                FROM public.file_data_normalized n
                JOIN public.file f ON f.id = n.file_id AND f.version = n.version
                WHERE n.file_id = $1
                  AND n.status = 'ready'
                  AND COALESCE(f.is_delete, false) = false
                ORDER BY n.source_row_id, n.id
                """,
                file_id,
            )
            overrides = {
                r["source_row_id"]: dict(r)
                for r in await conn.fetch(
                    "SELECT * FROM assistant.name_overrides WHERE file_id = $1", file_id
                )
            }

            updates: list[tuple[int, str, str | None]] = []
            named = flagged = curated = derived_communities = 0
            dated = imprecise = multi = 0
            skipped_clean = 0
            for record in rows:
                row = dict(record)
                payload = row.get("row_data_normalized")
                if isinstance(payload, str):
                    payload = json.loads(payload)
                row["row_data_normalized"] = payload or {}


                parts = parts_for_row(row, file_id, sources)

                # Optional generic migration mode for name fixes only.  It does not know
                # how many rows should change.  For the split-name Master list it simply
                # compares parsed semantics with the raw source fields and skips rows that
                # are already clean.  This prevents a first-time backfill from adding a
                # name_parts block to every ordinary single-name row.
                if only_name_issues and file_id in SPLIT_NAME_FILES:
                    raw_first = _source_cell(row, "first_name_source", sources, raw_only=True)
                    raw_middle = _source_cell(row, "middle_names_source", sources, raw_only=True)
                    raw_last = _source_cell(row, "last_name_source", sources, raw_only=True)
                    raw_indigenous = _source_cell(row, "indigenous_name", sources, raw_only=True)
                    if master_parts_match_clean_source(
                        parts,
                        first=raw_first,
                        middle=raw_middle,
                        last=raw_last,
                        indigenous=raw_indigenous,
                    ):
                        skipped_clean += 1
                        continue

                # Master rows are parser-only by default so a successful regeneration
                # cannot be masked by historical manual overrides. Overrides remain the
                # default for the other lists and can be explicitly re-enabled for Master.
                override = overrides.get(row["source_row_id"])
                if file_id in SPLIT_NAME_FILES and not use_master_name_overrides:
                    override = None
                block = block_for(parts, override)
                if block.get("first") or block.get("last"):
                    named += 1
                if block.get("flags"):
                    flagged += 1
                if block.get("source") == "curated":
                    curated += 1
                canonical = dict(row["row_data_normalized"].get("canonical") or {})
                canonical["name_parts"] = block
                column_update: str | None = None
                if community_path == "canonical.community" and not only_name_issues:
                    # A list whose names sit in their own cells has no whole row pasted into
                    # a name cell, so nothing there is read as a community.
                    column_update = apply_community(
                        canonical,
                        _source_cell(row, "community", sources, raw_only=True),
                        parts,
                        derive=file_id not in SPLIT_NAME_FILES,
                    )
                    if canonical.get("community_source") == "name_cell":
                        derived_communities += 1
                if not only_name_issues:
                    dates = date_block(row, date_fields, sources)
                    if dates:
                        canonical["date_parts"] = dates
                        dated += sum(1 for entry in dates.values() if entry.get("year"))
                        imprecise += sum(
                            1 for entry in dates.values() if entry.get("precision") != "day"
                        )
                        multi += sum(1 for entry in dates.values() if entry.get("extra"))
                merged = dict(row["row_data_normalized"])
                merged["canonical"] = canonical
                # Every variant feeds the search token list, so a query for an alias or an
                # alternate spelling reaches the record.
                #
                # Rebuilt, never merged. Merging made the index append-only: a token
                # written by an earlier, buggier run could never be removed, so fixing the
                # parser did not fix the data. It also inherits whatever the ingest folded
                # into canonical_name, which on the Potential list is the entire
                # contaminated cell -- that is how community words ended up matching people
                # by name. Deriving the index from the parsed parts alone keeps it
                # reproducible and drops both problems.
                # Assigned unconditionally, including when empty. A row whose name cell
                # holds no name -- a placeholder such as "TO BE REVIEWED" -- must not stay
                # findable by whatever an earlier run happened to leave behind.
                merged["names"] = sorted(
                    {token.casefold() for token in parts.searchable() if token}
                )
                # Generic/idempotent write gate: do not touch a row just because it was
                # inspected.  Only persist a row when normalization actually changes the
                # stored JSON payload.  This is what keeps already-clean single-word names
                # untouched without any reference list or expected row count.
                if merged != row["row_data_normalized"]:
                    updates.append(
                        (row["id"], json.dumps(merged, ensure_ascii=False), column_update)
                    )

            total = len(rows)
            print(
                f"file {file_id}: {total} rows | named {named} ({_pct(named, total)}) | "
                f"flagged {flagged} | curated {curated} | "
                f"community from name cell {derived_communities}"
            )
            if only_name_issues and file_id in SPLIT_NAME_FILES:
                print(f"  generic name-fix mode: skipped {skipped_clean} already-clean rows")
            else:
                print(
                    f"  dates: {dated} parsed across {len(date_fields)} date columns | "
                    f"{imprecise} below day precision | {multi} cells holding more than one date"
                )
            if dry_run:
                continue
            # Apply the changed set atomically. Rows whose computed payload is identical to
            # the stored payload never receive an UPDATE.
            async with conn.transaction():
                # The folded column is what filters match on, so a derived community has to
                # land there too; None leaves ingest's value, "" clears an earlier derivation.
                await conn.executemany(
                    "UPDATE public.file_data_normalized SET row_data_normalized = $2::jsonb, "
                    "canonical_community = CASE WHEN $3::text IS NULL THEN canonical_community "
                    "ELSE NULLIF($3::text, '') END "
                    "WHERE id = $1",
                    updates,
                )
            print(f"  wrote {len(updates)} rows")
        if not dry_run:
            try:
                # Fill rates tell the planner which field holds a value; a community just
                # derived changes that for its list.
                await conn.execute("SELECT assistant.refresh_fill_rates()")
            except asyncpg.PostgresError as exc:
                print(f"fill rates not refreshed: {exc}")
    finally:
        await conn.close()


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.0f}%" if whole else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", type=int, action="append", dest="file_ids")
    parser.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    parser.add_argument(
        "--use-master-name-overrides",
        action="store_true",
        help=(
            "opt in to assistant.name_overrides for Master List rows; by default Master "
            "is regenerated from raw source cells and parser rules only"
        ),
    )
    parser.add_argument(
        "--only-name-issues",
        action="store_true",
        help=(
            "generic name-fix migration: for split-name Master rows, skip source names "
            "whose parsed semantics already equal the clean raw first/middle/last/indigenous fields; "
            "date_parts are not changed in this mode"
        ),
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            tuple(args.file_ids or DEFAULT_FILES),
            dry_run=args.dry_run,
            use_master_name_overrides=args.use_master_name_overrides,
            only_name_issues=args.only_name_issues,
        )
    )


if __name__ == "__main__":
    main()
