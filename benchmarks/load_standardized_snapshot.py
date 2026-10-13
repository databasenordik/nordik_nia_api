"""Prepare an isolated database from released snapshot data and revised exports."""

from __future__ import annotations

import json
from pathlib import Path

from app.normalization.snapshot import released_csv_rows
from benchmarks.standardized_snapshot import FILES


def literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main():
    root = Path.cwd()
    normalized = {}
    for line in (
        (root / "database_review_work/released_normalized.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        row = json.loads(line)
        normalized[(int(row["file_id"]), int(row["source_row_id"]))] = row
    statements = [
        "BEGIN;",
        "DO $$ BEGIN IF current_database() <> 'nia_standardized_20261011' THEN RAISE EXCEPTION 'snapshot database required'; END IF; END $$;",
        "DELETE FROM public.file_data_normalized;",
        "DELETE FROM public.file_data;",
        "GRANT SELECT ON public.file_data_normalized, public.file_data, public.file TO assistant_migrator;",
        "GRANT UPDATE ON public.file_data_normalized TO assistant_migrator;",
    ]
    for file_id, filename in FILES.items():
        rows = list(
            released_csv_rows(root / "database_review_outputs/standardized" / filename, file_id)
        )
        version = int(rows[0]["__version"])
        statements.append(
            f"UPDATE public.file SET version={version},row_count={len(rows)} WHERE id={file_id};"
        )
        for row in rows:
            source_id = int(row["__id"])
            existing = normalized.get((file_id, source_id))
            if existing is None:
                raise ValueError(f"No released normalized snapshot row for {file_id}/{source_id}")
            raw = {key: value for key, value in row.items() if not key.startswith("__")}
            payload = existing["row_data_normalized"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload["fields"] = {
                key: {"raw": value, "normalized": value.strip().casefold()}
                for key, value in raw.items()
            }
            payload["canonical"].pop("name_parts", None)
            payload["canonical"].pop("date_parts", None)
            statements.append(
                f"INSERT INTO public.file_data(id,file_id,version,row_data) VALUES ({source_id},{file_id},{version},{literal(json.dumps(raw))}::jsonb);"
            )
            statements.append(
                "INSERT INTO public.file_data_normalized(source_row_id,file_id,version,status,row_data_normalized,canonical_name,canonical_community,canonical_school,search_text) VALUES "
                f"({source_id},{file_id},{version},'ready',{literal(json.dumps(payload))}::jsonb,{literal(existing.get('canonical_name') or '')},{literal(existing.get('canonical_community') or '')},{literal(existing.get('canonical_school') or '')},'');"
            )
    statements.append("COMMIT;")
    (root / "database_review_work/load_standardized.sql").write_text(
        "\n".join(statements), encoding="utf-8"
    )
    print("Prepared snapshot SQL for", len(normalized), "released normalized rows")


if __name__ == "__main__":
    main()
