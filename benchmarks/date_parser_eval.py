"""Score the date parser against every real date cell in the four lists.

There is no hand-curated gold set for dates, so the baseline is the parser already in
production: ``assistant_api._parse_date_value``. The gate is therefore two-sided.

* **Agreement** -- wherever the SQL parser produces a date and the cell was recorded to
  full day precision, the new parser must agree with it. A disagreement there is a
  regression.
* **Coverage** -- the new parser must not lose any cell the SQL parser could read, and is
  expected to recover precision the SQL parser discards: "June 1882" becomes 1882-01-01
  in SQL, which silently invents a day and moves the month to January.

Usage (from the backend container)::

    python -m benchmarks.date_parser_eval
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import asyncpg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.normalization.date_parser import parse_date  # noqa: E402


async def collect() -> list[tuple[int, str, str, str | None, str | None]]:
    conn = await asyncpg.connect(get_settings().assistant_migrator_database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT
                r.file_id,
                fr.semantic_field,
                assistant_api._field_display(
                    r.file_id, r.row_data_normalized, r.canonical_name,
                    r.canonical_community, r.canonical_school, fr.semantic_field
                ) AS raw,
                assistant_api._parse_date_value(
                    assistant_api._field_display(
                        r.file_id, r.row_data_normalized, r.canonical_name,
                        r.canonical_community, r.canonical_school, fr.semantic_field
                    )
                )::text AS sql_date,
                assistant_api._date_precision(
                    assistant_api._field_display(
                        r.file_id, r.row_data_normalized, r.canonical_name,
                        r.canonical_community, r.canonical_school, fr.semantic_field
                    )
                ) AS sql_precision
            FROM assistant_api.v_current_records r
            JOIN assistant.field_registry fr
              ON fr.file_id = r.file_id AND fr.semantic_type = 'date'
            """
        )
        return [
            (r["file_id"], r["semantic_field"], (r["raw"] or "").strip(),
             r["sql_date"], r["sql_precision"])
            for r in rows
            if (r["raw"] or "").strip()
        ]
    finally:
        await conn.close()


def evaluate(cells: list[tuple[int, str, str, str | None, str | None]]) -> int:
    agree = disagree = 0
    sql_only: list[tuple[str, str]] = []
    recovered = Counter()
    precision_counts: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    parsed = 0
    disagreements: list[tuple[str, str, str]] = []

    for _file_id, _field, raw, sql_date, sql_precision in cells:
        parts = parse_date(raw)
        if parts.year is not None:
            parsed += 1
            precision_counts[parts.precision] += 1
        for flag in parts.flags:
            flags[flag] += 1

        if sql_date is None:
            if parts.year is not None:
                recovered["cells SQL could not read"] += 1
            continue
        if parts.year is None:
            sql_only.append((raw, sql_date))
            continue
        # Compare only where the source really recorded a full date. Where it did not,
        # the SQL value contains invented components and is not a valid reference.
        if sql_precision == "day" and parts.precision == "day":
            if parts.iso == sql_date:
                agree += 1
            else:
                disagree += 1
                if len(disagreements) < 12:
                    disagreements.append((raw, sql_date, parts.iso))
        elif sql_precision in {"month", "year"}:
            # SQL pads these to the 1st of a month it may also have guessed.
            if parts.precision != "day":
                recovered[f"{sql_precision}-precision kept instead of padded"] += 1

    total = len(cells)
    print(f"date cells with a value: {total}\n")
    print("AGREEMENT WITH THE PRODUCTION SQL PARSER (full-date cells only)")
    checked = agree + disagree
    rate = agree / checked if checked else 1.0
    print(f"  {'OK ' if rate >= 0.99 else 'XX '}{agree}/{checked} = {rate:.2%}   (gate 99%)")

    print("\nCOVERAGE")
    print(f"  parsed by the new parser : {parsed}/{total} = {parsed / total:.1%}")
    print(f"  {'OK ' if not sql_only else 'XX '}cells SQL read but new parser lost: {len(sql_only)}   (gate 0)")
    for label, count in recovered.most_common():
        print(f"  recovered: {count} {label}")

    print("\nPRECISION ACTUALLY RECORDED")
    for name in ("day", "month", "year"):
        print(f"  {name:6} {precision_counts.get(name, 0)}")

    print("\nFLAGS")
    for name, count in flags.most_common():
        print(f"  {name:14} {count}")

    if disagreements:
        print("\nDISAGREEMENTS (first 12)")
        for raw, sql_value, mine in disagreements:
            print(f"  {raw[:44]!r:46} sql={sql_value} new={mine}")
    if sql_only:
        print("\nLOST (SQL read these, new parser did not)")
        for raw, sql_value in sql_only[:12]:
            print(f"  {raw[:44]!r:46} sql={sql_value}")

    ok = rate >= 0.99 and not sql_only
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    return evaluate(asyncio.run(collect()))


if __name__ == "__main__":
    raise SystemExit(main())
