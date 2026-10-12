"""Audit the four export sheets without materializing unreleased Potential rows."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from app.normalization.date_parser import parse_date
from app.normalization.snapshot import released_csv_rows

FILES = {
    49: "Shingwauk_And_Wawanosh_Students_Master_List_v5_latest.csv",
    91: "Confirmed-_Shingwauk_Wawanosh_v5_latest.csv",
    93: "Additional_Deaths_v2_latest.csv",
    94: "Potential_v2_latest.csv",
}


def audit(root: Path) -> dict:
    result = {}
    for file_id, filename in FILES.items():
        path = root / filename
        with path.open(encoding="utf-8-sig", newline="") as stream:
            headers = next(csv.reader(stream))
        rows = list(released_csv_rows(path, file_id))
        dates = [
            key
            for key in headers
            if not key.startswith("__")
            and key != "Date comments"
            and ("DATE" in key.upper() or key in ("Admitted", "Discharged"))
        ]
        names = (
            ["First Name", "Middle Name", "Last Name"]
            if "First Name" in headers
            else ["FIRST NAME", "MIDDLE NAME", "LAST NAME"]
        )
        bad_dates = []
        flags = Counter()
        shapes = Counter()
        for row in rows:
            for key in dates:
                raw = row.get(key, "").strip()
                if not raw or re.fullmatch(r"0+(?:[-/.]0+)*", raw):
                    shapes["unknown"] += 1
                    continue
                parts = parse_date(raw)
                flags.update(parts.flags)
                shapes[parts.precision or "unknown"] += 1
                if not re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}(?:\s*[|&:]\s*\d{4}-\d{2}-\d{2})*|\d{4}-\d{2}-\d{2}-\d{2}",
                    raw,
                ):
                    bad_dates.append(
                        {
                            "source_row_id": row.get("__id"),
                            "column": key,
                            "value": raw,
                            "parsed": parts.iso,
                            "flags": parts.flags,
                        }
                    )
        result[str(file_id)] = {
            "file": filename,
            "released_rows": len(rows),
            "headers": headers,
            "missing_name_columns": [key for key in names if key not in headers],
            "rows_without_split_names": sum(
                not any(row.get(key, "").strip() for key in names) for row in rows
            ),
            "date_shapes": dict(shapes),
            "date_flags": dict(flags),
            "nonstandard_dates": bad_dates,
            "name_legacy_values": {
                key: sum(bool(row.get(key, "").strip()) for row in rows)
                for key in headers
                if key
                in (
                    "First Names",
                    "Middle Names",
                    "Last Names",
                    "STUDENT NAME",
                    "INDIAN NAME",
                    "NAME PLACE HOLDER",
                )
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, default=Path("database_review_outputs"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/standardized-snapshot-audit.json")
    )
    args = parser.parse_args()
    report = audit(args.exports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for file_id, info in report.items():
        print(
            file_id,
            {
                key: value
                for key, value in info.items()
                if key not in ("headers", "nonstandard_dates")
            },
            "nonstandard_dates",
            len(info["nonstandard_dates"]),
        )


if __name__ == "__main__":
    main()
