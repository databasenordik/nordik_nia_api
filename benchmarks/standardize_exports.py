"""Produce reviewable standardized copies. Inputs and denied rows are never rewritten."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from app.normalization.date_parser import parse_date
from app.normalization.name_parser import parse_full_name
from app.normalization.snapshot import released_csv_rows
from benchmarks.standardized_snapshot import FILES, audit

ALIASES = {
    "FIRST NAME": "First Name",
    "MIDDLE NAME": "Middle Name",
    "LAST NAME": "Last Name",
    "First Names": "First Name",
    "Middle Names": "Middle Name",
    "Last Names": "Last Name",
    "DATE OF DISCHARGE": "DISCHARGE DATE",
    "COMMUNITY/RESEVE": "RESERVE/COMMUNITY",
    "First Nation/Community": "First Nation Community",
}


def is_date(key: str) -> bool:
    return (
        not key.startswith("__")
        and ("DATE" in key.upper() or key in ("Admitted", "Discharged"))
        and key != "Date comments"
    )


def padded_iso(text: str) -> str:
    part = parse_date(text)
    if not part.iso:
        return "0000-00-00"
    return f"{part.year or 0:04d}-{part.month or 0:02d}-{part.day or 0:02d}"


def revise_row(source: dict[str, str], file_id: int) -> tuple[dict[str, str], list[dict]]:
    row: dict[str, str] = {}
    issues = []
    originals = {}
    for key, value in source.items():
        clean = ALIASES.get(key, " ".join(key.split()))
        if clean in row and row[clean].strip() and value.strip() and row[clean] != value:
            originals[key] = value
            issues.append(
                {"column": clean, "reason": "conflicting duplicate column", "source": value}
            )
        elif value.strip() or clean not in row:
            row[clean] = value

    if not any(row.get(key, "").strip() for key in ("First Name", "Middle Name", "Last Name")):
        full = row.get("STUDENT NAME", "")
        # A report status is provenance, not a translation or part of the person's name.
        cleaned = re.sub(r"\(REPORT COMPLETE\)", "", full, flags=re.I).strip()
        parts = parse_full_name(cleaned)
        row.update(
            {
                "First Name": parts.first,
                "Middle Name": " ".join(parts.middle),
                "Last Name": parts.last,
            }
        )
        originals["STUDENT NAME"] = full
        if not parts.first and not parts.last:
            issues.append(
                {"column": "First Name", "reason": "name requires curation", "source": full}
            )
    if row.get("STUDENT NAME"):
        originals["STUDENT NAME"] = row.pop("STUDENT NAME")
    row.pop("STUDENT NAME", None)
    row["Name comments"] = row.get("Name comments", "") or row.get("VARIATIONS OF NAME", "")
    for key in ("First Name", "Middle Name", "Last Name"):
        recorded = row.get(key, "")
        if re.search(r"\(NCTR SOURCE\)", recorded, re.I):
            originals[key] = recorded
            row[key] = re.sub(r"\s*\(NCTR SOURCE\)\s*", " ", recorded, flags=re.I).strip()
            row["Name comments"] = "; ".join(value for value in (row["Name comments"], "Source note: NCTR SOURCE") if value)
            issues.append({"column": key, "reason": "editorial source annotation moved to Name comments", "source": recorded})

    date_notes = {}
    for key, raw in list(row.items()):
        if not is_date(key):
            continue
        if ":" in raw or re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}", raw.strip()):
            # Curator-defined conventions are interpreted in code, not rewritten.
            continue
        if (
            not raw.strip()
            or re.fullmatch(r"0+(?:[-/.]0+)*", raw.strip())
            or raw.strip().casefold() in ("unknown", "not known", "not listed")
        ):
            revised = "0000-00-00"
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:\s*[|&]\s*\d{4}-\d{2}-\d{2})*", raw.strip()):
            pieces = re.split(r"([|&])", raw)
            revised = " ".join(
                piece if piece in ("|", "&") else padded_iso(piece.strip()) for piece in pieces
            )
        else:
            parts = parse_date(raw)
            if (
                parts.extra
                or "range" in parts.flags
                or not parts.iso
                or re.search(r"\d{4}-\d{2}-\d{2}-\d", raw)
            ):
                revised = "0000-00-00"
                issues.append(
                    {
                        "column": key,
                        "reason": "ambiguous legacy date; original preserved in Date comments",
                        "source": raw,
                    }
                )
            else:
                revised = padded_iso(raw)
        if revised != raw:
            date_notes[key] = raw
        if "impossible_day" in parse_date(raw).flags:
            issues.append(
                {"column": key, "reason": "invalid calendar day; day kept unknown", "source": raw}
            )
        row[key] = revised
    if date_notes:
        row["Date comments"] = json.dumps(date_notes, ensure_ascii=False)
    if originals:
        row["Source revision notes"] = json.dumps(originals, ensure_ascii=False)
    return row, issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports", type=Path, default=Path("database_review_outputs"))
    parser.add_argument("--output", type=Path, default=Path("database_review_outputs/standardized"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "potential_policy": "Only ACCESS=TRUE rows were materialized; withheld rows were not reviewed or revised.",
        "files": {},
    }
    for file_id, filename in FILES.items():
        revised = []
        issues = []
        for source in released_csv_rows(args.exports / filename, file_id):
            row, found = revise_row(source, file_id)
            revised.append(row)
            issues.extend({"source_row_id": source["__id"], **issue} for issue in found)
        headers = list(dict.fromkeys(key for row in revised for key in row))
        for key in (
            "First Name",
            "Middle Name",
            "Last Name",
            "Date comments",
            "Source revision notes",
        ):
            if key not in headers:
                headers.append(key)
        with (args.output / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, headers)
            writer.writeheader()
            writer.writerows(revised)
        manifest["files"][str(file_id)] = {"rows": len(revised), "issues": issues}
        print(file_id, "revised rows", len(revised), "curation issues", len(issues))
    (args.output / "revision_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "audit.json").write_text(
        json.dumps(audit(args.output), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
