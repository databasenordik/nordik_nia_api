"""Read export snapshots with the release decision ahead of row materialization."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from app.security.row_access import access_is_true


def released_csv_rows(path: Path, file_id: int) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        access_index = headers.index("ACCESS") if "ACCESS" in headers else None
        if file_id == 94 and access_index is None:
            raise ValueError("Potential export is missing its mandatory ACCESS column")
        if file_id == 94 and headers.count("ACCESS") != 1:
            raise ValueError("Potential export has ambiguous duplicate ACCESS columns")
        for cells in reader:
            if file_id == 94 and (
                access_index >= len(cells) or not access_is_true(cells[access_index])
            ):
                continue
            if len(cells) != len(headers):
                raise ValueError("Export row width differs from its header")
            yield dict(zip(headers, cells, strict=True))
