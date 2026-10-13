"""Run the existing end-to-end matrix against the isolated standardized snapshot."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings
from benchmarks.database_question_matrix import main

if __name__ == "__main__":
    settings = get_settings()
    for key, value in (
        ("ASSISTANT_DATABASE_URL", settings.assistant_database_url),
        ("ASSISTANT_MIGRATOR_DATABASE_URL", settings.assistant_migrator_database_url),
    ):
        os.environ[key] = value.rsplit("/", 1)[0] + "/nia_standardized_20261011"
    os.environ["PLANNER_MODE"] = "ai"
    get_settings.cache_clear()
    raise SystemExit(asyncio.run(main()))
