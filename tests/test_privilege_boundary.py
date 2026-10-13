import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DB_TESTS") != "1",
    reason="Set RUN_DB_TESTS=1 against an isolated assistant_runtime database",
)


@pytest.mark.asyncio
async def test_runtime_cannot_select_forbidden_tables():
    from app.db.privilege_selftest import assert_runtime_isolation

    await assert_runtime_isolation()
