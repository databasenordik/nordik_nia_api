"""Where this checkout keeps the backend, and what to do when a file is not in it.

The same tests run in two repositories. The standalone monorepo keeps the API under
``backend/`` beside ``frontend/``, ``infra/`` and ``voice_cli.py``; the deployed API
repository holds ``app/`` at its root and nothing else. Tests that read backend *source*
should not care which one they are in, and tests that read a file only the monorepo has
should say so by skipping rather than failing: a missing docker-compose file in the API
repository is not a defect in the API.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# The directory that holds the ``app`` package, whichever layout this is.
BACKEND_ROOT = next(
    (path for path in (REPO_ROOT / "backend", REPO_ROOT) if (path / "app").is_dir()),
    REPO_ROOT,
)


# True in the standalone monorepo, where the backend is one directory among several.
IS_MONOREPO = BACKEND_ROOT != REPO_ROOT


def monorepo_file(*parts: str) -> Path:
    """A path outside the backend, skipping the test when this checkout lacks it.

    The existence check alone is not enough: a case-insensitive filesystem matches the API
    repository's own ReadME.md for README.md, and the test would then assert the monorepo's
    wording against a different document.
    """
    path = REPO_ROOT.joinpath(*parts)
    if not IS_MONOREPO or not path.exists():
        pytest.skip(f"{'/'.join(parts)} belongs to the standalone repository, not this one")
    return path
