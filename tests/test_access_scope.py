import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.security.access_scope import AccessScope


def test_scope_cannot_broaden():
    scope = AccessScope(
        principal_id="principal-researcher",
        allowed_file_ids=(49, 91),
        can_use_private_files=False,
    )
    narrowed = scope.narrow([49, 93, 94])
    assert narrowed.allowed_file_ids == (49,)
    assert not narrowed.allows_file(93, private=True)
    assert not narrowed.allows_file(94, private=True)


def test_scope_is_frozen():
    scope = AccessScope(principal_id="p", allowed_file_ids=(49,))
    try:
        scope.principal_id = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("AccessScope must be immutable")
