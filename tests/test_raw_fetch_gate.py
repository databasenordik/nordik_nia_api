"""Which source columns may be returned raw is the registry's decision, not the executor's.

Raw cells are the least filtered thing NIA can hand back, so the field registry carries
raw_fetch_allowed to say which fields may be read that way. It was declared, defaulted to
FALSE on every field of every list, and enforced nowhere: v_dataset_fields never selected
the column, so no runtime catalog saw it, and the executor asked for a hardcoded
["Notes", "Cause of Death", "Name"] regardless.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.execution.executor import _raw_fetch_keys  # noqa: E402
from app.planning.catalog import FieldCatalog, FieldSpec  # noqa: E402


def _catalog(*specs: FieldSpec) -> FieldCatalog:
    return FieldCatalog(datasets=(), fields=tuple(specs))


def _field(name: str, *, allowed: bool, keys: tuple[str, ...]) -> FieldSpec:
    return FieldSpec(
        file_id=49,
        semantic_field=name,
        human_label=name.replace("_", " ").title(),
        canonical_json_path=f"canonical.{name}",
        semantic_type="text",
        allowed_operators=("CONTAINS",),
        raw_json_keys=keys,
        raw_fetch_allowed=allowed,
    )


def test_a_field_that_is_not_granted_raw_access_is_never_fetched():
    catalog = _catalog(_field("notes", allowed=False, keys=("Notes",)))
    assert _raw_fetch_keys((49,), catalog) == []


def test_only_granted_fields_contribute_their_source_columns():
    catalog = _catalog(
        _field("notes", allowed=False, keys=("Notes",)),
        _field("cause_of_death", allowed=True, keys=("Cause of Death",)),
    )
    assert _raw_fetch_keys((49,), catalog) == ["Cause of Death"]


def test_the_keys_come_from_the_registry_not_from_a_literal():
    """The old code asked for Notes, Cause of Death and Name whatever the registry said."""
    catalog = _catalog(_field("student_name", allowed=True, keys=("Name", "Student Name")))
    assert _raw_fetch_keys((49,), catalog) == ["Name", "Student Name"]


def test_nothing_granted_anywhere_means_nothing_readable():
    """The policy every list currently carries, now actually enforced."""
    catalog = _catalog(
        _field("notes", allowed=False, keys=("Notes",)),
        _field("student_name", allowed=False, keys=("Name",)),
    )
    assert _raw_fetch_keys((49,), catalog) == []
