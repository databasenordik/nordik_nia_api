"""What the planner is told about each field, and what it is deliberately not told.

The catalog line is the planner's whole view of a field, so what goes in it is a design
decision with measured consequences. Two of them are pinned here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.ai_planner import _field_line  # noqa: E402
from app.planning.catalog import FieldSpec  # noqa: E402


def _sparse_field(**rules) -> FieldSpec:
    """File 94's Nation: labelled for a place of origin, filled in 3 rows of 56."""
    return FieldSpec(
        file_id=94,
        semantic_field="nation",
        human_label="Nation",
        canonical_json_path="canonical.nation",
        semantic_type="entity",
        aliases=("first nation", "band"),
        allowed_operators=("EQUALS", "CONTAINS"),
        validation_rules={"recorded_count": 3, "record_count": 56, **rules},
    )


def test_the_planner_is_told_how_much_of_a_field_is_recorded():
    """A label says what a field is for; only the fill rate says whether it was used.

    Without it, "how many potential records are from Garden River?" bound to this field and
    answered "0 records" -- a claim about the people that the records do not support.
    """
    assert "recorded=3/56" in _field_line(_sparse_field())


def test_the_planner_is_not_told_a_field_s_values():
    """Publishing the values fixed one case and broke four.

    Shown a field's categories the planner filtered on them unprompted, so "how many records
    are in the confirmed deaths list" answered 25 of 82. The compiler still reads
    recorded_values from the registry; the prompt does not.
    """
    line = _field_line(_sparse_field(recorded_values=["community member", "not confirmed"]))
    assert "recorded_values" not in line
    assert "community member" not in line


def test_a_field_with_no_counts_says_nothing_about_them():
    bare = FieldSpec(
        file_id=49,
        semantic_field="community",
        human_label="Community",
        canonical_json_path="canonical.community",
        semantic_type="entity",
        allowed_operators=("EQUALS",),
    )
    assert "recorded=" not in _field_line(bare)


def test_the_line_still_carries_what_the_planner_resolves_names_by():
    line = _field_line(_sparse_field())
    assert "nation" in line and "Nation" in line and "first nation" in line
