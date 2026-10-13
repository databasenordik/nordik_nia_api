"""A community written into a name cell is recovered by the cell's structure, never guessed.

A tester counted where the Potential list's children came from and found most origins
written inside the name cell -- rows pasted whole, "Name<TAB>date<TAB>community", or run
together with no separator at all. These tests use invented names in the same shapes. They
lock the rule that makes the recovery safe: a fragment is a place only when it follows a date
from the same pasted row or is introduced by "From", so a gloss, an editorial note or a
family sentence is never read as somebody's community.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.normalization.backfill_names import apply_community, fold_label  # noqa: E402
from app.normalization.name_parser import parse_full_name, place_from_residue  # noqa: E402


def test_a_place_pasted_after_a_date_in_the_same_cell_is_the_community():
    parts = parse_full_name("Mary Sample\tMarch 3, 1901\tRiver Bend (Lake People)")
    assert (parts.first, parts.last) == ("Mary", "Sample")
    assert place_from_residue(parts) == "River Bend (Lake People)"


def test_a_date_and_place_run_together_are_still_told_apart():
    parts = parse_full_name("John SampleApril 4, 1902Stony Point First Nation")
    assert place_from_residue(parts) == "Stony Point"


def test_a_from_note_after_a_death_note_names_the_community():
    parts = parse_full_name("Anne Sample (Died: May 5, 1903) — From the Stony Point community.")
    assert place_from_residue(parts) == "Stony Point"


@pytest.mark.parametrize(
    "cell",
    [
        "Peter Sample (Cloud Running in a Line)",
        "Wesley SAMPLE -(moved to additional list)",
        "TO BE REVIEWED",
        "The Sample Family: Emma was the third child from the family to attend",
        "Rose Sample (Died: June 1, 1904)",
        "Rose SAMPLE",
    ],
)
def test_nothing_else_in_a_name_cell_is_taken_for_a_place(cell):
    assert place_from_residue(parse_full_name(cell)) == ""


def test_a_derived_community_is_marked_and_folded_for_matching():
    canonical: dict = {}
    parts = parse_full_name("Mary Sample\tMarch 3, 1901\tRiver Bend (Lake People)")
    assert apply_community(canonical, "", parts, derive=True) == "river bend lake people"
    assert canonical == {"community": "River Bend (Lake People)", "community_source": "name_cell"}


def test_a_recorded_community_is_never_replaced_by_a_derived_one():
    canonical = {"community": "Walpole Island"}
    parts = parse_full_name("Mary Sample\tMarch 3, 1901\tRiver Bend")
    assert apply_community(canonical, "Walpole Island", parts, derive=True) is None
    assert canonical == {"community": "Walpole Island"}


def test_a_recorded_community_fills_a_missing_display_value():
    canonical: dict = {}
    assert apply_community(canonical, "Sault Ste. Marie", parse_full_name("Ann SAMPLE"), derive=True) is None
    assert canonical == {"community": "Sault Ste. Marie"}


def test_a_derivation_the_parser_no_longer_supports_is_withdrawn():
    canonical = {"community": "Old Place", "community_source": "name_cell"}
    assert apply_community(canonical, "", parse_full_name("Ann SAMPLE"), derive=True) == ""
    assert canonical == {}


def test_lists_whose_names_have_their_own_cells_are_not_derived():
    canonical: dict = {}
    parts = parse_full_name("Mary Sample\tMarch 3, 1901\tRiver Bend")
    assert apply_community(canonical, "", parts, derive=False) is None
    assert canonical == {}


def test_folded_labels_match_the_form_ingest_stores():
    assert fold_label("Sault Ste. Marie") == "sault ste marie"
    assert fold_label("Lac Seul #346") == "lac seul 346"
