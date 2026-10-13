"""Matching a closed set of categories exactly, rather than by substring.

A field whose recorded values are an enumeration is not free text. Matching one of its
categories with a substring search is wrong whenever one category contains another, and
deceased_status is exactly that shape: its values are yes, no and unknown, and "no" sits
inside "u-n-k-no-w-n".

That cost 665 records on one question and 665 on its mirror:

    "recorded as not deceased"           1,700 returned, 1,035 recorded "no"
    "unknown deceased status"               15 returned,   665 recorded "unknown"

Two layers had to agree for it to come out right, so both are pinned here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.planning.ai_catalog_coerce import _exact_category_match  # noqa: E402
from app.planning.catalog import FieldSpec  # noqa: E402
from app.planning.plan_schema import FilterOperator  # noqa: E402
from app.planning.turn_schema import FilterSpec  # noqa: E402
from app.planning.value_expansion import _expanded_operator  # noqa: E402

CATEGORY_OPS = (
    "EQUALS",
    "NOT_EQUALS",
    "IN",
    "NOT_IN",
    "CONTAINS",
    "NOT_CONTAINS",
    "CONTAINS_ANY",
    "NOT_CONTAINS_ANY",
    "IS_KNOWN",
    "IS_UNKNOWN",
)


def _categorical(**rules) -> FieldSpec:
    """A field shaped like deceased_status: three recorded categories, one inside another."""
    return FieldSpec(
        file_id=49,
        semantic_field="deceased_status",
        human_label="Deceased status",
        canonical_json_path="canonical.deceased_status",
        semantic_type="entity",
        allowed_operators=CATEGORY_OPS,
        validation_rules={"recorded_values": ["no", "unknown", "yes"], **rules},
    )


def _free_text(**rules) -> FieldSpec:
    return FieldSpec(
        file_id=91,
        semantic_field="cause_of_death",
        human_label="Cause of death",
        canonical_json_path="canonical.cause_of_death",
        semantic_type="text",
        allowed_operators=CATEGORY_OPS,
        validation_rules=dict(rules),
    )


# ---------------------------------------------------- the planner's own words, canonicalized


@pytest.mark.parametrize(
    "operator,value,expected_operator,expected_value",
    [
        ("CONTAINS_ANY", ["no"], "EQUALS", "no"),
        ("CONTAINS", "unknown", "EQUALS", "unknown"),
        ("CONTAINS_ANY", ["yes", "no"], "IN", ["yes", "no"]),
    ],
)
def test_a_category_asked_for_by_name_is_matched_exactly(
    operator, value, expected_operator, expected_value
):
    result = _exact_category_match(
        _categorical(), FilterSpec(field="deceased_status", operator=operator, value=value)
    )
    assert result == {"operator": expected_operator, "value": expected_value}


def _family() -> FieldSpec:
    """A field where one category is a longer form of another, not a different one."""
    return FieldSpec(
        file_id=91,
        semantic_field="location_of_death",
        human_label="Location of death",
        canonical_json_path="canonical.location_of_death",
        semantic_type="entity",
        allowed_operators=CATEGORY_OPS,
        validation_rules={
            "recorded_values": [
                "Community",
                "Hospital",
                "Sanitorium/Hospital",
                "School",
                "School (NCTR SOURCE)",
                "Unknown",
            ]
        },
    )


def test_a_category_that_is_a_word_of_another_keeps_substring_matching():
    """"Died at school" means both "School" and "School (NCTR SOURCE)".

    Pinning this to an exact match dropped the six rows carrying the provenance note --
    49 returned where 55 died at school.
    """
    assert _exact_category_match(
        _family(), FilterSpec(field="location_of_death", operator="CONTAINS", value="school")
    ) is None
    assert (
        _expanded_operator(_family(), FilterOperator.EQUALS, ["school"])
        is FilterOperator.CONTAINS_ANY
    )


def test_an_accidental_infix_still_matches_exactly():
    """The mirror of the above: "no" inside "unknown" is a letter collision, not a family."""
    assert _exact_category_match(
        _categorical(), FilterSpec(field="deceased_status", operator="CONTAINS", value="no")
    ) == {"operator": "EQUALS", "value": "no"}


def test_a_value_that_is_not_a_category_is_left_as_a_search():
    # "dece" is not one of the recorded values, so the user meant a substring search.
    assert _exact_category_match(
        _categorical(), FilterSpec(field="deceased_status", operator="CONTAINS", value="dece")
    ) is None


def test_a_field_without_an_enumeration_is_left_alone():
    assert _exact_category_match(
        _free_text(), FilterSpec(field="cause_of_death", operator="CONTAINS", value="tuberculosis")
    ) is None


# ---------------------------------------------------- and the same rule after value expansion


def test_expanding_within_an_enumeration_stays_exact():
    # Without this the expansion re-widened EQUALS 'no' back to CONTAINS_ANY ['no'],
    # undoing the coercion above and matching "unknown" all over again.
    assert (
        _expanded_operator(_categorical(), FilterOperator.EQUALS, ["no"]) is FilterOperator.IN
    )
    assert (
        _expanded_operator(_categorical(), FilterOperator.NOT_EQUALS, ["no"])
        is FilterOperator.NOT_IN
    )


def test_expanding_free_text_still_widens():
    # "tuberculosis" must still reach the records recorded as pulmonary tuberculosis.
    assert (
        _expanded_operator(_free_text(), FilterOperator.EQUALS, ["tuberculosis", "consumption"])
        is FilterOperator.CONTAINS_ANY
    )


def test_expanding_past_the_enumeration_widens():
    """A term outside the recorded set means the match cannot be an exact category match."""
    assert (
        _expanded_operator(_categorical(), FilterOperator.EQUALS, ["no", "alive"])
        is FilterOperator.CONTAINS_ANY
    )


def test_exact_matching_needs_the_operator_to_be_allowed():
    """Validation is strict about the operator list, so never widen into a banned operator."""
    narrow = FieldSpec(
        file_id=49,
        semantic_field="deceased_status",
        human_label="Deceased status",
        canonical_json_path="canonical.deceased_status",
        semantic_type="entity",
        allowed_operators=("EQUALS", "CONTAINS_ANY"),
        validation_rules={"recorded_values": ["no", "unknown", "yes"]},
    )
    assert (
        _expanded_operator(narrow, FilterOperator.EQUALS, ["no"]) is FilterOperator.CONTAINS_ANY
    )


# ---------------------------------------------------- a token index is not free text


def _token_index() -> FieldSpec:
    """name_search: one token per recorded form, rendered as a joined list."""
    return FieldSpec(
        file_id=91,
        semantic_field="name_search",
        human_label="Any recorded form of the name",
        canonical_json_path="names",
        semantic_type="text",
        allowed_operators=CATEGORY_OPS,
        validation_rules={"token_index": True},
    )


def test_a_multi_word_name_becomes_one_filter_per_token():
    """The index reads "albert, penance, pinnance", which contains no such phrase.

    Searching it for "Albert Penance" found nothing for a person who is in the list.
    Sibling filters are ANDed, so every token is required -- CONTAINS_ANY would return
    every Albert instead.
    """
    from app.planning.ai_catalog_coerce import _token_index_tokens

    assert _token_index_tokens(
        _token_index(),
        FilterSpec(field="name_search", operator="CONTAINS", value="Albert Penance"),
    ) == ["Albert", "Penance"]


def test_a_single_word_name_needs_no_splitting():
    from app.planning.ai_catalog_coerce import _token_index_tokens

    assert _token_index_tokens(
        _token_index(),
        FilterSpec(field="name_search", operator="CONTAINS", value="Penance"),
    ) is None


def test_an_ordinary_text_field_is_never_split():
    """Splitting free text would turn a phrase search into an unrelated conjunction."""
    from app.planning.ai_catalog_coerce import _token_index_tokens

    assert _token_index_tokens(
        _free_text(),
        FilterSpec(field="cause_of_death", operator="CONTAINS", value="pulmonary tuberculosis"),
    ) is None
