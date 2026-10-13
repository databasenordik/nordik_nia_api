"""Capability tests for the analytics surface added to answer computed questions.

Everything here runs against the in-process gateway, so the suite needs neither
PostgreSQL nor a reasoning provider. The Postgres implementations of the same
operations are covered by tests/test_analytics_sql.py.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.derivations import date_precision, numeric_value, parse_date, value_part
from app.execution.filters import matches_predicate_tree
from app.planning.catalog import static_catalog
from app.planning.plan_schema import FilterOperator, Predicate, any_of
from app.security.access_scope import AccessScope

SCOPE = AccessScope(principal_id="tests", allowed_file_ids=(49, 91))
CATALOG = static_catalog()


def _records() -> list[dict]:
    """A small master-list fixture with the shapes the real corpus contains."""
    def row(index, name, community, admitted=None, discharged=None, deceased="no", number=None):
        canonical = {
            "display_name": name,
            "community": community,
            "deceased_status": deceased,
            "dates": {},
        }
        if admitted:
            canonical["dates"]["admitted"] = admitted
        if discharged:
            canonical["dates"]["discharged"] = discharged
        if number:
            canonical["student_number"] = number
        return {
            "id": index,
            "source_row_id": 1000 + index,
            "file_id": 49,
            "version": 3,
            "canonical_name": name.lower(),
            "canonical_community": community.lower() if community else None,
            "canonical_school": None,
            "row_data_normalized": {"canonical": canonical},
        }

    return [
        row(1, "Mary Jane Fox", "Walpole Island", "1875-05-01", "1878-06-01", "yes", "1875.001"),
        row(2, "Mary Fox", "Walpole Island", "1884-01-01", "1890-01-01", "yes", "1875.001"),
        row(3, "John Fox", "Garden River", "1893-01-01", None, "no"),
        row(4, "John Fox", "Sarnia", "1947-01-01", "1950-01-01", "unknown"),
        row(5, "Alice Pine", None, None, None, None),
    ]


def _gateway() -> MemoryDataGateway:
    return MemoryDataGateway(records=_records(), catalog=CATALOG)


def test_or_group_matches_either_branch():
    """Filters used to be a pure conjunction, so either/or questions were unanswerable."""
    tree = any_of(
        Predicate(field="community", operator=FilterOperator.EQUALS, value="Sarnia"),
        Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"),
    )
    matched = [
        row["canonical_name"]
        for row in _records()
        if matches_predicate_tree(row, tree, CATALOG, 49)
    ]
    assert sorted(matched) == ["john fox", "john fox"]


def test_negative_operator_includes_records_with_no_value():
    """A record with no recorded community is genuinely "not from Sarnia"."""
    predicate = Predicate(
        field="community", operator=FilterOperator.NOT_CONTAINS, value="Sarnia"
    )
    matched = [
        row["canonical_name"]
        for row in _records()
        if matches_predicate_tree(row, predicate, CATALOG, 49)
    ]
    assert "alice pine" in matched
    assert matched.count("john fox") == 1


def test_contains_any_matches_a_family_of_spellings():
    predicate = Predicate(
        field="community",
        operator=FilterOperator.CONTAINS_ANY,
        value=["walpole", "sarnia"],
    )
    matched = [
        row["canonical_name"]
        for row in _records()
        if matches_predicate_tree(row, predicate, CATALOG, 49)
    ]
    assert len(matched) == 3


def test_ranking_orders_groups_and_honours_top_n():
    groups = asyncio.run(
        _gateway().group_values_multi(SCOPE, (49,), [], "community", top_n=2)
    )
    assert [(item.label, item.count) for item in groups] == [
        ("Walpole Island", 2),
        ("Garden River", 1),
    ]


def test_ranking_can_include_the_missing_bucket():
    groups = asyncio.run(
        _gateway().group_values_multi(SCOPE, (49,), [], "community", include_missing=True)
    )
    assert any(item.label is None and item.count == 1 for item in groups)


def test_having_floor_keeps_only_repeated_groups():
    groups = asyncio.run(
        _gateway().group_values_multi(SCOPE, (49,), [], "community", min_count=2)
    )
    assert [item.label for item in groups] == ["Walpole Island"]


def test_decade_bucket_groups_dates_without_a_new_field():
    groups = asyncio.run(
        _gateway().group_values_multi(SCOPE, (49,), [], "admitted_date", value_part="decade")
    )
    assert {item.label: item.count for item in groups} == {
        "1870s": 1,
        "1880s": 1,
        "1890s": 1,
        "1940s": 1,
    }


def test_name_part_grouping_counts_first_tokens():
    groups = asyncio.run(
        _gateway().group_values_multi(SCOPE, (49,), [], "student_name", value_part="first_token")
    )
    assert {item.label: item.count for item in groups}["Mary"] == 2


def test_statistics_report_median_and_mode_not_just_average():
    stats = asyncio.run(
        _gateway().field_stats(SCOPE, (49,), [], "admitted_date", value_part="year")
    )
    assert stats.known_count == 4
    assert stats.minimum == 1875
    assert stats.maximum == 1947
    assert stats.median == (1893 + 1884) / 2
    assert stats.mode_count == 1


def test_duplicate_detection_reports_groups_and_members():
    groups = asyncio.run(
        _gateway().duplicate_values(SCOPE, (49,), [], "student_number", companion_field="student_name")
    )
    assert len(groups) == 1
    assert groups[0].value == "1875.001"
    assert groups[0].count == 2
    assert sorted(groups[0].members) == ["Mary Fox", "Mary Jane Fox"]


def test_completeness_ranks_records_by_how_much_is_filled_in():
    most = asyncio.run(_gateway().record_completeness(SCOPE, (49,), [], direction="desc", limit=2))
    least = asyncio.run(_gateway().record_completeness(SCOPE, (49,), [], direction="asc", limit=1))
    assert {item.display_name for item in most} == {"Mary Jane Fox", "Mary Fox"}
    assert least[0].display_name == "Alice Pine"
    assert most[0].filled_fields > least[0].filled_fields
    assert most[0].total_fields == len(CATALOG.fields_for(49))


def test_interval_measures_elapsed_days_between_two_date_fields():
    rows = asyncio.run(
        _gateway().interval_rows(
            SCOPE, (49,), [], "admitted_date", "discharged_date", direction="desc", limit=5
        )
    )
    longest = rows[0]
    assert longest.display_name == "Mary Fox"
    assert longest.days == 2192.0
    stats = asyncio.run(
        _gateway().interval_stats(SCOPE, (49,), [], "admitted_date", "discharged_date")
    )
    assert stats.record_count == 3
    assert stats.minimum_days == 1096.0


def test_value_parts_and_date_parsing_match_the_stored_shapes():
    assert value_part("1889-12-24", "decade") == "1880s"
    assert value_part("1889-12-24", "year") == "1889"
    assert value_part("Albert PENANCE (Manitoba)", "last_token") == "(Manitoba)"
    assert numeric_value("14") == 14.0
    assert parse_date("01.12.1890").isoformat() == "1890-12-01"
    assert parse_date("1938-10-00").isoformat() == "1938-10-01"
    assert parse_date("1907").isoformat() == "1907-01-01"


def test_date_precision_separates_a_real_error_from_a_coarse_value():
    """A year-only value parses to 1 January, which can look like a negative span."""
    assert date_precision("1899-05-05") == "day"
    assert date_precision("1938-10-00") == "month"
    assert date_precision("1907") == "year"
    rows = asyncio.run(
        _gateway().interval_rows(
            SCOPE, (49,), [], "discharged_date", "admitted_date", direction="asc", limit=5
        )
    )
    assert all(item.is_impossible() for item in rows if item.year_delta is not None and item.year_delta < 0)
