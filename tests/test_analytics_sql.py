"""Integration checks for the analytics SQL, run only when PostgreSQL is reachable.

These assert the same behaviour as tests/test_analytics_capabilities.py, but through
the functions the production gateway actually calls.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.repository import AssistantDataGateway
from app.planning.plan_schema import FilterOperator, Predicate, any_of
from app.security.access_scope import AccessScope

DATABASE_URL = os.environ.get("ASSISTANT_TEST_DATABASE_URL", "")
SCOPE = AccessScope(principal_id="analytics-sql-tests", allowed_file_ids=(49, 91))


@pytest.fixture(scope="module")
def gateway():
    if not DATABASE_URL:
        pytest.skip("ASSISTANT_TEST_DATABASE_URL is not configured")
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        pytest.skip("asyncpg is not installed")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        pool = loop.run_until_complete(_build_pool())
    except Exception:
        pytest.skip("PostgreSQL is not reachable")
    if pool is None:
        pytest.skip("PostgreSQL is not reachable")
    yield AssistantDataGateway(pool), loop
    loop.run_until_complete(pool.close())
    loop.close()


async def _build_pool():
    import asyncpg

    return await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2, timeout=3)


def test_grouping_reports_the_stored_spelling_not_the_folded_lookup_value(gateway):
    api, loop = gateway
    groups = loop.run_until_complete(
        api.group_values_multi(SCOPE, (49,), [], "community", top_n=1)
    )
    assert groups[0].label == "Walpole Island"


def test_or_group_reaches_every_spelling_of_a_cause(gateway):
    api, loop = gateway
    tuberculosis = Predicate(
        field="cause_of_death",
        operator=FilterOperator.CONTAINS_ANY,
        value=["tuberculosis", "consumption", "phthisis", "scrofula", "tubercular"],
    )
    count = loop.run_until_complete(api.count_records(SCOPE, (91,), [tuberculosis]))
    assert count == 38


def test_negation_answers_somewhere_other_than(gateway):
    api, loop = gateway
    away = Predicate(
        field="location_of_death", operator=FilterOperator.NOT_CONTAINS, value="school"
    )
    assert loop.run_until_complete(api.count_records(SCOPE, (91,), [away])) == 27


def test_conjunction_keeps_both_conditions(gateway):
    api, loop = gateway
    predicates = [
        Predicate(field="admitted_date", operator=FilterOperator.IS_KNOWN),
        Predicate(field="discharged_date", operator=FilterOperator.IS_UNKNOWN),
    ]
    assert loop.run_until_complete(api.count_records(SCOPE, (49,), predicates)) == 735


def test_nested_boolean_group_is_evaluated(gateway):
    api, loop = gateway
    tree = any_of(
        Predicate(field="community", operator=FilterOperator.EQUALS, value="Sarnia"),
        Predicate(field="community", operator=FilterOperator.EQUALS, value="Garden River"),
    )
    both = loop.run_until_complete(api.count_records(SCOPE, (49,), [tree]))
    sarnia = loop.run_until_complete(
        api.count_records(
            SCOPE, (49,), [Predicate(field="community", operator=FilterOperator.EQUALS, value="Sarnia")]
        )
    )
    assert both > sarnia


def test_candidate_search_applies_community_filter_in_postgres(gateway):
    """Candidate retrieval must use the filter-aware overload before its LIMIT.

    The gRPC community gateway forwards this predicate to the repository.  Calling
    the real PostgreSQL gateway here catches a missing seven-argument SQL function
    as well as a function that limits candidates before applying the community.
    """
    api, loop = gateway
    rows = loop.run_until_complete(
        api.retrieve_candidates(
            SCOPE,
            (49,),
            "Shingwauk",
            "exact",
            limit=1,
            predicates=[
                Predicate(
                    field="community",
                    operator=FilterOperator.IN,
                    value=["Garden River"],
                )
            ],
        )
    )

    assert rows
    assert {row["canonical_community"] for row in rows} == {"Garden River"}


def test_statistics_match_the_recorded_ages(gateway):
    api, loop = gateway
    stats = loop.run_until_complete(api.field_stats(SCOPE, (91,), [], "age_at_death"))
    assert (stats.known_count, stats.minimum, stats.maximum) == (77, 7, 70)
    assert stats.median == 13
    assert stats.mode == 14


def test_duplicate_student_numbers_are_found(gateway):
    api, loop = gateway
    groups = loop.run_until_complete(
        api.duplicate_values(SCOPE, (49,), [], "student_number", companion_field="student_name")
    )
    assert len(groups) == 82


def test_decade_buckets_match_the_admission_years(gateway):
    api, loop = gateway
    groups = loop.run_until_complete(
        api.group_values_multi(SCOPE, (49,), [], "admitted_date", value_part="decade")
    )
    counts = {item.label: item.count for item in groups}
    assert counts["1940s"] == 231
    assert counts["1870s"] == 157


def test_interval_flags_an_impossible_ordering_but_not_a_coarse_date(gateway):
    api, loop = gateway
    rows = loop.run_until_complete(
        api.interval_rows(
            SCOPE, (49,), [], "admitted_date", "discharged_date",
            max_days=-1, direction="asc", limit=20,
        )
    )
    impossible = [item for item in rows if item.is_impossible()]
    assert [item.display_name for item in impossible] == ["William Maggrah"]
    assert len(rows) > len(impossible)
