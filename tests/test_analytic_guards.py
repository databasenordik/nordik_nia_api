"""A calculation must not be answered with an arbitrary page of records.

These guards run on every compiled turn, so the guarantees they enforce do not
depend on the language model choosing the right goal.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.analytic_guards import apply_analytic_guards, is_analytic_question
from app.planning.catalog import static_catalog
from app.planning.plan_coverage import plan_covers_question
from app.planning.query_plan_builder import build_query_plan
from app.planning.turn_schema import FilterSpec, QueryAction, TurnPlan
from app.security.access_scope import AccessScope

CATALOG = static_catalog()
SCOPE = AccessScope(principal_id="tests", allowed_file_ids=(49, 91))


def _guard(question: str, *, file_id: int = 49, goal: str = "list", **kwargs) -> QueryAction:
    turn = TurnPlan(
        actions=[QueryAction(datasets=[file_id], goal=goal, **kwargs)],
        final_response="deterministic",
        confidence=0.9,
    )
    return apply_analytic_guards(turn, question, CATALOG).query_actions()[0]


def test_how_many_never_stays_a_paginated_list():
    action = _guard("How many students have a community listed?")
    assert action.goal == "count"


def test_which_has_the_most_becomes_a_ranking_with_the_requested_depth():
    action = _guard(
        "What are the 20 communities with the highest number of deceased students?",
        group_by=["community"],
    )
    assert action.goal == "rank"
    assert action.top_n == 20


def test_what_percentage_becomes_a_percentage():
    action = _guard(
        "What percentage of students are listed as deceased?",
        filters=[FilterSpec(field="deceased_status", operator="IS_TRUE")],
    )
    assert action.goal == "percentage"


def test_a_percentage_is_not_invented_when_none_was_asked_for():
    action = _guard(
        "How many students are recorded as deceased?",
        goal="percentage",
        filters=[FilterSpec(field="deceased_status", operator="IS_TRUE")],
    )
    assert action.goal == "count"


def test_more_than_once_becomes_duplicate_detection():
    action = _guard("Which full names appear more than once?", group_by=["student_name"])
    assert action.goal == "duplicates"


def test_each_decade_becomes_a_grouped_breakdown():
    action = _guard("How many students were admitted during each decade?", goal="count")
    assert action.goal == "rank"
    assert action.group_by == ["admitted_date"]
    assert action.group_value_part == "decade"


def test_listing_the_buckets_of_a_field_does_not_filter_to_one_of_them():
    action = _guard(
        "How many students are listed as deceased, not deceased, unknown, "
        "or have no deceased status?",
        goal="count",
        group_by=["deceased_status"],
        filters=[FilterSpec(field="deceased_status", operator="IS_TRUE")],
    )
    assert action.goal == "rank"
    assert action.include_missing is True
    assert action.filters == []


def test_first_name_reaches_the_first_name_field_not_a_slice_of_the_full_name():
    action = _guard(
        "What are the 20 most common first names?",
        goal="rank",
        group_by=["student_name"],
        group_value_part="first_token",
    )
    assert action.group_by == ["first_name"]
    assert action.group_value_part is None


def test_elapsed_time_infers_the_two_dates_in_chronological_order():
    action = _guard(
        "Which student lived the longest after their first admission to the school?",
        file_id=91,
    )
    assert action.goal == "interval"
    assert action.interval_start == "first_admitted_date"
    assert action.interval_end == "death_date"
    assert action.sort_direction == "desc"


def test_shortest_span_sorts_ascending_and_excludes_impossible_orderings():
    action = _guard(
        "Which student died the shortest amount of time after being admitted?",
        file_id=91,
    )
    assert action.sort_direction == "asc"
    assert action.interval_min_days == 0.0


def test_a_synonym_family_widens_a_single_term():
    action = _guard(
        "Which students died from tuberculosis?",
        file_id=91,
        filters=[FilterSpec(field="cause_of_death", operator="CONTAINS", value="tuberculosis")],
    )
    assert action.filters[0].operator == "CONTAINS_ANY"
    assert "consumption" in action.filters[0].value


def test_a_field_reference_that_names_nothing_is_dropped_not_fatal():
    action = _guard(
        "What causes of death are recorded and how many died from each?",
        file_id=91,
        goal="rank",
        group_by=["cause_of_death"],
        sort_by="count",
    )
    assert action.sort_by is None
    assert action.group_by == ["cause_of_death"]


def test_analytic_questions_bypass_the_legacy_fast_paths():
    assert is_analytic_question("Which community has the largest number of students?")
    assert is_analytic_question("What communities are listed in the Student Master List?")
    assert is_analytic_question("How many students were admitted during each decade?")
    assert not is_analytic_question("How many students do not have a community listed?")


def test_a_fast_path_plan_that_drops_a_condition_is_rejected():
    """The legacy grammar recognised the field but not the value it was constrained to."""
    turn = TurnPlan(
        actions=[QueryAction(datasets=[91], goal="list", requested_fields=["cause_of_death"])],
        final_response="deterministic",
        confidence=1.0,
    )
    plan = build_query_plan(turn, scope=SCOPE, catalog=CATALOG)
    assert not plan_covers_question(plan, "Which students died from tuberculosis?", CATALOG)


def test_a_fast_path_plan_that_drops_a_named_field_is_rejected():
    turn = TurnPlan(
        actions=[
            QueryAction(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="admitted_date", operator="IS_KNOWN")],
            )
        ],
        final_response="deterministic",
        confidence=1.0,
    )
    plan = build_query_plan(turn, scope=SCOPE, catalog=CATALOG)
    assert not plan_covers_question(
        plan, "How many students have an admission date but no discharge date?", CATALOG
    )


def test_no_x_recorded_is_a_missing_value_not_a_present_one():
    """"with no First Nation recorded" was compiled as IS_KNOWN - the opposite question."""
    action = _guard(
        "Count the master list students with no First Nation recorded.",
        goal="count",
        filters=[FilterSpec(field="community", operator="IS_KNOWN")],
    )
    assert [(item.field, item.operator) for item in action.filters] == [
        ("community", "IS_UNKNOWN")
    ]


def test_polarity_correction_only_flips_the_negated_condition():
    action = _guard(
        "How many students have an admission date but no discharge date?",
        goal="count",
        filters=[
            FilterSpec(field="admitted_date", operator="IS_KNOWN"),
            FilterSpec(field="discharged_date", operator="IS_KNOWN"),
        ],
    )
    assert [(item.field, item.operator) for item in action.filters] == [
        ("admitted_date", "IS_KNOWN"),
        ("discharged_date", "IS_UNKNOWN"),
    ]


def test_several_values_for_one_field_are_alternatives_not_a_conjunction():
    """ANDing four causes onto one field can never match, so the answer was zero."""
    action = _guard(
        "How many confirmed students died of TB, consumption, phthisis or scrofula?",
        file_id=91,
        goal="count",
        filters=[
            FilterSpec(field="cause_of_death", operator="CONTAINS", value="TB"),
            FilterSpec(field="cause_of_death", operator="CONTAINS", value="consumption"),
            FilterSpec(field="cause_of_death", operator="CONTAINS", value="phthisis"),
        ],
    )
    assert len(action.filters) == 1
    assert action.filters[0].operator == "CONTAINS_ANY"
    assert {"consumption", "phthisis", "scrofula"} <= set(action.filters[0].value)
