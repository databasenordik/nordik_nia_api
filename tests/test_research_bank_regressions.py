"""General regressions exposed by the live research-bank review.

Fixtures are synthetic; expected answers never enter runtime planning.
"""
import math

import pytest

from app.planning.ai_catalog_coerce import coerce_planned_turn, coerce_stated_requirements
from app.planning.ai_turn_schema import PlannedTurn, bind_scope
from app.planning.catalog import FieldSpec, static_catalog
from app.planning.plan_validator import PlanValidationError, validate_filter_value
from app.planning.query_plan_builder import _predicates_for


def _matches(node, tokens):
    if node.op:
        results = [_matches(child, tokens) for child in node.items]
        return all(results) if node.op == "and" else any(results)
    return str(node.value).casefold() in tokens


@pytest.mark.parametrize("as_any", [False, True])
def test_alternative_full_names_preserve_each_name_conjunction(as_any):
    filters = ([{"field": "name_search", "operator": "CONTAINS_ANY", "value": ["Helen Lake", "Mary Stone"]}]
               if as_any else [
                   {"field": "name_search", "operator": "CONTAINS", "value": "Helen Lake"},
                   {"field": "name_search", "operator": "CONTAINS", "value": "Mary Stone"},
               ])
    requirements = [{"kind": "filter_logic", "action_index": 0, "text": "either full name", "filter_logic": "or"}]
    requirements += [{"kind": "filter", "action_index": 0, "text": "name", **item} for item in filters]
    planned = PlannedTurn.model_validate({"actions": [{"type": "query", "goal": "list", "filters": filters,
                                                       "filter_logic": "or"}], "stated_requirements": requirements,
                                          "final_response": "deterministic"})
    catalog = static_catalog()
    catalog = catalog.model_copy(update={"fields": (*catalog.fields, FieldSpec(
        file_id=49, semantic_field="name_search", human_label="Recorded name tokens",
        validation_rules={"token_index": True}, allowed_operators=("CONTAINS", "CONTAINS_ANY"),
    ))})
    coerced = coerce_planned_turn(planned, catalog, 49)
    assert coerce_stated_requirements(planned.stated_requirements, catalog, 49) == coerced.stated_requirements
    action = bind_scope(coerced, 49).actions[0]
    predicates = _predicates_for(action, (49,), catalog)
    for tokens, expected in [("helen lake", True), ("mary stone", True), ("helen stone", False),
                             ("mary lake", False), ("helen brook", False), ("jane stone", False)]:
        assert all(_matches(node, tokens.split()) for node in predicates) is expected


@pytest.mark.parametrize("operator,value", [
    ("EQUALS", None), ("CONTAINS", ""), ("IN", []), ("NUMBER_RANGE", [5, 2]),
    ("NUMBER_RANGE", [None, 2]), ("NUMBER_RANGE", [True, 4]), ("NUMBER_RANGE", [math.nan, 4]),
])
def test_malformed_filters_are_not_silently_zero_matches(operator, value):
    with pytest.raises(PlanValidationError):
        validate_filter_value(operator, value)


@pytest.mark.parametrize("operator", ["IS_KNOWN", "IS_UNKNOWN", "IS_TRUE", "IS_FALSE"])
def test_unary_filters_need_no_operand(operator):
    validate_filter_value(operator, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("forbidden,gate", [(True, True), (False, False), (True, False)])
async def test_startup_checks_permissions_and_gate_before_reading_content(monkeypatch, forbidden, gate):
    from unittest.mock import AsyncMock
    from app.db.privilege_selftest import assert_runtime_isolation

    conn = AsyncMock()
    conn.fetch.return_value = [{"name": "public.file_data"}] if forbidden else []
    conn.fetchval.return_value = gate
    monkeypatch.setattr("app.db.privilege_selftest.asyncpg.connect", AsyncMock(return_value=conn))
    with pytest.raises(PermissionError):
        await assert_runtime_isolation("postgresql://test")
    assert conn.fetch.await_count == 1
    assert "has_table_privilege" in conn.fetch.call_args.args[0]
    assert "SELECT *" not in conn.fetch.call_args.args[0]
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_rejected_role_never_creates_runtime_pool(monkeypatch):
    from unittest.mock import AsyncMock
    from app.db import pool

    monkeypatch.setattr(pool, "_pool", None)
    check = AsyncMock(side_effect=PermissionError("release gate missing"))
    create = AsyncMock()
    monkeypatch.setattr("app.db.privilege_selftest.assert_runtime_isolation", check)
    monkeypatch.setattr(pool.asyncpg, "create_pool", create)
    with pytest.raises(PermissionError):
        await pool.init_pool()
    create.assert_not_awaited()


@pytest.mark.parametrize("value", ["1972-00-00", "9?", "about 12", "12 to 14", "NaN", "Infinity"])
def test_numeric_measurements_do_not_extract_digits_from_other_values(value):
    from app.execution.derivations import numeric_value
    from app.execution.executor import _typed_aggregate_value

    assert numeric_value(value, "number") is None
    assert _typed_aggregate_value(value, "number") is None


def test_dates_and_decimal_measurements_keep_their_types():
    from app.execution.derivations import numeric_value
    from app.execution.executor import _typed_aggregate_value

    assert numeric_value("1972-00-00", "year") == 1972
    assert _typed_aggregate_value("12.5", "number") == 12.5
    assert _typed_aggregate_value("14 years", "number") == 14
    assert _typed_aggregate_value("1972-00-00", "date") == 1972


@pytest.mark.parametrize("primary", ["", None, {"raw": "", "normalized": "Stale value"}])
def test_blank_primary_source_cell_does_not_resurrect_a_legacy_value(primary):
    from app.normalization.backfill_names import _source_cell

    row = {"row_data_normalized": {"fields": {"Current": primary, "Legacy": "Old place"}}}
    sources = {"community": ["fields.Current", "fields.Legacy"]}
    assert _source_cell(row, "community", sources, raw_only=True) == ""
    del row["row_data_normalized"]["fields"]["Current"]
    assert _source_cell(row, "community", sources, raw_only=True) == "Old place"


def test_bare_recorded_name_variants_are_searchable_without_inventing_their_type():
    from app.normalization.name_parser import NameParts, merge_name_comments

    base = NameParts(first="Jane", last="Stone")
    alias = merge_name_comments(base, "Willowbird (little bird)", bare_variants=True)
    assert "Willowbird" in alias.alias
    assert not alias.indigenous
    ordinary_note = merge_name_comments(base, "Willowbird (little bird)")
    assert not ordinary_note.alias
    prose = merge_name_comments(base, "Mother lived in Willow City.", bare_variants=True)
    assert not prose.alias


@pytest.mark.parametrize("question,distinct", [
    ("How many students have different spellings recorded?", False),
    ("How many children went to different schools?", False),
    ("How many distinct communities are recorded?", True),
    ("How many unique values are present?", True),
    ("How many different names per community?", False),
])
def test_distinct_modifier_must_describe_the_counted_object(question, distinct):
    from app.planning.ai_catalog_coerce import _asks_distinct_count

    assert _asks_distinct_count(question) is distinct


def test_duplicate_requirement_error_identifies_the_repair_without_accepting_it():
    from app.planning.requirement_parity import check_requirement_parity

    condition = {"field": "community", "operator": "EQUALS", "value": "Lake Town"}
    requirement = {"kind": "filter", "action_index": 0, "text": "place", **condition}
    planned = PlannedTurn.model_validate({
        "actions": [{"type": "query", "goal": "count", "filters": [condition]}],
        "stated_requirements": [requirement, {**requirement, "text": "same place twice"}],
        "final_response": "deterministic",
    })
    with pytest.raises(PlanValidationError, match="duplicate requirement already accounted for"):
        check_requirement_parity(planned, bind_scope(planned, 49))


def test_an_operator_outside_the_schema_asks_the_reviewer_for_the_executable_form():
    from pydantic import ValidationError

    from app.planning.ai_planner import _review_schema_repair
    from app.planning.ai_turn_schema import PlanReviewResponse

    with pytest.raises(ValidationError) as raised:
        PlanReviewResponse.model_validate({
            "verdict": "complete",
            "reconstructed_requirements": [
                {"kind": "filter", "action_index": 0, "text": "both parents",
                 "field": "parents_names", "operator": "CONTAINS_ALL",
                 "value": ["Sabine Williams", "Emily Watson"]},
            ],
        })
    instruction = _review_schema_repair(raised.value)
    assert instruction is not None
    assert "CONTAINS_ALL" in instruction
    assert "one CONTAINS filter per term with filter_logic=and" in instruction
    assert "CONTAINS_ANY" in instruction  # named only to forbid weakening the condition


def test_an_unrepairable_review_error_still_fails_closed():
    from pydantic import ValidationError

    from app.planning.ai_planner import _review_schema_repair
    from app.planning.ai_turn_schema import PlanReviewResponse

    with pytest.raises(ValidationError) as raised:
        PlanReviewResponse.model_validate({"verdict": "corrected"})
    assert _review_schema_repair(raised.value) is None


@pytest.mark.parametrize("start,end,kept", [
    ("1940", "1945", False),
    ("1940-01-01", "1945-12-31", False),
    ("birth_date", "admitted_date", True),
    ("1940", "admitted_date", True),
])
def test_a_calendar_span_is_not_an_interval_between_two_date_fields(start, end, kept):
    """interval_start/interval_end name fields; a year pair restates the date filter.

    Live, "admitted from 1940 through 1945" was reconstructed as the DATE_RANGE filter and
    as an interval from "1940" to "1945". Parity then required the plan to express an
    interval whose only possible spelling named a field that cannot exist, and the turn
    ended in "that field is not available on the selected list".
    """
    requirements = [
        {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
        {"kind": "filter", "action_index": 0, "text": "1940 through 1945",
         "field": "admitted_date", "operator": "DATE_RANGE",
         "value": ["1940-01-01", "1945-12-31"]},
        {"kind": "interval", "action_index": 0, "text": "1940 through 1945",
         "interval_start": start, "interval_end": end},
    ]
    planned = PlannedTurn.model_validate({
        "actions": [{"type": "query", "goal": "count", "filters": [requirements[1]["field"]
                     and {"field": "admitted_date", "operator": "DATE_RANGE",
                          "value": ["1940-01-01", "1945-12-31"]}]}],
        "stated_requirements": requirements,
        "final_response": "deterministic",
    })
    catalog = static_catalog()
    coerced = coerce_stated_requirements(planned.stated_requirements, catalog, 49)
    assert any(item.kind == "interval" for item in coerced) is kept
    assert [item.kind for item in coerced if item.kind != "interval"] == ["goal", "filter"]
    assert coerce_planned_turn(planned, catalog, 49).stated_requirements == coerced


def test_an_interval_nobody_else_states_is_still_required():
    """Without a date condition carrying the span, the requirement stands and fails closed."""
    requirements = [
        {"kind": "goal", "action_index": 0, "text": "count", "goal": "count"},
        {"kind": "interval", "action_index": 0, "text": "1940 through 1945",
         "interval_start": "1940", "interval_end": "1945"},
    ]
    planned = PlannedTurn.model_validate({
        "actions": [{"type": "query", "goal": "count"}],
        "stated_requirements": requirements,
        "final_response": "deterministic",
    })
    coerced = coerce_stated_requirements(planned.stated_requirements, static_catalog(), 49)
    assert [item.kind for item in coerced] == ["goal", "interval"]


def test_clarifying_a_missing_column_comes_after_the_narrative_and_name_rules():
    """The anti-substitution rule must not short-circuit the rules that answer the question.

    Asking for hospital stays, families with siblings, or a misspelt person's siblings all
    came back as clarifying questions once "clarify that this list does not record it" read
    as unconditional, though the narrative-field and name-correction rules below it say what
    to do with exactly those requests.
    """
    from app.planning.ai_planner import AI_PLANNER_SYSTEM

    text = " ".join(AI_PLANNER_SYSTEM.split())
    assert "Clarify that this list does not record it only after the rules below leave" in text
    assert "neither a narrative field carrying the event nor a misspelled name is a reason" in text
    assert text.index("never substitute a related-sounding field") < text.index(
        "Distinguish a missing dedicated column from an event described"
    )


def test_both_prompts_say_interval_bounds_name_fields_not_dates():
    from app.planning.ai_planner import AI_PLANNER_SYSTEM, AI_REVIEW_SYSTEM

    for prompt in (AI_PLANNER_SYSTEM, AI_REVIEW_SYSTEM):
        text = " ".join(prompt.split())
        assert "Both name date FIELDS whose values each record carries, never a year or a date" in text
        assert 'a calendar span such as "from 1940 through 1945" is a DATE_RANGE filter' in text
