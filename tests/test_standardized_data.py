import json
from pathlib import Path

import pytest

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.memory.types import ConversationRecord
from app.normalization.backfill_names import date_block, parts_for_row
from app.normalization.date_expression import date_matches, recorded_event_count, shared_component
from app.normalization.date_parser import parse_date
from app.normalization.snapshot import released_csv_rows
from app.security.access_scope import AccessScope
from app.security.row_access import access_is_true
from benchmarks.standardize_exports import revise_row


@pytest.mark.asyncio
async def test_exact_repeat_reexecutes_validated_filter_without_model(monkeypatch):
    from app.config import get_settings
    from app.llm.fake import FakeReasoningProvider
    from app.planning.ai_planner import plan_user_turn_ai
    from app.planning.catalog import static_catalog
    from app.planning.conversation_resolver import ConversationContext
    from app.planning.turn_schema import ActiveQuery, FilterSpec

    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    provider = FakeReasoningProvider()
    scope = AccessScope(principal_id="test", allowed_file_ids=(49,))
    context = ConversationContext(
        active_question="Describe Alexander knegs",
        active_query=ActiveQuery(
            datasets=[49],
            goal="dossier",
            filters=[
                FilterSpec(
                    field="student_name", operator="EQUALS", value="Alexander Knaggs"
                )
            ],
        ),
    )
    result = await plan_user_turn_ai(
        "Describe Alexander knegs",
        scope,
        static_catalog(),
        selected_file_id=49,
        conversation=context,
        reasoner=provider,
    )
    assert result.status == "planned"
    assert result.planner_total_model_calls == 0
    assert result.turn_plan.verify_actions()
    assert any(
        predicate.value == "Alexander Knaggs"
        for step in result.plan.steps
        for predicate in step.where
    )
    assert not provider.structured_log
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1900-01-01 | 1901-01-01", None),
        ("1900-01-01 & 1901-01-01", None),
        ("1900-01-01 : 1901-01-01", None),
        ("1909-09-04-06", 1909.0),
    ],
)
def test_statistics_do_not_pick_first_year(value, expected):
    from app.execution.derivations import numeric_value

    assert numeric_value(value, "year") == expected
    assert numeric_value(value) == expected


@pytest.mark.parametrize("value,expected",[
    ("1900-05-01 : 1901-05-01", (None,None,None)),
    ("1900-05-01 : 1900-06-01", (1900,None,None)),
    ("1900-05-01 : 1900-05-14", (1900,5,None)),
    ("1900-06-01 : 1900-05-01", (None,None,None)),
])
def test_shared_components_describe_the_entire_range(value,expected):
    assert tuple(shared_component(value,part) for part in ("year","month","day")) == expected


def test_unknown_year_is_not_zero_in_numeric_date_statistics():
    from app.execution.derivations import numeric_value
    assert numeric_value("0000-09-00") is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1900-01-01 & 1901-01-01", 2),
        ("1900-01-01 | 1901-01-01", 1),
        ("1900-01-01 : 1901-01-01", 1),
        ("1909-09-04-06", 1),
        ("1900-01-01 & 1901-01-01 | 1902-01-01", 1),
        ("0000-00-00", 0),
        ("", 0),
    ],
)
def test_recorded_event_count_distinguishes_repeat_events_from_alternatives(
    value, expected
):
    assert recorded_event_count(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        ("TRUE", True),
        (" true ", True),
        (False, False),
        ("FALSE", False),
        ("yes", False),
        ("1", False),
        (1, False),
        (None, False),
        ("", False),
        ({"raw": "false", "normalized": "true"}, False),
        ({"raw": True}, False),
    ],
)
def test_only_explicit_true_releases_a_row(value, expected):
    assert access_is_true(value) is expected


def test_csv_gate_checks_permission_before_materializing_other_cells(tmp_path: Path):
    path = tmp_path / "Potential.csv"
    path.write_text(
        "ACCESS,First Name,__id\nFALSE,DO_NOT_MATERIALIZE,1\nTRUE,Allowed,2\n,Unknown,3\n",
        encoding="utf-8",
    )
    assert list(released_csv_rows(path, 94)) == [
        {"ACCESS": "TRUE", "First Name": "Allowed", "__id": "2"}
    ]
    path.write_text("First Name,__id\nMissing gate,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mandatory ACCESS"):
        list(released_csv_rows(path, 94))
    path.write_text("ACCESS,First Name,ACCESS\nTRUE,DO_NOT_MATERIALIZE,FALSE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate ACCESS"):
        list(released_csv_rows(path, 94))


def test_normalized_access_flag_cannot_release_a_row_without_current_source_permission():
    gateway = MemoryDataGateway(
        [
            {
                "file_id": 94,
                "row_data_normalized": {
                    "fields": {"ACCESS": {"raw": "TRUE"}},
                    "canonical": {"display_name": "STALE"},
                },
            }
        ]
    )
    assert gateway._records == []


@pytest.mark.asyncio
async def test_potential_catalog_never_reuses_cached_values_or_counts():
    from app.planning.catalog import FieldCatalog, FieldSpec

    catalog = FieldCatalog(
        fields=(
            FieldSpec(
                file_id=94,
                semantic_field="gender",
                human_label="Gender",
                validation_rules={
                    "recorded_values": ["STALE"],
                    "recorded_count": 24,
                    "record_count": 24,
                    "value_synonyms": {"female": ["F"]},
                },
            ),
        )
    )
    gateway = MemoryDataGateway([], catalog)
    fields = await gateway.get_dataset_fields(
        AccessScope(
            principal_id="test", allowed_file_ids=(94,), can_use_private_files=True
        ),
        94,
    )
    assert fields[0]["validation_rules"] == {"value_synonyms": {"female": ["F"]}}


@pytest.mark.asyncio
async def test_denied_rows_are_absent_from_gateway_storage_and_every_read():
    denied = {"file_id": 94, "id": 1, "source_row_id": 1, "raw": {"ACCESS": "FALSE"}}
    released = {
        "file_id": 94,
        "id": 2,
        "source_row_id": 2,
        "version": 2,
        "raw": {"ACCESS": "TRUE", "First Name": "Allowed"},
        "canonical_name": "Allowed",
        "row_data_normalized": {"canonical": {"display_name": "Allowed"}},
    }
    gateway = MemoryDataGateway([denied, released])
    assert gateway._records == [released]
    scope = AccessScope(
        principal_id="test", allowed_file_ids=(94,), can_use_private_files=True
    )
    assert await gateway.count_records(scope, (94,), []) == 1
    assert await gateway.get_records_by_source_ids(scope, [1]) == []
    assert await gateway.get_raw_fields(scope, [1], ["First Name"]) == []
    assert await gateway.suggest_name_matches(scope, (94,), "DO_NOT_MATERIALIZE") == []
    released["raw"]["ACCESS"] = "FALSE"
    assert await gateway.count_records(scope, (94,), []) == 0
    assert gateway._records == []


@pytest.mark.asyncio
async def test_revocation_invalidates_in_memory_answers_before_model_context():
    row = {"file_id": 94, "source_row_id": 1, "raw": {"ACCESS": "TRUE"}}
    gateway = MemoryDataGateway([row])
    conversation = ConversationRecord(
        id="test",
        principal_id="test",
        selected_file_id=94,
        row_access_revision=await gateway.potential_access_revision(),
        rolling_summary="old evidence",
        recent_turns=[
            {"role": "assistant", "text": "old evidence"},
            {"role": "user", "text": "what about this person?"},
        ],
    )
    row["raw"]["ACCESS"] = "FALSE"
    await AssistantTurnService(gateway)._refresh_potential_memory(conversation)
    assert conversation.rolling_summary == ""
    assert conversation.recent_turns == [
        {"role": "user", "text": "what about this person?"}
    ]


@pytest.mark.parametrize(
    "value,op,target,expected",
    [
        ("1900-00-00 | 1901-00-00", "YEAR_EQUALS", 1900, False),
        ("1900-01-01 | 1900-12-31", "YEAR_EQUALS", 1900, True),
        ("1900-00-00 & 1901-00-00", "YEAR_EQUALS", 1901, True),
        ("1900-00-00 & 1901-00-00", "YEAR_EQUALS", 1902, False),
        ("1900-00-00 | 1901-00-00", "DATE_RANGE", [1900, 1901], True),
        ("1900-00-00 | 0000-00-00", "YEAR_EQUALS", 1900, False),
        ("1900-00-00 & 0000-00-00", "YEAR_EQUALS", 1900, True),
        ("1900-06-15", "BEFORE", "1900-07-01", True),
        ("1900-06-15", "AFTER", "1900-05-01", True),
        ("1900-00-00", "BEFORE", "1900-07-01", False),
        ("1900-06-00", "DATE_RANGE", ["1900-06-01", "1900-06-30"], True),
        ("1900-06-00", "EQUALS", "1900-06-01", False),
        ("1900-06-15", "EQUALS", "1900-06-00", True),
        ("1900-06-15 | 1900-06-16", "NOT_EQUALS", "1900-06-15", False),
        ("1900-06-15 | 1900-06-16", "NOT_EQUALS", "1900-06-17", True),
        ("1901-00-00 : 1911-00-00", "YEAR_EQUALS", 1901, False),
        ("1901-00-00 : 1911-00-00", "DATE_RANGE", [1901, 1911], True),
        ("1901-00-00 : 1911-00-00", "DATE_RANGE", [1901, 1909], False),
        ("1911-00-00 : 1901-00-00", "DATE_RANGE", [1900, 1920], False),
        ("1909-09-04-06", "EQUALS", "1909-09-04", False),
        ("1909-09-04-06", "IN", ["1909-09-04", "1909-09-06"], True),
        ("1909-09-04-06", "YEAR_EQUALS", 1909, True),
        ("0000-00-00", "IS_UNKNOWN", None, True),
        ("0", "IS_UNKNOWN", None, True),
        ("", "IS_UNKNOWN", None, True),
        ("0000-09-00", "IS_KNOWN", None, True),
        ("1900-13-01", "IS_KNOWN", None, False),
        ("1900-00-15", "IS_UNKNOWN", None, True),
        ("1900-02-29", "IS_UNKNOWN", None, True),
        ("0000-02-31", "IS_UNKNOWN", None, True),
        ("0000-02-29", "IS_KNOWN", None, True),
        ("1900-01-01 : 1901-12-31 & 1902-01-01", "YEAR_EQUALS", 1902, True),
        ("1900-00-00 : 0000-00-00", "YEAR_EQUALS", 1900, False),
        ("1900-00-00 : 1901-13-00", "IS_UNKNOWN", None, True),
        ("1901-00-00 : 1911-00-00", "NOT_IN", [1900, 1901], False),
        ("1900-00-00", "NOT_EQUALS", "1900-01-01", False),
        ("1900-00-00", "NOT_EQUALS", "1901-01-01", True),
    ],
)
def test_date_conventions(value, op, target, expected):
    assert date_matches(value, op, target) is expected


@pytest.mark.parametrize("file_id", [49, 91, 93, 94])
def test_all_lists_parse_split_names_and_do_not_reuse_stale_canonical_names(file_id):
    row = {
        "row_data_normalized": {
            "fields": {
                "First Name": {"raw": "Ann"},
                "Middle Name": {"raw": "Jane"},
                "Last Name": {"raw": "Sample"},
            },
            "canonical": {"display_name": "STALE"},
        }
    }
    sources = {
        "first_name_source": ["fields.First Name"],
        "middle_names_source": ["fields.Middle Name"],
        "last_name_source": ["fields.Last Name"],
    }
    parts = parts_for_row(row, file_id, sources)
    assert (parts.first, parts.middle, parts.last) == ("Ann", ("Jane",), "Sample")


def test_backfill_keeps_expression_without_a_fabricated_canonical_year():
    row = {
        "row_data_normalized": {
            "fields": {"Admitted": {"raw": "1900-00-00 & 1901-00-00"}}
        }
    }
    block = date_block(row, ["admitted_date"], {"admitted_date": ["fields.Admitted"]})[
        "admitted_date"
    ]
    assert block["iso"] == "1900-00-00 & 1901-00-00"
    assert block["mode"] == "and"
    assert "year" not in block
    assert shared_component("1900-01-01 | 1900-12-31", "year") == 1900
    assert shared_component("1901-00-00 : 1911-00-00", "year") is None


def test_curator_shorthand_and_range_are_preserved_in_revised_copies():
    for value in ("1909-09-04-06", "1901-00-00 : 1911-00-00"):
        row, issues = revise_row({"First Name": "Ann", "DATE OF DEATH": value}, 93)
        assert row["DATE OF DEATH"] == value
        assert issues == []
    assert "alternative" in parse_date("1909-09-04-06").flags
    assert "range" in parse_date("1901-00-00 : 1911-00-00").flags


def test_partial_date_keeps_standardized_zero_components_in_canonical_value():
    row = {"row_data_normalized": {"fields": {"Date": {"raw": "0000-09-00"}}}}
    block = date_block(row, ["death_date"], {"death_date": ["fields.Date"]})
    assert block["death_date"]["iso"] == "0000-09-00"
    assert "year" not in block["death_date"]


def test_blank_raw_name_never_reuses_cached_normalized_name():
    row = {
        "row_data_normalized": {
            "fields": {"First Name": {"raw": "", "normalized": "STALE"}}
        }
    }
    parts = parts_for_row(row, 94, {"first_name_source": ["fields.First Name"]})
    assert not parts.first


def test_compound_names_remain_in_the_declared_standardized_slots():
    row = {
        "row_data_normalized": {
            "fields": {
                "First Name": {"raw": "Mary Ann"},
                "Middle Name": {"raw": "Jane"},
                "Last Name": {"raw": "Van Buren"},
                "Name comments": {"raw": "Middle Name: Joseph; Also known as: May"},
            }
        }
    }
    sources = {
        "first_name_source": ["fields.First Name"],
        "middle_names_source": ["fields.Middle Name"],
        "last_name_source": ["fields.Last Name"],
        "name_comments": ["fields.Name comments"],
    }
    parts = parts_for_row(row, 49, sources)
    assert (parts.first, parts.middle, parts.last) == (
        "Mary Ann",
        ("Jane",),
        "Van Buren",
    )


@pytest.mark.parametrize(
    "question,filters,direction,code",
    [
        (
            "Show all information for Albert Penance.",
            [{"field": "student_name", "operator": "CONTAINS", "value": "Albert"}],
            "asc",
            "name_contract",
        ),
        (
            "List the last 5 students alphabetically by full name.",
            [],
            "desc",
            "order_contract",
        ),
    ],
)
def test_explicit_request_contradictions_are_rejected(
    question, filters, direction, code
):
    from app.planning.ai_planner import _validate_explicit_request
    from app.planning.catalog import static_catalog
    from app.planning.plan_validator import PlanValidationError
    from app.planning.query_plan_builder import build_query_plan
    from app.planning.turn_schema import TurnPlan

    catalog = static_catalog()
    turn = TurnPlan(
        actions=[
            {
                "type": "query",
                "datasets": [49],
                "goal": "list",
                "filters": filters,
                "sort_by": "student_name",
                "sort_direction": direction,
            }
        ],
        final_response="deterministic",
        confidence=1,
    )
    plan = build_query_plan(
        turn,
        scope=AccessScope(principal_id="test", allowed_file_ids=(49,)),
        catalog=catalog,
    )
    with pytest.raises(PlanValidationError) as error:
        _validate_explicit_request(question, turn, plan, catalog)
    assert error.value.code == code


@pytest.mark.parametrize(
    "op,value,expected_op,expected_value",
    [
        ("EQUALS", 1901, "YEAR_EQUALS", 1901),
        ("GREATER_THAN", 1901, "AFTER", "1901-12-31"),
        ("LESS_THAN", 1901, "BEFORE", "1901-01-01"),
        ("NUMBER_RANGE", [1900, 1901], "DATE_RANGE", ["1900-01-01", "1901-12-31"]),
    ],
)
def test_event_year_filters_use_full_date_expression(
    op, value, expected_op, expected_value
):
    from app.planning.ai_catalog_coerce import coerce_planned_turn
    from app.planning.ai_turn_schema import PlannedTurn
    from app.planning.catalog import FieldCatalog, FieldSpec

    catalog = FieldCatalog(
        fields=(
            FieldSpec(
                file_id=49,
                semantic_field="admitted_date",
                human_label="Admitted",
                semantic_type="date",
            ),
            FieldSpec(
                file_id=49,
                semantic_field="admitted_date_year",
                human_label="Year",
                canonical_json_path="canonical.date_parts.admitted_date.year",
            ),
        )
    )
    planned = PlannedTurn(
        actions=[
            {
                "type": "query",
                "goal": "count",
                "filters": [
                    {"field": "admitted_date_year", "operator": op, "value": value}
                ],
            }
        ],
        final_response="deterministic",
    )
    spec = coerce_planned_turn(planned, catalog, 49).actions[0].filters[0]
    assert (spec.field, spec.operator, spec.value) == (
        "admitted_date",
        expected_op,
        expected_value,
    )


@pytest.mark.parametrize("recorded", ["Current Community", ""])
def test_standardized_community_replaces_stale_normalization(recorded):
    from app.normalization.backfill_names import apply_community, fold_label
    from app.normalization.name_parser import NameParts

    canonical = {"community": "Old Community", "community_source": "name_cell"}
    assert apply_community(canonical, recorded, NameParts(), derive=False, authoritative=True) == fold_label(recorded)
    assert canonical == {"community": recorded}


@pytest.mark.parametrize("operator", ["IN", "NOT_IN", "NOT_EQUALS", "NOT_CONTAINS"])
def test_gender_category_normalization_is_idempotent_and_preserves_negation(operator):
    from app.planning.ai_catalog_coerce import coerce_planned_turn
    from app.planning.ai_turn_schema import PlannedTurn
    from app.planning.catalog import static_catalog

    planned = PlannedTurn(actions=[{"type": "query", "goal": "count", "filters": [
        {"field": "gender", "operator": operator, "value": "female"}
    ]}], final_response="deterministic")
    catalog = static_catalog()
    first = coerce_planned_turn(planned, catalog, 91)
    assert coerce_planned_turn(first, catalog, 91) == first
    predicate = first.actions[0].filters[0]
    assert predicate.operator == ("NOT_IN" if operator.startswith("NOT_") else "IN")
    assert len(predicate.value) == len(set(predicate.value))


def test_name_search_projection_uses_display_name_in_action_and_requirements():
    from app.planning.ai_catalog_coerce import coerce_planned_turn
    from app.planning.ai_turn_schema import PlannedTurn
    from app.planning.catalog import FieldCatalog, FieldSpec

    planned = PlannedTurn(actions=[{"type": "query", "goal": "list", "requested_fields": ["name_search"]}],
        stated_requirements=[{"kind": "projection", "action_index": 0, "text": "student names",
                              "requested_fields": ["name_search"]}], final_response="deterministic")
    catalog = FieldCatalog(fields=(
        FieldSpec(file_id=49,semantic_field="student_name",human_label="Name"),
        FieldSpec(file_id=49,semantic_field="name_search",human_label="Name search",validation_rules={"token_index":True}),
    ))
    result = coerce_planned_turn(planned, catalog, 49)
    assert result.actions[0].requested_fields == ["student_name"]
    assert result.stated_requirements[0].requested_fields == ["student_name"]


def test_export_revision_keeps_editorial_source_note_out_of_the_name():
    row, issues = revise_row({"First Name": "Daniel (NCTR SOURCE)", "Middle Name": "",
                             "Last Name": "Sample"}, 93)
    assert row["First Name"] == "Daniel"
    assert row["Name comments"] == "Source note: NCTR SOURCE"
    assert json.loads(row["Source revision notes"])["First Name"] == "Daniel (NCTR SOURCE)"
    assert len(issues) == 1


@pytest.mark.asyncio
async def test_exact_duration_excludes_partial_and_uncertain_dates():
    from app.planning.catalog import static_catalog

    records = [{"file_id":49, "source_row_id":i,
                "row_data_normalized":{"canonical":{"display_name":f"Person {i}",
                    "dates":{"admitted":start,"discharged":"1901-01-01"}}}}
               for i,start in enumerate(["1900-01-01", "1900-00-00", "1900-06-00",
                   "0000-00-00", "1900-01-01 | 1900-02-01", "1900-01-01 & 1900-02-01"],1)]
    gateway = MemoryDataGateway(records=records, catalog=static_catalog())
    scope = AccessScope(principal_id="test",allowed_file_ids=(49,))
    rows = await gateway.interval_rows(scope,(49,),[],"admitted_date","discharged_date")
    assert [row.source_row_id for row in rows] == [1]
    assert rows[0].days == 365
    stats = await gateway.interval_stats(scope,(49,),[],"admitted_date","discharged_date")
    assert stats.record_count == 1


@pytest.mark.parametrize("question,convert", [
    ("How many distinct communities are recorded?",True),
    ("How many records per community?",False),
    ("How many different communities per school?",False),
])
def test_explicit_distinct_count_normalizes_one_selected_dimension(question,convert):
    from app.planning.ai_catalog_coerce import coerce_planned_turn, coerce_stated_requirements
    from app.planning.ai_turn_schema import PlannedTurn
    from app.planning.catalog import static_catalog

    planned = PlannedTurn(actions=[{"type":"query","goal":"count","group_by":["community"],
                                  "filters":[{"field":"deceased_status","operator":"IS_TRUE"}]}],
        stated_requirements=[{"kind":"goal","action_index":0,"text":"count","goal":"count"},
                             {"kind":"grouping","action_index":0,"text":"communities","field":"community"}],
        final_response="deterministic")
    result = coerce_planned_turn(planned,static_catalog(),49,question)
    assert result.actions[0].goal == ("aggregate" if convert else "count")
    assert result.actions[0].filters == planned.actions[0].filters
    assert coerce_stated_requirements(planned.stated_requirements,static_catalog(),49,question) == result.stated_requirements
    if convert:
        assert result.actions[0].aggregate.field == "community"
        assert result.actions[0].aggregate.function == "count_distinct"
        assert coerce_planned_turn(result,static_catalog(),49,question) == result


def test_all_source_attributes_expand_from_the_authorized_catalog():
    from app.planning.ai_catalog_coerce import (
        coerce_planned_turn,
        coerce_stated_requirements,
    )
    from app.planning.ai_turn_schema import PlannedTurn, ProjectionRequirement
    from app.planning.catalog import FieldCatalog, FieldSpec

    catalog = FieldCatalog(
        fields=(
            FieldSpec(file_id=94, semantic_field="student_name", human_label="Name"),
            FieldSpec(
                file_id=94,
                semantic_field="death_date",
                human_label="Death date",
                raw_json_keys=("DATE OF DEATH",),
            ),
            FieldSpec(
                file_id=94,
                semantic_field="private_helper",
                human_label="Internal",
                raw_json_keys=("SECRET",),
                exposure_policy="internal",
            ),
            FieldSpec(
                file_id=94, semantic_field="death_date_year", human_label="Derived year"
            ),
        )
    )
    projection = ProjectionRequirement(
        kind="projection",
        action_index=0,
        text="all information",
        requested_fields=["*"],
    )
    planned = PlannedTurn(
        actions=[{"type": "query", "goal": "list", "requested_fields": ["*"]}],
        stated_requirements=[projection],
        final_response="deterministic",
        confidence=1,
    )
    expected = ["student_name", "death_date"]
    assert (
        coerce_planned_turn(planned, catalog, 94).actions[0].requested_fields
        == expected
    )
    assert (
        coerce_stated_requirements([projection], catalog, 94)[0].requested_fields
        == expected
    )
    # An already-canonical action must still normalize its declarations before
    # reconciliation can fill a missing projection slot from them.
    declared_only = planned.model_copy(
        update={"actions": [planned.actions[0].model_copy(update={"requested_fields": ["student_name"]})]}
    )
    assert coerce_planned_turn(declared_only, catalog, 94).stated_requirements[0].requested_fields == expected


@pytest.mark.asyncio
async def test_restore_hash_numbers_keeps_the_active_distinct_query(monkeypatch):
    from app.config import get_settings
    from app.llm.fake import FakeReasoningProvider
    from app.planning.ai_planner import plan_user_turn_ai
    from app.planning.catalog import static_catalog
    from app.planning.conversation_resolver import ConversationContext
    from app.planning.turn_schema import ActiveQuery

    monkeypatch.setenv("PLANNER_MODE", "ai")
    get_settings.cache_clear()
    provider = FakeReasoningProvider()
    context = ConversationContext(
        active_query=ActiveQuery(
            datasets=[49],
            goal="distinct",
            group_by=["community"],
            group_value_part="base_name",
        )
    )
    result = await plan_user_turn_ai(
        "list them the hash numbers",
        AccessScope(principal_id="test", allowed_file_ids=(49,)),
        static_catalog(),
        selected_file_id=49,
        conversation=context,
        reasoner=provider,
    )
    assert result.status == "planned"
    assert result.planner_total_model_calls == 0
    assert result.turn_plan.modify_actions()[0].group_value_part == "raw"
    assert any(step.fields == ["community"] for step in result.plan.steps)
    get_settings.cache_clear()
