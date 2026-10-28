import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.planning.catalog import DatasetSpec, FieldCatalog, FieldSpec
from app.planning.plan_schema import FilterOperator, PlanOp
from app.planning.schema_query import parse_schema_query


def _field(
    name: str,
    label: str,
    *,
    aliases: tuple[str, ...] = (),
    semantic_type: str = "text",
    operators: tuple[str, ...] = ("EQUALS", "CONTAINS", "IS_KNOWN", "IS_UNKNOWN"),
    aggregatable: bool = True,
    sortable: bool = True,
    value_aliases: dict[str, str] | None = None,
) -> FieldSpec:
    return FieldSpec(
        file_id=7,
        semantic_field=name,
        human_label=label,
        aliases=aliases,
        semantic_type=semantic_type,
        allowed_operators=operators,
        aggregatable=aggregatable,
        sortable=sortable,
        validation_rules={"value_aliases": value_aliases or {}},
    )


def _catalog() -> FieldCatalog:
    return FieldCatalog(
        datasets=(DatasetSpec(file_id=7, user_facing_label="People records"),),
        fields=(
            _field(
                "student_name",
                "Student name",
                aliases=("name", "student"),
                operators=("EQUALS", "CONTAINS", "STARTS_WITH", "IS_KNOWN", "IS_UNKNOWN"),
                aggregatable=False,
            ),
            _field(
                "community",
                "Community",
                aliases=("reserve",),
            ),
            _field("first_name", "First name", aliases=("given name",)),
            _field("last_name", "Last name", aliases=("surname",)),
            _field("photos", "Photos", aliases=("photo", "image"), aggregatable=False, sortable=False),
            _field(
                "deceased_status",
                "Deceased status",
                aliases=("deceased", "died"),
                semantic_type="boolean",
                operators=("EQUALS", "IS_TRUE", "IS_FALSE", "IS_KNOWN", "IS_UNKNOWN"),
            ),
            _field(
                "gender",
                "Gender",
                aliases=("sex",),
                value_aliases={"female": "F", "male": "M"},
            ),
            _field(
                "place_of_death",
                "Place of death",
                aliases=("death place",),
            ),
            _field(
                "death_date",
                "Date of death",
                aliases=("death date", "died"),
                semantic_type="date",
                operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            ),
            _field("reason_for_discharge", "Reason for discharge", aliases=("discharge reason",)),
            _field("information_shared_with", "Information shared with", aliases=("shared with",)),
        ),
        default_people_file_id=7,
    )


def _plan(question: str):
    plan = parse_schema_query(question, file_id=7, catalog=_catalog())
    assert plan is not None
    return plan


def _filters(plan):
    return [predicate for step in plan.steps for predicate in step.where]


def test_unknown_literal_binds_to_following_field_not_record_noun():
    filters = _filters(_plan("How many students have an unknown deceased status?"))
    assert [(item.field, item.operator, item.value) for item in filters] == [
        ("deceased_status", FilterOperator.EQUALS, "unknown")
    ]


def test_missing_modifier_binds_to_nearest_following_field():
    filters = _filters(_plan("How many student records are missing a community?"))
    assert [(item.field, item.operator) for item in filters] == [
        ("community", FilterOperator.IS_UNKNOWN)
    ]


def test_registered_value_alias_becomes_an_equality_filter():
    filters = _filters(_plan("How many records are female?"))
    assert [(item.field, item.operator, item.value) for item in filters] == [
        ("gender", FilterOperator.EQUALS, "F")
    ]


def test_compound_field_head_plural_is_recognized():
    filters = _filters(_plan("How many places of death contain Shingwauk?"))
    assert [(item.field, item.operator, item.value) for item in filters] == [
        ("place_of_death", FilterOperator.CONTAINS, "shingwauk")
    ]


def test_requested_columns_are_not_converted_to_existence_filters():
    plan = _plan("List the first 5 students with their names and communities.")
    assert _filters(plan) == []
    assert plan.step_map()["sorted"].sort_field == "student_name"
    assert plan.step_map()["result"].fields == ["student_name", "community"]


def test_chronological_list_sorts_by_the_registered_date_field():
    plan = _plan("List the first 5 deaths chronologically with name and death date.")
    assert plan.step_map()["sorted"].sort_field == "death_date"
    assert plan.step_map()["sorted"].sort_direction == "ASC"


def test_date_distribution_can_group_by_year_without_question_specific_logic():
    plan = _plan("Which death year occurs most often?")
    grouped = next(step for step in plan.steps if step.op == PlanOp.GROUP_BY)
    assert grouped.fields == ["death_date"]
    assert grouped.value_part == "year"


def test_scalar_value_after_field_becomes_exact_equality():
    filters = _filters(_plan("How many students have the first name John?"))
    assert [(item.field, item.operator, item.value) for item in filters] == [
        ("first_name", FilterOperator.EQUALS, "john")
    ]


def test_scalar_value_before_as_field_becomes_exact_equality():
    filters = _filters(_plan("How many students have Williams as their last name?"))
    assert [(item.field, item.operator, item.value) for item in filters] == [
        ("last_name", FilterOperator.EQUALS, "williams")
    ]


def test_presence_and_absence_verbs_bind_to_registered_field():
    present = _filters(_plan("How many records include a photo reference?"))
    absent = _filters(_plan("How many students do not have a photo recorded?"))
    assert [(item.field, item.operator) for item in present] == [("photos", FilterOperator.IS_KNOWN)]
    assert [(item.field, item.operator) for item in absent] == [("photos", FilterOperator.IS_UNKNOWN)]


def test_generic_indefinite_pronoun_means_presence_not_literal_equality():
    filters = _filters(_plan("How many records say information was shared with someone?"))
    assert [(item.field, item.operator) for item in filters] == [
        ("information_shared_with", FilterOperator.IS_KNOWN)
    ]


def test_value_like_date_verb_is_not_treated_as_a_date_equality():
    filters = _filters(_plan("How many records have Died as the reason for discharge?"))
    assert ("reason_for_discharge", FilterOperator.EQUALS, "died") in [
        (item.field, item.operator, item.value) for item in filters
    ]
    assert not any(item.field == "death_date" for item in filters)
