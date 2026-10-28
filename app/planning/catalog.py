from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_LABEL_WORD = re.compile(r"[a-z0-9]+")


class FieldSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: int
    semantic_field: str
    human_label: str
    aliases: tuple[str, ...] = ()
    canonical_json_path: str | None = None
    raw_json_keys: tuple[str, ...] = ()
    semantic_type: str = "text"
    allowed_operators: tuple[str, ...] = ()
    searchable: bool = True
    aggregatable: bool = False
    sortable: bool = False
    quoteable: bool = False
    sensitivity: str = "research"
    # 'authorized' fields appear in the planner's catalog; 'internal' ones are read by
    # retrieval and projected into rows but never advertised to the planner, so matching
    # aids do not enlarge the prompt. The column has existed since 004_assistant_tables
    # and was unused until the preprocessed name columns needed it.
    exposure_policy: str = "authorized"
    evidence_allowed: bool = True
    raw_fetch_allowed: bool = False
    validation_rules: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_id: int
    user_facing_label: str
    aliases: tuple[str, ...] = ()
    is_default_people_scope: bool = False
    private: bool = False
    history_enabled: bool = False
    attachment_enabled: bool = False


class FieldCatalog(BaseModel):
    """Planner-visible schema. Never includes application table names."""

    model_config = ConfigDict(frozen=True)

    datasets: tuple[DatasetSpec, ...] = Field(default_factory=tuple)
    fields: tuple[FieldSpec, ...] = Field(default_factory=tuple)
    default_people_file_id: int = 49

    def dataset(self, file_id: int) -> DatasetSpec | None:
        return next((item for item in self.datasets if item.file_id == file_id), None)

    def fields_for(self, file_id: int) -> list[FieldSpec]:
        return [item for item in self.fields if item.file_id == file_id]

    def planner_fields_for(self, file_id: int) -> list[FieldSpec]:
        """Fields the planner is told about, excluding internal matching aids."""
        return [item for item in self.fields_for(file_id) if item.exposure_policy != "internal"]

    def resolve_field(self, file_id: int, name: str) -> FieldSpec | None:
        needle = name.strip().lower()
        for spec in self.fields_for(file_id):
            if spec.semantic_field == needle or needle == spec.human_label.lower():
                return spec
            if needle in {alias.lower() for alias in spec.aliases}:
                return spec
        return None

    def resolve_dataset_ids(self, text: str) -> list[int]:
        lowered = text.lower()
        hits: list[int] = []
        for dataset in self.datasets:
            labels = dataset_match_labels(dataset)
            if any(alias.lower() in lowered for alias in labels if alias):
                hits.append(dataset.file_id)
        return hits

    def for_scope(self, scope: Any) -> FieldCatalog:
        """Planner-visible slice. Unauthorized and ungranted private datasets disappear."""
        allowed = []
        for dataset in self.datasets:
            private = dataset.private
            if not scope.allows_file(dataset.file_id, private=private):
                continue
            allowed.append(dataset)
        allowed_ids = {item.file_id for item in allowed}
        fields = tuple(item for item in self.fields if item.file_id in allowed_ids)
        default_id = self.default_people_file_id
        if default_id not in allowed_ids:
            default_id = next((item.file_id for item in allowed if item.is_default_people_scope), 0)
            if not default_id and allowed:
                default_id = allowed[0].file_id
        return FieldCatalog(
            datasets=tuple(allowed),
            fields=fields,
            default_people_file_id=default_id,
        )


def dataset_match_labels(dataset: DatasetSpec) -> tuple[str, ...]:
    """Return safe textual dataset mentions, including a descriptive suffix.

    Database-backed catalogs do not always carry optional aliases. A label such
    as ``Student master list`` should still match the natural shorthand
    ``master list`` without maintaining a second hardcoded routing table.
    """

    labels = [dataset.user_facing_label, *dataset.aliases, f"file {dataset.file_id}"]
    words = _LABEL_WORD.findall(dataset.user_facing_label.lower())
    if len(words) >= 3:
        labels.append(" ".join(words[1:]))
    if words and words[-1].endswith("s"):
        labels.append(" ".join([*words[:-1], words[-1][:-1]]))
    return tuple(dict.fromkeys(label for label in labels if label))


def static_catalog() -> FieldCatalog:
    """Full seeded registry (tests + fallback). Call for_scope() before planning."""
    datasets = (
        DatasetSpec(
            file_id=49,
            user_facing_label="Student master list",
            aliases=(
                "shingwauk and wawanosh students master list",
                "master list",
                "student master",
            ),
            is_default_people_scope=True,
        ),
        DatasetSpec(
            file_id=91,
            user_facing_label="Confirmed deaths",
            aliases=("confirmed", "confirmed- shingwauk", "confirmed deaths"),
        ),
        DatasetSpec(
            file_id=93,
            user_facing_label="Additional deaths",
            aliases=("additional deaths",),
            private=True,
        ),
        DatasetSpec(
            file_id=94,
            user_facing_label="Potential records",
            aliases=("potential",),
            private=True,
        ),
    )
    seeded_fields = (
        FieldSpec(
            file_id=49,
            semantic_field="student_name",
            human_label="Student name",
            aliases=("name", "student", "full name"),
            canonical_json_path="canonical.display_name",
            raw_json_keys=("Name", "Student Name"),
            allowed_operators=("EQUALS", "STARTS_WITH", "CONTAINS", "FULL_TEXT_SEARCH", "FUZZY_SEARCH"),
            sortable=True,
            quoteable=True,
            evidence_allowed=True,
        ),
        FieldSpec(
            file_id=49,
            semantic_field="community",
            human_label="Community",
            aliases=("reserve", "first nation", "community"),
            canonical_json_path="canonical.community",
            raw_json_keys=("Community",),
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "FUZZY_SEARCH"),
            aggregatable=True,
            sortable=True,
        ),
        FieldSpec(
            file_id=49,
            semantic_field="admitted_date",
            human_label="Admitted date",
            aliases=("admitted", "admission"),
            canonical_json_path="canonical.dates.admitted",
            raw_json_keys=("Admitted", "Date of Admission"),
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_UNKNOWN", "IS_KNOWN"),
            aggregatable=True,
            sortable=True,
        ),
        FieldSpec(
            file_id=49,
            semantic_field="discharged_date",
            human_label="Discharged date",
            aliases=("discharged", "discharge"),
            canonical_json_path="canonical.dates.discharged",
            raw_json_keys=("Discharged",),
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "IS_UNKNOWN", "IS_KNOWN"),
            aggregatable=True,
            sortable=True,
        ),
        FieldSpec(
            file_id=49,
            semantic_field="deceased_status",
            human_label="Deceased status",
            aliases=("deceased", "died", "death"),
            canonical_json_path="canonical.deceased_status",
            raw_json_keys=("Deceased",),
            semantic_type="boolean",
            allowed_operators=("IS_TRUE", "IS_FALSE"),
            aggregatable=True,
        ),
        FieldSpec(
            file_id=49,
            semantic_field="notes",
            human_label="Notes",
            aliases=("note", "remarks"),
            canonical_json_path="chat.narrative_bundle.notes",
            raw_json_keys=("Notes",),
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "GET_QUOTE"),
            quoteable=True,
            evidence_allowed=True,
            validation_rules={"article_strip_safe": True},
        ),
        FieldSpec(
            file_id=91,
            semantic_field="student_name",
            human_label="Name",
            aliases=("name",),
            canonical_json_path="canonical.display_name",
            raw_json_keys=("Name",),
            allowed_operators=("EQUALS", "STARTS_WITH", "CONTAINS", "FUZZY_SEARCH"),
            sortable=True,
            quoteable=True,
            evidence_allowed=True,
        ),
        FieldSpec(
            file_id=91,
            semantic_field="cause_of_death",
            human_label="Cause of death",
            aliases=("cause",),
            canonical_json_path="canonical.cause_of_death",
            raw_json_keys=("Cause of Death",),
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH"),
            quoteable=True,
            evidence_allowed=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="student_name",
            human_label="Student name",
            aliases=("name", "student", "full name"),
            canonical_json_path="canonical.display_name",
            semantic_type="text",
            allowed_operators=("EQUALS", "STARTS_WITH", "CONTAINS", "FULL_TEXT_SEARCH", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="indigenous_name",
            human_label="Indigenous name",
            aliases=("indian name", "spirit name", "indigenous name"),
            canonical_json_path="fields.INDIAN NAME",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="gender",
            human_label="Gender",
            aliases=("sex",),
            canonical_json_path="fields.GENDER",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="nation",
            human_label="Nation",
            aliases=("first nation", "band"),
            canonical_json_path="fields.NATION",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="school",
            human_label="School",
            aliases=("residential school",),
            canonical_json_path="canonical.school",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="birth_date",
            human_label="Date of birth",
            aliases=("born", "dob", "birth date"),
            canonical_json_path="canonical.dates.birth",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="death_date",
            human_label="Date of death",
            aliases=("died on", "death date"),
            canonical_json_path="canonical.dates.death",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="cause_of_death",
            human_label="Cause of death",
            aliases=("cause", "died of"),
            canonical_json_path="canonical.cause_of_death",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="age_at_death",
            human_label="Age at death",
            aliases=("age",),
            canonical_json_path="fields.AGE AT DEATH",
            semantic_type="number",
            allowed_operators=("EQUALS", "GREATER_THAN", "LESS_THAN", "NUMBER_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="parents_names",
            human_label="Parents",
            aliases=("parent", "mother", "father"),
            canonical_json_path="canonical.parents_names",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=False,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="first_admitted_date",
            human_label="Date first admitted",
            aliases=("admitted", "first admitted"),
            canonical_json_path="fields.DATE FIRST ADMITTED",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="reason_for_discharge",
            human_label="Reason for discharge",
            aliases=("discharge reason",),
            canonical_json_path="fields.REASON FOR DISCHARGE",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="other_schools",
            human_label="Other schools / Institutions attended",
            aliases=("other institutions",),
            canonical_json_path="fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="community",
            human_label="First Nation / Community",
            aliases=("reserve", "community"),
            canonical_json_path="canonical.community",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=93,
            semantic_field="student_number",
            human_label="Student number",
            aliases=("student no", "id number"),
            canonical_json_path="canonical.student_number",
            semantic_type="text",
            allowed_operators=("EQUALS", "STARTS_WITH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="student_name",
            human_label="Student name",
            aliases=("name", "student", "full name"),
            canonical_json_path="canonical.display_name",
            semantic_type="text",
            allowed_operators=("EQUALS", "STARTS_WITH", "CONTAINS", "FULL_TEXT_SEARCH", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="indigenous_name",
            human_label="Indigenous name",
            aliases=("indian name", "spirit name", "indigenous name"),
            canonical_json_path="fields.INDIAN NAME",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="gender",
            human_label="Gender",
            aliases=("sex",),
            canonical_json_path="fields.GENDER",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="nation",
            human_label="Nation",
            aliases=("first nation", "band"),
            canonical_json_path="fields.NATION",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="school",
            human_label="School",
            aliases=("residential school",),
            canonical_json_path="canonical.school",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="birth_date",
            human_label="Date of birth",
            aliases=("born", "dob", "birth date"),
            canonical_json_path="canonical.dates.birth",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="death_date",
            human_label="Date of death",
            aliases=("died on", "death date"),
            canonical_json_path="canonical.dates.death",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="cause_of_death",
            human_label="Cause of death",
            aliases=("cause", "died of"),
            canonical_json_path="canonical.cause_of_death",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="age_at_death",
            human_label="Age at death",
            aliases=("age",),
            canonical_json_path="fields.AGE AT DEATH",
            semantic_type="number",
            allowed_operators=("EQUALS", "GREATER_THAN", "LESS_THAN", "NUMBER_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="parents_names",
            human_label="Parents",
            aliases=("parent", "mother", "father"),
            canonical_json_path="canonical.parents_names",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=False,
            sortable=False,
            quoteable=True,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="first_admitted_date",
            human_label="Date first admitted",
            aliases=("admitted", "first admitted"),
            canonical_json_path="fields.DATE FIRST ADMITTED",
            semantic_type="date",
            allowed_operators=("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=False,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="reason_for_discharge",
            human_label="Reason for discharge",
            aliases=("discharge reason",),
            canonical_json_path="fields.REASON FOR DISCHARGE",
            semantic_type="entity",
            allowed_operators=("EQUALS", "IN", "CONTAINS", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
        FieldSpec(
            file_id=94,
            semantic_field="other_schools",
            human_label="Other schools / Institutions attended",
            aliases=("other institutions",),
            canonical_json_path="fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED",
            semantic_type="text",
            allowed_operators=("CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN"),
            aggregatable=True,
            sortable=True,
            quoteable=True,
        ),
    )
    # The compact hard-coded catalog is a development/test fallback. Keep it
    # aligned with the database registry so fallback mode never loses a public
    # field or accepts a different operator surface.
    by_key = {(item.file_id, item.semantic_field): item for item in seeded_fields}
    for item in _comprehensive_public_fields():
        by_key[(item.file_id, item.semantic_field)] = item
    return FieldCatalog(
        datasets=datasets,
        fields=tuple(by_key.values()),
        default_people_file_id=49,
    )


def _comprehensive_public_fields() -> tuple[FieldSpec, ...]:
    text_ops = ("EQUALS", "CONTAINS", "FULL_TEXT_SEARCH", "IS_KNOWN", "IS_UNKNOWN")
    entity_ops = ("EQUALS", "IN", "CONTAINS", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN")
    date_ops = ("EQUALS", "YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "IS_KNOWN", "IS_UNKNOWN")
    number_ops = ("EQUALS", "GREATER_THAN", "LESS_THAN", "NUMBER_RANGE", "IS_KNOWN", "IS_UNKNOWN")
    boolean_ops = ("IS_TRUE", "IS_FALSE", "IS_KNOWN", "IS_UNKNOWN")

    definitions: list[tuple[int, str, str, str, str, tuple[str, ...], bool, bool]] = [
        (49, "student_name", "Student name", "canonical.display_name", "name", ("name", "student", "full name"), False, True),
        (49, "first_name", "First name", "canonical.first_name", "text", ("first names", "given name"), False, True),
        (49, "middle_names", "Middle names", "canonical.middle_names", "text", ("middle name",), False, True),
        (49, "last_name", "Last name", "canonical.last_name", "text", ("last names", "surname"), False, True),
        (49, "indigenous_name", "Indigenous name / Spirit name", "canonical.indigenous_name", "text", ("indian name", "spirit name"), False, True),
        (49, "community", "First Nation / Community", "canonical.community", "entity", ("reserve", "first nation", "community"), True, True),
        (49, "admitted_date", "Admitted date", "canonical.dates.admitted", "date", ("admitted", "admission date"), True, True),
        (49, "discharged_date", "Discharged date", "canonical.dates.discharged", "date", ("discharged", "discharge date"), True, True),
        (49, "birth_date", "Date of birth", "canonical.dates.birth", "date", ("birth", "birth date", "born", "dob"), True, True),
        # Registry migration 057 retyped this as a category, because the recorded values are
        # yes, no and unknown -- a tri-state does not fit in a flag. The offline catalog still
        # says boolean, so it and production disagree about this one field.
        #
        # Aligning them is a one-line change here, and it breaks nine tests: the fake planner
        # and several legacy suites assert IS_TRUE on this field. Most of those tests are
        # deleted by Phase 3, so rewriting them now is work thrown away, and the divergence is
        # dev-only -- _planning_catalog falls back to this catalog only outside production.
        # Left deliberately, to be closed with the Phase 3 test migration.
        (49, "deceased_status", "Deceased status", "canonical.deceased_status", "boolean", ("deceased", "died"), True, True),
        (49, "age", "Age", "fields.Age", "number", ("student age",), True, True),
        (49, "student_number", "Student number", "canonical.student_number", "text", ("student id",), False, True),
        (49, "parents_names", "Parents names", "canonical.parents_names", "text", ("parents", "family"), True, False),
        (49, "siblings", "Siblings", "fields.Siblings", "text", ("brothers", "sisters"), False, False),
        (49, "notes", "Notes", "chat.narrative_bundle.notes", "text", ("note", "remarks"), False, False),
        (49, "death_details", "Death details", "fields.Death details", "text", ("death information",), False, False),
        (49, "additional_information", "Additional information", "fields.Additional Information", "text", ("additional info",), False, False),
        (49, "mapping_location", "Mapping location", "canonical.mapping_location", "entity", ("map location", "location"), True, True),
        (49, "latitude", "Latitude", "canonical.lat", "number", ("lat",), True, True),
        (49, "longitude", "Longitude", "canonical.lng", "number", ("lng", "lon"), True, True),
        (49, "photos", "Photos", "fields.Photos", "text", ("photo", "images"), False, False),
        (49, "documents", "Documents", "fields.Documents", "text", ("document", "files"), False, False),
        (91, "student_name", "Student name", "canonical.display_name", "name", ("name", "child name"), False, True),
        (91, "cause_of_death", "Cause of death", "canonical.cause_of_death", "entity", ("cause", "causes of death", "reason of death", "reasons of death"), True, True),
        (91, "death_factor", "Death factor", "fields.DEATH FACTOR", "entity", ("factor",), True, True),
        (91, "birth_date", "Date of birth", "canonical.dates.birth", "date", ("birth", "birth date", "born", "dob"), True, True),
        (91, "death_date", "Date of death", "canonical.dates.death", "date", ("death date", "died", "dod"), True, True),
        (91, "burial_date", "Date of burial", "fields.DATE OF BURIAL", "date", ("burial date", "buried"), True, True),
        (91, "first_admitted_date", "Date first admitted", "fields.DATE FIRST ADMITTED", "date", ("first admitted", "admission date"), True, True),
        (91, "discharged_date", "Date of discharge", "fields.DATE OF DISCHARGE", "date", ("discharged", "discharge date"), True, True),
        (91, "death_registration_date", "Date of death registration", "fields.DATE OF DEATH \nREGISTRATION", "date", ("registration date",), True, True),
        (91, "age_at_death", "Age at death", "fields.AGE AT DEATH", "number", ("age", "death age"), True, True),
        (91, "gender", "Gender", "fields.GENDER", "entity", ("sex",), True, True),
        (91, "nation", "Nation", "fields.NATION", "entity", ("first nation",), True, True),
        (91, "community", "Community / Reserve", "fields.COMMUNITY/RESERVE", "entity", ("reserve", "community"), True, True),
        (91, "school", "School", "canonical.school", "entity", ("school name",), True, True),
        (91, "place_of_death", "Place of death", "fields.PLACE OF DEATH", "entity", ("death place", "died at"), True, True),
        (91, "location_of_death", "Location of death", "fields.LOCATION OF DEATH", "entity", ("death location",), True, True),
        (91, "place_of_burial", "Place of burial", "fields.PLACE OF BURIAL", "entity", ("burial place", "cemetery"), True, True),
        (91, "student_number", "Student number", "canonical.student_number", "text", ("student id",), False, True),
        (91, "indigenous_name", "Indigenous name", "fields.INDIAN NAME", "text", ("indian name", "spirit name"), False, True),
        (91, "parents_names", "Parents", "canonical.parents_names", "text", ("parents names", "family"), True, False),
        (91, "ancestry_information", "Ancestry information", "fields.ANCESTRY INFORMATION", "text", ("ancestry",), False, False),
        (91, "admission_method", "Admission method", "fields.ADMISSION METHOD", "entity", ("how admitted",), True, True),
        (91, "reason_for_discharge", "Reason for discharge", "fields.REASON FOR DISCHARGE", "entity", ("discharge reason",), True, True),
        (91, "documentation", "Documentation", "fields.DOCUMENTATION", "text", ("documents", "evidence"), False, False),
        (91, "investigative_notes", "Investigative notes", "fields.INVESTIGATIVE NOTES", "text", ("investigation notes",), False, False),
        (91, "specific_file_notes", "Specific file notes / References", "fields.SPECIFIC FILE NOTES/REFERENCES\n", "text", ("file notes", "references"), False, False),
        (91, "information_shared_with", "Information shared with", "fields.INFORMATION SHARED WITH ?", "text", ("shared with",), False, True),
        (91, "death_registration_number", "Death registration number", "fields.DEATH REGISTRATION \nNUMBER", "text", ("registration number",), False, True),
        (91, "other_schools", "Other schools / Institutions attended", "fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED", "text", ("other institutions", "other schools"), True, True),
        (91, "other_links", "Other links", "fields.Other Links", "text", ("links", "urls"), False, False),
        (91, "other_information", "Other information", "fields.Other \n", "text", ("other", "additional information"), False, False),
        (91, "census_documents_used", "Census documents used", "fields.INDICATED IF CENSUS DOCUMENTS USED", "boolean", ("census used",), True, True),
        (91, "census_year", "Census year used or required", "fields.INDICATE CENSUS YEAR USED OR REQUIRED", "number", ("census year",), True, True),
        (91, "nia_comments", "Nia comments", "fields.NIA Comments", "text", ("comments",), False, False),
    ]
    # Mirrors assistant.field_registry.validation_rules.value_synonyms so the
    # offline fallback catalog expands the same historical vocabulary as production.
    synonyms: dict[tuple[int, str], dict[str, list[str]]] = {
        (91, "cause_of_death"): {
            "tuberculosis": ["tuberculosis", "consumption", "phthisis", "scrofula", "tubercular"],
            "pneumonia": ["pneumonia"],
            "typhoid fever": ["typhoid"],
            "drowning": ["drowning", "drowned"],
        },
        (91, "location_of_death"): {
            "school": ["school"],
            "hospital": ["hospital", "sanitorium", "sanatorium", "sanitarium"],
            "sanatorium": ["hospital", "sanitorium", "sanatorium", "sanitarium"],
            "community": ["community"],
            "home community": ["community"],
        },
        (91, "place_of_burial"): {
            "shingwauk cemetery": ["shingwauk cemetery", "shingwauak cemetery"],
        },
    }
    result = []
    for file_id, semantic, label, path, kind, aliases, aggregate, sortable in definitions:
        semantic_type = "text" if kind == "name" else kind
        if kind == "name":
            operators = ("EQUALS", "STARTS_WITH", "CONTAINS", "FULL_TEXT_SEARCH", "FUZZY_SEARCH", "IS_KNOWN", "IS_UNKNOWN")
        elif kind == "entity":
            operators = entity_ops
        elif kind == "date":
            operators = date_ops
        elif kind == "number":
            operators = number_ops
        elif kind == "boolean":
            operators = boolean_ops
        else:
            operators = text_ops
        quoteable = semantic in {
            "student_name", "notes", "death_details", "additional_information",
            "cause_of_death", "documentation", "investigative_notes", "specific_file_notes",
            "ancestry_information", "other_information", "nia_comments",
        }
        if quoteable and "GET_QUOTE" not in operators:
            operators = (*operators, "GET_QUOTE")
        result.append(
            FieldSpec(
                file_id=file_id,
                semantic_field=semantic,
                human_label=label,
                aliases=aliases,
                canonical_json_path=path,
                semantic_type=semantic_type,
                allowed_operators=operators,
                searchable=True,
                aggregatable=aggregate,
                sortable=sortable,
                quoteable=quoteable,
                evidence_allowed=True,
                validation_rules=_static_validation_rules(file_id, semantic, synonyms),
            )
        )
    return tuple(result)


_NOTE_FIELDS = frozenset(
    {
        "notes",
        "investigative_notes",
        "specific_file_notes",
        "remarks",
        "comments",
        "additional_notes",
        "narrative",
        "nia_comments",
        "additional_information",
        "other_information",
    }
)


def _static_validation_rules(
    file_id: int, semantic: str, synonyms: dict[tuple[int, str], dict[str, list[str]]]
) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    if (file_id, semantic) in synonyms:
        rules["value_synonyms"] = synonyms[(file_id, semantic)]
    if semantic in _NOTE_FIELDS:
        rules["article_strip_safe"] = True
    return rules


def catalog_from_rows(
    datasets: Iterable[dict[str, Any]],
    fields: Iterable[dict[str, Any]],
    default_people_file_id: int = 49,
) -> FieldCatalog:
    parsed_datasets = []
    for row in datasets:
        parsed_datasets.append(
            DatasetSpec(
                file_id=int(row["file_id"]),
                user_facing_label=row.get("user_facing_label") or row.get("filename") or str(row["file_id"]),
                aliases=tuple(row.get("aliases") or ()),
                is_default_people_scope=bool(row.get("is_default_people_scope")),
                private=bool(row.get("private")),
            )
        )
    parsed_fields = []
    for row in fields:
        parsed_fields.append(
            FieldSpec(
                file_id=int(row["file_id"]),
                semantic_field=row["semantic_field"],
                human_label=row["human_label"],
                aliases=tuple(row.get("aliases") or ()),
                canonical_json_path=row.get("canonical_json_path"),
                raw_json_keys=tuple(row.get("raw_json_keys") or ()),
                semantic_type=row.get("semantic_type") or "text",
                allowed_operators=tuple(row.get("allowed_operators") or ()),
                searchable=bool(row.get("searchable", True)),
                aggregatable=bool(row.get("aggregatable", False)),
                sortable=bool(row.get("sortable", False)),
                quoteable=bool(row.get("quoteable", False)),
                sensitivity=str(row.get("sensitivity") or "research"),
                exposure_policy=str(row.get("exposure_policy") or "authorized"),
                evidence_allowed=bool(row.get("evidence_allowed", True)),
                raw_fetch_allowed=bool(row.get("raw_fetch_allowed", False)),
                validation_rules=dict(row.get("validation_rules") or {}),
            )
        )
    return FieldCatalog(
        datasets=tuple(parsed_datasets),
        fields=tuple(parsed_fields),
        default_people_file_id=default_people_file_id,
    )
