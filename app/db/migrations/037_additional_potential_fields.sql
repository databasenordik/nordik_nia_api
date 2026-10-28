-- Bring Additional Deaths (93) and Potential (94) up to the same semantic
-- coverage that migration 024 gave the Master (49) and Confirmed (91) lists.
-- Both share the 24-column deaths-record raw schema, so the canonical/fields
-- paths mirror file 91.
--
-- Deliberate asymmetry, taken from the inspected database: file 93 normalizes
-- communities and identifiers; file 94 does NOT. Registering community or
-- student_number for 94 would advertise a field that dataset cannot answer, so
-- they are omitted rather than resolving to unknown for every row.
--
-- This registers semantics only. It grants no table access: the runtime still
-- reads rows through assistant_api, and 93/94 remain grant_required.

INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
SELECT * FROM (VALUES
    -- ---------- 93 Additional Deaths ----------
    (93, 'student_name', 'Student name', ARRAY['name','student','full name'], 'canonical.display_name', ARRAY['STUDENT NAME','Student Name'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','FULL_TEXT_SEARCH','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (93, 'indigenous_name', 'Indigenous name', ARRAY['indian name','spirit name','indigenous name'], 'fields.INDIAN NAME', ARRAY['INDIAN NAME'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (93, 'gender', 'Gender', ARRAY['sex'], 'fields.GENDER', ARRAY['GENDER'], 'entity', ARRAY['EQUALS','IN','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'nation', 'Nation', ARRAY['first nation','band'], 'fields.NATION', ARRAY['NATION'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'community', 'First Nation / Community', ARRAY['reserve','community'], 'canonical.community', ARRAY['COMMUNITY/RESERVE'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'school', 'School', ARRAY['residential school'], 'canonical.school', ARRAY['SCHOOL'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'birth_date', 'Date of birth', ARRAY['born','dob','birth date'], 'canonical.dates.birth', ARRAY['DATE OF BIRTH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'death_date', 'Date of death', ARRAY['died on','death date'], 'canonical.dates.death', ARRAY['DATE OF DEATH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'cause_of_death', 'Cause of death', ARRAY['cause','died of'], 'canonical.cause_of_death', ARRAY['CAUSE OF DEATH'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (93, 'age_at_death', 'Age at death', ARRAY['age'], 'fields.AGE AT DEATH', ARRAY['AGE AT DEATH'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'parents_names', 'Parents', ARRAY['parent','mother','father'], 'canonical.parents_names', ARRAY['PARENTS'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (93, 'student_number', 'Student number', ARRAY['student no','id number'], 'canonical.student_number', ARRAY['STUDENT NUMBER'], 'text', ARRAY['EQUALS','STARTS_WITH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, FALSE, TRUE, FALSE),
    (93, 'first_admitted_date', 'Date first admitted', ARRAY['admitted','first admitted'], 'fields.DATE FIRST ADMITTED', ARRAY['DATE FIRST ADMITTED'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (93, 'reason_for_discharge', 'Reason for discharge', ARRAY['discharge reason'], 'fields.REASON FOR DISCHARGE', ARRAY['REASON FOR DISCHARGE'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (93, 'other_schools', 'Other schools / Institutions attended', ARRAY['other institutions'], 'fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED', ARRAY['OTHER SCHOOLS/INSTITUTIONS ATTENDED'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),

    -- ---------- 94 Potential ----------
    -- No community/student_number: this dataset does not normalize them.
    (94, 'student_name', 'Student name', ARRAY['name','student','full name'], 'canonical.display_name', ARRAY['STUDENT NAME','Student Name'], 'text', ARRAY['EQUALS','STARTS_WITH','CONTAINS','FULL_TEXT_SEARCH','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (94, 'indigenous_name', 'Indigenous name', ARRAY['indian name','spirit name','indigenous name'], 'fields.INDIAN NAME', ARRAY['INDIAN NAME'], 'text', ARRAY['EQUALS','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, TRUE, TRUE, TRUE, FALSE),
    (94, 'gender', 'Gender', ARRAY['sex'], 'fields.GENDER', ARRAY['GENDER'], 'entity', ARRAY['EQUALS','IN','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'nation', 'Nation', ARRAY['first nation','band'], 'fields.NATION', ARRAY['NATION'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'school', 'School', ARRAY['residential school'], 'canonical.school', ARRAY['SCHOOL'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FUZZY_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'birth_date', 'Date of birth', ARRAY['born','dob','birth date'], 'canonical.dates.birth', ARRAY['DATE OF BIRTH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'death_date', 'Date of death', ARRAY['died on','death date'], 'canonical.dates.death', ARRAY['DATE OF DEATH'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'cause_of_death', 'Cause of death', ARRAY['cause','died of'], 'canonical.cause_of_death', ARRAY['CAUSE OF DEATH'], 'entity', ARRAY['EQUALS','IN','CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (94, 'age_at_death', 'Age at death', ARRAY['age'], 'fields.AGE AT DEATH', ARRAY['AGE AT DEATH'], 'number', ARRAY['EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'parents_names', 'Parents', ARRAY['parent','mother','father'], 'canonical.parents_names', ARRAY['PARENTS'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE, TRUE, FALSE),
    (94, 'first_admitted_date', 'Date first admitted', ARRAY['admitted','first admitted'], 'fields.DATE FIRST ADMITTED', ARRAY['DATE FIRST ADMITTED'], 'date', ARRAY['EQUALS','YEAR_EQUALS','BEFORE','AFTER','DATE_RANGE','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE, TRUE, FALSE),
    (94, 'reason_for_discharge', 'Reason for discharge', ARRAY['discharge reason'], 'fields.REASON FOR DISCHARGE', ARRAY['REASON FOR DISCHARGE'], 'entity', ARRAY['EQUALS','IN','CONTAINS','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE),
    (94, 'other_schools', 'Other schools / Institutions attended', ARRAY['other institutions'], 'fields.OTHER SCHOOLS/INSTITUTIONS ATTENDED', ARRAY['OTHER SCHOOLS/INSTITUTIONS ATTENDED'], 'text', ARRAY['CONTAINS','FULL_TEXT_SEARCH','IS_KNOWN','IS_UNKNOWN'], TRUE, TRUE, TRUE, TRUE, TRUE, FALSE)
) AS v(
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
ON CONFLICT (file_id, semantic_field) DO UPDATE SET
    human_label = EXCLUDED.human_label,
    aliases = EXCLUDED.aliases,
    canonical_json_path = EXCLUDED.canonical_json_path,
    raw_json_keys = EXCLUDED.raw_json_keys,
    semantic_type = EXCLUDED.semantic_type,
    allowed_operators = EXCLUDED.allowed_operators,
    searchable = EXCLUDED.searchable,
    aggregatable = EXCLUDED.aggregatable,
    sortable = EXCLUDED.sortable,
    quoteable = EXCLUDED.quoteable,
    evidence_allowed = EXCLUDED.evidence_allowed,
    raw_fetch_allowed = EXCLUDED.raw_fetch_allowed;

-- Match the labels used by the inspected database so dataset naming resolves.
UPDATE assistant.dataset_policy
SET user_facing_label = 'Additional Deaths', updated_at = now()
WHERE file_id = 93;

UPDATE assistant.dataset_policy
SET user_facing_label = 'Potential', updated_at = now()
WHERE file_id = 94;
