-- Domain vocabulary lives in the field registry, not in application code.
--
-- The golden questions use historical and colloquial wording that the stored values
-- never spell the same way: tuberculosis is recorded as consumption, phthisis, or
-- scrofula; "a hospital or sanatorium" is stored as Sanitorium/Hospital; "died at the
-- school" is School or School (NCTR SOURCE). Encoding those families in Python would
-- hardcode one dataset's vocabulary into the planner, so they are catalog metadata
-- that the planner reads and expands into CONTAINS_ANY / NOT_CONTAINS_ANY filters.

-- Expose validation_rules to the planner-visible catalog view.
CREATE OR REPLACE VIEW assistant_api.v_dataset_fields AS
SELECT
    r.file_id,
    r.semantic_field,
    r.human_label,
    r.aliases,
    r.canonical_json_path,
    r.raw_json_keys,
    r.semantic_type,
    r.allowed_operators,
    r.searchable,
    r.aggregatable,
    r.sortable,
    r.quoteable,
    r.sensitivity,
    r.exposure_policy,
    r.validation_rules
FROM assistant.field_registry r;

CREATE OR REPLACE FUNCTION assistant.set_value_synonyms(
    p_file_id INTEGER,
    p_semantic_field TEXT,
    p_synonyms JSONB
)
RETURNS VOID
LANGUAGE sql
AS $$
    UPDATE assistant.field_registry
    SET validation_rules = jsonb_set(
        COALESCE(validation_rules, '{}'::jsonb), '{value_synonyms}', p_synonyms, TRUE
    )
    WHERE file_id = p_file_id AND semantic_field = p_semantic_field;
$$;

SELECT assistant.set_value_synonyms(91, 'cause_of_death', '{
  "tuberculosis": ["tuberculosis", "consumption", "phthisis", "scrofula", "tubercular"],
  "pneumonia": ["pneumonia"],
  "typhoid fever": ["typhoid"],
  "drowning": ["drowning", "drowned"],
  "meningitis": ["meningitis"],
  "unknown": ["unknown"]
}'::jsonb);

SELECT assistant.set_value_synonyms(91, 'location_of_death', '{
  "school": ["school"],
  "hospital": ["hospital", "sanitorium", "sanatorium", "sanitarium"],
  "sanatorium": ["hospital", "sanitorium", "sanatorium", "sanitarium"],
  "community": ["community"],
  "home community": ["community"]
}'::jsonb);

SELECT assistant.set_value_synonyms(91, 'place_of_burial', '{
  "shingwauk cemetery": ["shingwauk cemetery", "shingwauak cemetery"],
  "sault ste. marie": ["shingwauk cemetery", "shingwauak cemetery", "sault ste", "sault st"]
}'::jsonb);

SELECT assistant.set_value_synonyms(93, 'cause_of_death', '{
  "tuberculosis": ["tuberculosis", "consumption", "phthisis", "scrofula", "tubercular"]
}'::jsonb);

-- Natural wording the golden questions use that the registry did not carry.
UPDATE assistant.field_registry SET aliases = ARRAY[
    'burial place', 'buried at', 'buried in', 'cemetery', 'burial location',
    'where buried', 'place of burial'
] WHERE file_id = 91 AND semantic_field = 'place_of_burial';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'death location', 'died at', 'at school', 'where they died', 'location of death',
    'death setting'
] WHERE file_id = 91 AND semantic_field = 'location_of_death';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'cause', 'causes of death', 'reason of death', 'reasons of death', 'died from',
    'died of', 'cause of death'
] WHERE file_id = 91 AND semantic_field = 'cause_of_death';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'death information', 'death notes', 'details about their death',
    'information about their death', 'death details'
] WHERE file_id = 49 AND semantic_field = 'death_details';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'brothers', 'sisters', 'brothers or sisters', 'brothers and sisters', 'siblings'
] WHERE file_id = 49 AND semantic_field = 'siblings';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'parents', 'parent names', 'family', 'mother', 'father', 'parents names'
] WHERE file_id IN (49, 91) AND semantic_field = 'parents_names';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'lat', 'map coordinates', 'coordinates', 'map coordinate', 'latitude'
] WHERE file_id = 49 AND semantic_field = 'latitude';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'lng', 'lon', 'map coordinates', 'coordinates', 'map coordinate', 'longitude'
] WHERE file_id = 49 AND semantic_field = 'longitude';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'first names', 'given name', 'given names', 'first name'
] WHERE file_id = 49 AND semantic_field = 'first_name';

UPDATE assistant.field_registry SET aliases = ARRAY[
    'last names', 'surname', 'family name', 'last name'
] WHERE file_id = 49 AND semantic_field = 'last_name';

-- Negative and multi-term operators are now implemented, so the fields the questions
-- filter on must advertise them.
UPDATE assistant.field_registry
SET allowed_operators = (
    SELECT ARRAY(SELECT DISTINCT item FROM unnest(
        allowed_operators || ARRAY['CONTAINS', 'CONTAINS_ANY', 'NOT_CONTAINS',
                                   'NOT_CONTAINS_ANY', 'NOT_EQUALS', 'NOT_IN',
                                   'IS_KNOWN', 'IS_UNKNOWN']
    ) item)
)
WHERE semantic_type IN ('entity', 'text');

UPDATE assistant.field_registry
SET allowed_operators = (
    SELECT ARRAY(SELECT DISTINCT item FROM unnest(
        allowed_operators || ARRAY['IS_KNOWN', 'IS_UNKNOWN', 'NOT_EQUALS']
    ) item)
)
WHERE semantic_type IN ('date', 'number', 'boolean');

DROP FUNCTION assistant.set_value_synonyms(INTEGER, TEXT, JSONB);
