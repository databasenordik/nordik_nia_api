-- Tell the planner what a low-cardinality field actually contains.
--
-- "How many potential records are from Garden River?" on file 94 compiled to
-- nation CONTAINS 'Garden River' and answered "0 records". The planner's reasoning was
-- sound given what the catalog told it: the field is labelled "Nation" with the alias
-- "first nation", and Garden River is a First Nation. What the catalog did not say is that
-- nation on file 94 is empty in 53 of 56 rows and its only recorded values are
-- "community member", "not confirmed" and "Shingwauk".
--
-- So the field cannot answer the question, and "0 records" is not a small error -- it reads
-- as "no potential records are from Garden River" when the truth is that this list does not
-- record where anyone is from. The prompt already forbids exactly this ("a confident zero
-- from the wrong field is worse than saying the list does not record it") and instructs the
-- planner to judge from "label, aliases, type, and value_families". The rule was right and
-- the evidence was missing.
--
-- This publishes the recorded values for fields that have few enough of them to enumerate,
-- which is where substitution errors happen -- an enum-like column looks plausible from its
-- label alone. High-cardinality columns (names, notes, dates) are excluded: listing them
-- would be both useless and enormous.
--
-- It stays in the catalog rather than the prompt, so no dataset vocabulary is hardcoded.
--
-- Counts today: 2 such fields on 49, 11 on 91, 9 on 93, 8 on 94.

CREATE OR REPLACE FUNCTION assistant.refresh_recorded_values(p_max_values integer DEFAULT 12)
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
DECLARE
    touched INTEGER := 0;
BEGIN
    -- Recomputed from the data, so re-running after an ingest keeps it honest. A field that
    -- grows past the cap loses the key rather than keeping a stale list.
    WITH candidate AS (
        SELECT r.id, r.file_id, r.canonical_json_path
        FROM assistant.field_registry r
        WHERE r.canonical_json_path IS NOT NULL
          AND r.semantic_type <> 'number'
          AND r.canonical_json_path NOT LIKE 'canonical.name_parts.%'
          AND r.canonical_json_path NOT LIKE 'canonical.date_parts.%'
          AND r.canonical_json_path <> 'names'
    ),
    valued AS (
        SELECT c.id,
               array_agg(DISTINCT left(v.value, 60) ORDER BY left(v.value, 60)) AS values,
               count(DISTINCT v.value) AS n
        FROM candidate c
        JOIN assistant_api.v_current_records rec ON rec.file_id = c.file_id
        CROSS JOIN LATERAL (
            SELECT NULLIF(btrim(assistant_api._field_display_at_path(
                rec.row_data_normalized, rec.canonical_name,
                rec.canonical_community, rec.canonical_school, c.canonical_json_path
            )), '') AS value
        ) v
        WHERE v.value IS NOT NULL
        GROUP BY c.id
    ),
    applied AS (
        UPDATE assistant.field_registry r
        SET validation_rules = CASE
                WHEN valued.n IS NOT NULL AND valued.n <= p_max_values
                THEN COALESCE(r.validation_rules, '{}'::jsonb)
                     || jsonb_build_object('recorded_values', to_jsonb(valued.values))
                ELSE COALESCE(r.validation_rules, '{}'::jsonb) - 'recorded_values'
            END
        FROM candidate c
        LEFT JOIN valued ON valued.id = c.id
        WHERE r.id = c.id
        RETURNING r.id
    )
    SELECT count(*) INTO touched FROM applied;
    RETURN touched;
END;
$function$;

SELECT assistant.refresh_recorded_values();
