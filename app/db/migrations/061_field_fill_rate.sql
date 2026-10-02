-- Tell the planner how much of each field is actually recorded.
--
-- "How many potential records are from Garden River?" binds to nation and answers
-- "0 records". On file 94 nation is filled in 3 rows of 56, and those three say "community
-- member", "not confirmed" and "Shingwauk" -- it has never held a place of origin. The zero
-- reads as "no potential records are from Garden River" when the truth is that this list
-- does not record where anyone is from. On residential-school records that distinction is
-- the whole point.
--
-- The planner had no way to know. A field's label and aliases describe what it is *for*;
-- nothing said whether anyone ever filled it in. Publishing the fill rate gives it the one
-- fact that separates "the field that holds this" from "the field that sounds like it".
--
-- Deliberately a count and not a list of values. An earlier attempt published each field's
-- recorded values, which fixed this same case and broke four others: shown a field's
-- categories, the planner began filtering on them unprompted, so "how many records are in
-- the confirmed deaths list" came back 25 of 82. A proportion carries the evidence without
-- offering anything to filter on.
--
-- Fill rate is about choosing which field holds a value the user named. It is not a reason
-- to refuse a question about the field itself: "how many potential records have a recorded
-- cause of death" is a fair question whose true answer is 2, and an empty field answering
-- zero is correct.

CREATE OR REPLACE FUNCTION assistant.refresh_fill_rates()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
DECLARE
    touched INTEGER := 0;
BEGIN
    WITH counted AS (
        SELECT
            r.id,
            count(*) AS total,
            count(*) FILTER (
                WHERE NULLIF(btrim(assistant_api._field_display_at_path(
                    rec.row_data_normalized, rec.canonical_name,
                    rec.canonical_community, rec.canonical_school, r.canonical_json_path
                )), '') IS NOT NULL
            ) AS filled
        FROM assistant.field_registry r
        JOIN assistant_api.v_current_records rec ON rec.file_id = r.file_id
        WHERE r.canonical_json_path IS NOT NULL
        GROUP BY r.id
    ),
    applied AS (
        UPDATE assistant.field_registry r
        SET validation_rules = COALESCE(r.validation_rules, '{}'::jsonb)
                               || jsonb_build_object(
                                      'recorded_count', counted.filled,
                                      'record_count', counted.total
                                  )
        FROM counted
        WHERE r.id = counted.id
        RETURNING r.id
    )
    SELECT count(*) INTO touched FROM applied;
    RETURN touched;
END;
$function$;

SELECT assistant.refresh_fill_rates();
