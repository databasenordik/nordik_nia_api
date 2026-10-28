-- Put structured_completeness back the way it was. The hoist that helped the other five
-- aggregates makes this one slower, twice over.
--
-- 051 applied the same treatment here as everywhere else. Measured on file 49 (2,781 rows,
-- 41 registered fields, no filters), best of three alternating warm runs:
--
--   original, three correlated subqueries      1511 ms
--   051, join to field_registry                2957 ms
--   lateral with the field test computed once  2567 ms
--
-- Both rewrites lose, for different reasons. Joining the registry to the records widens
-- every row into one row per field and each copy carries row_data_normalized, roughly 5 KB,
-- so file 49 builds a 2,781 x 41 intermediate before grouping it back down. Moving that
-- into a LATERAL stops the copying but still builds a small tuplestore per row, 2,781 of
-- them, and that costs more than it saves.
--
-- The original looks wasteful -- it asks the registry three separate questions per row, so
-- it tests every field of every row twice over -- but Postgres evaluates correlated scalar
-- subqueries in the target list without materialising anything, and that beats both shapes
-- comfortably. Doing less work in a more expensive shape is still more expensive.
--
-- So this restores the definition 051 replaced, verbatim. The other five functions 051
-- changed keep their 3-9x gains; only this one goes back. No caller sees any difference:
-- output was byte-identical across all three shapes.
--
-- Anyone tempted to optimise this again should know where the time actually goes: it is
-- 2,781 x 41 evaluations of _field_display_at_path, each re-running
-- string_to_array(canonical_json_path, '.') for a path that is constant per field. Hoisting
-- that split is the change with a real prospect of paying off, and it means either teaching
-- _field_display_at_path to take a pre-split path or duplicating its fallback logic -- which
-- is why it is not being done here for a code path no benchmark case reaches.

CREATE OR REPLACE FUNCTION assistant_api.structured_completeness(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_direction text,
    p_limit integer
) RETURNS TABLE(
    file_id integer,
    source_row_id bigint,
    display_name text,
    filled_fields bigint,
    total_fields bigint,
    filled_field_labels text[]
)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH scored AS (
        SELECT
            records.file_id,
            records.source_row_id,
            COALESCE(
                assistant_api._field_display(
                    records.file_id, records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, 'student_name'
                ),
                records.canonical_name
            ) AS display_name,
            (
                SELECT COUNT(*)
                FROM assistant.field_registry r
                WHERE r.file_id = records.file_id
                  AND NULLIF(btrim(COALESCE(
                      assistant_api._field_display_at_path(
                          records.row_data_normalized, records.canonical_name,
                          records.canonical_community, records.canonical_school,
                          r.canonical_json_path
                      ), '')), '') IS NOT NULL
            ) AS filled_fields,
            (
                SELECT COUNT(*) FROM assistant.field_registry r WHERE r.file_id = records.file_id
            ) AS total_fields,
            (
                SELECT array_agg(r.human_label ORDER BY r.human_label)
                FROM assistant.field_registry r
                WHERE r.file_id = records.file_id
                  AND NULLIF(btrim(COALESCE(
                      assistant_api._field_display_at_path(
                          records.row_data_normalized, records.canonical_name,
                          records.canonical_community, records.canonical_school,
                          r.canonical_json_path
                      ), '')), '') IS NOT NULL
            ) AS filled_field_labels
        FROM assistant_api.v_current_records records
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    )
    SELECT
        scored.file_id,
        scored.source_row_id,
        scored.display_name,
        scored.filled_fields,
        scored.total_fields,
        scored.filled_field_labels
    FROM scored
    ORDER BY
        CASE WHEN lower(COALESCE(p_direction, 'desc')) = 'asc' THEN scored.filled_fields END ASC,
        CASE WHEN lower(COALESCE(p_direction, 'desc')) <> 'asc' THEN scored.filled_fields END DESC,
        lower(scored.display_name) ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 10), 200), 1);
$function$;

-- The measurement scaffolding, so a rerun of this file leaves nothing behind.
DROP FUNCTION IF EXISTS assistant_api.structured_completeness_orig(
    text, integer[], boolean, jsonb, text, integer
);
DROP FUNCTION IF EXISTS assistant_api.structured_completeness_flat(
    text, integer[], boolean, jsonb, text, integer
);
