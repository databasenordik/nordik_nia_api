-- Resolve each aggregate's field paths once per file instead of once per row.
--
-- 050 did this for filtering. The aggregate entry points project values the same way, and
-- were paying the same cost: _field_display(file_id, ..., 'community') runs a registry
-- lookup for every row, and because it is SECURITY DEFINER with SET search_path the planner
-- cannot inline it. Several of these also carried a correlated EXISTS against the registry
-- as a second per-row lookup.
--
-- assistant.field_registry is 243 rows. Grouping it by file_id costs nothing and turns both
-- per-row lookups into one hash join, after which the projection calls
-- _field_display_at_path -- IMMUTABLE, no SET, so it inlines.
--
-- Measured on file 49 (2,781 current rows, no filters), before this migration:
--   structured_completeness   1690 ms
--   structured_interval        998 ms
--   structured_duplicates      746 ms
--   structured_group_values    474 ms
--   structured_group_values2   457 ms
--   structured_stats           188 ms
--
-- Equivalence rests on two facts, both checked against the current definitions:
--   * _field_display returns NULL for a field that is not registered, and
--     _field_display_at_path returns NULL when handed a NULL path, so a LEFT JOIN that
--     finds no registry row produces exactly the value the old call produced;
--   * has_field reproduces the EXISTS guard, including its behaviour when the field
--     argument is NULL (no match, so the row is excluded).
-- Row counts, ordering and every other expression are carried over untouched.
--
-- structured_completeness additionally stops asking the registry three separate correlated
-- questions per row -- filled count, total count, filled labels -- and answers all three
-- from one join, which is where most of its 1.7 s went.

-- ---------------------------------------------------------------- completeness

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
    WITH name_paths AS (
        SELECT r.file_id,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = 'student_name'
               ) AS name_path
        FROM assistant.field_registry r
        GROUP BY r.file_id
    ),
    matched AS (
        SELECT
            records.file_id,
            records.source_row_id,
            records.row_data_normalized,
            records.canonical_name,
            records.canonical_community,
            records.canonical_school
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
    ),
    per_field AS (
        SELECT
            matched.file_id,
            matched.source_row_id,
            r.human_label,
            NULLIF(btrim(COALESCE(
                assistant_api._field_display_at_path(
                    matched.row_data_normalized, matched.canonical_name,
                    matched.canonical_community, matched.canonical_school,
                    r.canonical_json_path
                ), '')), '') IS NOT NULL AS filled
        FROM matched
        JOIN assistant.field_registry r ON r.file_id = matched.file_id
    ),
    tallied AS (
        SELECT
            per_field.file_id,
            per_field.source_row_id,
            COUNT(*) FILTER (WHERE per_field.filled) AS filled_fields,
            COUNT(*) AS total_fields,
            array_agg(per_field.human_label ORDER BY per_field.human_label)
                FILTER (WHERE per_field.filled) AS filled_field_labels
        FROM per_field
        GROUP BY per_field.file_id, per_field.source_row_id
    ),
    scored AS (
        SELECT
            matched.file_id,
            matched.source_row_id,
            COALESCE(
                assistant_api._field_display_at_path(
                    matched.row_data_normalized, matched.canonical_name,
                    matched.canonical_community, matched.canonical_school,
                    name_paths.name_path
                ),
                matched.canonical_name
            ) AS display_name,
            -- A file with no registry rows at all produced 0 and 0 from the old
            -- subqueries; the join drops it, so restore those counts here.
            COALESCE(tallied.filled_fields, 0) AS filled_fields,
            COALESCE(tallied.total_fields, 0) AS total_fields,
            tallied.filled_field_labels
        FROM matched
        LEFT JOIN name_paths ON name_paths.file_id = matched.file_id
        LEFT JOIN tallied
               ON tallied.file_id = matched.file_id
              AND tallied.source_row_id = matched.source_row_id
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

-- ---------------------------------------------------------------- group_values

CREATE OR REPLACE FUNCTION assistant_api.structured_group_values(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_field text
) RETURNS TABLE(file_id integer, value text, record_count bigint)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH field_paths AS (
        SELECT r.file_id,
               bool_or(r.semantic_field = p_field) AS has_field,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_field
               ) AS field_path
        FROM assistant.field_registry r
        GROUP BY r.file_id
    )
    SELECT
        matched.file_id,
        MODE() WITHIN GROUP (ORDER BY matched.value) AS value,
        COUNT(*) AS record_count
    FROM (
        SELECT
            records.file_id,
            NULLIF(
                btrim(
                    assistant_api._field_display_at_path(
                        records.row_data_normalized,
                        records.canonical_name,
                        records.canonical_community,
                        records.canonical_school,
                        fp.field_path
                    )
                ),
                ''
            ) AS value
        FROM assistant_api.v_current_records records
        LEFT JOIN field_paths fp ON fp.file_id = records.file_id
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND COALESCE(fp.has_field, FALSE)
          AND assistant_api._matches_group(
              records.file_id,
              records.row_data_normalized,
              records.canonical_name,
              records.canonical_community,
              records.canonical_school,
              p_filters
          )
    ) matched
    GROUP BY matched.file_id, lower(matched.value)
    ORDER BY COUNT(*) DESC, lower(MODE() WITHIN GROUP (ORDER BY matched.value)) ASC NULLS LAST,
             matched.file_id;
$function$;

-- ---------------------------------------------------------------- group_values2

CREATE OR REPLACE FUNCTION assistant_api.structured_group_values2(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_field_1 text,
    p_part_1 text,
    p_field_2 text,
    p_part_2 text,
    p_min_count bigint,
    p_top_n integer,
    p_per_group_top_n integer,
    p_include_missing boolean
) RETURNS TABLE(file_id integer, label_1 text, label_2 text, record_count bigint)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH field_paths AS (
        SELECT r.file_id,
               bool_or(r.semantic_field = p_field_1) AS has_field_1,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_field_1
               ) AS path_1,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_field_2
               ) AS path_2
        FROM assistant.field_registry r
        GROUP BY r.file_id
    ),
    projected AS (
        SELECT
            records.file_id,
            assistant_api._value_part(
                assistant_api._field_display_at_path(
                    records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, fp.path_1
                ),
                p_part_1
            ) AS label_1,
            CASE WHEN p_field_2 IS NULL THEN NULL ELSE
                assistant_api._value_part(
                    assistant_api._field_display_at_path(
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school, fp.path_2
                    ),
                    p_part_2
                )
            END AS label_2
        FROM assistant_api.v_current_records records
        LEFT JOIN field_paths fp ON fp.file_id = records.file_id
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND COALESCE(fp.has_field_1, FALSE)
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    ),
    grouped AS (
        SELECT
            projected.file_id,
            MODE() WITHIN GROUP (ORDER BY projected.label_1) AS label_1,
            MODE() WITHIN GROUP (ORDER BY projected.label_2) AS label_2,
            COUNT(*) AS record_count
        FROM projected
        WHERE COALESCE(p_include_missing, FALSE)
           OR (projected.label_1 IS NOT NULL
               AND (p_field_2 IS NULL OR projected.label_2 IS NOT NULL))
        GROUP BY projected.file_id, lower(projected.label_1), lower(projected.label_2)
        HAVING COUNT(*) >= GREATEST(COALESCE(p_min_count, 1), 1)
    ),
    ranked AS (
        SELECT
            grouped.*,
            ROW_NUMBER() OVER (
                PARTITION BY grouped.file_id, lower(grouped.label_1)
                ORDER BY grouped.record_count DESC, lower(grouped.label_2) ASC NULLS LAST
            ) AS within_group
        FROM grouped
    )
    SELECT ranked.file_id, ranked.label_1, ranked.label_2, ranked.record_count
    FROM ranked
    WHERE COALESCE(p_per_group_top_n, 0) <= 0 OR ranked.within_group <= p_per_group_top_n
    ORDER BY
        CASE WHEN COALESCE(p_per_group_top_n, 0) > 0 THEN lower(ranked.label_1) END ASC NULLS LAST,
        ranked.record_count DESC,
        lower(ranked.label_1) ASC NULLS LAST,
        lower(ranked.label_2) ASC NULLS LAST
    LIMIT CASE WHEN COALESCE(p_top_n, 0) > 0 THEN p_top_n ELSE NULL END;
$function$;

-- ---------------------------------------------------------------- duplicates

CREATE OR REPLACE FUNCTION assistant_api.structured_duplicates(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_field text,
    p_part text,
    p_companion_field text,
    p_min_count bigint,
    p_member_limit integer,
    p_limit integer
) RETURNS TABLE(
    file_id integer,
    value text,
    record_count bigint,
    distinct_companions bigint,
    members jsonb
)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH field_paths AS (
        SELECT r.file_id,
               bool_or(r.semantic_field = p_field) AS has_field,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_field
               ) AS field_path,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = COALESCE(p_companion_field, 'student_name')
               ) AS companion_path
        FROM assistant.field_registry r
        GROUP BY r.file_id
    ),
    projected AS (
        SELECT
            records.file_id,
            records.source_row_id,
            assistant_api._value_part(
                assistant_api._field_display_at_path(
                    records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, fp.field_path
                ),
                p_part
            ) AS value,
            COALESCE(
                assistant_api._field_display_at_path(
                    records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, fp.companion_path
                ),
                records.canonical_name
            ) AS member_label
        FROM assistant_api.v_current_records records
        LEFT JOIN field_paths fp ON fp.file_id = records.file_id
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND COALESCE(fp.has_field, FALSE)
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    ),
    grouped AS (
        SELECT
            projected.file_id,
            projected.value,
            COUNT(*) AS record_count,
            COUNT(DISTINCT projected.member_label) AS distinct_companions,
            jsonb_agg(
                projected.member_label
                ORDER BY lower(projected.member_label), projected.source_row_id
            ) AS members
        FROM projected
        WHERE projected.value IS NOT NULL
        GROUP BY projected.file_id, projected.value
        HAVING COUNT(*) >= GREATEST(COALESCE(p_min_count, 2), 2)
    )
    SELECT
        grouped.file_id,
        grouped.value,
        grouped.record_count,
        grouped.distinct_companions,
        CASE
            WHEN COALESCE(p_member_limit, 0) > 0
            THEN COALESCE(
                (
                    SELECT jsonb_agg(item)
                    FROM (
                        SELECT item
                        FROM jsonb_array_elements(grouped.members) AS members(item)
                        LIMIT p_member_limit
                    ) capped
                ),
                '[]'::jsonb
            )
            ELSE grouped.members
        END AS members
    FROM grouped
    ORDER BY grouped.record_count DESC, lower(grouped.value) ASC
    LIMIT CASE WHEN COALESCE(p_limit, 0) > 0 THEN p_limit ELSE NULL END;
$function$;

-- ---------------------------------------------------------------- interval

CREATE OR REPLACE FUNCTION assistant_api.structured_interval(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_start_field text,
    p_end_field text,
    p_min_days numeric,
    p_max_days numeric,
    p_direction text,
    p_limit integer
) RETURNS TABLE(
    file_id integer,
    source_row_id bigint,
    display_name text,
    start_value text,
    end_value text,
    interval_days numeric,
    start_precision text,
    end_precision text,
    year_delta integer
)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH field_paths AS (
        SELECT r.file_id,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = 'student_name'
               ) AS name_path,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_start_field
               ) AS start_path,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_end_field
               ) AS end_path
        FROM assistant.field_registry r
        GROUP BY r.file_id
    ),
    spans AS (
        SELECT
            records.file_id,
            records.source_row_id,
            COALESCE(
                assistant_api._field_display_at_path(
                    records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, fp.name_path
                ),
                records.canonical_name
            ) AS display_name,
            assistant_api._field_display_at_path(
                records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, fp.start_path
            ) AS start_value,
            assistant_api._field_display_at_path(
                records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, fp.end_path
            ) AS end_value
        FROM assistant_api.v_current_records records
        LEFT JOIN field_paths fp ON fp.file_id = records.file_id
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    ),
    measured AS (
        SELECT
            spans.file_id,
            spans.source_row_id,
            spans.display_name,
            spans.start_value,
            spans.end_value,
            (
                assistant_api._parse_date_value(spans.end_value)
                - assistant_api._parse_date_value(spans.start_value)
            )::NUMERIC AS interval_days,
            assistant_api._date_precision(spans.start_value) AS start_precision,
            assistant_api._date_precision(spans.end_value) AS end_precision,
            (
                EXTRACT(YEAR FROM assistant_api._parse_date_value(spans.end_value))
                - EXTRACT(YEAR FROM assistant_api._parse_date_value(spans.start_value))
            )::INTEGER AS year_delta
        FROM spans
        WHERE assistant_api._parse_date_value(spans.start_value) IS NOT NULL
          AND assistant_api._parse_date_value(spans.end_value) IS NOT NULL
    )
    SELECT
        measured.file_id,
        measured.source_row_id,
        measured.display_name,
        measured.start_value,
        measured.end_value,
        measured.interval_days,
        measured.start_precision,
        measured.end_precision,
        measured.year_delta
    FROM measured
    WHERE (p_min_days IS NULL OR measured.interval_days >= p_min_days)
      AND (p_max_days IS NULL OR measured.interval_days <= p_max_days)
    ORDER BY
        CASE WHEN lower(COALESCE(p_direction, 'desc')) = 'asc' THEN measured.interval_days END ASC,
        CASE WHEN lower(COALESCE(p_direction, 'desc')) <> 'asc' THEN measured.interval_days END DESC,
        lower(measured.display_name) ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 10), 500), 1);
$function$;

-- ---------------------------------------------------------------- stats

CREATE OR REPLACE FUNCTION assistant_api.structured_stats(
    p_principal_id text,
    p_requested_file_ids integer[],
    p_can_use_private boolean,
    p_filters jsonb,
    p_field text,
    p_part text
) RETURNS TABLE(
    record_count bigint,
    known_count bigint,
    min_value numeric,
    max_value numeric,
    avg_value numeric,
    median_value numeric,
    mode_value numeric,
    mode_count bigint
)
LANGUAGE sql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
    WITH field_paths AS (
        SELECT r.file_id,
               bool_or(r.semantic_field = p_field) AS has_field,
               MAX(r.canonical_json_path) FILTER (
                   WHERE r.semantic_field = p_field
               ) AS field_path
        FROM assistant.field_registry r
        GROUP BY r.file_id
    ),
    projected AS (
        SELECT assistant_api._numeric_value(
            assistant_api._field_display_at_path(
                records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, fp.field_path
            ),
            p_part
        ) AS value
        FROM assistant_api.v_current_records records
        LEFT JOIN field_paths fp ON fp.file_id = records.file_id
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND COALESCE(fp.has_field, FALSE)
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    ),
    modal AS (
        SELECT value, COUNT(*) AS n
        FROM projected
        WHERE value IS NOT NULL
        GROUP BY value
        ORDER BY COUNT(*) DESC, value ASC
        LIMIT 1
    )
    SELECT
        COUNT(*) AS record_count,
        COUNT(projected.value) AS known_count,
        MIN(projected.value) AS min_value,
        MAX(projected.value) AS max_value,
        ROUND(AVG(projected.value), 4) AS avg_value,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY projected.value) AS median_value,
        (SELECT value FROM modal) AS mode_value,
        (SELECT n FROM modal) AS mode_count
    FROM projected;
$function$;
