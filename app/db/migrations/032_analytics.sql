-- Ranking, statistics, duplicates, completeness, and date intervals.
--
-- The API could count, list, and group a single field. Every question that asked
-- "which has the most", "what percentage", "what was the average/typical", "which
-- value appears more than once", "who has the most information", or "how long
-- between X and Y" therefore degraded into a bare count or an arbitrary first page
-- of rows. These functions add the missing computations, all catalog-driven: the
-- caller names semantic fields, never columns.

-- Grouped counts over one or two semantic fields, with optional derived buckets
-- (year, decade, first/last name token), a HAVING floor, an overall top-N, and a
-- per-group top-N ("the most common cause of death in each decade").
CREATE OR REPLACE FUNCTION assistant_api.structured_group_values2(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_field_1 TEXT,
    p_part_1 TEXT,
    p_field_2 TEXT,
    p_part_2 TEXT,
    p_min_count BIGINT,
    p_top_n INTEGER,
    p_per_group_top_n INTEGER,
    p_include_missing BOOLEAN
)
RETURNS TABLE (
    file_id INTEGER,
    label_1 TEXT,
    label_2 TEXT,
    record_count BIGINT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH projected AS (
        SELECT
            records.file_id,
            assistant_api._value_part(
                assistant_api._field_display(
                    records.file_id, records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, p_field_1
                ),
                p_part_1
            ) AS label_1,
            CASE WHEN p_field_2 IS NULL THEN NULL ELSE
                assistant_api._value_part(
                    assistant_api._field_display(
                        records.file_id, records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school, p_field_2
                    ),
                    p_part_2
                )
            END AS label_2
        FROM assistant_api.v_current_records records
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND EXISTS (
              SELECT 1 FROM assistant.field_registry r
              WHERE r.file_id = records.file_id AND r.semantic_field = p_field_1
          )
          AND assistant_api._matches_group(
              records.file_id, records.row_data_normalized, records.canonical_name,
              records.canonical_community, records.canonical_school, p_filters
          )
    ),
    grouped AS (
        SELECT
            projected.file_id,
            projected.label_1,
            projected.label_2,
            COUNT(*) AS record_count
        FROM projected
        WHERE COALESCE(p_include_missing, FALSE)
           OR (projected.label_1 IS NOT NULL
               AND (p_field_2 IS NULL OR projected.label_2 IS NOT NULL))
        GROUP BY projected.file_id, projected.label_1, projected.label_2
        HAVING COUNT(*) >= GREATEST(COALESCE(p_min_count, 1), 1)
    ),
    ranked AS (
        SELECT
            grouped.*,
            ROW_NUMBER() OVER (
                PARTITION BY grouped.file_id, grouped.label_1
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
$$;

-- Descriptive statistics over the numeric projection of a semantic field:
-- youngest / oldest / average / median (typical) / mode (most frequent).
CREATE OR REPLACE FUNCTION assistant_api.structured_stats(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_field TEXT,
    p_part TEXT
)
RETURNS TABLE (
    record_count BIGINT,
    known_count BIGINT,
    min_value NUMERIC,
    max_value NUMERIC,
    avg_value NUMERIC,
    median_value NUMERIC,
    mode_value NUMERIC,
    mode_count BIGINT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH projected AS (
        SELECT assistant_api._numeric_value(
            assistant_api._field_display(
                records.file_id, records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, p_field
            ),
            p_part
        ) AS value
        FROM assistant_api.v_current_records records
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND EXISTS (
              SELECT 1 FROM assistant.field_registry r
              WHERE r.file_id = records.file_id AND r.semantic_field = p_field
          )
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
$$;

-- Values shared by more than one record, with the records that carry them.
-- Answers "are any student numbers assigned to more than one student", "which full
-- names appear more than once", "are any communities listed with the same
-- coordinates".
CREATE OR REPLACE FUNCTION assistant_api.structured_duplicates(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_field TEXT,
    p_part TEXT,
    p_companion_field TEXT,
    p_min_count BIGINT,
    p_member_limit INTEGER,
    p_limit INTEGER
)
RETURNS TABLE (
    file_id INTEGER,
    value TEXT,
    record_count BIGINT,
    distinct_companions BIGINT,
    members JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH projected AS (
        SELECT
            records.file_id,
            records.source_row_id,
            assistant_api._value_part(
                assistant_api._field_display(
                    records.file_id, records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school, p_field
                ),
                p_part
            ) AS value,
            COALESCE(
                assistant_api._field_display(
                    records.file_id, records.row_data_normalized, records.canonical_name,
                    records.canonical_community, records.canonical_school,
                    COALESCE(p_companion_field, 'student_name')
                ),
                records.canonical_name
            ) AS member_label
        FROM assistant_api.v_current_records records
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND EXISTS (
              SELECT 1 FROM assistant.field_registry r
              WHERE r.file_id = records.file_id AND r.semantic_field = p_field
          )
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
$$;

-- How many of a record's catalog fields carry a value. Powers "which records have
-- the most / least information recorded about them" without hardcoding a field list.
CREATE OR REPLACE FUNCTION assistant_api.structured_completeness(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_direction TEXT,
    p_limit INTEGER
)
RETURNS TABLE (
    file_id INTEGER,
    source_row_id BIGINT,
    display_name TEXT,
    filled_fields BIGINT,
    total_fields BIGINT,
    filled_field_labels TEXT[]
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
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
$$;

-- Elapsed time between two date fields on the same record. Answers "who stayed
-- longest", "how long were students at the school before they died", "how long
-- after death were they buried", and surfaces impossible orderings (a discharge
-- before an admission) rather than silently dropping them.
CREATE OR REPLACE FUNCTION assistant_api.structured_interval(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_start_field TEXT,
    p_end_field TEXT,
    p_min_days NUMERIC,
    p_max_days NUMERIC,
    p_direction TEXT,
    p_limit INTEGER
)
RETURNS TABLE (
    file_id INTEGER,
    source_row_id BIGINT,
    display_name TEXT,
    start_value TEXT,
    end_value TEXT,
    interval_days NUMERIC,
    start_precision TEXT,
    end_precision TEXT,
    year_delta INTEGER
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH spans AS (
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
            assistant_api._field_display(
                records.file_id, records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, p_start_field
            ) AS start_value,
            assistant_api._field_display(
                records.file_id, records.row_data_normalized, records.canonical_name,
                records.canonical_community, records.canonical_school, p_end_field
            ) AS end_value
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
$$;

CREATE OR REPLACE FUNCTION assistant_api.structured_interval_stats(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_start_field TEXT,
    p_end_field TEXT,
    p_min_days NUMERIC,
    p_max_days NUMERIC
)
RETURNS TABLE (
    record_count BIGINT,
    min_days NUMERIC,
    max_days NUMERIC,
    avg_days NUMERIC,
    median_days NUMERIC,
    negative_count BIGINT,
    impossible_count BIGINT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH measured AS (
        SELECT interval_days, year_delta
        FROM assistant_api.structured_interval(
            p_principal_id, p_requested_file_ids, p_can_use_private, p_filters,
            p_start_field, p_end_field, NULL, NULL, 'desc', 500
        )
    ),
    bounded AS (
        SELECT interval_days
        FROM measured
        WHERE (p_min_days IS NULL OR interval_days >= p_min_days)
          AND (p_max_days IS NULL OR interval_days <= p_max_days)
    )
    SELECT
        (SELECT COUNT(*) FROM bounded) AS record_count,
        (SELECT MIN(interval_days) FROM bounded) AS min_days,
        (SELECT MAX(interval_days) FROM bounded) AS max_days,
        (SELECT ROUND(AVG(interval_days), 4) FROM bounded) AS avg_days,
        (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY interval_days) FROM bounded) AS median_days,
        (SELECT COUNT(*) FROM measured WHERE interval_days < 0) AS negative_count,
        (SELECT COUNT(*) FROM measured WHERE year_delta < 0) AS impossible_count;
$$;

ALTER FUNCTION assistant_api.structured_group_values2(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER, BOOLEAN
) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_stats(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_duplicates(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER
) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_completeness(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, INTEGER)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_interval(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC, TEXT, INTEGER
) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_interval_stats(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC
) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api.structured_group_values2(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER, BOOLEAN
) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_stats(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_duplicates(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER
) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_completeness(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_interval(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC, TEXT, INTEGER
) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_interval_stats(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api.structured_group_values2(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER, BOOLEAN
) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_stats(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_duplicates(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, TEXT, BIGINT, INTEGER, INTEGER
) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_completeness(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, INTEGER) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_interval(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC, TEXT, INTEGER
) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_interval_stats(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, NUMERIC, NUMERIC
) TO assistant_runtime;
