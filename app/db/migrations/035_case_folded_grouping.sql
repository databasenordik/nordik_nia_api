-- Group values case-insensitively, and report the spelling the records use most.
--
-- The corpus stores the same value with different capitalisation ("Unknown" and
-- "unknown"), which split one bucket into two in every breakdown. Folding the group
-- key merges them; mode() picks the dominant original spelling so the answer still
-- reads the way the records are written. Genuinely distinct labels are unaffected:
-- the Master List still has 370 community labels either way.

CREATE OR REPLACE FUNCTION assistant_api.structured_group_values(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_field TEXT
)
RETURNS TABLE (file_id INTEGER, value TEXT, record_count BIGINT)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT
        matched.file_id,
        MODE() WITHIN GROUP (ORDER BY matched.value) AS value,
        COUNT(*) AS record_count
    FROM (
        SELECT
            records.file_id,
            NULLIF(
                btrim(
                    assistant_api._field_display(
                        records.file_id,
                        records.row_data_normalized,
                        records.canonical_name,
                        records.canonical_community,
                        records.canonical_school,
                        p_field
                    )
                ),
                ''
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
$$;

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
$$;
