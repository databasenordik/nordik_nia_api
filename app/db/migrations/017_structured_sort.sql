CREATE OR REPLACE FUNCTION assistant_api.structured_list(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_sort_field TEXT,
    p_sort_direction TEXT,
    p_limit INTEGER,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE (
    id BIGINT,
    source_row_id BIGINT,
    file_id INTEGER,
    version INTEGER,
    canonical_name TEXT,
    canonical_community TEXT,
    canonical_school TEXT,
    row_data_normalized JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH matched AS (
        SELECT
            r.id,
            r.source_row_id,
            r.file_id,
            r.version,
            r.canonical_name,
            r.canonical_community,
            r.canonical_school,
            r.row_data_normalized,
            CASE
                WHEN p_sort_field IS NULL THEN lower(r.canonical_name)
                WHEN registry.sortable THEN lower(
                    assistant_api._field_text(
                        r.file_id,
                        r.row_data_normalized,
                        r.canonical_name,
                        r.canonical_community,
                        r.canonical_school,
                        p_sort_field
                    )
                )
                ELSE NULL
            END AS sort_value
        FROM assistant_api.v_current_records r
        LEFT JOIN assistant.field_registry registry
          ON registry.file_id = r.file_id
         AND registry.semantic_field = p_sort_field
        WHERE r.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND assistant_api._record_matches(
              r.file_id,
              r.row_data_normalized,
              r.canonical_name,
              r.canonical_community,
              r.canonical_school,
              p_filters
          )
    )
    SELECT
        matched.id,
        matched.source_row_id,
        matched.file_id,
        matched.version,
        matched.canonical_name,
        matched.canonical_community,
        matched.canonical_school,
        matched.row_data_normalized
    FROM matched
    ORDER BY
        CASE
            WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_value
        END DESC NULLS LAST,
        CASE
            WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_value
        END ASC NULLS LAST,
        lower(matched.canonical_name) ASC NULLS LAST,
        matched.source_row_id ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 25), 50), 1)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0);
$$;

ALTER FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) TO assistant_runtime;
