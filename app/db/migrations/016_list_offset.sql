DROP FUNCTION IF EXISTS assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER);

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
    SELECT
        r.id,
        r.source_row_id,
        r.file_id,
        r.version,
        r.canonical_name,
        r.canonical_community,
        r.canonical_school,
        r.row_data_normalized
    FROM assistant_api.v_current_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND assistant_api._record_matches(
          r.file_id, r.row_data_normalized, r.canonical_name, r.canonical_community, r.canonical_school, p_filters
      )
    ORDER BY
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC' THEN r.canonical_name END DESC,
        r.canonical_name ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 25), 50), 1)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0);
$$;

ALTER FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER)
    OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER)
    TO assistant_runtime;
