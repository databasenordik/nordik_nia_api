CREATE OR REPLACE FUNCTION assistant_api.structured_sample_page(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_limit INTEGER
)
RETURNS TABLE (
    id BIGINT,
    source_row_id BIGINT,
    file_id INTEGER,
    version INTEGER,
    canonical_name TEXT,
    canonical_community TEXT,
    canonical_school TEXT,
    row_data_normalized JSONB,
    total_count BIGINT
)
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    WITH matched AS MATERIALIZED (
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
        m.id,
        m.source_row_id,
        m.file_id,
        m.version,
        m.canonical_name,
        m.canonical_community,
        m.canonical_school,
        m.row_data_normalized,
        COUNT(*) OVER () AS total_count
    FROM matched m
    ORDER BY random(), m.source_row_id
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 5), 50), 1);
$$;

ALTER FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) TO assistant_runtime;
