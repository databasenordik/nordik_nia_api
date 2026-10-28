CREATE OR REPLACE FUNCTION assistant_api.retrieve_candidates(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_query TEXT,
    p_method TEXT,
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
    search_text TEXT,
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
        r.search_text,
        r.row_data_normalized
    FROM assistant_api.v_current_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND (
            p_method = 'exact'
            AND (
                lower(coalesce(r.canonical_name, '')) = lower(p_query)
                OR lower(coalesce(r.canonical_community, '')) = lower(p_query)
                OR lower(coalesce(r.canonical_school, '')) = lower(p_query)
            )
         OR p_method = 'fts'
            AND to_tsvector('simple', coalesce(r.search_text, ''))
                @@ plainto_tsquery('simple', coalesce(p_query, ''))
         OR p_method IN ('trgm', 'fuzzy')
            AND (
                similarity(coalesce(r.canonical_name, ''), p_query) > 0.25
                OR similarity(coalesce(r.canonical_community, ''), p_query) > 0.25
                OR similarity(coalesce(r.search_text, ''), p_query) > 0.15
            )
      )
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 20), 50), 1);
$$;

ALTER FUNCTION assistant_api.retrieve_candidates(TEXT, INTEGER[], BOOLEAN, TEXT, TEXT, INTEGER)
    OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.retrieve_candidates(TEXT, INTEGER[], BOOLEAN, TEXT, TEXT, INTEGER) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.retrieve_candidates(TEXT, INTEGER[], BOOLEAN, TEXT, TEXT, INTEGER)
    TO assistant_runtime;
