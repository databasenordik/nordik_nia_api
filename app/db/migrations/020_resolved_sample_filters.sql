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
    WITH authorized AS MATERIALIZED (
        SELECT unnest(
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        ) AS file_id
    ),
    filter_specs AS MATERIALIZED (
        SELECT item AS spec, ordinal
        FROM jsonb_array_elements(COALESCE(p_filters, '[]'::jsonb))
             WITH ORDINALITY AS filters(item, ordinal)
    ),
    resolved_filters AS MATERIALIZED (
        SELECT
            authorized.file_id,
            filter_specs.ordinal,
            registry.canonical_json_path,
            filter_specs.spec->>'operator' AS operator,
            CASE
                WHEN jsonb_typeof(filter_specs.spec->'value') = 'string'
                    THEN filter_specs.spec->>'value'
                ELSE trim(both '"' from (filter_specs.spec->'value')::text)
            END AS expected
        FROM authorized
        CROSS JOIN filter_specs
        LEFT JOIN assistant.field_registry registry
          ON registry.file_id = authorized.file_id
         AND registry.semantic_field = filter_specs.spec->>'field'
    ),
    matched AS MATERIALIZED (
        SELECT
            records.id,
            records.source_row_id,
            records.file_id,
            records.version,
            records.canonical_name,
            records.canonical_community,
            records.canonical_school,
            records.row_data_normalized
        FROM assistant_api.v_current_records records
        JOIN authorized USING (file_id)
        WHERE NOT EXISTS (
            SELECT 1
            FROM resolved_filters resolved
            WHERE resolved.file_id = records.file_id
              AND (
                  resolved.canonical_json_path IS NULL
                  OR NOT COALESCE(
                      assistant_api._apply_op(
                          CASE
                              WHEN resolved.canonical_json_path IN (
                                  'canonical.name', 'canonical.display_name'
                              ) THEN records.canonical_name
                              WHEN resolved.canonical_json_path = 'canonical.community'
                                  THEN records.canonical_community
                              WHEN resolved.canonical_json_path = 'canonical.school'
                                  THEN records.canonical_school
                              ELSE records.row_data_normalized #>>
                                  string_to_array(resolved.canonical_json_path, '.')
                          END,
                          resolved.operator,
                          resolved.expected
                      ),
                      FALSE
                  )
              )
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
        matched.row_data_normalized,
        COUNT(*) OVER () AS total_count
    FROM matched
    ORDER BY random(), matched.source_row_id
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
