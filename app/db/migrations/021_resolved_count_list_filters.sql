CREATE OR REPLACE FUNCTION assistant_api._field_text_at_path(
    p_row JSONB,
    p_canonical_name TEXT,
    p_community TEXT,
    p_school TEXT,
    p_canonical_json_path TEXT
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE
        WHEN p_canonical_json_path IN ('canonical.name', 'canonical.display_name')
            THEN p_canonical_name
        WHEN p_canonical_json_path = 'canonical.community' THEN p_community
        WHEN p_canonical_json_path = 'canonical.school' THEN p_school
        WHEN p_canonical_json_path IS NULL THEN NULL
        ELSE p_row #>> string_to_array(p_canonical_json_path, '.')
    END;
$$;

CREATE OR REPLACE FUNCTION assistant_api.structured_count(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB
)
RETURNS BIGINT
LANGUAGE sql
STABLE
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
    )
    SELECT COUNT(*)
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
                      assistant_api._field_text_at_path(
                          records.row_data_normalized,
                          records.canonical_name,
                          records.canonical_community,
                          records.canonical_school,
                          resolved.canonical_json_path
                      ),
                      resolved.operator,
                      resolved.expected
                  ),
                  FALSE
              )
          )
    );
$$;

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
    resolved_sort AS MATERIALIZED (
        SELECT
            authorized.file_id,
            registry.canonical_json_path,
            COALESCE(registry.sortable, FALSE) AS sortable
        FROM authorized
        LEFT JOIN assistant.field_registry registry
          ON registry.file_id = authorized.file_id
         AND registry.semantic_field = p_sort_field
    ),
    matched AS (
        SELECT
            records.id,
            records.source_row_id,
            records.file_id,
            records.version,
            records.canonical_name,
            records.canonical_community,
            records.canonical_school,
            records.row_data_normalized,
            CASE
                WHEN p_sort_field IS NULL THEN lower(records.canonical_name)
                WHEN resolved_sort.sortable THEN lower(
                    assistant_api._field_text_at_path(
                        records.row_data_normalized,
                        records.canonical_name,
                        records.canonical_community,
                        records.canonical_school,
                        resolved_sort.canonical_json_path
                    )
                )
                ELSE NULL
            END AS sort_value
        FROM assistant_api.v_current_records records
        JOIN authorized USING (file_id)
        JOIN resolved_sort USING (file_id)
        WHERE NOT EXISTS (
            SELECT 1
            FROM resolved_filters resolved
            WHERE resolved.file_id = records.file_id
              AND (
                  resolved.canonical_json_path IS NULL
                  OR NOT COALESCE(
                      assistant_api._apply_op(
                          assistant_api._field_text_at_path(
                              records.row_data_normalized,
                              records.canonical_name,
                              records.canonical_community,
                              records.canonical_school,
                              resolved.canonical_json_path
                          ),
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

ALTER FUNCTION assistant_api._field_text_at_path(JSONB, TEXT, TEXT, TEXT, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api._field_text_at_path(JSONB, TEXT, TEXT, TEXT, TEXT)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) TO assistant_runtime;
