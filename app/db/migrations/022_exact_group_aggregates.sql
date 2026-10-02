CREATE OR REPLACE FUNCTION assistant_api.structured_group_count(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB
)
RETURNS TABLE (file_id INTEGER, record_count BIGINT)
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
    SELECT records.file_id, COUNT(*) AS record_count
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
    )
    GROUP BY records.file_id
    ORDER BY records.file_id;
$$;

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
    WITH authorized AS MATERIALIZED (
        SELECT unnest(
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        ) AS file_id
    ),
    field_specs AS MATERIALIZED (
        SELECT authorized.file_id, registry.canonical_json_path
        FROM authorized
        JOIN assistant.field_registry registry
          ON registry.file_id = authorized.file_id
         AND registry.semantic_field = p_field
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
            records.file_id,
            NULLIF(
                btrim(
                    assistant_api._field_text_at_path(
                        records.row_data_normalized,
                        records.canonical_name,
                        records.canonical_community,
                        records.canonical_school,
                        field_specs.canonical_json_path
                    )
                ),
                ''
            ) AS value
        FROM assistant_api.v_current_records records
        JOIN field_specs USING (file_id)
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
    SELECT matched.file_id, matched.value, COUNT(*) AS record_count
    FROM matched
    GROUP BY matched.file_id, matched.value
    ORDER BY COUNT(*) DESC, lower(matched.value) ASC NULLS LAST, matched.file_id;
$$;

ALTER FUNCTION assistant_api.structured_group_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_group_values(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT)
    OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api.structured_group_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_group_values(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT)
    FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api.structured_group_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_group_values(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT)
    TO assistant_runtime;
