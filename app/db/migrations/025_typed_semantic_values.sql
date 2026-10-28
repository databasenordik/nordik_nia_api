-- Treat normalized values as typed scalars consistently across filters,
-- grouping, sorting, sampling, projection, and aggregates. Source rows store
-- many values as objects such as {"raw":"14","normalized":"14"} or
-- {"iso":"1903-07-30","year":1903}; comparing the serialized object was
-- both incorrect and database-shape dependent.

CREATE OR REPLACE FUNCTION assistant_api._semantic_text(p_value JSONB)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT CASE jsonb_typeof(p_value)
        WHEN 'object' THEN COALESCE(
            NULLIF(p_value->>'iso', ''),
            NULLIF(p_value->>'raw', ''),
            NULLIF(p_value->>'display', ''),
            NULLIF(p_value->>'value', ''),
            NULLIF(p_value->>'year', ''),
            NULLIF(p_value->>'normalized', '')
        )
        WHEN 'array' THEN NULLIF(
            (SELECT string_agg(value, ', ' ORDER BY ordinal)
             FROM jsonb_array_elements_text(p_value) WITH ORDINALITY AS item(value, ordinal)),
            ''
        )
        WHEN 'null' THEN NULL
        ELSE NULLIF(p_value #>> '{}', '')
    END;
$$;

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
            THEN NULLIF(p_canonical_name, '')
        WHEN p_canonical_json_path = 'canonical.community' THEN NULLIF(p_community, '')
        WHEN p_canonical_json_path = 'canonical.school' THEN NULLIF(p_school, '')
        WHEN p_canonical_json_path IS NULL THEN NULL
        ELSE assistant_api._semantic_text(
            p_row #> string_to_array(p_canonical_json_path, '.')
        )
    END;
$$;

-- Value aliases remain metadata. The planner reads these rules and does not
-- embed dataset-specific encodings such as F/M into its architecture.
UPDATE assistant.field_registry
SET validation_rules = jsonb_set(
        COALESCE(validation_rules, '{}'::jsonb),
        '{value_aliases}',
        '{"female":"F","females":"F","woman":"F","women":"F","girl":"F","girls":"F","male":"M","males":"M","man":"M","men":"M","boy":"M","boys":"M"}'::jsonb,
        TRUE
    ),
    aliases = ARRAY['sex','gender']
WHERE file_id = 91 AND semantic_field = 'gender';

UPDATE assistant.field_registry
SET aliases = ARRAY['age','death age','age at death','years old','old at death','older','younger']
WHERE file_id = 91 AND semantic_field = 'age_at_death';

UPDATE assistant.field_registry
SET aliases = ARRAY['death date','died','dod','death year','year of death']
WHERE file_id = 91 AND semantic_field = 'death_date';

UPDATE assistant.field_registry
SET aliases = ARRAY['death location','died at','at school']
WHERE file_id = 91 AND semantic_field = 'location_of_death';

UPDATE assistant.field_registry
SET aliases = ARRAY['death place','place where died']
WHERE file_id = 91 AND semantic_field = 'place_of_death';

UPDATE assistant.field_registry
SET aliases = ARRAY['other institutions','other schools','another school','another institution']
WHERE file_id = 91 AND semantic_field = 'other_schools';

UPDATE assistant.field_registry
SET allowed_operators = (
    SELECT ARRAY(SELECT DISTINCT item FROM unnest(allowed_operators || ARRAY['EQUALS']) item)
)
WHERE file_id = 49 AND semantic_field = 'deceased_status';

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
            registry.semantic_type,
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
            END AS sort_text,
            CASE
                WHEN resolved_sort.sortable AND resolved_sort.semantic_type = 'number'
                     AND assistant_api._field_text_at_path(
                         records.row_data_normalized,
                         records.canonical_name,
                         records.canonical_community,
                         records.canonical_school,
                         resolved_sort.canonical_json_path
                     ) ~ '-?[0-9]+(?:\.[0-9]+)?'
                THEN substring(
                    assistant_api._field_text_at_path(
                        records.row_data_normalized,
                        records.canonical_name,
                        records.canonical_community,
                        records.canonical_school,
                        resolved_sort.canonical_json_path
                    ) from '-?[0-9]+(?:\.[0-9]+)?'
                )::NUMERIC
                ELSE NULL
            END AS sort_number
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
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_number END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_number END ASC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_text END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_text END ASC NULLS LAST,
        lower(matched.canonical_name) ASC NULLS LAST,
        matched.source_row_id ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 25), 50), 1)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0);
$$;

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
        SELECT records.*
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

ALTER FUNCTION assistant_api._semantic_text(JSONB) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._field_text_at_path(JSONB, TEXT, TEXT, TEXT, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api._semantic_text(JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._field_text_at_path(JSONB, TEXT, TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api._semantic_text(JSONB) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._field_text_at_path(JSONB, TEXT, TEXT, TEXT, TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_sample_page(
    TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER
) TO assistant_runtime;
