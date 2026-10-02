-- Sort registered date fields by recognizable calendar components instead of
-- lexicographic raw text. Mixed source formats remain displayable, while
-- malformed values without a plausible year sort as unknown (NULLS LAST).

UPDATE assistant.field_registry
SET aliases = array_append(aliases, 'birth')
WHERE semantic_field = 'birth_date'
  AND NOT ('birth' = ANY(aliases));

CREATE OR REPLACE FUNCTION assistant_api._date_sort_key(p_value TEXT)
RETURNS BIGINT
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    parts TEXT[];
    parsed_year INTEGER;
    parsed_month INTEGER := 0;
    parsed_day INTEGER := 0;
    month_text TEXT;
BEGIN
    IF p_value IS NULL THEN RETURN NULL; END IF;
    parsed_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
    IF parsed_year IS NULL THEN RETURN NULL; END IF;

    parts := regexp_match(
        p_value,
        '(1[6-9][0-9]{2}|20[0-9]{2})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})'
    );
    IF parts IS NOT NULL THEN
        parsed_month := parts[2]::INTEGER;
        parsed_day := parts[3]::INTEGER;
    ELSE
        parts := regexp_match(
            p_value,
            '([0-9]{1,2})[-/.]([0-9]{1,2})[-/.](1[6-9][0-9]{2}|20[0-9]{2})'
        );
        IF parts IS NOT NULL THEN
            parsed_day := parts[1]::INTEGER;
            parsed_month := parts[2]::INTEGER;
        ELSE
            parts := regexp_match(
                lower(p_value),
                '([0-9]{1,2})[[:space:]]+(january|february|march|april|may|june|july|august|september|october|november|december)[[:space:]]+(1[6-9][0-9]{2}|20[0-9]{2})'
            );
            IF parts IS NOT NULL THEN
                parsed_day := parts[1]::INTEGER;
                month_text := parts[2];
            ELSE
                parts := regexp_match(
                    lower(p_value),
                    '(january|february|march|april|may|june|july|august|september|october|november|december)[[:space:]]+(1[6-9][0-9]{2}|20[0-9]{2})'
                );
                IF parts IS NOT NULL THEN month_text := parts[1]; END IF;
            END IF;
            IF month_text IS NOT NULL THEN
                parsed_month := array_position(
                    ARRAY['january','february','march','april','may','june',
                          'july','august','september','october','november','december'],
                    month_text
                );
            END IF;
        END IF;
    END IF;
    IF parsed_month NOT BETWEEN 0 AND 12 OR parsed_day NOT BETWEEN 0 AND 31 THEN
        RETURN NULL;
    END IF;
    RETURN parsed_year::BIGINT * 10000 + parsed_month * 100 + parsed_day;
END;
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
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school,
                        resolved_sort.canonical_json_path
                    )
                )
                ELSE NULL
            END AS sort_text,
            CASE
                WHEN resolved_sort.sortable AND resolved_sort.semantic_type = 'number'
                     AND assistant_api._field_text_at_path(
                         records.row_data_normalized, records.canonical_name,
                         records.canonical_community, records.canonical_school,
                         resolved_sort.canonical_json_path
                     ) ~ '-?[0-9]+(?:\.[0-9]+)?'
                THEN substring(
                    assistant_api._field_text_at_path(
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school,
                        resolved_sort.canonical_json_path
                    ) from '-?[0-9]+(?:\.[0-9]+)?'
                )::NUMERIC
                ELSE NULL
            END AS sort_number,
            CASE
                WHEN resolved_sort.sortable AND resolved_sort.semantic_type = 'date'
                THEN assistant_api._date_sort_key(
                    assistant_api._field_text_at_path(
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school,
                        resolved_sort.canonical_json_path
                    )
                )
                ELSE NULL
            END AS sort_date
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
                              records.row_data_normalized, records.canonical_name,
                              records.canonical_community, records.canonical_school,
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
        matched.id, matched.source_row_id, matched.file_id, matched.version,
        matched.canonical_name, matched.canonical_community,
        matched.canonical_school, matched.row_data_normalized
    FROM matched
    ORDER BY
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_number END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_number END ASC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_date END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_date END ASC NULLS LAST,
        CASE WHEN matched.sort_number IS NULL AND matched.sort_date IS NULL
                  AND upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_text END DESC NULLS LAST,
        CASE WHEN matched.sort_number IS NULL AND matched.sort_date IS NULL
                  AND upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_text END ASC NULLS LAST,
        lower(matched.canonical_name) ASC NULLS LAST,
        matched.source_row_id ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 25), 50), 1)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0);
$$;
