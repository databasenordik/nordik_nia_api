-- Boolean filter groups and a wider operator set.
--
-- Every structured_* function evaluated filters as a pure conjunction
-- (NOT EXISTS over the filter list), so OR, negation, and multi-term matching were
-- inexpressible. Questions such as "tuberculosis or consumption or phthisis or
-- scrofula", "a hospital or sanatorium", and "somewhere other than the school"
-- could not be represented at all.
--
-- The filter payload is now a boolean tree:
--   {"op": "and" | "or" | "not", "items": [ <node>, ... ]}
-- A leaf is the existing {"field", "operator", "value"} shape, and a bare JSON
-- array is still read as an implicit AND, so stored plans keep working.

CREATE OR REPLACE FUNCTION assistant_api._apply_op(
    p_value TEXT,
    p_operator TEXT,
    p_expected TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
AS $$
DECLARE
    value_year INTEGER;
    expected_year INTEGER;
    years INTEGER[];
    value_number NUMERIC;
    expected_number NUMERIC;
    range_parts TEXT[];
    expected_json JSONB;
    folded TEXT;
BEGIN
    IF p_operator = 'IS_UNKNOWN' THEN
        RETURN p_value IS NULL OR btrim(p_value) = '';
    END IF;
    IF p_operator = 'IS_KNOWN' THEN
        RETURN p_value IS NOT NULL AND btrim(p_value) <> '';
    END IF;
    IF p_operator = 'IS_TRUE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('true','t','1','yes','y','deceased');
    END IF;
    IF p_operator = 'IS_FALSE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('false','f','0','no','n');
    END IF;

    folded := lower(btrim(coalesce(p_value, '')));

    -- Negative operators are answered before the NULL guard: a record with no
    -- recorded cause of death genuinely "did not die of tuberculosis".
    IF p_operator = 'NOT_EQUALS' THEN
        RETURN folded <> lower(btrim(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'NOT_CONTAINS' THEN
        RETURN position(lower(btrim(coalesce(p_expected, ''))) IN folded) = 0;
    END IF;
    IF p_operator IN ('CONTAINS_ANY', 'NOT_CONTAINS_ANY', 'STARTS_WITH_ANY', 'NOT_IN') THEN
        BEGIN
            expected_json := p_expected::jsonb;
        EXCEPTION WHEN OTHERS THEN
            expected_json := to_jsonb(ARRAY[coalesce(p_expected, '')]);
        END;
        IF jsonb_typeof(expected_json) <> 'array' THEN
            expected_json := jsonb_build_array(expected_json);
        END IF;
        IF p_operator = 'CONTAINS_ANY' THEN
            RETURN folded <> '' AND EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(expected_json) item
                WHERE btrim(item) <> '' AND position(lower(btrim(item)) IN folded) > 0
            );
        END IF;
        IF p_operator = 'NOT_CONTAINS_ANY' THEN
            RETURN NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(expected_json) item
                WHERE btrim(item) <> '' AND position(lower(btrim(item)) IN folded) > 0
            );
        END IF;
        IF p_operator = 'STARTS_WITH_ANY' THEN
            RETURN folded <> '' AND EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(expected_json) item
                WHERE btrim(item) <> '' AND starts_with(folded, lower(btrim(item)))
            );
        END IF;
        RETURN NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(expected_json) item
            WHERE lower(btrim(item)) = folded
        );
    END IF;

    IF p_value IS NULL THEN
        RETURN FALSE;
    END IF;
    IF p_operator = 'EQUALS' THEN
        RETURN folded = lower(btrim(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'STARTS_WITH' THEN
        RETURN starts_with(lower(p_value), lower(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'CONTAINS' THEN
        RETURN position(lower(coalesce(p_expected, '')) IN lower(p_value)) > 0;
    END IF;
    IF p_operator = 'IN' THEN
        BEGIN
            expected_json := p_expected::jsonb;
            RETURN EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(expected_json) item
                WHERE lower(btrim(item)) = folded
            );
        EXCEPTION WHEN OTHERS THEN
            RETURN FALSE;
        END;
    END IF;
    IF p_operator IN ('YEAR_EQUALS','BEFORE','AFTER') THEN
        value_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        expected_year := substring(coalesce(p_expected, '') from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        IF value_year IS NULL OR expected_year IS NULL THEN RETURN FALSE; END IF;
        IF p_operator = 'YEAR_EQUALS' THEN RETURN value_year = expected_year; END IF;
        IF p_operator = 'BEFORE' THEN RETURN value_year < expected_year; END IF;
        RETURN value_year > expected_year;
    END IF;
    IF p_operator = 'DATE_RANGE' THEN
        SELECT array_agg(match[1]::INTEGER)
        INTO years
        FROM regexp_matches(coalesce(p_expected, ''), '(1[6-9][0-9]{2}|20[0-9]{2})', 'g') match;
        value_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        RETURN value_year IS NOT NULL AND cardinality(years) >= 2
            AND value_year BETWEEN years[1] AND years[2];
    END IF;
    IF p_operator IN ('GREATER_THAN','LESS_THAN','NUMBER_RANGE') THEN
        BEGIN
            value_number := assistant_api._numeric_value(p_value, 'number');
            IF value_number IS NULL THEN RETURN FALSE; END IF;
            IF p_operator = 'NUMBER_RANGE' THEN
                range_parts := regexp_split_to_array(
                    regexp_replace(coalesce(p_expected, ''), '[^0-9.,-]', '', 'g'), ','
                );
                RETURN cardinality(range_parts) >= 2
                    AND value_number BETWEEN range_parts[1]::NUMERIC AND range_parts[2]::NUMERIC;
            END IF;
            expected_number := NULLIF(regexp_replace(coalesce(p_expected, ''), '[^0-9.-]', '', 'g'), '')::NUMERIC;
            IF expected_number IS NULL THEN RETURN FALSE; END IF;
            IF p_operator = 'GREATER_THAN' THEN RETURN value_number > expected_number; END IF;
            RETURN value_number < expected_number;
        EXCEPTION WHEN OTHERS THEN
            RETURN FALSE;
        END;
    END IF;
    RETURN FALSE;
END;
$$;

-- `_field_text` returned `p_row #>> path` directly, so any registry path pointing at
-- a normalized object ("fields.PLACE OF BURIAL" is {"raw":...,"tokens":[...]}) was
-- compared as serialized JSON. Route it through the same scalar collapse the rest of
-- the API uses, and prefer the canonical JSON over the folded columns.
CREATE OR REPLACE FUNCTION assistant_api._field_text(
    p_file_id INTEGER,
    p_row JSONB,
    p_canonical_name TEXT,
    p_community TEXT,
    p_school TEXT,
    p_semantic_field TEXT
)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT assistant_api._field_display_at_path(
        p_row, p_canonical_name, p_community, p_school, r.canonical_json_path
    )
    FROM assistant.field_registry r
    WHERE r.file_id = p_file_id
      AND r.semantic_field = p_semantic_field
    LIMIT 1;
$$;

-- Recursive boolean evaluation. Unknown fields fail closed.
CREATE OR REPLACE FUNCTION assistant_api._matches_group(
    p_file_id INTEGER,
    p_row JSONB,
    p_canonical_name TEXT,
    p_community TEXT,
    p_school TEXT,
    p_node JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
DECLARE
    node_type TEXT;
    op TEXT;
    items JSONB;
    child JSONB;
    result BOOLEAN;
    field_name TEXT;
    known BOOLEAN;
    expected TEXT;
BEGIN
    IF p_node IS NULL THEN
        RETURN TRUE;
    END IF;
    node_type := jsonb_typeof(p_node);
    IF node_type = 'null' THEN
        RETURN TRUE;
    END IF;

    IF node_type = 'array' THEN
        IF jsonb_array_length(p_node) = 0 THEN
            RETURN TRUE;
        END IF;
        FOR child IN SELECT value FROM jsonb_array_elements(p_node) LOOP
            IF NOT assistant_api._matches_group(
                p_file_id, p_row, p_canonical_name, p_community, p_school, child
            ) THEN
                RETURN FALSE;
            END IF;
        END LOOP;
        RETURN TRUE;
    END IF;

    IF node_type <> 'object' THEN
        RETURN FALSE;
    END IF;

    op := lower(coalesce(p_node->>'op', ''));
    IF op IN ('and', 'or', 'not') THEN
        items := COALESCE(p_node->'items', '[]'::jsonb);
        IF jsonb_typeof(items) <> 'array' OR jsonb_array_length(items) = 0 THEN
            RETURN TRUE;
        END IF;
        IF op = 'not' THEN
            RETURN NOT assistant_api._matches_group(
                p_file_id, p_row, p_canonical_name, p_community, p_school, items->0
            );
        END IF;
        result := (op = 'and');
        FOR child IN SELECT value FROM jsonb_array_elements(items) LOOP
            IF op = 'and' THEN
                IF NOT assistant_api._matches_group(
                    p_file_id, p_row, p_canonical_name, p_community, p_school, child
                ) THEN
                    RETURN FALSE;
                END IF;
            ELSE
                IF assistant_api._matches_group(
                    p_file_id, p_row, p_canonical_name, p_community, p_school, child
                ) THEN
                    RETURN TRUE;
                END IF;
            END IF;
        END LOOP;
        RETURN result;
    END IF;

    field_name := p_node->>'field';
    IF field_name IS NULL THEN
        RETURN FALSE;
    END IF;
    SELECT TRUE INTO known
    FROM assistant.field_registry r
    WHERE r.file_id = p_file_id AND r.semantic_field = field_name
    LIMIT 1;
    IF NOT COALESCE(known, FALSE) THEN
        RETURN FALSE;
    END IF;

    expected := CASE
        WHEN p_node->'value' IS NULL THEN NULL
        WHEN jsonb_typeof(p_node->'value') = 'string' THEN p_node->>'value'
        WHEN jsonb_typeof(p_node->'value') = 'array' THEN (p_node->'value')::text
        ELSE trim(both '"' from (p_node->'value')::text)
    END;

    RETURN COALESCE(
        assistant_api._apply_op(
            assistant_api._field_display(
                p_file_id, p_row, p_canonical_name, p_community, p_school, field_name
            ),
            p_node->>'operator',
            expected
        ),
        FALSE
    );
END;
$$;

-- Keep the historical entry point working; it now delegates to the group evaluator.
CREATE OR REPLACE FUNCTION assistant_api._record_matches(
    p_file_id INTEGER,
    p_row JSONB,
    p_canonical_name TEXT,
    p_community TEXT,
    p_school TEXT,
    p_filters JSONB
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT assistant_api._matches_group(
        p_file_id, p_row, p_canonical_name, p_community, p_school, p_filters
    );
$$;

ALTER FUNCTION assistant_api._matches_group(INTEGER, JSONB, TEXT, TEXT, TEXT, JSONB)
    OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api._matches_group(INTEGER, JSONB, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api._matches_group(INTEGER, JSONB, TEXT, TEXT, TEXT, JSONB) TO assistant_runtime;

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
    SELECT COUNT(*)
    FROM assistant_api.v_current_records records
    WHERE records.file_id = ANY (
        assistant_api._authorized_file_ids(
            p_principal_id, p_requested_file_ids, p_can_use_private
        )
    )
      AND assistant_api._matches_group(
          records.file_id,
          records.row_data_normalized,
          records.canonical_name,
          records.canonical_community,
          records.canonical_school,
          p_filters
      );
$$;

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
    SELECT records.file_id, COUNT(*) AS record_count
    FROM assistant_api.v_current_records records
    WHERE records.file_id = ANY (
        assistant_api._authorized_file_ids(
            p_principal_id, p_requested_file_ids, p_can_use_private
        )
    )
      AND assistant_api._matches_group(
          records.file_id,
          records.row_data_normalized,
          records.canonical_name,
          records.canonical_community,
          records.canonical_school,
          p_filters
      )
    GROUP BY records.file_id
    ORDER BY records.file_id;
$$;

-- Grouped labels now come from the canonical JSON, so "Walpole Island" is no longer
-- reported as "walpole island" and case-distinct labels stop collapsing.
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
        matched.value,
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
    GROUP BY matched.file_id, matched.value
    ORDER BY COUNT(*) DESC, lower(matched.value) ASC NULLS LAST, matched.file_id;
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
    WITH matched AS MATERIALIZED (
        SELECT records.*
        FROM assistant_api.v_current_records records
        WHERE records.file_id = ANY (
            assistant_api._authorized_file_ids(
                p_principal_id, p_requested_file_ids, p_can_use_private
            )
        )
          AND assistant_api._matches_group(
              records.file_id,
              records.row_data_normalized,
              records.canonical_name,
              records.canonical_community,
              records.canonical_school,
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
        matched.row_data_normalized,
        COUNT(*) OVER () AS total_count
    FROM matched
    ORDER BY random(), matched.source_row_id
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 5), 50), 1);
$$;

-- Listing keeps its typed sort but reads display text, and the hard 50-row ceiling
-- becomes 500 so an exhaustive enumeration ("name every student who ...") can be
-- answered in full instead of as an arbitrary first page.
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
                    assistant_api._field_display_at_path(
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school,
                        resolved_sort.canonical_json_path
                    )
                )
                ELSE NULL
            END AS sort_text,
            CASE
                WHEN resolved_sort.sortable AND resolved_sort.semantic_type = 'number'
                THEN assistant_api._numeric_value(
                    assistant_api._field_display_at_path(
                        records.row_data_normalized, records.canonical_name,
                        records.canonical_community, records.canonical_school,
                        resolved_sort.canonical_json_path
                    ),
                    'number'
                )
                ELSE NULL
            END AS sort_number,
            CASE
                WHEN resolved_sort.sortable AND resolved_sort.semantic_type = 'date'
                THEN assistant_api._date_sort_key(
                    assistant_api._field_display_at_path(
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
        WHERE assistant_api._matches_group(
            records.file_id,
            records.row_data_normalized,
            records.canonical_name,
            records.canonical_community,
            records.canonical_school,
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
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_number END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_number END ASC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_date END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_date END ASC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) = 'DESC'
            THEN matched.sort_text END DESC NULLS LAST,
        CASE WHEN upper(coalesce(p_sort_direction, 'ASC')) <> 'DESC'
            THEN matched.sort_text END ASC NULLS LAST,
        lower(matched.canonical_name) ASC NULLS LAST,
        matched.source_row_id ASC
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 25), 500), 1)
    OFFSET GREATEST(COALESCE(p_offset, 0), 0);
$$;

ALTER FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_group_count(TEXT, INTEGER[], BOOLEAN, JSONB)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_group_values(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_sample_page(TEXT, INTEGER[], BOOLEAN, JSONB, INTEGER)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_list(
    TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER, INTEGER
) OWNER TO assistant_gateway_owner;
