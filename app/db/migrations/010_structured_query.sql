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
    SELECT CASE
        WHEN r.canonical_json_path = 'canonical.name' THEN p_canonical_name
        WHEN r.canonical_json_path = 'canonical.community' THEN p_community
        WHEN r.canonical_json_path = 'canonical.school' THEN p_school
        WHEN r.canonical_json_path IS NULL THEN NULL
        ELSE p_row #>> string_to_array(r.canonical_json_path, '.')
    END
    FROM assistant.field_registry r
    WHERE r.file_id = p_file_id
      AND r.semantic_field = p_semantic_field
    LIMIT 1;
$$;

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
    lowered TEXT;
BEGIN
    IF p_operator = 'IS_UNKNOWN' THEN
        RETURN p_value IS NULL OR btrim(p_value) = '';
    END IF;
    IF p_operator = 'IS_TRUE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('true', 't', '1', 'yes', 'deceased');
    END IF;
    IF p_operator = 'IS_FALSE' THEN
        RETURN lower(coalesce(p_value, '')) IN ('false', 'f', '0', 'no');
    END IF;
    IF p_value IS NULL THEN
        RETURN FALSE;
    END IF;
    lowered := lower(p_value);
    IF p_operator = 'EQUALS' THEN
        RETURN lowered = lower(coalesce(p_expected, ''));
    END IF;
    IF p_operator = 'STARTS_WITH' THEN
        RETURN starts_with(lowered, lower(coalesce(p_expected, '')));
    END IF;
    IF p_operator = 'CONTAINS' THEN
        RETURN position(lower(coalesce(p_expected, '')) IN lowered) > 0;
    END IF;
    IF p_operator IN ('YEAR_EQUALS', 'BEFORE', 'AFTER') THEN
        value_year := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        expected_year := substring(coalesce(p_expected, '') from '(1[6-9][0-9]{2}|20[0-9]{2})')::INTEGER;
        IF value_year IS NULL OR expected_year IS NULL THEN
            RETURN FALSE;
        END IF;
        IF p_operator = 'YEAR_EQUALS' THEN
            RETURN value_year = expected_year;
        END IF;
        IF p_operator = 'BEFORE' THEN
            RETURN value_year < expected_year;
        END IF;
        RETURN value_year > expected_year;
    END IF;
    RETURN FALSE;
END;
$$;

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
    SELECT COALESCE(
        (
            SELECT bool_and(
                assistant_api._apply_op(
                    assistant_api._field_text(
                        p_file_id, p_row, p_canonical_name, p_community, p_school, f->>'field'
                    ),
                    f->>'operator',
                    CASE
                        WHEN jsonb_typeof(f->'value') = 'string' THEN f->>'value'
                        ELSE trim(both '"' from (f->'value')::text)
                    END
                )
            )
            FROM jsonb_array_elements(COALESCE(p_filters, '[]'::jsonb)) AS f
        ),
        TRUE
    );
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
    SELECT COUNT(*)
    FROM assistant_api.v_current_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND assistant_api._record_matches(
          r.file_id, r.row_data_normalized, r.canonical_name, r.canonical_community, r.canonical_school, p_filters
      );
$$;

CREATE OR REPLACE FUNCTION assistant_api.structured_list(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_filters JSONB,
    p_sort_field TEXT,
    p_sort_direction TEXT,
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
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 20), 50), 1);
$$;

ALTER FUNCTION assistant_api._field_text(INTEGER, JSONB, TEXT, TEXT, TEXT, TEXT) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._apply_op(TEXT, TEXT, TEXT) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._record_matches(INTEGER, JSONB, TEXT, TEXT, TEXT, JSONB) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api._field_text(INTEGER, JSONB, TEXT, TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._apply_op(TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._record_matches(INTEGER, JSONB, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api.structured_count(TEXT, INTEGER[], BOOLEAN, JSONB) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.structured_list(TEXT, INTEGER[], BOOLEAN, JSONB, TEXT, TEXT, INTEGER) TO assistant_runtime;
