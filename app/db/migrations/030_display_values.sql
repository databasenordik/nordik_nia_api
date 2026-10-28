-- Display text, date parsing, and derived value parts.
--
-- `_field_text_at_path` short-circuits canonical.display_name / canonical.community /
-- canonical.school to the denormalized `canonical_*` columns. Those columns are
-- lowercased and punctuation-stripped, so every grouped label and every listed name
-- was rendered in a lossy form and distinct community labels collapsed from 370 to
-- 369. Matching still uses the folded columns; anything the user reads now comes
-- from the canonical JSON.

CREATE OR REPLACE FUNCTION assistant_api._field_display_at_path(
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
        WHEN p_canonical_json_path IS NULL THEN NULL
        ELSE COALESCE(
            NULLIF(
                btrim(
                    assistant_api._semantic_text(
                        p_row #> string_to_array(p_canonical_json_path, '.')
                    )
                ),
                ''
            ),
            CASE p_canonical_json_path
                WHEN 'canonical.name' THEN NULLIF(btrim(p_canonical_name), '')
                WHEN 'canonical.display_name' THEN NULLIF(btrim(p_canonical_name), '')
                WHEN 'canonical.community' THEN NULLIF(btrim(p_community), '')
                WHEN 'canonical.school' THEN NULLIF(btrim(p_school), '')
                ELSE NULL
            END
        )
    END;
$$;

CREATE OR REPLACE FUNCTION assistant_api._field_display(
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

-- Best-effort date parsing for the shapes this corpus actually stores:
-- 1889-12-24, 1903/07/30, 01.12.1890, 1938-10-00, 1907, and any of those with
-- trailing source annotations such as "1902-09-14 (NCTR SOURCE)".
CREATE OR REPLACE FUNCTION assistant_api._parse_date_value(p_value TEXT)
RETURNS DATE
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    parts TEXT[];
    y INTEGER;
    m INTEGER;
    d INTEGER;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN
        RETURN NULL;
    END IF;

    parts := regexp_match(p_value, '(1[6-9][0-9]{2}|20[0-9]{2})[-/.]([0-9]{1,2})[-/.]([0-9]{1,2})');
    IF parts IS NOT NULL THEN
        y := parts[1]::INTEGER;
        m := GREATEST(LEAST(parts[2]::INTEGER, 12), 1);
        d := GREATEST(LEAST(parts[3]::INTEGER, 31), 1);
        BEGIN
            RETURN make_date(y, m, d);
        EXCEPTION WHEN OTHERS THEN
            RETURN make_date(y, m, 1);
        END;
    END IF;

    parts := regexp_match(p_value, '([0-9]{1,2})[-/.]([0-9]{1,2})[-/.](1[6-9][0-9]{2}|20[0-9]{2})');
    IF parts IS NOT NULL THEN
        y := parts[3]::INTEGER;
        m := GREATEST(LEAST(parts[2]::INTEGER, 12), 1);
        d := GREATEST(LEAST(parts[1]::INTEGER, 31), 1);
        BEGIN
            RETURN make_date(y, m, d);
        EXCEPTION WHEN OTHERS THEN
            RETURN make_date(y, m, 1);
        END;
    END IF;

    parts := regexp_match(p_value, '(1[6-9][0-9]{2}|20[0-9]{2})[-/.]([0-9]{1,2})');
    IF parts IS NOT NULL THEN
        y := parts[1]::INTEGER;
        m := GREATEST(LEAST(parts[2]::INTEGER, 12), 1);
        RETURN make_date(y, m, 1);
    END IF;

    parts := regexp_match(p_value, '(1[6-9][0-9]{2}|20[0-9]{2})');
    IF parts IS NOT NULL THEN
        RETURN make_date(parts[1]::INTEGER, 1, 1);
    END IF;

    RETURN NULL;
END;
$$;

-- How precisely a stored date is written. Year- and month-only values are common in
-- this corpus, and an interval computed from them can look negative purely because
-- the missing day defaults to the first of the period. Callers use this to separate
-- a genuine ordering error from a precision artefact.
CREATE OR REPLACE FUNCTION assistant_api._date_precision(p_value TEXT)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN
        RETURN NULL;
    END IF;
    IF p_value ~ '(1[6-9][0-9]{2}|20[0-9]{2})[-/.][0-9]{1,2}[-/.][0-9]{1,2}'
       OR p_value ~ '[0-9]{1,2}[-/.][0-9]{1,2}[-/.](1[6-9][0-9]{2}|20[0-9]{2})' THEN
        IF p_value ~ '[-/.]0{1,2}([^0-9]|$)' THEN
            RETURN 'month';
        END IF;
        RETURN 'day';
    END IF;
    IF p_value ~ '(1[6-9][0-9]{2}|20[0-9]{2})[-/.][0-9]{1,2}'
       OR p_value ~ '[[:alpha:]]{3,}[[:space:]]+(1[6-9][0-9]{2}|20[0-9]{2})' THEN
        RETURN 'month';
    END IF;
    IF p_value ~ '[0-9]{1,2}[[:space:]]*[[:alpha:]]{3,}[[:space:]]*(1[6-9][0-9]{2}|20[0-9]{2})' THEN
        RETURN 'day';
    END IF;
    IF p_value ~ '(1[6-9][0-9]{2}|20[0-9]{2})' THEN
        RETURN 'year';
    END IF;
    RETURN NULL;
END;
$$;

-- Derived presentation buckets. `raw` is the stored label; the rest are computed so
-- the planner can group by year, decade, or a name part without new columns.
CREATE OR REPLACE FUNCTION assistant_api._value_part(p_value TEXT, p_part TEXT)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    year_text TEXT;
    tokens TEXT[];
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN
        RETURN NULL;
    END IF;
    IF p_part IS NULL OR p_part = 'raw' THEN
        RETURN btrim(p_value);
    END IF;
    IF p_part IN ('year', 'decade') THEN
        year_text := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})');
        IF year_text IS NULL THEN
            RETURN NULL;
        END IF;
        IF p_part = 'year' THEN
            RETURN year_text;
        END IF;
        RETURN ((year_text::INTEGER / 10) * 10)::TEXT || 's';
    END IF;
    IF p_part IN ('first_token', 'last_token') THEN
        tokens := regexp_split_to_array(btrim(p_value), '[[:space:]]+');
        IF cardinality(tokens) = 0 THEN
            RETURN NULL;
        END IF;
        IF p_part = 'first_token' THEN
            RETURN tokens[1];
        END IF;
        RETURN tokens[cardinality(tokens)];
    END IF;
    IF p_part = 'number' THEN
        RETURN substring(p_value from '-?[0-9]+(?:\.[0-9]+)?');
    END IF;
    RETURN btrim(p_value);
END;
$$;

-- Numeric projection used by statistics and ordering.
CREATE OR REPLACE FUNCTION assistant_api._numeric_value(p_value TEXT, p_part TEXT)
RETURNS NUMERIC
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    candidate TEXT;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN
        RETURN NULL;
    END IF;
    IF p_part IN ('year', 'decade') THEN
        candidate := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})');
    ELSE
        candidate := substring(p_value from '-?[0-9]+(?:\.[0-9]+)?');
    END IF;
    IF candidate IS NULL THEN
        RETURN NULL;
    END IF;
    BEGIN
        RETURN candidate::NUMERIC;
    EXCEPTION WHEN OTHERS THEN
        RETURN NULL;
    END;
END;
$$;

ALTER FUNCTION assistant_api._field_display_at_path(JSONB, TEXT, TEXT, TEXT, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._field_display(INTEGER, JSONB, TEXT, TEXT, TEXT, TEXT)
    OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._parse_date_value(TEXT) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._date_precision(TEXT) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._value_part(TEXT, TEXT) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api._numeric_value(TEXT, TEXT) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api._field_display_at_path(JSONB, TEXT, TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._field_display(INTEGER, JSONB, TEXT, TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._parse_date_value(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._date_precision(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._value_part(TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api._numeric_value(TEXT, TEXT) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION assistant_api._field_display_at_path(JSONB, TEXT, TEXT, TEXT, TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._field_display(INTEGER, JSONB, TEXT, TEXT, TEXT, TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._parse_date_value(TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._date_precision(TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._value_part(TEXT, TEXT) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api._numeric_value(TEXT, TEXT) TO assistant_runtime;
