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
    IF p_operator = 'IS_KNOWN' THEN
        RETURN p_value IS NOT NULL AND btrim(p_value) <> '';
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

ALTER FUNCTION assistant_api._apply_op(TEXT, TEXT, TEXT) OWNER TO assistant_gateway_owner;

UPDATE assistant.field_registry
SET allowed_operators = array_append(allowed_operators, 'IS_KNOWN')
WHERE semantic_field = 'discharged_date'
  AND NOT ('IS_KNOWN' = ANY (allowed_operators));
