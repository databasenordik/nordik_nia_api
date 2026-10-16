-- A numeric measurement cannot be inferred from the leading digits of a date or
-- an uncertain/narrative cell. Preserve explicit date-year statistics separately.
CREATE OR REPLACE FUNCTION assistant_api._numeric_value(p_value text, p_part text)
RETURNS numeric LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE candidate text;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN RETURN NULL; END IF;
    IF p_part = 'number' THEN
        IF btrim(p_value) !~* '^-?[0-9]+(?:\.[0-9]+)?(?:\s+(?:years?|yrs?|days?|months?))?$' THEN RETURN NULL; END IF;
        candidate := substring(p_value from '-?[0-9]+(?:\.[0-9]+)?');
    ELSIF p_part IN ('year','decade') THEN
        candidate := assistant_api._value_part(p_value,'year');
    ELSIF p_part IS NULL AND btrim(p_value) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN
        IF NOT assistant_api._date_matches(p_value,'IS_KNOWN','null'::jsonb) THEN RETURN NULL; END IF;
        candidate := assistant_api._value_part(p_value,'year');
    ELSE
        candidate := substring(p_value from '-?[0-9]+(?:\.[0-9]+)?');
    END IF;
    RETURN candidate::numeric;
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN RETURN NULL;
END $$;

-- Keep the existing predicate implementation and its date/Boolean semantics.
-- Add the numeric type check to its leaf dispatch, without another field lookup.
DO $$ DECLARE definition text; marker text; replacement text;
BEGIN
    definition := pg_get_functiondef('assistant_api._matches_group(integer,jsonb,text,text,text,jsonb)'::regprocedure);
    marker := 'IF field_type = ''date'' THEN';
    replacement := 'IF field_type = ''number'' AND p_node->>''operator'' IN (''GREATER_THAN'',''LESS_THAN'',''NUMBER_RANGE'')
        AND assistant_api._numeric_value(assistant_api._field_display_at_path(
            p_row,p_canonical_name,p_community,p_school,field_path),''number'') IS NULL THEN
        RETURN FALSE;
    END IF;
    IF field_type = ''date'' THEN';
    IF position('p_node->>''operator'' IN (''GREATER_THAN''' in definition) = 0 THEN
        IF position(marker in definition) = 0 THEN RAISE EXCEPTION 'Unknown predicate function; cannot apply typed numeric guard'; END IF;
        EXECUTE replace(definition,marker,replacement);
    END IF;
END $$;
