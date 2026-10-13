-- Date matching uses intervals at the recorded precision. | denotes alternatives;
-- & denotes real events. A definite match must hold for every alternative branch.
CREATE OR REPLACE FUNCTION assistant_api._date_bounds(p_value text)
RETURNS date[] LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE parts text[]; y integer; m integer; d integer; low_date date; high_date date; start_bounds date[]; end_bounds date[];
BEGIN
    IF position(':' in p_value) > 0 THEN
        parts := string_to_array(p_value, ':');
        IF cardinality(parts) <> 2 THEN RETURN NULL; END IF;
        start_bounds := assistant_api._date_bounds(parts[1]);
        end_bounds := assistant_api._date_bounds(parts[2]);
        IF start_bounds IS NULL OR end_bounds IS NULL OR start_bounds[1] > end_bounds[2] THEN RETURN NULL; END IF;
        RETURN ARRAY[start_bounds[1],end_bounds[2]];
    END IF;
    parts := regexp_match(btrim(coalesce(p_value,'')), '^([0-9]{4})(?:-([0-9]{2})(?:-([0-9]{2}))?)?$');
    IF parts IS NULL THEN RETURN NULL; END IF;
    y := parts[1]::integer; m := coalesce(parts[2]::integer,0); d := coalesce(parts[3]::integer,0);
    IF y < 1600 OR y > 2100 OR m > 12 OR d > 31 OR (m = 0 AND d <> 0) THEN RETURN NULL; END IF;
    low_date := make_date(y, greatest(m,1), greatest(d,1));
    high_date := CASE WHEN d <> 0 THEN low_date
                     WHEN m = 0 THEN make_date(y,12,31)
                     ELSE (make_date(y,m,1) + interval '1 month' - interval '1 day')::date END;
    RETURN ARRAY[low_date,high_date];
EXCEPTION WHEN datetime_field_overflow THEN RETURN NULL;
END $$;

CREATE OR REPLACE FUNCTION assistant_api._date_leaf_matches(p_value text, p_operator text, p_expected jsonb)
RETURNS boolean LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE actual date[]; target date[]; finish date[]; item jsonb;
BEGIN
    actual := assistant_api._date_bounds(p_value);
    IF actual IS NULL THEN RETURN false; END IF;
    IF p_operator = 'DATE_RANGE' THEN
        IF jsonb_typeof(p_expected) <> 'array' OR jsonb_array_length(p_expected) <> 2 THEN RETURN false; END IF;
        target := assistant_api._date_bounds(p_expected->>0);
        finish := assistant_api._date_bounds(p_expected->>1);
        RETURN coalesce(target[1] <= actual[1] AND actual[2] <= finish[2], false);
    END IF;
    IF p_operator = 'IN' THEN
        IF jsonb_typeof(p_expected) <> 'array' THEN RETURN false; END IF;
        FOR item IN SELECT value FROM jsonb_array_elements(p_expected) LOOP
            IF assistant_api._date_leaf_matches(p_value,'EQUALS',item) THEN RETURN true; END IF;
        END LOOP;
        RETURN false;
    END IF;
    IF p_operator = 'NOT_IN' THEN
        IF jsonb_typeof(p_expected) <> 'array' OR jsonb_array_length(p_expected) = 0 THEN RETURN false; END IF;
        FOR item IN SELECT value FROM jsonb_array_elements(p_expected) LOOP
            IF NOT assistant_api._date_leaf_matches(p_value,'NOT_EQUALS',item) THEN RETURN false; END IF;
        END LOOP;
        RETURN true;
    END IF;
    target := assistant_api._date_bounds(p_expected #>> '{}');
    IF target IS NULL THEN RETURN false; END IF;
    RETURN CASE p_operator
        WHEN 'YEAR_EQUALS' THEN extract(year FROM actual[1]) = extract(year FROM target[1])
                               AND extract(year FROM actual[2]) = extract(year FROM target[1])
        WHEN 'EQUALS' THEN target[1] <= actual[1] AND actual[2] <= target[2]
        WHEN 'NOT_EQUALS' THEN actual[2] < target[1] OR actual[1] > target[2]
        WHEN 'BEFORE' THEN actual[2] < target[1]
        WHEN 'AFTER' THEN actual[1] > target[2]
        ELSE false END;
END $$;

CREATE OR REPLACE FUNCTION assistant_api._date_matches(p_value text, p_operator text, p_expected jsonb)
RETURNS boolean LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE branch text; event text; hit boolean; known boolean := true; branch_known boolean;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN RETURN p_operator = 'IS_UNKNOWN'; END IF;
    IF btrim(p_value) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}$' THEN
        p_value := left(btrim(p_value),10) || ' | ' || left(btrim(p_value),8) || right(btrim(p_value),2);
    END IF;
    IF p_operator IN ('IS_KNOWN','IS_UNKNOWN') THEN
        FOREACH branch IN ARRAY string_to_array(coalesce(p_value,''),'|') LOOP
            branch_known := false;
            FOREACH event IN ARRAY string_to_array(branch,'&') LOOP
                branch_known := branch_known OR (assistant_api._date_bounds(event) IS NOT NULL)
                    OR (btrim(event) ~ '^0000-(0[1-9]|1[0-2])-(00|0[1-9]|[12][0-9]|3[01])$'
                        AND assistant_api._date_bounds(regexp_replace(btrim(event),'^0000-','2000-')) IS NOT NULL);
            END LOOP;
            known := known AND branch_known;
        END LOOP;
        RETURN CASE WHEN p_operator = 'IS_KNOWN' THEN known ELSE NOT known END;
    END IF;
    IF p_value IS NULL OR btrim(p_value) = '' THEN RETURN false; END IF;
    FOREACH branch IN ARRAY string_to_array(p_value,'|') LOOP
        hit := false;
        FOREACH event IN ARRAY string_to_array(branch,'&') LOOP
            IF p_operator IN ('NOT_EQUALS','NOT_IN') THEN
                IF NOT assistant_api._date_leaf_matches(event,p_operator,p_expected) THEN RETURN false; END IF;
                hit := true;
            ELSE
                hit := hit OR assistant_api._date_leaf_matches(event,p_operator,p_expected);
            END IF;
        END LOOP;
        IF NOT hit THEN RETURN false; END IF;
    END LOOP;
    RETURN true;
END $$;

-- An expression cannot be reduced to a single date for ordering or interval arithmetic.
CREATE OR REPLACE FUNCTION assistant_api._parse_date_value(p_value text)
RETURNS date LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    -- This helper measures elapsed days. Partial dates cannot establish an exact point.
    SELECT CASE WHEN p_value ~ '[|&:]|^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}$'
                     OR (assistant_api._date_bounds(p_value))[1] IS DISTINCT FROM (assistant_api._date_bounds(p_value))[2] THEN NULL
                ELSE (assistant_api._date_bounds(p_value))[1] END;
$$;
CREATE OR REPLACE FUNCTION assistant_api._date_precision(p_value text)
RETURNS text LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE WHEN assistant_api._date_bounds(p_value) IS NULL OR p_value ~ '[:|&]' THEN NULL
        WHEN btrim(p_value) ~ '^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$' THEN 'day'
        WHEN btrim(p_value) ~ '^[0-9]{4}-(0[1-9]|1[0-2])(?:-00)?$' THEN 'month'
        ELSE 'year' END;
$$;

CREATE OR REPLACE FUNCTION assistant_api._matches_group(
    p_file_id integer,
    p_row jsonb,
    p_canonical_name text,
    p_community text,
    p_school text,
    p_node jsonb
) RETURNS boolean
LANGUAGE plpgsql
STABLE SECURITY DEFINER
SET search_path TO 'assistant_api', 'assistant', 'public'
AS $function$
DECLARE
    node_type TEXT;
    op TEXT;
    items JSONB;
    child JSONB;
    result BOOLEAN;
    field_name TEXT;
    field_path TEXT;
    field_type TEXT;
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

    -- One lookup, and it carries the path. An unregistered field finds no row and is
    -- rejected here, which is what the separate known check did.
    SELECT r.canonical_json_path, r.semantic_type INTO field_path, field_type
    FROM assistant.field_registry r
    WHERE r.file_id = p_file_id AND r.semantic_field = field_name
    LIMIT 1;
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    expected := CASE
        WHEN p_node->'value' IS NULL THEN NULL
        WHEN jsonb_typeof(p_node->'value') = 'string' THEN p_node->>'value'
        WHEN jsonb_typeof(p_node->'value') = 'array' THEN (p_node->'value')::text
        ELSE trim(both '"' from (p_node->'value')::text)
    END;

    IF field_type = 'date' THEN
        RETURN assistant_api._date_matches(
            assistant_api._field_display_at_path(p_row,p_canonical_name,p_community,p_school,field_path),
            p_node->>'operator',p_node->'value');
    END IF;

    RETURN COALESCE(
        assistant_api._apply_op(
            assistant_api._field_display_at_path(
                p_row, p_canonical_name, p_community, p_school, field_path
            ),
            p_node->>'operator',
            expected
        ),
        FALSE
    );
END;
$function$;

CREATE OR REPLACE FUNCTION assistant_api._value_part(p_value TEXT, p_part TEXT)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
DECLARE
    year_text TEXT;
    tokens TEXT[];
    stripped TEXT;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN
        RETURN NULL;
    END IF;
    IF p_part IS NULL OR p_part = 'raw' THEN
        RETURN btrim(p_value);
    END IF;
    IF p_part IN ('year', 'decade') THEN
        IF p_value ~ '[|&:]|^[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}$' THEN
            SELECT CASE WHEN count(DISTINCT extract(year from span[1]))=1
                              AND bool_and(span IS NOT NULL AND extract(year from span[1])=extract(year from span[2]))
                        THEN min(extract(year from span[1]))::integer::text END
            INTO year_text FROM regexp_split_to_table(regexp_replace(p_value,
                '^([0-9]{4}-[0-9]{2}-)([0-9]{2})-([0-9]{2})$', E'\\1\\2 | \\1\\3'), '[|&]') event
            CROSS JOIN LATERAL (SELECT assistant_api._date_bounds(btrim(event)) AS span) bounds;
        ELSE
            year_text := substring(p_value from '(1[6-9][0-9]{2}|20[0-9]{2})');
        END IF;
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
    IF p_part = 'base_name' THEN
        -- Drop "#12", "#12/136" and "#2 - 46" style band numbers anywhere in
        -- the label, then tidy the separators they leave behind.
        stripped := regexp_replace(
            p_value,
            '[[:space:]]*#[[:space:]]*[0-9]+([[:space:]]*[/-][[:space:]]*[0-9]+)*',
            '',
            'g'
        );
        stripped := regexp_replace(stripped, '[[:space:]]+', ' ', 'g');
        stripped := btrim(stripped);
        stripped := btrim(stripped, '/- ');
        IF stripped = '' THEN
            -- A label that was only a band number keeps its original text
            -- rather than collapsing every such row into one empty group.
            RETURN btrim(p_value);
        END IF;
        RETURN stripped;
    END IF;
    RETURN btrim(p_value);
END;
$$;

-- Statistics must not pick an arbitrary year from alternatives or multiple events.
CREATE OR REPLACE FUNCTION assistant_api._numeric_value(p_value TEXT, p_part TEXT)
RETURNS NUMERIC LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE candidate TEXT;
BEGIN
    IF p_value IS NULL OR btrim(p_value) = '' THEN RETURN NULL; END IF;
    IF p_part IN ('year','decade') THEN
        candidate := assistant_api._value_part(p_value, 'year');
    ELSIF p_part IS NULL AND btrim(p_value) ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN
        IF NOT assistant_api._date_matches(p_value,'IS_KNOWN','null'::jsonb) THEN RETURN NULL; END IF;
        candidate := assistant_api._value_part(p_value,'year');
    ELSE
        candidate := substring(p_value from '-?[0-9]+(?:\.[0-9]+)?');
    END IF;
    IF candidate IS NULL THEN RETURN NULL; END IF;
    RETURN candidate::NUMERIC;
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN RETURN NULL;
END;
$$;
