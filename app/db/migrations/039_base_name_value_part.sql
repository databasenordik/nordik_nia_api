-- Add a "base_name" value part: the community/reserve label with its band
-- number removed, so "Chisasibi #66", "Chisasibi #100" and "Chisasibi" group
-- as one community.
--
-- Why this is needed: listing distinct communities in file 49 returns 370
-- values, 191 of which carry a "#<number>" band suffix. Asking for the list
-- "without the hash numbers" was previously inexpressible, so the request was
-- silently ignored. Collapsing them yields 198 distinct communities.
--
-- This is a presentation/grouping projection only. Stored rows are untouched
-- and "raw" remains the default, so no existing answer changes.

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

ALTER FUNCTION assistant_api._value_part(TEXT, TEXT) OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api._value_part(TEXT, TEXT) FROM PUBLIC;
