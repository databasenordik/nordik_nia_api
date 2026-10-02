-- Resolve a filtered field's JSON path once per row instead of twice, and stop routing it
-- through a function Postgres cannot inline.
--
-- Every structured_* entry point filters rows by calling _matches_group once per row. At
-- each leaf predicate that did two lookups against assistant.field_registry: one to decide
-- the field was known, then _field_display to fetch canonical_json_path and read the value.
-- _field_display is SECURITY DEFINER with SET search_path, so the planner cannot inline it
-- and each call pays a role switch, a GUC save/restore and a nested query.
--
-- Measured on file 49 (8,347 stored rows, CONTAINS on community):
--   raw scan                                     56 ms
--   _field_display_at_path + ILIKE               93 ms   <- the actual work
--   _field_display + ILIKE                      547 ms   <- + per-row registry indirection
--   _record_matches (the production path)       692 ms
--
-- So roughly four fifths of a filtered scan was the indirection, not the comparison. This
-- reads canonical_json_path directly -- _matches_group is already SECURITY DEFINER with the
-- same search_path, and already queried this table for the known check -- and hands it to
-- _field_display_at_path, which is what _field_display would have called anyway.
--
-- Behaviour is unchanged at every leaf:
--   * field absent from the registry -> FALSE, as the known check did;
--   * field present with a NULL canonical_json_path -> _field_display_at_path(..., NULL),
--     exactly what _field_display returned for that row;
--   * field present with a path -> the same value from the same path.
-- Only _matches_group changes, so all twelve callers benefit without being touched.

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
    SELECT r.canonical_json_path INTO field_path
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
