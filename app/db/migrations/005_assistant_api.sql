-- Internal current-record view. Runtime does not get SELECT on this object.

CREATE OR REPLACE VIEW assistant_api.v_current_records AS
SELECT
    n.id,
    n.source_row_id,
    n.file_id,
    n.version,
    n.row_data_normalized,
    n.search_text,
    n.canonical_name,
    n.canonical_community,
    n.canonical_school,
    n.source_created_at,
    n.source_updated_at,
    f.filename,
    f.private
FROM public.file_data_normalized n
JOIN public.file f
  ON f.id = n.file_id
 AND f.version = n.version
WHERE COALESCE(f.is_delete, false) = false
  AND n.status = 'ready';

CREATE OR REPLACE VIEW assistant_api.v_current_raw_records AS
SELECT
    d.id,
    d.file_id,
    d.version,
    d.row_data,
    f.filename,
    f.private
FROM public.file_data d
JOIN public.file f
  ON f.id = d.file_id
 AND f.version = d.version
WHERE COALESCE(f.is_delete, false) = false;

CREATE OR REPLACE VIEW assistant_api.v_datasets AS
SELECT
    f.id AS file_id,
    f.filename,
    f.description,
    f.version AS current_version,
    f.private,
    p.enabled AS assistant_enabled,
    p.user_facing_label,
    p.is_default_people_scope
FROM public.file f
JOIN assistant.dataset_policy p ON p.file_id = f.id
WHERE COALESCE(f.is_delete, false) = false;

CREATE OR REPLACE VIEW assistant_api.v_dataset_fields AS
SELECT
    r.file_id,
    r.semantic_field,
    r.human_label,
    r.aliases,
    r.canonical_json_path,
    r.raw_json_keys,
    r.semantic_type,
    r.allowed_operators,
    r.searchable,
    r.aggregatable,
    r.sortable,
    r.quoteable,
    r.sensitivity,
    r.exposure_policy
FROM assistant.field_registry r;

CREATE OR REPLACE VIEW assistant_api.v_reference_entities AS
SELECT 'community'::text AS entity_type, id, name, variants FROM public.communities
UNION ALL
SELECT 'day_school', id, name, variants FROM public.day_schools
UNION ALL
SELECT 'indian_hospital', id, name, variants FROM public.indian_hospitals
UNION ALL
SELECT 'province', id, name, COALESCE(variants, ARRAY[code]) FROM public.provinces;

CREATE OR REPLACE FUNCTION assistant_api._authorized_file_ids(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN
)
RETURNS INTEGER[]
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT COALESCE(array_agg(f.id), ARRAY[]::INTEGER[])
    FROM public.file f
    JOIN assistant.dataset_policy p ON p.file_id = f.id
    WHERE p.enabled = TRUE
      AND COALESCE(f.is_delete, false) = false
      AND f.id = ANY (p_requested_file_ids)
      AND (
            f.private = FALSE
            OR (
                p_can_use_private
                AND EXISTS (
                    SELECT 1
                    FROM assistant.access_grants g
                    WHERE g.principal_id = p_principal_id
                      AND g.file_id = f.id
                      AND g.grant_status = 'active'
                      AND (g.expires_at IS NULL OR g.expires_at > now())
                )
            )
      );
$$;

CREATE OR REPLACE FUNCTION assistant_api.list_datasets(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN
)
RETURNS TABLE (
    file_id INTEGER,
    filename TEXT,
    description TEXT,
    current_version INTEGER,
    private BOOLEAN,
    assistant_enabled BOOLEAN,
    user_facing_label TEXT,
    is_default_people_scope BOOLEAN
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT
        d.file_id,
        d.filename,
        d.description,
        d.current_version,
        d.private,
        d.assistant_enabled,
        d.user_facing_label,
        d.is_default_people_scope
    FROM assistant_api.v_datasets d
    WHERE d.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
    ORDER BY d.file_id;
$$;

CREATE OR REPLACE FUNCTION assistant_api.get_dataset_fields(
    p_principal_id TEXT,
    p_file_id INTEGER,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN
)
RETURNS SETOF assistant_api.v_dataset_fields
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT f.*
    FROM assistant_api.v_dataset_fields f
    WHERE f.file_id = p_file_id
      AND p_file_id = ANY (
          assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
      );
$$;

CREATE OR REPLACE FUNCTION assistant_api.search_current_records(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_query TEXT,
    p_limit INTEGER DEFAULT 20
)
RETURNS TABLE (
    id BIGINT,
    source_row_id BIGINT,
    file_id INTEGER,
    version INTEGER,
    canonical_name TEXT,
    canonical_community TEXT,
    canonical_school TEXT,
    search_text TEXT,
    private BOOLEAN
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
        left(r.search_text, 400),
        r.private
    FROM assistant_api.v_current_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND (
            p_query IS NULL
            OR p_query = ''
            OR r.search_text ILIKE '%' || p_query || '%'
            OR r.canonical_name ILIKE '%' || p_query || '%'
      )
    LIMIT GREATEST(LEAST(COALESCE(p_limit, 20), 50), 1);
$$;

CREATE OR REPLACE FUNCTION assistant_api.get_current_records_by_ids(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_source_row_ids BIGINT[]
)
RETURNS TABLE (
    id BIGINT,
    source_row_id BIGINT,
    file_id INTEGER,
    version INTEGER,
    row_data_normalized JSONB,
    canonical_name TEXT,
    canonical_community TEXT,
    canonical_school TEXT
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
        r.row_data_normalized,
        r.canonical_name,
        r.canonical_community,
        r.canonical_school
    FROM assistant_api.v_current_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND r.source_row_id = ANY (p_source_row_ids);
$$;

CREATE OR REPLACE FUNCTION assistant_api.get_raw_fields_by_ids(
    p_principal_id TEXT,
    p_requested_file_ids INTEGER[],
    p_can_use_private BOOLEAN,
    p_source_row_ids BIGINT[],
    p_keys TEXT[]
)
RETURNS TABLE (
    id BIGINT,
    file_id INTEGER,
    version INTEGER,
    selected JSONB
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT
        r.id,
        r.file_id,
        r.version,
        (
            SELECT jsonb_object_agg(k, r.row_data -> k)
            FROM unnest(p_keys) AS k
        ) AS selected
    FROM assistant_api.v_current_raw_records r
    WHERE r.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
      AND r.id = ANY (p_source_row_ids);
$$;

CREATE OR REPLACE FUNCTION assistant_api.resolve_reference_entity(
    p_principal_id TEXT,
    p_entity_type TEXT,
    p_text TEXT
)
RETURNS TABLE (
    entity_type TEXT,
    id INTEGER,
    name TEXT,
    variants TEXT[]
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public
AS $$
    SELECT e.entity_type, e.id, e.name, e.variants
    FROM assistant_api.v_reference_entities e
    WHERE e.entity_type = p_entity_type
      AND (
            lower(e.name) = lower(p_text)
            OR lower(p_text) = ANY (SELECT lower(v) FROM unnest(e.variants) v)
      )
    LIMIT 5;
$$;
