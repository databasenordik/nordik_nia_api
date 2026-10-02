-- Dataset aliases were modelled in the catalog but never stored or exposed, so the
-- planner-visible datasets carried an empty alias list. A question naming
-- "the Confirmed Shingwauk list" therefore matched nothing and could not be routed
-- to, or fanned out across, the right list.

ALTER TABLE assistant.dataset_policy
    ADD COLUMN IF NOT EXISTS aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

UPDATE assistant.dataset_policy
SET aliases = ARRAY[
    'shingwauk and wawanosh students master list',
    'student master list',
    'master list',
    'student master',
    'master'
]
WHERE file_id = 49;

UPDATE assistant.dataset_policy
SET aliases = ARRAY[
    'confirmed- shingwauk (wawanosh)',
    'confirmed shingwauk list',
    'confirmed shingwauk',
    'confirmed deaths',
    'confirmed list',
    'confirmed'
]
WHERE file_id = 91;

UPDATE assistant.dataset_policy
SET aliases = ARRAY['additional deaths', 'additional death list']
WHERE file_id = 93;

UPDATE assistant.dataset_policy
SET aliases = ARRAY['potential records', 'potential list', 'potential']
WHERE file_id = 94;

CREATE OR REPLACE VIEW assistant_api.v_datasets AS
SELECT
    f.id AS file_id,
    f.filename,
    f.description,
    f.version AS current_version,
    f.private,
    p.enabled AS assistant_enabled,
    p.user_facing_label,
    p.is_default_people_scope,
    p.aliases
FROM public.file f
JOIN assistant.dataset_policy p ON p.file_id = f.id
WHERE COALESCE(f.is_delete, false) = false;

DROP FUNCTION IF EXISTS assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN);

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
    is_default_people_scope BOOLEAN,
    aliases TEXT[]
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
        d.is_default_people_scope,
        d.aliases
    FROM assistant_api.v_datasets d
    WHERE d.file_id = ANY (
        assistant_api._authorized_file_ids(p_principal_id, p_requested_file_ids, p_can_use_private)
    )
    ORDER BY d.file_id;
$$;

ALTER FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN)
    OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN) TO assistant_runtime;
