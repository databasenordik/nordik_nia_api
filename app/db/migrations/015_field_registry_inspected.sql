ALTER TABLE assistant.field_registry
    ADD COLUMN IF NOT EXISTS evidence_allowed BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE assistant.field_registry
    ADD COLUMN IF NOT EXISTS raw_fetch_allowed BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE assistant.field_registry
SET canonical_json_path = 'canonical.display_name'
WHERE semantic_field = 'student_name';

UPDATE assistant.field_registry
SET canonical_json_path = 'canonical.deceased_status'
WHERE semantic_field = 'deceased_status';

UPDATE assistant.field_registry
SET canonical_json_path = 'chat.narrative_bundle.notes'
WHERE semantic_field = 'notes';

DELETE FROM assistant.field_registry
WHERE file_id = 49 AND semantic_field IN ('school', 'cause_of_death');

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
        WHEN r.canonical_json_path IN ('canonical.name', 'canonical.display_name') THEN p_canonical_name
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
