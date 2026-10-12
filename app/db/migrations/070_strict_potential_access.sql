-- Release decisions come from the current raw row, never a cached normalized flag.
-- TRUE alone releases a Potential row. Missing policy, source, or flag denies access.
ALTER TABLE assistant.dataset_policy ALTER COLUMN row_access_allowed SET DEFAULT ARRAY['true']::text[];
UPDATE assistant.dataset_policy SET row_access_allowed = ARRAY['true']::text[]
WHERE row_access_key IS NOT NULL;

CREATE OR REPLACE FUNCTION assistant_api._row_released(p_file_id integer, p_value text)
RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = assistant_api, assistant, public, pg_temp AS $$
    SELECT CASE WHEN p_file_id = 94 THEN lower(btrim(coalesce(p_value, ''))) = 'true'
                WHEN p.row_access_key IS NULL THEN true
                ELSE lower(btrim(coalesce(p_value, ''))) = ANY(p.row_access_allowed) END
    FROM (SELECT 1) one LEFT JOIN assistant.dataset_policy p ON p.file_id = p_file_id;
$$;

CREATE OR REPLACE VIEW assistant_api.v_current_raw_records WITH (security_barrier = true) AS
SELECT d.id, d.file_id, d.version, d.row_data, f.filename, f.private
FROM public.file_data d
JOIN public.file f ON f.id = d.file_id AND f.version = d.version
LEFT JOIN assistant.dataset_policy p ON p.file_id = d.file_id
WHERE coalesce(f.is_delete, false) = false
  AND assistant_api._row_released(d.file_id, d.row_data->>(
      CASE WHEN d.file_id = 94 THEN 'ACCESS' ELSE coalesce(p.row_access_key, 'ACCESS') END));

CREATE OR REPLACE VIEW assistant_api.v_current_records WITH (security_barrier = true) AS
SELECT n.id, n.source_row_id, n.file_id, n.version, n.row_data_normalized,
       n.search_text, n.canonical_name, n.canonical_community, n.canonical_school,
       n.source_created_at, n.source_updated_at, f.filename, f.private
FROM public.file_data_normalized n
JOIN public.file f ON f.id = n.file_id AND f.version = n.version
LEFT JOIN assistant.dataset_policy p ON p.file_id = n.file_id
WHERE coalesce(f.is_delete, false) = false AND n.status = 'ready'
  AND ((n.file_id <> 94 AND p.row_access_key IS NULL) OR EXISTS (
      SELECT 1 FROM public.file_data d
      WHERE d.id = n.source_row_id AND d.file_id = n.file_id AND d.version = n.version
        AND assistant_api._row_released(d.file_id, d.row_data->>(
            CASE WHEN d.file_id = 94 THEN 'ACCESS' ELSE coalesce(p.row_access_key, 'ACCESS') END))
  ));

REVOKE ALL ON FUNCTION assistant_api._row_released(integer, text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api._row_released(integer, text) TO assistant_gateway_owner;

-- Data-derived catalog caches may predate a revocation. Never return their values or
-- counts for Potential; static schema and validation rules remain available.
CREATE OR REPLACE VIEW assistant_api.v_dataset_fields AS
SELECT file_id,semantic_field,human_label,aliases,canonical_json_path,raw_json_keys,
       semantic_type,allowed_operators,searchable,aggregatable,sortable,quoteable,
       sensitivity,exposure_policy,
       CASE WHEN file_id=94 THEN validation_rules - ARRAY['recorded_values','recorded_count','record_count']
            ELSE validation_rules END AS validation_rules,
       evidence_allowed,raw_fetch_allowed
FROM assistant.field_registry;
SELECT assistant.refresh_fill_rates();
SELECT assistant.refresh_recorded_values();
