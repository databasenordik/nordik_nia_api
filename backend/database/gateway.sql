-- Administrator migration. Never run through the assistant connection.
-- Requires the reviewed assistant_api schema from postgres-live-2026-10-15.dump.
BEGIN;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'opencode_research_ro') THEN
    CREATE ROLE opencode_research_ro NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
  END IF;
END $$;
CREATE SCHEMA IF NOT EXISTS research_api AUTHORIZATION assistant_gateway_owner;
REVOKE ALL ON SCHEMA research_api FROM PUBLIC;
CREATE TABLE IF NOT EXISTS research_api.installation (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  id uuid NOT NULL DEFAULT public.gen_random_uuid(),
  contract integer NOT NULL DEFAULT 1 CHECK (contract = 1)
);
INSERT INTO research_api.installation(singleton) VALUES (true) ON CONFLICT DO NOTHING;
ALTER TABLE research_api.installation OWNER TO assistant_gateway_owner;
REVOKE ALL ON ALL TABLES IN SCHEMA research_api FROM PUBLIC;

CREATE OR REPLACE FUNCTION research_api.identity() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, research_api, pg_temp
AS $$ SELECT jsonb_build_object('database', id::text, 'contract', contract, 'dates', 'dates-2') FROM research_api.installation $$;

CREATE OR REPLACE FUNCTION research_api.catalog(p_principal text, p_file_id integer, p_private boolean)
RETURNS jsonb LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, research_api, assistant_api, assistant, public, pg_temp
AS $$
DECLARE result jsonb;
BEGIN
  IF p_file_id NOT IN (49,91,93,94) OR NOT (p_file_id = ANY(assistant_api._authorized_file_ids(p_principal, ARRAY[p_file_id], p_private))) THEN
    RAISE EXCEPTION 'List unavailable or access denied' USING ERRCODE='42501';
  END IF;
  SELECT jsonb_build_object('fileID', d.file_id, 'label', d.user_facing_label, 'version', d.current_version,
    'fields', COALESCE((SELECT jsonb_agg(jsonb_build_object(
      'name', f.semantic_field, 'label', f.human_label, 'type', f.semantic_type,
      'operators', f.allowed_operators, 'aggregatable', f.aggregatable, 'sortable', f.sortable,
      'quoteable', f.quoteable, 'evidence', f.evidence_allowed
    ) ORDER BY f.semantic_field) FROM assistant.field_registry f
      WHERE f.file_id=p_file_id AND f.exposure_policy='authorized'), '[]'::jsonb))
  INTO result FROM assistant_api.list_datasets(p_principal, ARRAY[p_file_id], p_private) d WHERE d.file_id=p_file_id;
  IF result IS NULL THEN RAISE EXCEPTION 'List unavailable or access denied' USING ERRCODE='42501'; END IF;
  RETURN result;
END $$;

DROP FUNCTION IF EXISTS research_api.snapshot(text,integer,boolean);
CREATE OR REPLACE FUNCTION research_api.snapshot(p_principal text, p_file_id integer, p_private boolean, p_fields text[] DEFAULT NULL)
RETURNS jsonb LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, research_api, assistant_api, assistant, public, pg_temp
-- Published list sizes differ greatly; cached generic plans repeatedly exceeded the deadline on Master.
SET plan_cache_mode = force_custom_plan
AS $$
DECLARE dataset jsonb; records jsonb; raw_count bigint; normalized_count bigint; unique_count bigint; stale_count bigint; revision text;
BEGIN
  dataset := research_api.catalog(p_principal,p_file_id,p_private);
  SELECT count(*) INTO raw_count FROM assistant_api.v_current_raw_records WHERE file_id=p_file_id;
  SELECT count(*), count(DISTINCT r.source_row_id), count(*) FILTER(WHERE d.id IS NULL OR d.file_id<>r.file_id OR d.version<>r.version OR d.updated_at<>r.source_updated_at)
  INTO normalized_count, unique_count, stale_count FROM assistant_api.v_current_records r
    LEFT JOIN public.file_data d ON d.id=r.source_row_id
    WHERE r.file_id=p_file_id;
  IF raw_count<>normalized_count OR unique_count<>normalized_count OR stale_count<>0 THEN RAISE EXCEPTION 'Current normalization is incomplete or stale' USING ERRCODE='55000'; END IF;
  -- EXCEPT verifies the released identities without a quadratic join under the views' low row estimates.
  IF EXISTS(SELECT source_row_id FROM assistant_api.v_current_records WHERE file_id=p_file_id
            EXCEPT SELECT id FROM assistant_api.v_current_raw_records WHERE file_id=p_file_id) THEN
    RAISE EXCEPTION 'Current normalization has conflicting release membership' USING ERRCODE='55000';
  END IF;
  IF raw_count>50000 THEN RAISE EXCEPTION 'List exceeds the qualified population budget' USING ERRCODE='54000'; END IF;
  -- Resolve registry paths once per snapshot, not once per field on every source row.
  WITH projected_fields AS MATERIALIZED (
    SELECT f.semantic_field,f.canonical_json_path,f.raw_json_keys,
      f.semantic_type='date' OR f.semantic_field LIKE '%\_date' ESCAPE '\' AS date_field,
      f.semantic_field ~ '_date_(year|month|day)$' AS date_component,
      string_to_array(f.canonical_json_path,'.') AS raw_path,
      CASE WHEN f.canonical_json_path LIKE 'fields.%' THEN
        string_to_array(btrim(regexp_replace(f.canonical_json_path,'[[:space:]]+',' ','g')),'.') END AS normalized_path,
      ARRAY(SELECT btrim(regexp_replace(k.key,'[[:space:]]+',' ','g')) FROM unnest(f.raw_json_keys) WITH ORDINALITY k(key,ordinal) ORDER BY k.ordinal) AS normalized_keys
    FROM assistant.field_registry f WHERE f.file_id=p_file_id AND f.exposure_policy='authorized'
      AND (p_fields IS NULL OR f.semantic_field=ANY(p_fields))
  )
  SELECT COALESCE(jsonb_agg(jsonb_build_object('sourceRowID',r.source_row_id::text,'values',
    CASE WHEN cardinality(p_fields)=0 THEN '{}'::jsonb ELSE COALESCE((
    SELECT jsonb_object_agg(f.semantic_field,
      CASE WHEN f.semantic_field='first_name' THEN COALESCE(r.row_data_normalized #>> '{canonical,name_parts,first}',r.row_data_normalized #>> '{canonical,first_name}')
           WHEN f.semantic_field='last_name' THEN COALESCE(r.row_data_normalized #>> '{canonical,name_parts,last}',r.row_data_normalized #>> '{canonical,last_name}')
           -- The adapter derives components from the parent source expression; old metadata is lossy.
           WHEN f.date_component THEN NULL
           -- Date evidence must retain source precision/operators; normalized iso/kind can be lossy.
           WHEN f.date_field THEN COALESCE(
             (r.row_data_normalized #> f.raw_path)->>'raw',
             (r.row_data_normalized #> f.normalized_path)->>'raw',
             (SELECT COALESCE(r.row_data_normalized #>> ARRAY['fields',k.key,'raw'],
                         r.row_data_normalized #>> ARRAY['fields',f.normalized_keys[k.ordinal::integer],'raw'])
                FROM unnest(f.raw_json_keys) WITH ORDINALITY AS k(key,ordinal)
                -- Some registered headers contain newlines; stored field keys use single spaces.
                WHERE COALESCE(r.row_data_normalized #>> ARRAY['fields',k.key,'raw'],
                         r.row_data_normalized #>> ARRAY['fields',f.normalized_keys[k.ordinal::integer],'raw']) IS NOT NULL
                ORDER BY k.ordinal LIMIT 1),
             assistant_api._field_display_at_path(r.row_data_normalized,r.canonical_name,r.canonical_community,r.canonical_school,f.canonical_json_path))
           ELSE assistant_api._field_display_at_path(r.row_data_normalized,r.canonical_name,r.canonical_community,r.canonical_school,f.canonical_json_path) END)
    FROM projected_fields f
  ),'{}'::jsonb) END) ORDER BY r.source_row_id), '[]'::jsonb)
  INTO records FROM assistant_api.v_current_records r WHERE r.file_id=p_file_id;
  IF octet_length(records::text)>33554432 THEN RAISE EXCEPTION 'List exceeds the qualified byte budget' USING ERRCODE='54000'; END IF;
  -- Include field/permission policy, same-version edits and source membership; no cross-question result cache.
  SELECT md5(COALESCE(string_agg(r.source_row_id::text || ':' || r.row_data_normalized::text || ':' || r.source_updated_at::text,'|' ORDER BY r.source_row_id),'')
    || dataset::text || COALESCE((SELECT row_to_json(p)::text FROM assistant.dataset_policy p WHERE p.file_id=p_file_id),''))
    INTO revision FROM assistant_api.v_current_records r WHERE r.file_id=p_file_id;
  RETURN jsonb_build_object('binding',jsonb_build_object('database',(SELECT id::text FROM research_api.installation),'fileID',p_file_id),
    'dataset',dataset,'at',transaction_timestamp()::text,'fingerprint',revision,'records',records);
END $$;

ALTER FUNCTION research_api.identity() OWNER TO assistant_gateway_owner;
ALTER FUNCTION research_api.catalog(text,integer,boolean) OWNER TO assistant_gateway_owner;
ALTER FUNCTION research_api.snapshot(text,integer,boolean,text[]) OWNER TO assistant_gateway_owner;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA research_api FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA research_api FROM opencode_research_ro;
GRANT USAGE ON SCHEMA research_api TO opencode_research_ro;
GRANT EXECUTE ON FUNCTION research_api.identity(),research_api.catalog(text,integer,boolean),research_api.snapshot(text,integer,boolean,text[]) TO opencode_research_ro;
ALTER ROLE opencode_research_ro SET default_transaction_read_only=on;
COMMIT;
