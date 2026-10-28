-- Rows a list marks as not for release must be invisible to Nia, not filtered per question.
--
-- The standardized Potential list carries an ACCESS column: TRUE for the 23 rows cleared
-- for use, FALSE for the 28 that are not. A planner-side filter would be the wrong place
-- for that -- it would depend on every question remembering to ask, on the reviewer
-- agreeing, and on no other path (count, search, evidence fetch, name suggestion, fill
-- rate) reading the table directly. assistant_api.v_current_records and its raw twin are
-- the two objects every read goes through, so the gate belongs there and applies once.
--
-- The rule is configuration, not a column name in SQL: dataset_policy names the key and
-- the values that mean "released", so the next list to need this is a one-row update.
-- A row whose flag is missing or blank is withheld: an unset flag is not a clearance, and
-- for records of children who died at a residential school the safe default is silence.

ALTER TABLE assistant.dataset_policy
    ADD COLUMN IF NOT EXISTS row_access_key TEXT,
    ADD COLUMN IF NOT EXISTS row_access_allowed TEXT[] NOT NULL
        DEFAULT ARRAY['true', 'yes', '1']::TEXT[];

COMMENT ON COLUMN assistant.dataset_policy.row_access_key IS
    'Raw column that says whether a row may be read. NULL means every row may be read.';
COMMENT ON COLUMN assistant.dataset_policy.row_access_allowed IS
    'Lower-cased values of row_access_key that release a row; anything else withholds it.';

CREATE OR REPLACE FUNCTION assistant_api._row_released(
    p_file_id INTEGER,
    p_value TEXT
) RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = assistant_api, assistant, public, pg_temp
AS $$
    SELECT CASE
        WHEN p.row_access_key IS NULL THEN TRUE
        ELSE lower(btrim(COALESCE(p_value, ''))) = ANY (p.row_access_allowed)
    END
    FROM (SELECT 1) one
    LEFT JOIN assistant.dataset_policy p ON p.file_id = p_file_id;
$$;

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
LEFT JOIN assistant.dataset_policy p
  ON p.file_id = n.file_id
WHERE COALESCE(f.is_delete, false) = false
  AND n.status = 'ready'
  AND (
        p.row_access_key IS NULL
        OR lower(btrim(COALESCE(
               n.row_data_normalized -> 'fields' -> p.row_access_key ->> 'raw',
               n.row_data_normalized -> 'fields' ->> p.row_access_key,
               ''
           ))) = ANY (p.row_access_allowed)
      );

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
LEFT JOIN assistant.dataset_policy p
  ON p.file_id = d.file_id
WHERE COALESCE(f.is_delete, false) = false
  AND (
        p.row_access_key IS NULL
        OR lower(btrim(COALESCE(d.row_data ->> p.row_access_key, ''))) = ANY (p.row_access_allowed)
      );

-- The Potential list is the one that carries the column today.
UPDATE assistant.dataset_policy
   SET row_access_key = 'ACCESS',
       row_access_allowed = ARRAY['true', 'yes', '1']::TEXT[],
       updated_at = now()
 WHERE file_id = 94;

-- Coverage counts and category lists are both computed from the view, so they follow.
SELECT assistant.refresh_fill_rates();
SELECT assistant.refresh_recorded_values();
