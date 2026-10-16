-- Some registered source headers are aliases (for example name comments versus
-- variations of name). Resolve only an unambiguous, actually present source key.
-- A present primary header remains authoritative, including its blank cells.
-- Work happens during migration, never in a user's question path.
WITH headers AS (
    SELECT DISTINCT file_id, jsonb_object_keys(row_data_normalized->'fields') AS key
    FROM assistant_api.v_current_records
), candidates AS (
    SELECT r.file_id,r.semantic_field,min(h.key) AS key
    FROM assistant.field_registry r
    JOIN headers h ON h.file_id=r.file_id AND h.key=ANY(r.raw_json_keys)
    WHERE r.canonical_json_path LIKE 'fields.%'
      AND NOT EXISTS (SELECT 1 FROM headers primary_header
          WHERE primary_header.file_id=r.file_id
            AND primary_header.key=substring(r.canonical_json_path FROM 8))
    GROUP BY r.file_id,r.semantic_field
    HAVING count(DISTINCT h.key)=1
)
UPDATE assistant.field_registry r SET canonical_json_path='fields.' || c.key
FROM candidates c WHERE r.file_id=c.file_id AND r.semantic_field=c.semantic_field;

SELECT assistant.refresh_fill_rates();
SELECT assistant.refresh_recorded_values();
