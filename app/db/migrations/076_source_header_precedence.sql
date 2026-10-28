-- Header aliases are ordered fallbacks for a missing column, not for an empty
-- cell. Preserve the current standardized header before its legacy spellings.
UPDATE assistant.field_registry
SET raw_json_keys = ARRAY['First Nation Community','Community','First Nation/Community']
WHERE file_id=49 AND semantic_field='community';

UPDATE assistant.field_registry
SET raw_json_keys = CASE semantic_field
    WHEN 'first_name' THEN ARRAY['First Name','FIRST NAME']
    WHEN 'first_name_source' THEN ARRAY['First Name','FIRST NAME']
    WHEN 'middle_names' THEN ARRAY['Middle Name','MIDDLE NAME']
    WHEN 'middle_names_source' THEN ARRAY['Middle Name','MIDDLE NAME']
    WHEN 'last_name' THEN ARRAY['Last Name','LAST NAME']
    WHEN 'last_name_source' THEN ARRAY['Last Name','LAST NAME'] END
WHERE file_id IN (49,91,93,94) AND semantic_field IN (
    'first_name','first_name_source','middle_names','middle_names_source','last_name','last_name_source');

-- Run app.normalization.backfill_names after this metadata migration. The source
-- cells themselves are never rewritten.
