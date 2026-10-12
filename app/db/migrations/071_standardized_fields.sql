-- All four lists use separate name columns. Only standardized spellings are inputs.
-- The standardized death lists share their source schema, not just date columns.
INSERT INTO assistant.field_registry(file_id,semantic_field,human_label,aliases,canonical_json_path,
    raw_json_keys,semantic_type,allowed_operators,searchable,aggregatable,sortable,quoteable,
    sensitivity,exposure_policy,validation_rules,evidence_allowed,raw_fetch_allowed)
SELECT target,semantic_field,human_label,aliases,canonical_json_path,raw_json_keys,semantic_type,
    allowed_operators,searchable,aggregatable,sortable,quoteable,sensitivity,exposure_policy,
    validation_rules,evidence_allowed,raw_fetch_allowed
FROM assistant.field_registry CROSS JOIN unnest(ARRAY[93,94]) target
WHERE file_id = 91 AND cardinality(raw_json_keys) > 0
ON CONFLICT (file_id,semantic_field) DO NOTHING;
UPDATE assistant.field_registry
SET raw_json_keys = CASE semantic_field
    WHEN 'first_name' THEN ARRAY['First Name', 'FIRST NAME']
    WHEN 'middle_names' THEN ARRAY['Middle Name', 'MIDDLE NAME']
    WHEN 'last_name' THEN ARRAY['Last Name', 'LAST NAME']
    END
WHERE file_id IN (49,91,93,94) AND semantic_field IN ('first_name','middle_names','last_name');
UPDATE assistant.field_registry
SET raw_json_keys = CASE semantic_field
    WHEN 'first_name_source' THEN ARRAY['First Name', 'FIRST NAME']
    WHEN 'middle_names_source' THEN ARRAY['Middle Name', 'MIDDLE NAME']
    WHEN 'last_name_source' THEN ARRAY['Last Name', 'LAST NAME']
    END
WHERE file_id IN (49,91,93,94) AND semantic_field IN ('first_name_source','middle_names_source','last_name_source');

-- Registration/header whitespace varied in older spreadsheet templates.
UPDATE assistant.field_registry r SET raw_json_keys = (
    SELECT array_agg(DISTINCT btrim(regexp_replace(key, '[[:space:]]+', ' ', 'g')))
    FROM unnest(r.raw_json_keys) AS key
) WHERE file_id IN (49,91,93,94) AND cardinality(raw_json_keys) > 0;
UPDATE assistant.field_registry
SET canonical_json_path = 'fields.' || btrim(regexp_replace(substring(canonical_json_path FROM 8), '[[:space:]]+', ' ', 'g'))
WHERE file_id IN (49,91,93,94) AND canonical_json_path LIKE 'fields.%';

UPDATE assistant.field_registry SET raw_json_keys = ARRAY['RESERVE/COMMUNITY','COMMUNITY/RESERVE']
WHERE file_id IN (91,93,94) AND semantic_field='community';
UPDATE assistant.field_registry SET canonical_json_path='canonical.community'
WHERE file_id IN (49,91,93,94) AND semantic_field='community';

-- Source attributes omitted by the legacy registry remain distinct from other columns.
DO $$ DECLARE r record; BEGIN
FOR r IN SELECT * FROM (VALUES
    (49,'source_reference','Source','Source'),
    (49,'indigenous_community_name','Indigenous Community Name','Indigenous Community Name'),
    (93,'workflow_checklist','Workflow checklist','WORKFLOW CHECKLIST'),
    (93,'name_place_holder','Name place holder','NAME PLACE HOLDER')
) AS fields(file_id,semantic_field,label,raw_key) LOOP
    PERFORM assistant.register_name_field(r.file_id,r.semantic_field,r.label,
        ARRAY[r.label],'fields.' || r.raw_key,
        ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'],false,false,'authorized');
    UPDATE assistant.field_registry SET raw_json_keys=ARRAY[r.raw_key],
        evidence_allowed=true,quoteable=true
    WHERE file_id=r.file_id AND semantic_field=r.semantic_field;
END LOOP; END $$;

UPDATE assistant.field_registry SET raw_json_keys = ARRAY['DISCHARGE DATE']
WHERE file_id IN (91,93,94) AND semantic_field = 'discharged_date';

-- Our backfill retains operators and precision; upstream canonical dates may not.
UPDATE assistant.field_registry SET canonical_json_path = 'canonical.date_parts.' || semantic_field || '.iso'
WHERE file_id IN (49,91,93,94) AND semantic_type = 'date' AND cardinality(raw_json_keys) > 0;

DO $$ DECLARE r record; BEGIN
FOR r IN SELECT file_id,semantic_field,human_label FROM assistant.field_registry
         WHERE file_id IN (49,91,93,94) AND semantic_type = 'date' LOOP
    PERFORM assistant.register_name_field(r.file_id, r.semantic_field || '_event_count',
        'Number of recorded ' || lower(r.human_label) || ' events',
        CASE WHEN r.semantic_field = 'admitted_date' THEN ARRAY['number of recorded admission dates','repeated admissions','admitted twice']
             ELSE ARRAY[r.human_label || ' event count'] END,
        'canonical.date_parts.' || r.semantic_field || '.event_count',
        ARRAY['EQUALS','NOT_EQUALS','GREATER_THAN','LESS_THAN','NUMBER_RANGE','IS_KNOWN','IS_UNKNOWN'],
        true,true,'authorized');
    UPDATE assistant.field_registry SET semantic_type='number', raw_json_keys=ARRAY[]::text[]
    WHERE file_id=r.file_id AND semantic_field=r.semantic_field || '_event_count';
END LOOP; END $$;

DO $$ DECLARE target integer; BEGIN
FOREACH target IN ARRAY ARRAY[49,91,93,94] LOOP
    PERFORM assistant.register_name_field(target, 'name_comments', 'Name comments',
        ARRAY['name notes','variations of name'], 'fields.Name comments',
        ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], false, false, 'authorized');
    UPDATE assistant.field_registry SET raw_json_keys = ARRAY['Name comments','VARIATIONS OF NAME'],
        evidence_allowed = true, quoteable = true
    WHERE file_id = target AND semantic_field = 'name_comments';
    PERFORM assistant.register_name_field(target, 'date_comments', 'Date comments',
        ARRAY['date notes','date provenance'], 'fields.Date comments',
        ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], false, false, 'authorized');
    UPDATE assistant.field_registry SET raw_json_keys = ARRAY['Date comments'],
        evidence_allowed = true, quoteable = true
    WHERE file_id = target AND semantic_field = 'date_comments';
    PERFORM assistant.register_name_field(target, 'source_revision_notes', 'Source revision notes',
        ARRAY['source corrections'], 'fields.Source revision notes',
        ARRAY['CONTAINS','IS_KNOWN','IS_UNKNOWN'], false, false, 'authorized');
    UPDATE assistant.field_registry SET raw_json_keys = ARRAY['Source revision notes'],
        evidence_allowed = true, quoteable = true
    WHERE file_id = target AND semantic_field = 'source_revision_notes';
END LOOP; END $$;
