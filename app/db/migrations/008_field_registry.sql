INSERT INTO assistant.dataset_policy (
    file_id, enabled, user_facing_label, is_default_people_scope,
    history_enabled, attachment_enabled, private_policy_mode
) VALUES
    (49, TRUE, 'Student master list', TRUE, FALSE, FALSE, 'public'),
    (91, TRUE, 'Confirmed deaths', FALSE, FALSE, FALSE, 'public'),
    (93, TRUE, 'Additional deaths', FALSE, FALSE, FALSE, 'grant_required'),
    (94, TRUE, 'Potential records', FALSE, FALSE, FALSE, 'grant_required')
ON CONFLICT (file_id) DO UPDATE SET
    user_facing_label = EXCLUDED.user_facing_label,
    is_default_people_scope = EXCLUDED.is_default_people_scope,
    private_policy_mode = EXCLUDED.private_policy_mode,
    updated_at = now();

INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable
)
SELECT *
FROM (
    VALUES
        (49, 'student_name', 'Student name', ARRAY['name', 'student', 'full name'], 'canonical.display_name', ARRAY['Name', 'Student Name'], 'text', ARRAY['EQUALS', 'STARTS_WITH', 'CONTAINS', 'FULL_TEXT_SEARCH', 'FUZZY_SEARCH'], TRUE, FALSE, TRUE, TRUE),
        (49, 'community', 'Community', ARRAY['reserve', 'first nation', 'community'], 'canonical.community', ARRAY['Community'], 'entity', ARRAY['EQUALS', 'IN', 'FUZZY_SEARCH'], TRUE, TRUE, TRUE, FALSE),
        (49, 'admitted_date', 'Admitted date', ARRAY['admitted', 'admission'], 'canonical.dates.admitted', ARRAY['Admitted', 'Date of Admission'], 'date', ARRAY['EQUALS', 'YEAR_EQUALS', 'BEFORE', 'AFTER', 'DATE_RANGE'], TRUE, TRUE, TRUE, FALSE),
        (49, 'discharged_date', 'Discharged date', ARRAY['discharged', 'discharge'], 'canonical.dates.discharged', ARRAY['Discharged'], 'date', ARRAY['EQUALS', 'YEAR_EQUALS', 'BEFORE', 'AFTER', 'IS_UNKNOWN'], TRUE, TRUE, TRUE, FALSE),
        (49, 'deceased_status', 'Deceased status', ARRAY['deceased', 'died', 'death'], 'canonical.deceased_status', ARRAY['Deceased'], 'boolean', ARRAY['IS_TRUE', 'IS_FALSE'], TRUE, TRUE, FALSE, FALSE),
        (49, 'notes', 'Notes', ARRAY['note', 'remarks'], 'chat.narrative_bundle.notes', ARRAY['Notes'], 'text', ARRAY['CONTAINS', 'FULL_TEXT_SEARCH', 'GET_QUOTE'], TRUE, FALSE, FALSE, TRUE),
        (91, 'student_name', 'Name', ARRAY['name'], 'canonical.display_name', ARRAY['Name'], 'text', ARRAY['EQUALS', 'STARTS_WITH', 'CONTAINS', 'FUZZY_SEARCH'], TRUE, FALSE, TRUE, TRUE),
        (91, 'cause_of_death', 'Cause of death', ARRAY['cause'], 'canonical.cause_of_death', ARRAY['Cause of Death'], 'text', ARRAY['CONTAINS', 'FULL_TEXT_SEARCH'], TRUE, FALSE, FALSE, TRUE),
        (93, 'student_name', 'Name', ARRAY['name'], 'canonical.display_name', ARRAY['Name'], 'text', ARRAY['EQUALS', 'STARTS_WITH'], TRUE, FALSE, TRUE, TRUE),
        (93, 'cause_of_death', 'Cause of death', ARRAY['cause'], 'canonical.cause_of_death', ARRAY['Cause of Death'], 'text', ARRAY['CONTAINS'], TRUE, FALSE, FALSE, TRUE),
        (94, 'student_name', 'Name', ARRAY['name'], 'canonical.display_name', ARRAY['Name'], 'text', ARRAY['EQUALS', 'STARTS_WITH'], TRUE, FALSE, TRUE, TRUE),
        (94, 'notes', 'Notes', ARRAY['note'], 'chat.narrative_bundle.notes', ARRAY['Notes'], 'text', ARRAY['CONTAINS', 'IS_UNKNOWN'], TRUE, FALSE, FALSE, TRUE)
) AS v(
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable
)
ON CONFLICT (file_id, semantic_field) DO UPDATE SET
    human_label = EXCLUDED.human_label,
    aliases = EXCLUDED.aliases,
    canonical_json_path = EXCLUDED.canonical_json_path,
    raw_json_keys = EXCLUDED.raw_json_keys,
    allowed_operators = EXCLUDED.allowed_operators;
