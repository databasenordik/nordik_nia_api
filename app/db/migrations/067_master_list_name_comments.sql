-- Read the Master list's name comments column.
--
-- The standardization moved out of the name cells everything that was not the name:
-- "Sahguj (also spelt Saguj)" became "Sahguj" plus the comment "Also spelled: Saguj",
-- and 285 rows that carried a parenthetical now carry none. Those parentheticals were
-- where the Indigenous name, its meaning, the spelling variants and the aliases came
-- from, so on the standardized data every one of those fields empties out: Indigenous
-- names fall from 148 rows to 4, their meanings from 130 to 1, spelling variants from
-- 95 to 7. The comments column holds the same facts, now labelled.
--
-- Registered as a parser input, not as a question the planner can ask: it is exposed
-- 'internal' so the prompt does not grow, and what a researcher searches by continues
-- to be the name fields themselves, which name_parser fills from these comments.

INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    exposure_policy, evidence_allowed, raw_fetch_allowed
)
SELECT
    49, 'name_comments', 'Name comments', ARRAY[]::TEXT[],
    'fields.Name comments', ARRAY['Name comments'], 'text',
    ARRAY['CONTAINS', 'FULL_TEXT_SEARCH', 'IS_KNOWN', 'IS_UNKNOWN'],
    TRUE, FALSE, FALSE, TRUE, 'internal', TRUE, FALSE
WHERE NOT EXISTS (
    SELECT 1 FROM assistant.field_registry WHERE file_id = 49 AND semantic_field = 'name_comments'
);

SELECT assistant.refresh_fill_rates();
