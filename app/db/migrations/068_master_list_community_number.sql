-- Register the band number the standardization split out of the community column.
--
-- Version 5 of the Master list writes "Chisasibi #30" as the community "Chisasibi" plus a
-- separate "First Nation Community Number" of 30, filled for 213 rows. Until it is
-- registered the number is not in the catalog, so a question about a band number has no
-- field to bind to and is correctly refused -- while the number sits in the data.
--
-- Typed as text rather than a quantity: a band number identifies a community, it is never
-- summed or compared by magnitude, and the same reasoning already types student_number.
-- Left aggregatable because counting students per band is a real question, and grouping by
-- the number is the one way to group communities that were recorded under more than one
-- name. The earlier shape, where the number was part of the community value, means rows
-- loaded before this standardization will read as not recorded, which is what they are.

INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
SELECT
    49, 'community_number', 'First Nation community number',
    ARRAY['band number', 'community number', 'reserve number', 'band no'],
    'fields.First Nation Community Number', ARRAY['First Nation Community Number'], 'text',
    ARRAY['EQUALS', 'IN', 'CONTAINS', 'IS_KNOWN', 'IS_UNKNOWN'],
    TRUE, TRUE, TRUE, FALSE, TRUE, FALSE
WHERE NOT EXISTS (
    SELECT 1 FROM assistant.field_registry
    WHERE file_id = 49 AND semantic_field = 'community_number'
);

SELECT assistant.refresh_fill_rates();
