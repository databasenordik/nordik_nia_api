-- The Potential list records where children came from: register the field that holds it.
--
-- Migration 037 left community off file 94 on the belief that the list "does NOT normalize
-- communities", and 061 went further, describing origin as something this list "has never
-- held". Both were wrong about the data. COMMUNITY/RESERVE is filled for ten children and
-- ingest already folds it into canonical_community; another twenty-one rows carry the
-- community inside the name cell itself -- a whole row pasted as "Name<TAB>date<TAB>community",
-- or "(Died: date) -- From X First Nation" -- which the name parser already sets aside as
-- residue. A tester counted origins for 31 of the list's 49 children from exactly those two
-- sources, while NIA, with no community field to use, grouped by NATION and reported three
-- filled rows reading "community member", "not confirmed" and "Shingwauk".
--
-- Registered the way 037 registered it for file 93. The values carried in name cells are
-- written by backfill_names, which derives them from parsed residue by structure alone and
-- marks them canonical.community_source = 'name_cell'.

INSERT INTO assistant.field_registry (
    file_id, semantic_field, human_label, aliases, canonical_json_path, raw_json_keys,
    semantic_type, allowed_operators, searchable, aggregatable, sortable, quoteable,
    evidence_allowed, raw_fetch_allowed
)
SELECT
    94, 'community', 'First Nation / Community', ARRAY['reserve', 'community'],
    'canonical.community', ARRAY['COMMUNITY/RESERVE'], 'entity',
    ARRAY['EQUALS', 'IN', 'CONTAINS', 'FUZZY_SEARCH', 'IS_KNOWN', 'IS_UNKNOWN'],
    TRUE, TRUE, TRUE, FALSE, TRUE, FALSE
WHERE NOT EXISTS (
    SELECT 1 FROM assistant.field_registry WHERE file_id = 94 AND semantic_field = 'community'
);

SELECT assistant.refresh_fill_rates();
