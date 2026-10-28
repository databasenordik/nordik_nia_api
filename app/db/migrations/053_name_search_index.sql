-- One field that searches every preprocessed name column at once, and the parsed
-- Indigenous name Master was missing.
--
-- The derived name columns exist so a person can be found by any form of their name: the
-- canonical first and last, the alternates, the recorded spellings, the Indigenous name and
-- its spellings. The backfill already folds all of them into row_data_normalized.names --
-- casefolded, one token per form -- but nothing exposed that array to the planner, so a
-- name lookup had to guess a column. Measured before this migration:
--
--   "Tell me about Pahpahmaush"  -> student_name CONTAINS 'Pahpahmaush'  -- found, by luck:
--                                   the display cell happens to carry the Indigenous name
--   "Who is Wm. Fisher?"         -> student_name CONTAINS 'Wm. Fisher'   -- NO MATCH, even
--                                   though 'wm.' and 'fisher' are both in names[]
--
-- name_search fixes the second case. It renders as "fisher, garnet, william, wm." so a
-- CONTAINS on any single recorded form matches, whichever column that form came from.
--
-- It is an index, not content: evidence_allowed is false, so it never appears in the row
-- handed back to the model. Same for every other derived name and date column -- once a
-- record is located, what comes back is the row as it was recorded. Being non-evidence
-- does not make a column unusable; filtering, sorting and grouping are unaffected.

DO $$
DECLARE
    target INTEGER;
BEGIN
    FOREACH target IN ARRAY ARRAY[49, 91, 93, 94] LOOP
        PERFORM assistant.register_name_field(
            target,
            'name_search',
            'Any recorded form of the name',
            ARRAY[
                'name', 'any name', 'called', 'known as', 'goes by',
                'spelled', 'spelling', 'alternate name', 'other name'
            ],
            'names',
            ARRAY[
                'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY', 'NOT_CONTAINS_ANY',
                'EQUALS', 'NOT_EQUALS', 'IN', 'NOT_IN',
                'FULL_TEXT_SEARCH', 'FUZZY_SEARCH', 'IS_KNOWN', 'IS_UNKNOWN'
            ],
            FALSE,   -- sortable: it is a bag of forms, so an order over it means nothing
            FALSE,   -- aggregatable: grouping by it would group by the whole bag
            'authorized'
        );
    END LOOP;

    -- Master parses an Indigenous name for 148 rows and stores it, but never registered
    -- the column, so it could not be filtered on. The three death lists already have it.
    PERFORM assistant.register_name_field(
        49,
        'indigenous_name_normalized',
        'Indigenous name (normalised)',
        ARRAY['indigenous name', 'native name', 'traditional name'],
        'canonical.name_parts.indigenous',
        ARRAY[
            'EQUALS', 'NOT_EQUALS', 'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY',
            'NOT_CONTAINS_ANY', 'STARTS_WITH', 'STARTS_WITH_ANY', 'IS_KNOWN',
            'IS_UNKNOWN', 'IN', 'NOT_IN', 'FULL_TEXT_SEARCH', 'FUZZY_SEARCH'
        ],
        TRUE,
        TRUE,
        'authorized'
    );
END
$$;

-- A derived column is how a row is found, never what the row says.
UPDATE assistant.field_registry
SET evidence_allowed = FALSE
WHERE canonical_json_path LIKE 'canonical.name_parts.%'
   OR canonical_json_path LIKE 'canonical.date_parts.%'
   OR canonical_json_path = 'names';
