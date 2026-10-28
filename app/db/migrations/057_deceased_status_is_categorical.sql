-- deceased_status records three categories, not a true/false.
--
-- "How many students have an unknown deceased status?" answers 15. The right answer is 665:
--
--     yes      1066
--     no       1035
--     unknown   665      <- recorded, and what the question is about
--     (blank)    15      <- what IS_UNKNOWN finds
--
-- A 44x undercount, on a question about whether children died. 665 students whose fate was
-- never established is itself one of the more significant facts this list holds, and the
-- answer erased it.
--
-- The cause is upstream of the planner. The field is typed 'boolean' and offers IS_TRUE /
-- IS_FALSE / IS_UNKNOWN, so "unknown" reads as the absence of a value rather than as one of
-- the values -- and for a boolean that is the correct reading. It is only wrong because the
-- column is not a boolean: it is a three-way category, and a tri-state does not fit in a
-- flag. The prompt already tells the planner that a named category beats IS_UNKNOWN, and
-- the catalog already publishes recorded_values [no, unknown, yes]; neither helps while the
-- declared type says the field has two states and a null.
--
-- Typing it as the category it is lets every reading work through the same door:
--   "deceased"        -> EQUALS 'yes'      (value_synonyms already maps this)
--   "not deceased"    -> EQUALS 'no'
--   "unknown status"  -> EQUALS 'unknown'  -- the case that was wrong
--   "no status"       -> IS_UNKNOWN        -- still available, still means the 15 blanks
--
-- ai_catalog_coerce rewrites EQUALS 'yes' to IS_TRUE for boolean fields; that coercion
-- simply stops applying here, which is the point. IS_TRUE / IS_FALSE leave the operator
-- list because they cannot express a third category and their presence is what invited the
-- binary framing.
--
-- Scope: file 49 only. 91's census_documents_used is a real yes/no and stays boolean.
-- The static catalog in backend/app/planning/catalog.py still declares this field boolean;
-- it is a test fixture and offline fallback, and is left alone until this is measured.

UPDATE assistant.field_registry
SET semantic_type = 'entity',
    allowed_operators = ARRAY[
        'EQUALS', 'NOT_EQUALS', 'IN', 'NOT_IN', 'IS_KNOWN', 'IS_UNKNOWN'
    ]
WHERE file_id = 49
  AND semantic_field = 'deceased_status';
