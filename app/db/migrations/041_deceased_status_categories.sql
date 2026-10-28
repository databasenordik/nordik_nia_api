-- deceased_status is registered as boolean but stores a ternary category.
--
-- The recorded values on the master list are yes (1,066), no (1,035), and a literal
-- unknown (665), alongside 15 NULLs that mean the cell is empty. Without
-- value_synonyms the planner has no way to tell that "unknown deceased status" names
-- a recorded category rather than a missing value, so it compiled IS_UNKNOWN and
-- answered 15 instead of 665 -- the opposite of what was asked.
--
-- The vocabulary belongs here rather than in the planner prompt or in Python: the
-- planner reads value_synonyms from the catalog and expands a named category into a
-- value match, exactly as cause_of_death and location_of_death already do in
-- 033_domain_metadata.sql. Nothing about this dataset is encoded in code.
--
-- The UPDATE is written out rather than calling assistant.set_value_synonyms: that
-- helper is not present in every environment that has 033 recorded as applied, and a
-- migration must not depend on an earlier migration's helper still existing.

UPDATE assistant.field_registry
SET validation_rules = jsonb_set(
    COALESCE(validation_rules, '{}'::jsonb),
    '{value_synonyms}',
    '{
       "yes": ["yes"],
       "deceased": ["yes"],
       "no": ["no"],
       "not deceased": ["no"],
       "alive": ["no"],
       "living": ["no"],
       "unknown": ["unknown"]
     }'::jsonb,
    TRUE
)
WHERE file_id = 49 AND semantic_field = 'deceased_status';
