-- Mark name_search as a token index, so a multi-word name is matched token by token.
--
-- The field stores one token per recorded form of a name and renders as a comma-joined
-- list: "albert, penance, pinnance". So CONTAINS 'Albert Penance' looks for that phrase
-- inside a string that never contains it, and "show all information for Albert Penance"
-- on the Confirmed deaths list found nothing at all -- for a person who is in it. The same
-- flaw made "Who is Wm. Fisher?" return no rows.
--
-- The field is not free text and should not be matched as though it were. A person's name
-- arrives as several words and the index holds them separately, so the question is whether
-- the row carries all of those tokens, not whether it contains that phrase.
--
-- Declared here rather than special-cased in code: any field whose stored value is a bag of
-- tokens can carry this and get the same treatment. assistant.field_registry is the only
-- place the pipeline learns what a field is.

UPDATE assistant.field_registry
SET validation_rules = COALESCE(validation_rules, '{}'::jsonb)
                       || jsonb_build_object('token_index', true)
WHERE semantic_field = 'name_search';
