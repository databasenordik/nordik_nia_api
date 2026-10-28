-- Mark genuine free-text note fields as safe for leading-article stripping.
-- Do not apply this to names, communities, schools, nations, or place fields.

UPDATE assistant.field_registry
SET validation_rules = COALESCE(validation_rules, '{}'::jsonb)
    || jsonb_build_object('article_strip_safe', true)
WHERE semantic_field IN (
    'notes',
    'investigative_notes',
    'remarks',
    'comments',
    'additional_notes',
    'narrative'
);
