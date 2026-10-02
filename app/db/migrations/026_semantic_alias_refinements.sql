-- Additional natural-language aliases remain catalog data so new deployments,
-- schemas, or encodings can change them without changing planner architecture.
UPDATE assistant.field_registry
SET aliases = ARRAY[
    'census used',
    'census documents',
    'census documents used',
    'census documents were used'
]
WHERE file_id = 91 AND semantic_field = 'census_documents_used';
