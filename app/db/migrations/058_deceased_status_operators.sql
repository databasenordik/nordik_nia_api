-- Give the retyped deceased_status the operators an entity field actually gets asked for.
--
-- 057 retyped it and set a narrow operator list. That broke the two cases which had been
-- working: "recorded as deceased" and "recorded as not deceased" both came back
-- "operator CONTAINS_ANY is not allowed on deceased_status for file 49". Once the field
-- stopped being a boolean the planner stopped reaching for IS_TRUE and reached for the
-- ordinary text operators instead -- reasonably, since it is now a category with three
-- named values -- and CONTAINS_ANY was not in the list.
--
-- This restores them so the retyping can be judged on its own merits rather than on an
-- operator list that was too narrow. IS_TRUE and IS_FALSE stay out: they cannot express a
-- third category.

UPDATE assistant.field_registry
SET allowed_operators = ARRAY[
        'EQUALS', 'NOT_EQUALS', 'IN', 'NOT_IN',
        'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY', 'NOT_CONTAINS_ANY',
        'IS_KNOWN', 'IS_UNKNOWN'
    ]
WHERE file_id = 49
  AND semantic_field = 'deceased_status';
