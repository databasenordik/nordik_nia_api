-- Parent-name values are categorical text and support grouping/counting just
-- like other registry-defined entity/text fields. The planner still controls
-- limits and never exposes an unregistered column.

UPDATE assistant.field_registry
SET aggregatable = TRUE
WHERE semantic_field = 'parents_names';
