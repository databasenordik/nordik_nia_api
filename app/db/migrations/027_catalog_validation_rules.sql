-- Expose schema-defined value aliases and validation metadata to the runtime
-- catalog. The planner must receive this metadata through the same authorized
-- interface as the rest of each public field definition.

CREATE OR REPLACE VIEW assistant_api.v_dataset_fields AS
SELECT
    r.file_id,
    r.semantic_field,
    r.human_label,
    r.aliases,
    r.canonical_json_path,
    r.raw_json_keys,
    r.semantic_type,
    r.allowed_operators,
    r.searchable,
    r.aggregatable,
    r.sortable,
    r.quoteable,
    r.sensitivity,
    r.exposure_policy,
    r.validation_rules
FROM assistant.field_registry r;
