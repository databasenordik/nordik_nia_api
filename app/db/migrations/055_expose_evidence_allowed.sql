-- Let evidence_allowed reach the code that reads it.
--
-- assistant.field_registry has carried an evidence_allowed column since 015, and
-- plan_validator, canonical_query and query_plan_builder all branch on it. But
-- v_dataset_fields -- the only way the catalog is loaded at runtime -- never selected it,
-- so catalog_from_rows fell through to its `row.get("evidence_allowed", True)` default and
-- every field came back evidence-allowed regardless of what the registry said.
--
-- The column was therefore inert in production: the checks ran, and always passed. It
-- surfaced when 053 marked the derived name and date columns non-evidence and they kept
-- appearing in the row handed to the model anyway.
--
-- CREATE OR REPLACE VIEW rather than DROP ... CASCADE: a column may be appended to the end
-- of a view's list in place, which keeps the view's owner, keeps its grants
-- (assistant_gateway_owner and assistant_runtime hold SELECT; it is not public), and
-- leaves get_dataset_fields -- declared RETURNS SETOF this view -- standing. Dropping the
-- view would take the function with it and hand both back with different ownership, which
-- on a SECURITY DEFINER path is a privilege change, not a refactor.
--
-- Nothing else changes: fields that were never marked keep the TRUE they already defaulted
-- to, so this only starts honouring the rows 053 set to FALSE.

CREATE OR REPLACE VIEW assistant_api.v_dataset_fields AS
SELECT file_id,
    semantic_field,
    human_label,
    aliases,
    canonical_json_path,
    raw_json_keys,
    semantic_type,
    allowed_operators,
    searchable,
    aggregatable,
    sortable,
    quoteable,
    sensitivity,
    exposure_policy,
    validation_rules,
    evidence_allowed
   FROM assistant.field_registry r;
