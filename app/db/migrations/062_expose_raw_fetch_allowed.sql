-- Let raw_fetch_allowed reach the code, and make GET_RAW_FIELDS obey it.
--
-- assistant.field_registry has carried raw_fetch_allowed since 015 and FieldSpec declares
-- it, but v_dataset_fields never selected it, so catalog_from_rows fell through to its
-- default and no runtime catalog has ever seen the real value. The same defect as
-- evidence_allowed, fixed in 055, on a column that guards more.
--
-- It guards raw source cells -- the least filtered thing NIA can return. Every field on
-- every list is currently raw_fetch_allowed = FALSE, so the registry's policy is that
-- nothing may be fetched raw. The executor was not asking: it requested
-- ["Notes", "Cause of Death", "Name"] as a hardcoded list, in a system whose whole design
-- is that the registry decides what a field is and who may see it.
--
-- After this the executor derives those keys from the registry, so today GET_RAW_FIELDS
-- returns nothing -- which is the policy actually written down. Granting a field raw access
-- becomes a deliberate registry change with an audit trail, rather than a literal in an
-- executor branch.
--
-- CREATE OR REPLACE VIEW, not DROP ... CASCADE: appending a column keeps the view's owner
-- and grants and leaves get_dataset_fields standing. On a SECURITY DEFINER path, dropping
-- and recreating would be a privilege change rather than a refactor.

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
    evidence_allowed,
    raw_fetch_allowed
   FROM assistant.field_registry r;
