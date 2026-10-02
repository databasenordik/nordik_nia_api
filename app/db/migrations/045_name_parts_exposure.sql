-- Make the name variant columns visible to the planner.
--
-- 043 kept the matching aids (other names, spellings, alias, flags) at
-- exposure_policy='internal' to hold the planner prompt down, because an oversized
-- SELECTED FIELDS block has measurably degraded planning accuracy before.
--
-- That was too cautious for what these columns are for. A researcher does ask "what other
-- names was this person recorded under" and "how else is this name spelled", and the
-- assistant can only answer from fields it can see and project. Keeping them hidden also
-- made them absent from projected rows, so an all-information request returned a record
-- with its own alternate names missing.
--
-- The measured cost is small: the planner catalog goes from 22 to 32 fields on the master
-- list and 34 to 47 on confirmed deaths. exposure_policy stays wired end to end, so the
-- mechanism is available the moment a genuinely internal column needs it.

UPDATE assistant.field_registry
SET exposure_policy = 'authorized'
WHERE canonical_json_path LIKE 'canonical.name_parts.%';
