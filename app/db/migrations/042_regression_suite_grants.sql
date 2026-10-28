-- Local/demo only. Registered in migrate.STUB_FILES, so APPLY_APP_STUBS=false skips
-- this file and a production database is never auto-granted access to the private
-- research datasets.
--
-- The scoped AI-planner suite (docs/ai-planner-scoped-suite.json) covers all four
-- lists, and its file 93 / 94 cases are marked private. benchmarks/regression.py runs
-- under its own principal rather than the demo researcher, which is deliberate --
-- benchmark conversations and result sets stay isolated from the demo principal's --
-- but that principal has no grant, so --allow-private-files failed preflight with
-- "missing active grants [93, 94]".
--
-- The grant is seeded here rather than by the harness itself on purpose. A benchmark
-- that could grant itself access would be escalating its own privileges, and the
-- authorization boundary is one of the things this suite is meant to verify. Keeping
-- it in an APPLY_APP_STUBS-gated migration keeps the grant explicit and auditable,
-- and keeps production unable to acquire it.
--
-- Principal must match benchmarks/regression.py --principal (default regression-suite).
-- No research rows are inserted here.

INSERT INTO assistant.access_grants (principal_id, file_id, grant_status, source, source_version)
VALUES
    ('regression-suite', 93, 'active', 'regression-suite', 'seed'),
    ('regression-suite', 94, 'active', 'regression-suite', 'seed')
ON CONFLICT (principal_id, file_id) DO UPDATE SET
    grant_status = EXCLUDED.grant_status,
    source = EXCLUDED.source,
    source_version = EXCLUDED.source_version;
