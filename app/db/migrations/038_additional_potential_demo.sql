-- Local/demo only. Registered in migrate.STUB_FILES, so APPLY_APP_STUBS=false
-- skips this file and a production database is never auto-granted access to
-- the private research datasets.
--
-- Files 93 and 94 are private and grant_required. The gateway function
-- assistant_api._authorized_file_ids requires BOTH an AccessScope carrying
-- can_use_private_files AND an active row here, so this seeds the grant that
-- the website authorization layer would normally synchronize.
--
-- This file deliberately inserts NO research rows. An earlier revision seeded
-- synthetic rows for these datasets; that is unsafe because APPLY_APP_STUBS is
-- commonly true against a database that already holds the real 23 (file 93)
-- and 56 (file 94) records, where fake rows would corrupt real counts. Stub
-- rows for a bare local database already come from 007_seed.sql.

INSERT INTO assistant.access_grants (principal_id, file_id, grant_status, source, source_version)
VALUES
    ('principal-researcher', 93, 'active', 'standalone-demo', 'seed'),
    ('principal-researcher', 94, 'active', 'standalone-demo', 'seed')
ON CONFLICT (principal_id, file_id) DO UPDATE SET
    grant_status = EXCLUDED.grant_status,
    source = EXCLUDED.source,
    source_version = EXCLUDED.source_version;
