-- Roles used by the assistant. Fresh local installs receive passwords through
-- session-local settings populated by app.db.migrate; no credential is stored
-- in this migration. On an existing cluster, provision roles separately.

DO $$
DECLARE
    runtime_password TEXT := current_setting('nia.assistant_runtime_password', true);
    migrator_password TEXT := current_setting('nia.assistant_migrator_password', true);
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assistant_gateway_owner') THEN
        CREATE ROLE assistant_gateway_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assistant_runtime') THEN
        IF coalesce(runtime_password, '') = '' THEN
            RAISE EXCEPTION 'ASSISTANT_RUNTIME_PASSWORD is required to create assistant_runtime';
        END IF;
        EXECUTE format('CREATE ROLE assistant_runtime LOGIN PASSWORD %L', runtime_password);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assistant_migrator') THEN
        IF coalesce(migrator_password, '') = '' THEN
            RAISE EXCEPTION 'ASSISTANT_MIGRATOR_PASSWORD is required to create assistant_migrator';
        END IF;
        EXECUTE format('CREATE ROLE assistant_migrator LOGIN PASSWORD %L', migrator_password);
    END IF;
END
$$;

GRANT assistant_gateway_owner TO assistant_migrator;

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO assistant_runtime', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO assistant_migrator', current_database());
END
$$;
