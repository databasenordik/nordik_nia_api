GRANT USAGE ON SCHEMA public TO assistant_gateway_owner;

DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'file', 'data_config', 'file_data', 'file_data_normalized',
        'communities', 'day_schools', 'indian_hospitals', 'provinces'
    ]
    LOOP
        IF to_regclass('public.' || tbl) IS NOT NULL THEN
            EXECUTE format('GRANT SELECT ON TABLE public.%I TO assistant_gateway_owner', tbl);
        END IF;
    END LOOP;
END
$$;

GRANT SELECT ON TABLE
    assistant.dataset_policy,
    assistant.field_registry,
    assistant.access_grants
TO assistant_gateway_owner;

ALTER FUNCTION assistant_api._authorized_file_ids(TEXT, INTEGER[], BOOLEAN) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.get_dataset_fields(TEXT, INTEGER, INTEGER[], BOOLEAN) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.search_current_records(TEXT, INTEGER[], BOOLEAN, TEXT, INTEGER) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.get_current_records_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[]) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.get_raw_fields_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[], TEXT[]) OWNER TO assistant_gateway_owner;
ALTER FUNCTION assistant_api.resolve_reference_entity(TEXT, TEXT, TEXT) OWNER TO assistant_gateway_owner;

REVOKE ALL ON FUNCTION assistant_api._authorized_file_ids(TEXT, INTEGER[], BOOLEAN) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.get_dataset_fields(TEXT, INTEGER, INTEGER[], BOOLEAN) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.search_current_records(TEXT, INTEGER[], BOOLEAN, TEXT, INTEGER) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.get_current_records_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.get_raw_fields_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[], TEXT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION assistant_api.resolve_reference_entity(TEXT, TEXT, TEXT) FROM PUBLIC;

GRANT USAGE ON SCHEMA assistant TO assistant_runtime;
GRANT USAGE ON SCHEMA assistant TO assistant_gateway_owner;
GRANT USAGE ON SCHEMA assistant_api TO assistant_runtime;
GRANT USAGE ON SCHEMA assistant_api TO assistant_gateway_owner;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA assistant TO assistant_runtime;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA assistant TO assistant_runtime;

ALTER DEFAULT PRIVILEGES IN SCHEMA assistant
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO assistant_runtime;
ALTER DEFAULT PRIVILEGES IN SCHEMA assistant
    GRANT USAGE, SELECT ON SEQUENCES TO assistant_runtime;

GRANT SELECT ON assistant_api.v_datasets TO assistant_gateway_owner;
GRANT SELECT ON assistant_api.v_dataset_fields TO assistant_gateway_owner;
GRANT SELECT ON assistant_api.v_reference_entities TO assistant_gateway_owner;

GRANT SELECT ON assistant_api.v_datasets TO assistant_runtime;
GRANT SELECT ON assistant_api.v_dataset_fields TO assistant_runtime;
GRANT SELECT ON assistant_api.v_reference_entities TO assistant_runtime;

REVOKE ALL ON assistant_api.v_current_records FROM PUBLIC;
REVOKE ALL ON assistant_api.v_current_raw_records FROM PUBLIC;
REVOKE ALL ON assistant_api.v_current_records FROM assistant_runtime;
REVOKE ALL ON assistant_api.v_current_raw_records FROM assistant_runtime;

GRANT SELECT ON assistant_api.v_current_records TO assistant_gateway_owner;
GRANT SELECT ON assistant_api.v_current_raw_records TO assistant_gateway_owner;

GRANT EXECUTE ON FUNCTION assistant_api._authorized_file_ids(TEXT, INTEGER[], BOOLEAN) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.list_datasets(TEXT, INTEGER[], BOOLEAN) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.get_dataset_fields(TEXT, INTEGER, INTEGER[], BOOLEAN) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.search_current_records(TEXT, INTEGER[], BOOLEAN, TEXT, INTEGER) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.get_current_records_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[]) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.get_raw_fields_by_ids(TEXT, INTEGER[], BOOLEAN, BIGINT[], TEXT[]) TO assistant_runtime;
GRANT EXECUTE ON FUNCTION assistant_api.resolve_reference_entity(TEXT, TEXT, TEXT) TO assistant_runtime;

-- Runtime must not read application base tables.
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'file', 'data_config', 'file_data', 'file_data_normalized',
        'communities', 'day_schools', 'indian_hospitals', 'provinces',
        'users', 'roles', 'otps', 'logs', 'support_requests',
        'file_access', 'file_edit_request', 'form_submissions'
    ]
    LOOP
        IF to_regclass('public.' || tbl) IS NOT NULL THEN
            EXECUTE format('REVOKE ALL ON TABLE public.%I FROM assistant_runtime', tbl);
        END IF;
    END LOOP;
END
$$;
