-- Old answers may contain rows released at the time but withheld now.
-- The signature uses only released source rows. Missing signatures deny reuse.
CREATE OR REPLACE FUNCTION assistant_api.potential_access_revision()
RETURNS text LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = assistant_api, assistant, public AS $$
    SELECT 'strict-true-v1:' || md5(coalesce(string_agg(
        id::text || ':' || version::text || ':' || row_data::text, ',' ORDER BY id
    ), '')) FROM assistant_api.v_current_raw_records WHERE file_id = 94;
$$;
ALTER FUNCTION assistant_api.potential_access_revision() OWNER TO assistant_gateway_owner;
REVOKE ALL ON FUNCTION assistant_api.potential_access_revision() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION assistant_api.potential_access_revision() TO assistant_runtime;
