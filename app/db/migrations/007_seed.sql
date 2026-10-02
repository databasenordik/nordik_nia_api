INSERT INTO public.file (id, filename, description, version, private, is_delete, row_count)
VALUES
    (49, 'Shingwauk And Wawanosh Students Master List', 'Primary student master list', 3, FALSE, FALSE, 8),
    (91, 'Confirmed- Shingwauk (Wawanosh)', 'Confirmed deaths dataset', 1, FALSE, FALSE, 2),
    (93, 'Additional Deaths', 'Private additional deaths dataset', 1, TRUE, FALSE, 1),
    (94, 'Potential', 'Private potential records', 1, TRUE, FALSE, 1)
ON CONFLICT (id) DO UPDATE SET
    filename = EXCLUDED.filename,
    description = EXCLUDED.description,
    version = EXCLUDED.version,
    private = EXCLUDED.private,
    is_delete = EXCLUDED.is_delete;

INSERT INTO assistant.dataset_policy (
    file_id, enabled, user_facing_label, is_default_people_scope,
    history_enabled, attachment_enabled, private_policy_mode
) VALUES
    (49, TRUE, 'Student master list', TRUE, FALSE, FALSE, 'public'),
    (91, TRUE, 'Confirmed deaths', FALSE, FALSE, FALSE, 'public'),
    (93, TRUE, 'Additional deaths', FALSE, FALSE, FALSE, 'grant_required'),
    (94, TRUE, 'Potential records', FALSE, FALSE, FALSE, 'grant_required')
ON CONFLICT (file_id) DO UPDATE SET
    user_facing_label = EXCLUDED.user_facing_label,
    is_default_people_scope = EXCLUDED.is_default_people_scope,
    private_policy_mode = EXCLUDED.private_policy_mode,
    updated_at = now();

INSERT INTO public.communities (name, variants)
SELECT 'Garden River', ARRAY['Garden River First Nation', 'Garden Rvr']
WHERE NOT EXISTS (SELECT 1 FROM public.communities WHERE name = 'Garden River');

INSERT INTO public.day_schools (name, variants)
SELECT 'Shingwauk', ARRAY['Shingwauk Home', 'Shingwauk Residential School']
WHERE NOT EXISTS (SELECT 1 FROM public.day_schools WHERE name = 'Shingwauk');

INSERT INTO public.indian_hospitals (name, variants)
SELECT 'Fort William', ARRAY['Fort William Indian Hospital']
WHERE NOT EXISTS (SELECT 1 FROM public.indian_hospitals WHERE name = 'Fort William');

INSERT INTO public.provinces (name, code, variants)
SELECT 'Ontario', 'ON', ARRAY['Ont.', 'Ontario']
WHERE NOT EXISTS (SELECT 1 FROM public.provinces WHERE name = 'Ontario');

-- Historical copies (version 2) plus current version 3 for file 49.
INSERT INTO public.file_data (id, file_id, version, row_data)
VALUES
    (1001, 49, 2, '{"Name":"Historical Only","Notes":"old version"}'),
    (1101, 49, 3, '{"Name":"Samuel A","Notes":"admitted 1910","Community":"Garden River","Deceased":false}'),
    (1102, 49, 3, '{"Name":"Sarah B","Notes":"tuberculosis mentioned","Community":"Garden River","Deceased":true,"Cause of Death":"tuberculosis"}'),
    (1103, 49, 3, '{"Name":"Thomas C","Notes":"discharged 1912","Community":"Sault Ste. Marie","Deceased":false}'),
    (1104, 49, 3, '{"Name":"Mary D","Notes":"","Community":"Garden River","Deceased":true,"Cause of Death":"influenza"}'),
    (1105, 49, 3, '{"Name":"Joseph E","Notes":"admitted 1911","Community":"Garden River","Deceased":false}'),
    (1106, 91, 1, '{"Name":"Sarah B","Cause of Death":"tuberculosis"}'),
    (1107, 91, 1, '{"Name":"Unconfirmed F","Cause of Death":"unknown"}'),
    (1108, 93, 1, '{"Name":"Private Name","Cause of Death":"restricted"}'),
    (1109, 94, 1, '{"Name":"Sparse Record","Notes":null}')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.file_data_normalized (
    source_row_id, file_id, version, status, row_data_normalized, search_text,
    canonical_name, canonical_community, canonical_school, source_created_at, source_updated_at
)
SELECT v.source_row_id, v.file_id, v.version, v.status, v.row_data_normalized, v.search_text,
       v.canonical_name, v.canonical_community, v.canonical_school, now(), now()
FROM (
    VALUES
        (1001, 49, 2, 'ready', '{"canonical":{"name":"Historical Only"}}'::jsonb,
         'Historical Only old version', 'Historical Only', NULL, 'Shingwauk'),
        (1101, 49, 3, 'ready', '{"canonical":{"name":"Samuel A","community":"Garden River","dates":{"admitted":"1910"},"deceased":false},"chat":{"default_bundle":{"name":"Samuel A"}}}'::jsonb,
         'Samuel A Garden River admitted 1910', 'Samuel A', 'Garden River', 'Shingwauk'),
        (1102, 49, 3, 'ready', '{"canonical":{"name":"Sarah B","community":"Garden River","deceased":true,"cause_of_death":"tuberculosis"},"chat":{"default_bundle":{"name":"Sarah B"}}}'::jsonb,
         'Sarah B Garden River tuberculosis deceased', 'Sarah B', 'Garden River', 'Shingwauk'),
        (1103, 49, 3, 'ready', '{"canonical":{"name":"Thomas C","community":"Sault Ste. Marie","dates":{"discharged":"1912"},"deceased":false}}'::jsonb,
         'Thomas C Sault Ste. Marie discharged 1912', 'Thomas C', 'Sault Ste. Marie', 'Shingwauk'),
        (1104, 49, 3, 'ready', '{"canonical":{"name":"Mary D","community":"Garden River","deceased":true,"cause_of_death":"influenza"}}'::jsonb,
         'Mary D Garden River influenza deceased', 'Mary D', 'Garden River', 'Shingwauk'),
        (1105, 49, 3, 'ready', '{"canonical":{"name":"Joseph E","community":"Garden River","dates":{"admitted":"1911"},"deceased":false}}'::jsonb,
         'Joseph E Garden River admitted 1911', 'Joseph E', 'Garden River', 'Shingwauk'),
        (1106, 91, 1, 'ready', '{"canonical":{"name":"Sarah B","cause_of_death":"tuberculosis"}}'::jsonb,
         'Sarah B tuberculosis confirmed', 'Sarah B', NULL, 'Shingwauk'),
        (1107, 91, 1, 'ready', '{"canonical":{"name":"Unconfirmed F","cause_of_death":"unknown"}}'::jsonb,
         'Unconfirmed F unknown', 'Unconfirmed F', NULL, 'Shingwauk'),
        (1108, 93, 1, 'ready', '{"canonical":{"name":"Private Name","cause_of_death":"restricted"}}'::jsonb,
         'Private Name restricted', 'Private Name', NULL, NULL),
        (1109, 94, 1, 'ready', '{"canonical":{"name":"Sparse Record"}}'::jsonb,
         'Sparse Record', 'Sparse Record', NULL, NULL)
) AS v(source_row_id, file_id, version, status, row_data_normalized, search_text, canonical_name, canonical_community, canonical_school)
WHERE NOT EXISTS (
    SELECT 1 FROM public.file_data_normalized n
    WHERE n.source_row_id = v.source_row_id AND n.file_id = v.file_id AND n.version = v.version
);

CREATE INDEX IF NOT EXISTS idx_fdn_file_version_status
    ON public.file_data_normalized (file_id, version, status);
CREATE INDEX IF NOT EXISTS idx_fdn_canonical_name
    ON public.file_data_normalized (canonical_name);
CREATE INDEX IF NOT EXISTS idx_fdn_search_trgm
    ON public.file_data_normalized USING gin (search_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fdn_search_fts
    ON public.file_data_normalized USING gin (to_tsvector('simple', coalesce(search_text, '')));
