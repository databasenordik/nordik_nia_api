DO $$
BEGIN
    IF to_regclass('public.file_data_normalized') IS NOT NULL THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_fdn_file_version_status ON public.file_data_normalized (file_id, version, status)';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_fdn_canonical_name ON public.file_data_normalized (canonical_name)';
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_fdn_search_trgm ON public.file_data_normalized USING gin (search_text gin_trgm_ops)';
        EXECUTE $i$CREATE INDEX IF NOT EXISTS idx_fdn_search_fts ON public.file_data_normalized USING gin (to_tsvector('simple', coalesce(search_text, '')))$i$;
    END IF;
END
$$;
