CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON assistant.background_jobs (idempotency_key)
    WHERE idempotency_key IS NOT NULL;
