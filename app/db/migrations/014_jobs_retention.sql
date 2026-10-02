-- Compatibility migration: an earlier deployment recorded this index as 014
-- while the standalone source tree carried it as 013. Keeping the idempotent
-- migration under both recorded names reconciles existing and fresh databases.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON assistant.background_jobs (idempotency_key)
    WHERE idempotency_key IS NOT NULL;
