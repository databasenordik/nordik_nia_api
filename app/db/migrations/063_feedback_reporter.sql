-- Let a tester sign the report they file.
--
-- assistant.feedback has existed since the early tables and nothing ever wrote to it. The
-- testing build now asks people to click Report, write their name and any comments, and
-- submit -- so the row needs somewhere to put the name, and somewhere to record which build
-- they were looking at when they wrote it. Without the second, a report that arrives a week
-- later cannot be told from one filed against a fix that has since landed.
--
-- Both are nullable: a report with no name is still worth having, and refusing it would
-- lose the finding to protect a field nobody needs.

ALTER TABLE assistant.feedback
    ADD COLUMN IF NOT EXISTS reporter_name text,
    ADD COLUMN IF NOT EXISTS principal_id text,
    ADD COLUMN IF NOT EXISTS app_version text;

-- Reports are read newest-first when someone sits down to triage them.
CREATE INDEX IF NOT EXISTS feedback_created_at_idx
    ON assistant.feedback (created_at DESC);
