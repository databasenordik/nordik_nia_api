-- Keep a tester's report when the chat it is about is deleted, and let it carry files.
--
-- assistant.feedback.conversation_id was declared ON DELETE CASCADE, so deleting a chat from
-- the sidebar -- added in the same testing round -- silently deleted every report filed about
-- it. A report outlives the conversation that prompted it: the link is cleared, the finding
-- stays.
--
-- Testers also asked to attach "a picture of the issue here and/or document". Attachments
-- live in their own table, bounded by the API (a few files, a few megabytes in all), and go
-- with the report they belong to.

ALTER TABLE assistant.feedback
    DROP CONSTRAINT IF EXISTS feedback_conversation_id_fkey;
ALTER TABLE assistant.feedback
    ADD CONSTRAINT feedback_conversation_id_fkey
    FOREIGN KEY (conversation_id) REFERENCES assistant.conversations (id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS assistant.feedback_attachments (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id uuid NOT NULL REFERENCES assistant.feedback (id) ON DELETE CASCADE,
    filename text NOT NULL,
    content_type text NOT NULL,
    byte_size integer NOT NULL CHECK (byte_size > 0),
    content bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_attachments_feedback_idx
    ON assistant.feedback_attachments (feedback_id);

-- 006 granted on the tables that existed then; a table created later needs its own grant.
GRANT SELECT, INSERT ON assistant.feedback_attachments TO assistant_runtime;
