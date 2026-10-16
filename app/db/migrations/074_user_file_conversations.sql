-- NEW CHANGES TO EXECUTE: user/file chat history and question/answer exchange IDs.
-- This migration is mirrored in nordikdriveapi/init.sql and in the NEW CHANGES
-- section of nordikdriveapi/assistant.sql. Use ONE upgrade route per database.
-- Extend existing NIA storage; public.users/public.file remain authoritative.
-- Legacy conversations have no trustworthy application-user mapping and stay NULL.
ALTER TABLE assistant.conversations ADD COLUMN IF NOT EXISTS user_id BIGINT;
-- Preserve the dataset binding used by NIA; expose the stored application file ID
-- without maintaining two independently editable values for the same file.
ALTER TABLE assistant.conversations ADD COLUMN IF NOT EXISTS file_id INTEGER
    GENERATED ALWAYS AS (selected_file_id) STORED;

CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_user_file
    ON assistant.conversations (user_id, file_id)
    WHERE user_id IS NOT NULL;

COMMENT ON COLUMN assistant.conversations.user_id IS
    'Authenticated Nordik Drive user ID forwarded by its trusted gRPC adapter.';
COMMENT ON COLUMN assistant.conversations.selected_file_id IS
    'Nordik Drive file ID; together with user_id identifies one conversation.';
COMMENT ON COLUMN assistant.conversations.file_id IS
    'Stored application file ID, generated from the existing selected_file_id binding.';

ALTER TABLE assistant.messages ADD COLUMN IF NOT EXISTS exchange_id UUID;
ALTER TABLE assistant.messages ADD COLUMN IF NOT EXISTS communities TEXT[] NOT NULL DEFAULT '{}';
ALTER TABLE assistant.messages ADD COLUMN IF NOT EXISTS row_access_revision TEXT;

-- Backfill identifiable adjacent Q/A pairs without inventing user ownership for
-- legacy conversations. A message keeps its own id; both messages share exchange_id.
WITH ordered AS (
    SELECT id, role,
           lag(id) OVER w AS previous_id,
           lag(role) OVER w AS previous_role
    FROM assistant.messages
    WINDOW w AS (PARTITION BY conversation_id ORDER BY
        CASE WHEN turn_id ~ '^turn-[0-9]+$'
             THEN substring(turn_id from '[0-9]+')::bigint ELSE 0 END,
        created_at, id)
)
UPDATE assistant.messages m
SET exchange_id = CASE WHEN ordered.role = 'assistant' AND previous_role = 'user'
                       THEN previous_id ELSE ordered.id END
FROM ordered WHERE m.id = ordered.id AND m.exchange_id IS NULL;

ALTER TABLE assistant.messages ALTER COLUMN exchange_id SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_exchange_role
    ON assistant.messages (conversation_id, exchange_id, role);
COMMENT ON COLUMN assistant.messages.exchange_id IS
    'Shared ID for a question and its answer; id remains the individual message ID.';
