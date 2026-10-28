ALTER TABLE assistant.conversations
    ADD COLUMN IF NOT EXISTS selected_file_id INTEGER;

ALTER TABLE assistant.voice_sessions
    ADD COLUMN IF NOT EXISTS selected_file_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_conversations_dataset
    ON assistant.conversations (principal_id, selected_file_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_dataset
    ON assistant.voice_sessions (conversation_id, selected_file_id, updated_at DESC);

COMMENT ON COLUMN assistant.conversations.selected_file_id IS
    'Permanent single-dataset binding. NULL identifies a legacy unscoped conversation.';

COMMENT ON COLUMN assistant.voice_sessions.selected_file_id IS
    'Dataset binding copied from the conversation/LiveKit token for observability.';
