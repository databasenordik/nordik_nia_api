-- Human corrections to the parsed name columns.
--
-- The parser is measured, not perfect: on the curated Master sample it places 97.3% of
-- variants in the right column and extracts 96.4% of them, and some cells are simply
-- ambiguous -- "Sheilah/Sheila" gives no rule for which spelling is canonical, and the
-- curated workbook itself chose the first form as often as the second.
--
-- So curation has to outrank the parser, and it has to survive re-parsing. The backfill
-- writes parser output to canonical.name_parts.*; any column present here replaces it.
-- Re-running the backfill can never destroy a person's work.
--
-- Rows are keyed by the source row, not by name, so a correction stays attached when the
-- name text itself is what changed.

CREATE TABLE IF NOT EXISTS assistant.name_overrides (
    file_id INTEGER NOT NULL,
    source_row_id BIGINT NOT NULL,
    first_name TEXT,
    middle_names TEXT,
    last_name TEXT,
    other_first_names TEXT,
    other_last_names TEXT,
    first_name_spellings TEXT,
    last_name_spellings TEXT,
    first_name_meaning TEXT,
    last_name_meaning TEXT,
    indigenous_name TEXT,
    indigenous_name_spellings TEXT,
    indigenous_name_meaning TEXT,
    name_alias TEXT,
    name_flags TEXT,
    note TEXT,
    curated_by TEXT NOT NULL DEFAULT 'unknown',
    curated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (file_id, source_row_id)
);

COMMENT ON TABLE assistant.name_overrides IS
    'Hand-curated name parts. Any non-null column wins over the parser output for that row.';

CREATE INDEX IF NOT EXISTS idx_name_overrides_file ON assistant.name_overrides (file_id);

-- The runtime never writes these; curation runs as the migrator. Reading is enough for
-- the gateway to apply an override when projecting a row.
GRANT SELECT ON assistant.name_overrides TO assistant_runtime;
