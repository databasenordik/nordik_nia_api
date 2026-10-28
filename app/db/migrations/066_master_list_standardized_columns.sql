-- The Master list's spreadsheet columns were restandardized at version 5: the name
-- columns lost their plural spelling ("First Names" -> "First Name"), and the community
-- column split in two ("Chisasibi #30" -> "Chisasibi" plus a separate band number).
--
-- The registry records which spreadsheet column each semantic field was read from, and
-- backfill_names reads the Master list's name cells through exactly those keys: names are
-- regenerated raw-only, because the canonical values are this parser's own output and must
-- never become its input. A renamed column therefore reads as an empty cell -- on the
-- standardized data the backfill named 30 of 2,814 rows, the 30 that still carry the old
-- spelling -- and first, middle and last name empty out for everybody else.
--
-- Both spellings are kept so that rows of either shape parse. They never co-occur in one
-- row, so the order only decides which is looked at first; the standardized name leads.

UPDATE assistant.field_registry
SET raw_json_keys = ARRAY['First Name', 'First Names']
WHERE file_id = 49 AND semantic_field = 'first_name';

UPDATE assistant.field_registry
SET raw_json_keys = ARRAY['Middle Name', 'Middle Names']
WHERE file_id = 49 AND semantic_field = 'middle_names';

UPDATE assistant.field_registry
SET raw_json_keys = ARRAY['Last Name', 'Last Names']
WHERE file_id = 49 AND semantic_field = 'last_name';

UPDATE assistant.field_registry
SET raw_json_keys = ARRAY['First Nation Community', 'First Nation/Community', 'Community']
WHERE file_id = 49 AND semantic_field = 'community';

SELECT assistant.refresh_fill_rates();
