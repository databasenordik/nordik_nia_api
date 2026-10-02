-- One column for the parts of a name cell that are not a name.
--
-- The name columns added in 043 hold only names. Everything else a transcriber wrote into
-- the same cell -- a provenance note, a community, a date of death, a status marker, a
-- sentence about the family -- had nowhere to go, so it was recognised, flagged, and then
-- dropped. That is lossy in a way this project should not be: the cell could no longer be
-- reconstructed from the derived columns, and a reviewer could not see what was set aside.
--
-- canonical.name_parts.residue now keeps those fragments verbatim. It is deliberately not
-- part of the name: it never feeds the name search index, so a student is never findable
-- by their community, by a date, or by the word "removed" -- which is exactly what was
-- happening before.
--
-- Internal, because it describes the record rather than answering a question about a
-- person; it is still projected into rows so an answer can quote it.

SELECT assistant.register_name_field(
    file_id, 'name_residue', 'Name cell residue', ARRAY[]::TEXT[],
    'canonical.name_parts.residue',
    ARRAY['EQUALS', 'NOT_EQUALS', 'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY',
          'NOT_CONTAINS_ANY', 'IS_KNOWN', 'IS_UNKNOWN'],
    FALSE, FALSE, 'internal'
)
FROM (SELECT UNNEST(ARRAY[49, 91, 93, 94]) AS file_id) t;
