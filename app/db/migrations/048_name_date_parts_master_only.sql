-- Derived name and date columns are for the Master list only.
--
-- 043/046/047 registered them on all four lists so a query planner could filter on parsed
-- parts. Confirmed, Additional Deaths and Potential are now answered by handing the model
-- every row instead, so on those three the derived columns buy nothing -- and they cost
-- something real. Every misclassification the last review found sat on these lists: an
-- English surname respelling filed as an Indigenous name (Veters, Gray), a single-word
-- gloss filed as a surname (Duck), a bracketed Indigenous name filed as a spelling. The
-- model reading the source cell has none of those problems, because it never has to decide
-- which column a fragment belongs in.
--
-- The parsers and their accuracy gate are untouched: the Master list still uses them, and
-- benchmarks/name_parser_eval.py still measures them against the curated workbook.
--
-- Confirmed's planner catalog returns from 65 fields to 34.

DELETE FROM assistant.field_registry
WHERE file_id IN (91, 93, 94)
  AND (canonical_json_path LIKE 'canonical.name_parts.%'
       OR canonical_json_path LIKE 'canonical.date_parts.%');

-- Drop the stored blocks too, so a row on these lists carries only what the source
-- recorded plus the ingest's own canonical values.
UPDATE public.file_data_normalized
SET row_data_normalized = jsonb_set(
        row_data_normalized,
        '{canonical}',
        (row_data_normalized->'canonical') - 'name_parts' - 'date_parts'
    )
WHERE file_id IN (91, 93, 94)
  AND row_data_normalized ? 'canonical'
  AND ((row_data_normalized->'canonical') ? 'name_parts'
       OR (row_data_normalized->'canonical') ? 'date_parts');
