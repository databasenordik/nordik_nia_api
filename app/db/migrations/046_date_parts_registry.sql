-- Preprocessed date columns, identical on all four lists.
--
-- The lists record dates in at least a dozen shapes -- 1889-12-24, June 1882, 1874,
-- 23.04.1940, 9Aug1877, "28 August 1879 2nd time August 1881", 1868-00-00~ -- and the
-- production parser flattens all of them into a single DATE. That flattening invents
-- data: "June 1882" matches only its bare-year branch and is stored as 1882-01-01, so
-- the month is discarded and a day nobody recorded is asserted. 666 cells were padded to
-- 1 January and 246 to the 1st of their month.
--
-- app/normalization/date_parser.py keeps what the source actually said and writes it to
-- canonical.date_parts.<field>.*. The original cell is untouched and remains what is
-- displayed and cited, and the existing date field keeps working exactly as before.
--
-- Every date field on every list gets the same treatment, derived from the registry
-- rather than a hardcoded list, so a new date column is covered automatically.
--
-- Visibility: year, month and day are planner-visible because they are what a researcher
-- groups and filters by ("deaths per year", "admissions in June"). The bookkeeping --
-- iso, precision, extra, flags -- is exposure_policy='internal', read by retrieval and
-- projected into rows but kept out of the planner prompt, which is size-sensitive.

DO $$
DECLARE
    field RECORD;
    number_ops TEXT[] := ARRAY[
        'EQUALS', 'NOT_EQUALS', 'GREATER_THAN', 'LESS_THAN', 'NUMBER_RANGE',
        'IN', 'NOT_IN', 'IS_KNOWN', 'IS_UNKNOWN'
    ];
    text_ops TEXT[] := ARRAY[
        'EQUALS', 'NOT_EQUALS', 'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY',
        'NOT_CONTAINS_ANY', 'STARTS_WITH', 'IS_KNOWN', 'IS_UNKNOWN'
    ];
    base TEXT;
BEGIN
    FOR field IN
        SELECT file_id, semantic_field, human_label
        FROM assistant.field_registry
        WHERE semantic_type = 'date'
          AND semantic_field NOT LIKE '%\_year'
          AND semantic_field NOT LIKE '%\_month'
          AND semantic_field NOT LIKE '%\_day'
    LOOP
        base := 'canonical.date_parts.' || field.semantic_field || '.';

        -- The three components the request asked for. Numeric so they can be grouped,
        -- compared and averaged directly.
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_year', field.human_label || ' year',
            ARRAY[]::TEXT[], base || 'year', number_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_month', field.human_label || ' month',
            ARRAY[]::TEXT[], base || 'month', number_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_day', field.human_label || ' day',
            ARRAY[]::TEXT[], base || 'day', number_ops, TRUE, TRUE, 'authorized');

        -- One normalised string at the precision actually recorded: 1889-12-24, 1882-06
        -- or 1874. Never padded, so it can be shown without asserting a false day.
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_iso', field.human_label || ' (normalised)',
            ARRAY[]::TEXT[], base || 'iso', text_ops, TRUE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_precision', field.human_label || ' precision',
            ARRAY[]::TEXT[], base || 'precision', text_ops, FALSE, TRUE, 'internal');
        -- Further dates recorded in the same cell: a second or third admission is a real
        -- event, and the production parser discards everything after the first date.
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_extra', field.human_label || ' (other dates)',
            ARRAY[]::TEXT[], base || 'extra', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            field.file_id, field.semantic_field || '_flags', field.human_label || ' flags',
            ARRAY[]::TEXT[], base || 'flags', text_ops, FALSE, TRUE, 'internal');
    END LOOP;
END
$$;

-- The components are numbers, not text; register_name_field defaults to text.
UPDATE assistant.field_registry
SET semantic_type = 'number'
WHERE canonical_json_path LIKE 'canonical.date_parts.%'
  AND (canonical_json_path LIKE '%.year'
       OR canonical_json_path LIKE '%.month'
       OR canonical_json_path LIKE '%.day');
