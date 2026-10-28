-- Restore the derived name and date columns on 91/93/94.
--
-- 048 removed them because the whole-list answering path was meant to replace the query
-- path on those lists. Measured against benchmarks/database_question_matrix.py it did not:
-- 26 of 74 cases passed against the query path's 62-66, with 25 genuinely wrong answers --
-- mostly counts, where an exact SQL aggregate beats a model reading 82 rows -- plus three
-- timeouts from the roughly 28k-token payload.
--
-- The path stays in the codebase behind WHOLE_LIST_ANSWER, which is off. While it is off
-- these lists are served by the query planner, and the query planner needs these columns:
-- without them Confirmed cannot group or sort by surname at all. Leaving 048 in place
-- would have made the death lists strictly worse than before either change.

DO $$
DECLARE
    target INTEGER;
    field RECORD;
    base TEXT;
    text_ops TEXT[] := ARRAY[
        'EQUALS', 'NOT_EQUALS', 'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY',
        'NOT_CONTAINS_ANY', 'STARTS_WITH', 'STARTS_WITH_ANY', 'IS_KNOWN', 'IS_UNKNOWN',
        'IN', 'NOT_IN', 'FULL_TEXT_SEARCH', 'FUZZY_SEARCH'
    ];
    number_ops TEXT[] := ARRAY[
        'EQUALS', 'NOT_EQUALS', 'GREATER_THAN', 'LESS_THAN', 'NUMBER_RANGE',
        'IN', 'NOT_IN', 'IS_KNOWN', 'IS_UNKNOWN'
    ];
BEGIN
    FOREACH target IN ARRAY ARRAY[91, 93, 94] LOOP
        PERFORM assistant.register_name_field(target, 'first_name', 'First name',
            ARRAY['given name', 'forename'], 'canonical.name_parts.first',
            text_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(target, 'middle_names', 'Middle names',
            ARRAY['middle name'], 'canonical.name_parts.middle',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'last_name', 'Last name',
            ARRAY['surname', 'family name'], 'canonical.name_parts.last',
            text_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(target, 'first_name_meaning',
            'First name meaning', ARRAY[]::TEXT[], 'canonical.name_parts.first_meaning',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'last_name_meaning',
            'Last name meaning', ARRAY[]::TEXT[], 'canonical.name_parts.last_meaning',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'indigenous_name_meaning',
            'Indigenous name meaning', ARRAY[]::TEXT[],
            'canonical.name_parts.indigenous_meaning', text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'other_first_names',
            'Other given names', ARRAY[]::TEXT[], 'canonical.name_parts.extra_first',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'other_last_names',
            'Other surnames', ARRAY[]::TEXT[], 'canonical.name_parts.extra_last',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'first_name_spellings',
            'First name spellings', ARRAY[]::TEXT[], 'canonical.name_parts.first_spelling',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'last_name_spellings',
            'Last name spellings', ARRAY[]::TEXT[], 'canonical.name_parts.last_spelling',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'indigenous_name_spellings',
            'Indigenous name spellings', ARRAY[]::TEXT[],
            'canonical.name_parts.indigenous_spelling', text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'name_alias', 'Name alias',
            ARRAY[]::TEXT[], 'canonical.name_parts.alias',
            text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(target, 'name_residue', 'Name cell residue',
            ARRAY[]::TEXT[], 'canonical.name_parts.residue',
            text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(target, 'name_flags', 'Name quality flags',
            ARRAY[]::TEXT[], 'canonical.name_parts.flags',
            text_ops, FALSE, TRUE, 'internal');

        FOR field IN
            SELECT semantic_field, human_label
            FROM assistant.field_registry
            WHERE file_id = target
              AND semantic_type = 'date'
              AND semantic_field NOT LIKE '%_year'
              AND semantic_field NOT LIKE '%_month'
              AND semantic_field NOT LIKE '%_day'
        LOOP
            base := 'canonical.date_parts.' || field.semantic_field || '.';
            PERFORM assistant.register_name_field(target, field.semantic_field || '_year',
                field.human_label || ' year', ARRAY[]::TEXT[], base || 'year',
                number_ops, TRUE, TRUE, 'authorized');
            PERFORM assistant.register_name_field(target, field.semantic_field || '_month',
                field.human_label || ' month', ARRAY[]::TEXT[], base || 'month',
                number_ops, TRUE, TRUE, 'authorized');
            PERFORM assistant.register_name_field(target, field.semantic_field || '_day',
                field.human_label || ' day', ARRAY[]::TEXT[], base || 'day',
                number_ops, TRUE, TRUE, 'authorized');
            PERFORM assistant.register_name_field(target, field.semantic_field || '_iso',
                field.human_label || ' (normalised)', ARRAY[]::TEXT[], base || 'iso',
                text_ops, TRUE, FALSE, 'internal');
            PERFORM assistant.register_name_field(target,
                field.semantic_field || '_precision', field.human_label || ' precision',
                ARRAY[]::TEXT[], base || 'precision', text_ops, FALSE, TRUE, 'internal');
            PERFORM assistant.register_name_field(target, field.semantic_field || '_extra',
                field.human_label || ' (other dates)', ARRAY[]::TEXT[], base || 'extra',
                text_ops, FALSE, FALSE, 'internal');
            PERFORM assistant.register_name_field(target, field.semantic_field || '_flags',
                field.human_label || ' flags', ARRAY[]::TEXT[], base || 'flags',
                text_ops, FALSE, TRUE, 'internal');
        END LOOP;
    END LOOP;
END
$$;

UPDATE assistant.field_registry
SET semantic_type = 'number'
WHERE canonical_json_path LIKE 'canonical.date_parts.%'
  AND (canonical_json_path LIKE '%.year'
       OR canonical_json_path LIKE '%.month'
       OR canonical_json_path LIKE '%.day');
