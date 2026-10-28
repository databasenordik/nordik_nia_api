-- Preprocessed name columns, identical on all four lists.
--
-- Every list stores names differently: the Master list splits them across Last/First/
-- Middle, while the three death lists pack everything into one STUDENT NAME cell. Worse,
-- a single cell routinely carries several facts at once -- "Fletcher (Souliere)" is two
-- surnames, "Campau (also spelt Compo)" is one surname twice, "Assiginak (A Starling)" is
-- a surname and its translation. Nia cannot match a person by name through that.
--
-- app/normalization/name_parser.py splits each cell into one fact per column and writes
-- the result to canonical.name_parts.*. The originals are never modified: student_name,
-- first_name's source cell and the rest still hold exactly what the source recorded, and
-- they remain what is displayed and cited.
--
-- The container is canonical.name_parts rather than canonical.name because
-- _field_display_at_path already special-cases 'canonical.name' as the display-name
-- fallback; nesting an object under that key would collide with it.
--
-- Multi-valued columns are stored as "; "-joined text, the same convention the curation
-- workbook uses, so CONTAINS and CONTAINS_ANY match a single variant inside them.
--
-- Visibility is deliberately split. The parts a researcher asks questions about are in
-- the planner catalog; the matching aids are exposure_policy='internal' so they stay out
-- of the planner prompt while still being read by retrieval and projected into rows.
-- Catalog size is load-bearing: an oversized SELECTED FIELDS block measurably degraded
-- planning accuracy, so only fields that answer questions are advertised.

CREATE OR REPLACE FUNCTION assistant.register_name_field(
    p_file_id INTEGER,
    p_semantic_field TEXT,
    p_human_label TEXT,
    p_aliases TEXT[],
    p_path TEXT,
    p_operators TEXT[],
    p_sortable BOOLEAN,
    p_aggregatable BOOLEAN,
    p_exposure TEXT
)
RETURNS VOID
LANGUAGE sql
AS $$
    INSERT INTO assistant.field_registry (
        file_id, semantic_field, human_label, aliases, canonical_json_path,
        semantic_type, allowed_operators, searchable, aggregatable, sortable,
        quoteable, sensitivity, exposure_policy
    )
    VALUES (
        p_file_id, p_semantic_field, p_human_label, COALESCE(p_aliases, '{}'), p_path,
        'text', p_operators, TRUE, p_aggregatable, p_sortable,
        FALSE, 'research', p_exposure
    )
    ON CONFLICT (file_id, semantic_field) DO UPDATE SET
        human_label = EXCLUDED.human_label,
        aliases = EXCLUDED.aliases,
        canonical_json_path = EXCLUDED.canonical_json_path,
        allowed_operators = EXCLUDED.allowed_operators,
        sortable = EXCLUDED.sortable,
        aggregatable = EXCLUDED.aggregatable,
        exposure_policy = EXCLUDED.exposure_policy;
$$;

DO $$
DECLARE
    target INTEGER;
    text_ops TEXT[] := ARRAY[
        'EQUALS', 'NOT_EQUALS', 'CONTAINS', 'NOT_CONTAINS', 'CONTAINS_ANY',
        'NOT_CONTAINS_ANY', 'STARTS_WITH', 'STARTS_WITH_ANY', 'IS_KNOWN', 'IS_UNKNOWN',
        'IN', 'NOT_IN', 'FULL_TEXT_SEARCH', 'FUZZY_SEARCH'
    ];
BEGIN
    FOREACH target IN ARRAY ARRAY[49, 91, 93, 94] LOOP
        -- Planner-visible. These are the parts a researcher asks about, and they give the
        -- three death lists the surname grouping and sorting that only the Master list
        -- could do before.
        PERFORM assistant.register_name_field(
            target, 'first_name', 'First name', ARRAY['given name', 'forename'],
            'canonical.name_parts.first', text_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(
            target, 'middle_names', 'Middle names', ARRAY['middle name'],
            'canonical.name_parts.middle', text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(
            target, 'last_name', 'Last name', ARRAY['surname', 'family name'],
            'canonical.name_parts.last', text_ops, TRUE, TRUE, 'authorized');
        PERFORM assistant.register_name_field(
            target, 'first_name_meaning', 'First name meaning', ARRAY['given name meaning'],
            'canonical.name_parts.first_meaning', text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(
            target, 'last_name_meaning', 'Last name meaning', ARRAY['surname meaning'],
            'canonical.name_parts.last_meaning', text_ops, FALSE, FALSE, 'authorized');
        PERFORM assistant.register_name_field(
            target, 'indigenous_name_meaning', 'Indigenous name meaning', ARRAY['spirit name meaning'],
            'canonical.name_parts.indigenous_meaning', text_ops, FALSE, FALSE, 'authorized');

        -- Matching aids. Kept out of the planner catalog so the prompt stays small; still
        -- read by retrieval and projected into rows, so an answer can quote them.
        PERFORM assistant.register_name_field(
            target, 'other_first_names', 'Other given names', ARRAY[]::TEXT[],
            'canonical.name_parts.extra_first', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'other_last_names', 'Other surnames', ARRAY[]::TEXT[],
            'canonical.name_parts.extra_last', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'first_name_spellings', 'First name spellings', ARRAY[]::TEXT[],
            'canonical.name_parts.first_spelling', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'last_name_spellings', 'Last name spellings', ARRAY[]::TEXT[],
            'canonical.name_parts.last_spelling', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'indigenous_name_spellings', 'Indigenous name spellings', ARRAY[]::TEXT[],
            'canonical.name_parts.indigenous_spelling', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'name_alias', 'Name alias', ARRAY[]::TEXT[],
            'canonical.name_parts.alias', text_ops, FALSE, FALSE, 'internal');
        PERFORM assistant.register_name_field(
            target, 'name_flags', 'Name quality flags', ARRAY[]::TEXT[],
            'canonical.name_parts.flags', text_ops, FALSE, TRUE, 'internal');
    END LOOP;
END
$$;
