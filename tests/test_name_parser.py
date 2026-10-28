"""Behaviour of the preprocessed name columns.

These lock the decisions the parser makes about what each fragment of a name *is*, which
is what makes a person findable. They deliberately assert on shape and on findability
rather than on any particular record: the curated workbook is the accuracy gate
(``benchmarks/name_parser_eval.py``), and no name from these records is asserted here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.normalization.name_parser import (  # noqa: E402
    classify_variant,
    looks_like_meaning,
    parse_full_name,
    parse_master,
)


def test_parenthetical_surname_is_a_second_identity():
    parts = parse_master(first="Alice", last="Fletcher (Souliere)")
    assert parts.last == "Fletcher"
    assert "Souliere" in parts.extra_last
    assert "Souliere" in parts.searchable()


def test_also_spelt_is_an_orthography_not_another_name():
    parts = parse_master(first="Wesley", last="Campau (also spelt Compo)")
    assert parts.last == "Campau"
    assert "Compo" in parts.last_spelling
    assert not parts.extra_last


def test_english_gloss_is_a_meaning_and_never_a_name():
    parts = parse_master(first="Francis", last="Assiginak (A Starling)")
    assert parts.last == "Assiginak"
    # Assiginak is itself an Indigenous surname, so the gloss is filed as its meaning
    # rather than as a meaning of the English surname slot. Either way the requirement is
    # that the translation is recorded as a meaning and is never a name.
    assert "A Starling" in parts.last_meaning + parts.indigenous_meaning
    assert "A Starling" not in parts.searchable()


def test_editorial_brackets_are_an_insertion_not_a_variant():
    parts = parse_master(first="Constance", last="Bott[omles]")
    assert parts.last == "Bottomles"
    assert "Bott[omles]" in parts.last_spelling


def test_ordinary_second_given_name_is_a_middle_name():
    parts = parse_master(first="Robert Nolan", last="Abotossaway")
    assert parts.first == "Robert"
    assert parts.middle == ("Nolan",)
    assert not parts.extra_first


def test_gloss_marks_the_preceding_token_as_an_indigenous_name():
    parts = parse_master(first="Joseph Ogauns (Little Pickerel)", last="Smith")
    assert parts.first == "Joseph"
    # The gloss is what identifies Ogauns as an Indigenous name, so it lands in the
    # indigenous slot rather than among the alternate given names -- which is what this
    # test is named for. It stays searchable either way, and is still not a middle name.
    assert "Ogauns" in parts.extra_first + (parts.indigenous,)
    assert "Ogauns" in parts.searchable()
    assert "Ogauns" not in parts.middle


def test_uncertain_variants_go_to_alias_rather_than_a_guess():
    # A parenthetical too close to be a different name and too far to be a respelling has
    # no safe classification, so it must stay findable instead of being guessed.
    verdict = classify_variant("Susannah", "Susan", "paren")
    assert verdict in {"extra", "alias"}
    parts = parse_master(first="Susannah (Susan)", last="Kechego")
    assert "Susan" in parts.searchable()


def test_diminutive_is_an_alternate_name_not_a_respelling():
    assert classify_variant("John", "Johnny", "paren") == "extra"


def test_respelling_is_not_mistaken_for_a_diminutive():
    assert classify_variant("Kezhikgobness", "Kezhikgobnes", "slash") == "spelling"


def test_uppercase_surname_splits_a_single_name_cell():
    parts = parse_full_name("Charles Mortimer GREEN")
    assert (parts.first, parts.last) == ("Charles", "GREEN")
    assert parts.middle == ("Mortimer",)


def test_comma_form_puts_the_surname_after_the_comma():
    parts = parse_full_name("Betsey Jane, WHITE")
    assert (parts.first, parts.last) == ("Betsey", "WHITE")


def test_provenance_annotation_is_flagged_not_treated_as_a_name():
    parts = parse_full_name("Albert PENANCE (Manitoba Vital Stats)")
    assert (parts.first, parts.last) == ("Albert", "PENANCE")
    assert "annotated" in parts.flags
    assert not any("Vital" in token for token in parts.searchable())


def test_a_fused_trailing_field_is_split_off_and_flagged():
    parts = parse_full_name("Edward ShingwaukFebruary 11, 1905Garden River First Nation")
    assert parts.first == "Edward"
    assert parts.last == "Shingwauk"
    assert "concatenated" in parts.flags


def test_surname_taken_without_a_cue_is_marked_inferred():
    parts = parse_full_name("Tonsay Peltier")
    assert parts.last == "Peltier"
    assert "inferred_surname" in parts.flags


def test_indigenous_name_gloss_and_respellings():
    parts = parse_full_name("John SMITH", indigenous="Micowatch, Mecowatch, Mucowatch")
    assert parts.indigenous == "Micowatch"
    assert "Mecowatch" in parts.searchable()
    hyphenated = parse_full_name("A B", indigenous="Shahwanenooqua-lady of the south wind")
    assert hyphenated.indigenous == "Shahwanenooqua"
    assert "lady of the south wind" in hyphenated.indigenous_meaning


def test_meaning_detection_ignores_relation_words():
    # "or Scutzman" and "also spelt Compo" read as English but state a relation.
    assert not looks_like_meaning("or Scutzman")
    assert looks_like_meaning("Cloud Running in a Line")


def test_suffix_is_not_an_alternate_given_name():
    parts = parse_master(first="Charles Sr.", last="Baxter")
    assert parts.first == "Charles"
    assert "Sr." not in parts.extra_first
    assert any(flag.startswith("suffix:") for flag in parts.flags)


def test_children_note_is_not_a_given_name():
    parts = parse_master(first="Mrs. Willie (Children)", last="Cree")
    assert parts.first == "Willie"
    assert "Children" not in parts.extra_first
    assert "annotated" in parts.flags
    assert "titled" in parts.flags


def test_question_mark_is_uncertainty_not_a_spelling():
    parts = parse_master(first="Robby (?)", last="Kahbege (?)")
    assert parts.first == "Robby"
    assert parts.last == "Kahbege"
    assert "uncertain" in parts.flags
    assert not any("?" in token for token in parts.searchable())


def test_sp_query_is_not_a_surname():
    parts = parse_master(first="Lacchaeus", last="Lainawap (sp?)")
    assert parts.last == "Lainawap"
    assert "sp?" not in parts.extra_last
    assert "uncertain" in parts.flags


def test_tab_separated_date_and_community_are_not_name_parts():
    parts = parse_full_name("Hannah Kechego \tJanuary 21, 1905\tMuncey (Chippewas of the Thames)")
    assert (parts.first, parts.last) == ("Hannah", "Kechego")
    assert not any(token in {"January", "1905", "Muncey"} for token in parts.searchable())


def test_family_narrative_extracts_the_named_person():
    parts = parse_full_name(
        "The Isaacs Family: Emma was the fifth child from the Isaacs family to die at Shingwauk"
    )
    assert (parts.first, parts.last) == ("Emma", "Isaacs")
    assert "narrative" in parts.flags
    assert parts.first != "The"


def test_placeholder_is_not_a_surname():
    parts = parse_full_name("TO BE REVIEWED")
    assert parts.first == ""
    assert parts.last == ""
    assert "placeholder" in parts.flags


def test_newline_variants_stay_separate():
    parts = parse_full_name("Frank SPANIEL", indigenous="ESPANIEL\nESPAGNOL")
    found = parts.searchable()
    assert "ESPANIEL" in found
    assert "ESPAGNOL" in found
    assert "ESPANIEL ESPAGNOL" not in found
    assert parts.indigenous != "ESPANIEL ESPAGNOL"


def test_explicit_meaning_is_not_a_spelling():
    parts = parse_full_name("John RODD", indigenous='Wahwegahbowh (meaning "Stand in Middle")')
    assert parts.indigenous == "Wahwegahbowh"
    assert "Stand in Middle" in parts.indigenous_meaning
    assert not any("Stand" in token for token in parts.indigenous_spelling)


def test_empty_input_is_safe():
    parts = parse_master()
    assert parts.first == "" and parts.last == ""
    assert parts.searchable() == ()


def test_backfill_reads_the_raw_cell_not_the_cleaned_one():
    """The pipeline must parse what a person typed, not what ingest already cleaned.

    canonical.first_name holds "Willie Pashegezhik" where the source column reads
    "Willie Pashegezhik (Cloud Running in a Line)". Parsing the canonical value silently
    discarded every translation on the master list; the parser's own tests could not
    catch it because they never exercised source-cell selection.
    """
    from app.normalization.backfill_names import _source_cell

    row = {
        "row_data_normalized": {
            "fields": {"First Names": {"raw": "Willie Pashegezhik (Cloud Running in a Line)"}},
            "canonical": {"first_name": "Willie Pashegezhik"},
        }
    }
    sources = {"first_name_source": ["fields.First Names", "canonical.first_name"]}
    assert _source_cell(row, "first_name_source", sources).endswith("(Cloud Running in a Line)")

    # With no raw column the canonical path is still a valid fallback.
    bare = {"row_data_normalized": {"canonical": {"first_name": "Willie"}}}
    assert _source_cell(bare, "first_name_source", sources) == "Willie"


def test_normalized_field_objects_do_not_leak_python_repr():
    from app.normalization.backfill_names import _as_text

    assert _as_text({"raw": "Pinnance", "normalized": "pinnance", "tokens": ["pinnance"]}) == "Pinnance"
    assert _as_text({"normalized": "pinnance"}) == "pinnance"
    assert _as_text({"tokens": ["x"]}) == ""


def test_labelled_comment_spelling_lands_on_the_name_it_resembles():
    parts = parse_master(first="Christine", last="Abbot", comments="Also spelled: Christina")
    assert parts.first == "Christine"
    assert "Christina" in parts.first_spelling
    assert "Christina" in parts.searchable()
    assert not parts.last_spelling


def test_labelled_comment_indigenous_name_and_its_meaning_are_separated():
    parts = parse_master(
        first="Willie",
        last="Adams",
        comments="Indian name: Pashegezhik; Meaning of Indian name: Cloud Running in a Line",
    )
    assert parts.indigenous == "Pashegezhik"
    assert parts.indigenous_meaning == ("Cloud Running in a Line",)
    assert "Cloud Running in a Line" not in parts.searchable()


def test_labelled_comment_alias_is_a_name_and_an_alternate_surname_is_a_surname():
    parts = parse_master(
        first="Leon", last="Nelson", comments="Also known as: Tuffy; Alternate surname: Isaac"
    )
    assert "Tuffy" in parts.alias
    assert "Isaac" in parts.extra_last
    assert {"Tuffy", "Isaac"} <= set(parts.searchable())


def test_a_spelling_that_resembles_no_parsed_name_is_not_recorded_as_one():
    parts = parse_master(first="Mary", last="Bearskin", comments="Also spelled: Kahgegwonabe")
    assert not parts.first_spelling
    assert not parts.last_spelling
    assert "Kahgegwonabe" in parts.alias


def test_unlabelled_comment_prose_is_kept_without_becoming_searchable():
    parts = parse_master(first="John", last="Smith", comments="Died at the school in 1901")
    assert parts.residue == ("Died at the school in 1901",)
    assert "Died" not in " ".join(parts.searchable())


def test_bare_aka_comment_is_read_as_an_alias():
    parts = parse_master(first="Thomas", last="Shilling", comments="AKA Tommy")
    assert "Tommy" in parts.alias


def test_original_recorded_name_is_a_spelling_not_a_second_given_name():
    parts = parse_master(first="Sahguj", last="Schell", comments="Original first name recorded as: [Sen]")
    assert parts.first == "Sahguj"
    assert "Sen" not in parts.extra_first
    assert "Sen" in parts.searchable()


def test_comments_do_not_disturb_a_row_that_has_none():
    with_comment = parse_master(first="Ann", last="Pine", comments="")
    assert with_comment == parse_master(first="Ann", last="Pine")
