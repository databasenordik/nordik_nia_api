"""A misspelt name is answered or asked about, never reported as nothing found.

Someone reading a handwritten register types what they see, and what they see is often a
letter or two off. "No matching records" is then wrong twice over: the person is in the
list, and the researcher cannot tell a typo from a genuine absence.

How close the guess was decides the response, which is the distinction a person would draw:
one edit and one candidate is a typo worth correcting in place; anything less certain is a
question only the researcher can settle.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.data_gateway.memory import MemoryDataGateway  # noqa: E402
from app.execution.name_correction import (  # noqa: E402
    NameCorrection,
    NameSuggestion,
    ask_text,
    corrected_prefix,
    from_rows,
    is_name_field,
    rewrite_for,
    searched_name,
)
from app.planning.catalog import static_catalog  # noqa: E402
from app.planning.plan_schema import (  # noqa: E402
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)
from app.security.access_scope import AccessScope  # noqa: E402


def _scope() -> AccessScope:
    return AccessScope(principal_id="researcher", allowed_file_ids=(49, 91))


def _plan(*where: Predicate) -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=("lookup",),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(id="filter", op=PlanOp.FILTER, input="rows", where=list(where)),
        ],
    )


def _leaf(field: str, value: str) -> Predicate:
    return Predicate(field=field, operator=FilterOperator.CONTAINS, value=value)


# ---------------------------------------------------------------- reading the plan


def test_a_name_field_is_recognised_by_what_it_holds_not_by_its_name():
    catalog = static_catalog()
    assert is_name_field("student_name", (49,), catalog)
    assert not is_name_field("community", (49,), catalog)


def test_a_multi_word_name_is_reassembled_from_its_token_filters():
    """A token index is matched one filter per word, so the words are joined back."""
    plan = _plan(_leaf("student_name", "Albert"), _leaf("student_name", "Penence"))
    assert searched_name(plan, static_catalog()) == "Albert Penence"


def test_presence_filters_are_not_a_searched_name():
    plan = _plan(Predicate(field="student_name", operator=FilterOperator.IS_KNOWN))
    assert searched_name(plan, static_catalog()) == ""


def test_a_question_with_no_name_filter_has_nothing_to_correct():
    plan = _plan(_leaf("community", "Walpole Island"))
    assert searched_name(plan, static_catalog()) == ""


# ---------------------------------------------------------------- deciding what to do


def _suggest(*pairs: tuple[str, int]) -> NameCorrection:
    return NameCorrection(
        searched_for="Albert Penence",
        suggestions=tuple(
            NameSuggestion(display_name=name, file_id=49, source_row_id=i, distance=d)
            for i, (name, d) in enumerate(pairs)
        ),
    )


def test_one_person_one_edit_away_is_a_typo():
    assert _suggest(("Albert Penance", 1)).certain.display_name == "Albert Penance"


def test_the_only_person_within_reach_is_answered_at_two_edits_too():
    """Nothing is being guessed when there is nothing else it could be.

    The index is searched with a bounded edit distance per word, so one candidate means no
    other person is within that bound. Asking "did you mean Rita Abotossaway?" of a search
    for "Rita Abottasaway" put a question whose answer was already known, and the answer
    states the correction either way.
    """
    assert _suggest(("Albert Penance", 2)).certain.display_name == "Albert Penance"


def test_several_people_spelt_differently_are_asked_about():
    """"Penence" is close to two different names; only the researcher knows which."""
    correction = _suggest(("Bertha F Fisher", 1), ("Charles Pike", 1))
    assert correction.certain is None
    assert correction.corrected_word is None


def test_one_surname_shared_by_every_candidate_needs_no_question():
    """A surname is typed to ask about a family, not to pick one of its children."""
    correction = NameCorrection(
        searched_for="Abotossay",
        suggestions=tuple(
            NameSuggestion(display_name=name, file_id=49, source_row_id=index, distance=2)
            for index, name in enumerate(
                ["Christine Abotossaway", "Clayton Abotossaway", "Rita Abotossaway"]
            )
        ),
    )
    assert correction.certain is None
    assert correction.corrected_word == "Abotossaway"


def test_a_multiword_search_is_not_corrected_one_word_at_a_time():
    """Which person was meant is the question again as soon as a given name is in play."""
    correction = _suggest(("Albert Penance", 2), ("Alberta Penance", 2))
    assert correction.corrected_word is None


def test_a_name_the_list_records_is_never_corrected():
    """An exact match proves the name was not misspelt, so there is nothing to offer."""
    assert from_rows("Albert Penance", [{"display_name": "Albert Penance", "distance": 0}]) is None


def test_an_exact_match_silences_the_near_ones():
    """The real case this guards, taken from a live thread.

    Asked about four students named York, a plan bound "York" to the wrong name field and
    returned nothing. This then offered "Did you mean Albert Voerk?" -- inventing a spelling
    problem, hiding the real one, and pointing at someone else entirely. A zero-distance row
    anywhere in the index means this feature has no business in the turn.
    """
    rows = [
        {"display_name": "Edward York", "distance": 0},
        {"display_name": "Albert Voerk", "distance": 1},
        {"display_name": "Cassie Yorke", "distance": 1},
    ]
    assert from_rows("York", rows) is None


def test_nothing_close_enough_leaves_the_original_answer_alone():
    assert from_rows("Zzzz", []) is None


# ---------------------------------------------------------------- what the user reads


def test_a_single_candidate_is_offered_by_name():
    text = ask_text(_suggest(("Albert Penance", 2)))
    assert "Albert Penance" in text and "Did you mean" in text


def test_several_candidates_are_all_listed():
    text = ask_text(_suggest(("Bertha F Fisher", 1), ("Charles Fisher", 1)))
    assert "Bertha F Fisher" in text and "Charles Fisher" in text


def test_a_correction_is_stated_never_silent():
    text = corrected_prefix("Albert Penence", "Albert Penance")
    assert "Albert Penence" in text and "Albert Penance" in text


# ---------------------------------------------------------------- re-asking the question


def test_the_corrected_plan_asks_the_same_question_of_the_right_person():
    plan = _plan(_leaf("student_name", "Albert"), _leaf("student_name", "Penence"))
    rewritten = rewrite_for(plan, static_catalog(), "Albert Penance")
    values = [
        leaf.value
        for step in rewritten.steps
        for predicate in step.where
        for leaf in predicate.leaves()
    ]
    assert values == ["Albert", "Penance"]
    assert [step.op for step in rewritten.steps] == [step.op for step in plan.steps]


def test_conditions_other_than_the_name_survive_the_rewrite():
    """Correcting who was asked about must not drop what was asked."""
    plan = _plan(_leaf("student_name", "Penence"), _leaf("community", "Walpole Island"))
    rewritten = rewrite_for(plan, static_catalog(), "Albert Penance")
    kept = [
        (leaf.field, leaf.value)
        for step in rewritten.steps
        for predicate in step.where
        for leaf in predicate.leaves()
    ]
    assert ("community", "Walpole Island") in kept


def _search_plan(query: str) -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=(49,)),
        goals=("lookup",),
        steps=[
            PlanStep(id="rows", op=PlanOp.USE_CURRENT_VERSION),
            PlanStep(id="search", op=PlanOp.FULL_TEXT_SEARCH, input="rows", query=query),
        ],
    )


def test_a_name_searched_as_free_text_is_still_recognised():
    """The planner may search a name instead of filtering on it; the typo is the same."""
    plan = _search_plan("Albert Penence")
    assert searched_name(plan, static_catalog(), "Show the community for Albert Penence") == (
        "Albert Penence"
    )


def test_a_topic_searched_as_free_text_is_not_taken_for_a_name():
    """Only words written capitalised in the question read as a name."""
    plan = _search_plan("transferred")
    assert searched_name(plan, static_catalog(), "How many were transferred to schools?") == ""


def test_the_corrected_search_asks_for_the_corrected_name():
    plan = _search_plan("Albert Penence")
    rewritten = rewrite_for(
        plan, static_catalog(), "Albert Penance", searched_for="Albert Penence"
    )
    assert rewritten.steps[1].query == "Albert Penance"


# ---------------------------------------------------------------- against the gateway


@pytest.mark.parametrize(
    "typed,expected",
    [("Samuell A", "Samuel A"), ("Samuel A", None)],
)
def test_the_gateway_finds_near_misses_and_ignores_exact_ones(typed, expected):
    gateway = MemoryDataGateway()
    rows = asyncio.run(gateway.suggest_name_matches(_scope(), (49, 91), typed))
    correction = from_rows(typed, rows)
    if expected is None:
        assert correction is None
    else:
        assert correction is not None
        assert correction.certain is not None
        assert correction.certain.display_name == expected


def test_the_feature_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("NAME_CORRECTION", "false")
    get_settings.cache_clear()
    try:
        assert get_settings().name_correction is False
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------- pointing at an offer


def test_an_offered_list_can_be_indexed_by_a_follow_up():
    """"Did you mean one of these?" is answered by pointing: "the second one".

    That needs the offer registered as a result set. It normally falls out of running a
    QueryPlan, and this turn has none, so the correction path registers it directly --
    without this the follow-up came back "no listed result set to index".
    """
    from app.memory.followups import resolve_listed_ordinal
    from app.memory.types import ConversationRecord
    from app.memory.updates import record_candidate_list

    conversation = ConversationRecord(id="c1", principal_id="researcher")
    record_candidate_list(
        conversation,
        file_ids=(49,),
        records=[(49, 1101), (49, 1102), (49, 1103)],
        key="name_correction",
    )
    # index is zero-based at this level; the turn service maps "the second one" onto it.
    resolved = resolve_listed_ordinal(conversation, index=1)
    assert resolved.kind == "ordinal"
    assert resolved.source_ids == ["file:49:v1:row:1102"]


def test_an_empty_offer_registers_nothing():
    from app.memory.types import ConversationRecord
    from app.memory.updates import record_candidate_list

    conversation = ConversationRecord(id="c2", principal_id="researcher")
    record_candidate_list(conversation, file_ids=(49,), records=[], key="name_correction")
    assert conversation.active_frame() is None


def test_a_source_note_in_a_display_name_is_not_searched_as_part_of_the_name():
    from app.execution.name_correction import name_tokens

    assert name_tokens("Alexander KNAGGS (Manitoba Vital Stats)") == ["Alexander", "KNAGGS"]
    assert name_tokens("Betsey Jane, WHITE") == ["Betsey", "Jane", "WHITE"]
    assert name_tokens("Mary O'Brien-Smith") == ["Mary", "O'Brien-Smith"]
