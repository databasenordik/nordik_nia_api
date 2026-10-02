"""High-confidence follow-up shapes the compiler must not get wrong."""

from __future__ import annotations

import re

from app.planning.analytic_guards import _STRIP_BAND_NUMBER, apply_analytic_guards
from app.planning.catalog import FieldCatalog
from app.planning.turn_schema import (
    ANALYTIC_GOALS,
    MAX_WINDOW_SIZE,
    ClarifyAction,
    ConversationAction,
    FilterGroupSpec,
    FilterSpec,
    ModifyPreviousAction,
    QueryAction,
    ReferenceAction,
    ResultWindow,
    TurnPlan,
    VerifyPreviousAction,
)
from app.planning.user_name import (
    extract_followup_name,
    extract_introduced_name,
    is_assistant_identity_question,
    is_bare_affirmation,
    is_confusion,
    is_conversational_correction,
    is_explicit_row_reference,
    is_speak_request,
    is_user_name_question,
    looks_like_research,
)

_GREETING = re.compile(
    r"^(?:hi|hey|hello|hiya|yo|howdy)(?:\s+there)?[.!?]*$",
    re.IGNORECASE,
)
_THANKS = re.compile(r"^(?:thanks|thank you|thx|ty)[.!?]*$", re.IGNORECASE)
_FAREWELL = re.compile(r"^(?:bye|goodbye|good bye|see you|later)[.!?]*$", re.IGNORECASE)
_NEXT = re.compile(
    r"^(?:next(?:\s+\d+)?|keep going|show more|continue|next page)[.!?]*$",
    re.IGNORECASE,
)
_PREV = re.compile(
    r"^(?:previous(?:\s+\d+)?|go back|back|show previous|previous page)[.!?]*$",
    re.IGNORECASE,
)
_CHALLENGE = re.compile(
    r"^(?:i believe there(?:'s| is| are)? more(?: than that)?|there(?:'s| is| are) more than that|"
    r"that'?s wrong|still wrong|check again|are you sure\??)[.!?]*$",
    re.IGNORECASE,
)
_LIST_PREVIOUS = re.compile(
    r"\b(?:list (?:them|those|these)|one by one|organized|alphabet)",
    re.IGNORECASE,
)
_LIST_STUDENTS = re.compile(
    r"(?<![-])\blist(?:\s+all)?(?:\s+the)?\s+students\b",
    re.IGNORECASE,
)
_PURE_LIST_STUDENTS = re.compile(
    r"^(?:please\s+)?list(?:\s+all)?(?:\s+the)?\s+students[.!?]*$",
    re.IGNORECASE,
)
_RESULT_WINDOW = re.compile(
    r"\b(?P<anchor>last|final|first|initial)\s+(?P<size>\d{1,4})\b",
    re.IGNORECASE,
)
_FIELD_PROJECTION = re.compile(
    r"^(?:please\s+)?(?:put|add|include|show)\s+(?P<fields>.+?)\s+"
    r"(?:beside|besides|alongside|next\s+to)\s+(?:each|every|the)\b",
    re.IGNORECASE,
)
_ALL_INFORMATION = re.compile(
    r"\b(?:all|every|complete|full)\s+(?:available\s+)?(?:information|info|details?|fields?|columns?)\b"
    r"|\beverything\s+(?:recorded\s+)?(?:about|for|on)\b",
    re.IGNORECASE,
)
_AVAILABLE_DATA = re.compile(
    r"\b(?:what|which)\b.{0,45}\b(?:information|data|records|datasets?|fields?)\b"
    r".{0,45}\b(?:available|access|contain|include|exist)\b"
    r"|\b(?:what|which)\b.{0,30}\b(?:available|accessible)\b.{0,30}"
    r"\b(?:information|data|records|datasets?|fields?)\b|"
    r"\bwhat\s+(?:is|are)?\s*(?:in|inside)\s+(?:the\s+)?"
    r"(?:(?:research|authorized)\s+)?(?:records|datasets?)\b",
    re.IGNORECASE,
)
_CAUSE_SUMMARY = re.compile(
    r"\b(?:summarize|summary\s+of|summarise|overview\s+of)\b.{0,60}"
    r"\bcauses?\s+of\s+death\b|\bcauses?\s+of\s+death\b.{0,40}\bsummary\b",
    re.IGNORECASE,
)
_SING = re.compile(r"^(?:can|could|would|will)\s+you\s+sing(?:\s+for\s+me)?[.!?]*$", re.IGNORECASE)
_CATALOG_FIELD_REQUEST = re.compile(
    r"\b(?:what|which|list|show)\b[^?]{0,35}\b(communities|schools)\b",
    re.IGNORECASE,
)
_RANDOM_SAMPLE = re.compile(r"\b(?:random|randomly|sample|pick\s+any)\b", re.IGNORECASE)
_STUDENT_REQUEST = re.compile(r"\b(?:student|students|master\s+list)\b", re.IGNORECASE)
_KNOWN_BOTH_DATES = re.compile(
    r"\b(?:have|has|with|where)\b.{0,80}\b(?:admitted|admission)\b.{0,50}"
    r"\b(?:discharged|discharge)\b|"
    r"\b(?:have|has|with|where)\b.{0,80}\b(?:discharged|discharge)\b.{0,50}"
    r"\b(?:admitted|admission)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def apply_trusted_overrides(
    turn: TurnPlan,
    text: str,
    *,
    has_active_query: bool,
    catalog: FieldCatalog | None = None,
    last_assistant_text: str | None = None,
) -> TurnPlan:
    lowered = " ".join(text.strip().split())
    last = (last_assistant_text or "").lower()
    if accepts_offered_name(text, last_assistant_text):
        return _replace(turn, [ReferenceAction(selector="first", target="row")])
    if _GREETING.fullmatch(lowered):
        return _respond(turn, "greeting")
    if _THANKS.fullmatch(lowered):
        return _respond(turn, "thanks")
    if _FAREWELL.fullmatch(lowered):
        return _respond(turn, "farewell")
    if is_assistant_identity_question(lowered):
        return _respond(turn, "assistant_identity")
    introduced = extract_introduced_name(lowered)
    if introduced:
        return _respond(turn, "user_introduction", response=introduced)
    if is_user_name_question(lowered):
        return _respond(turn, "user_identity")
    if is_speak_request(lowered):
        return _respond(turn, "capability", response="__voice__")
    if _SING.fullmatch(lowered):
        return _respond(turn, "capability", response="__sing__")
    if _AVAILABLE_DATA.search(lowered):
        return _respond(turn, "describe_available_data", response="__describe_authorized_data__")
    cause_summary = _enforce_cause_summary(turn, lowered)
    if cause_summary is not None:
        return cause_summary
    unavailable = _unavailable_catalog_field(lowered, catalog=catalog)
    if unavailable:
        return _respond(turn, "general_conversation", response=unavailable)
    if _RANDOM_SAMPLE.search(lowered) and _STUDENT_REQUEST.search(lowered):
        sampled = _apply_random_sample(turn, lowered, catalog=catalog)
        if sampled is not None:
            return sampled
    requested_fields = (
        _all_public_fields(catalog)
        if _ALL_INFORMATION.search(lowered)
        else _extract_requested_fields(lowered)
    )
    if requested_fields:
        projected = _apply_requested_fields(
            turn,
            requested_fields,
            has_active_query=has_active_query,
            catalog=catalog,
        )
        if projected is not None:
            return projected
    window = _extract_result_window(lowered)
    if window is not None:
        windowed = _apply_result_window(turn, window, has_active_query=has_active_query)
        if windowed is not None:
            return windowed
    if is_bare_affirmation(lowered):
        if "connect voice" in last or "speak" in last or "voice" in last:
            return _respond(turn, "capability", response="__voice__")
        if has_active_query and _offers_next_page(last):
            return _replace(turn, [ModifyPreviousAction(goal="list", page="next")])
        return _respond(turn, "acknowledgement")
    # "without the hash numbers" regroups the previous distinct list by the
    # label with its band number removed. Without this the request has no
    # expressible edit and the same list is returned unchanged.
    if has_active_query and _STRIP_BAND_NUMBER.search(lowered):
        return _replace(turn, [ModifyPreviousAction(group_value_part="base_name")])

    if is_confusion(lowered):
        return _respond(turn, "correction", response="__explain_previous__")
    if is_conversational_correction(lowered) and (
        not looks_like_research(lowered) or _EXPLICIT_COMPLAINT.search(lowered)
    ):
        return _respond(turn, "correction", response="__acknowledge_correction__")
    if turn.reference_actions() and not is_explicit_row_reference(lowered):
        if any((item.target or "row") == "row" for item in turn.reference_actions()):
            return _respond(turn, "correction")
    if _LIST_STUDENTS.search(lowered) and not (
        has_active_query and _LIST_PREVIOUS.search(lowered)
    ) and not turn.modify_actions() and not turn.verify_actions():
        if turn.clarify_actions() or turn.confidence < 0.75 or not turn.query_actions():
            file_id = catalog.default_people_file_id if catalog is not None else 49
            return _replace(
                turn,
                [
                    QueryAction(
                        datasets=[file_id] if file_id else [49],
                        goal="list",
                        exhaustive=True,
                        presentation="numbered_list",
                        sort_by="student_name",
                        sort_direction="asc",
                        limit=25,
                        offset=0,
                    )
                ],
            )
    if has_active_query and _NEXT.fullmatch(lowered):
        return _replace(
            turn,
            [ModifyPreviousAction(goal="list", page="next")],
        )
    if has_active_query and _PREV.fullmatch(lowered):
        return _replace(
            turn,
            [ModifyPreviousAction(goal="list", page="previous")],
        )
    if has_active_query and _CHALLENGE.fullmatch(lowered):
        return _replace(turn, [VerifyPreviousAction()])
    if has_active_query and _LIST_PREVIOUS.search(lowered):
        return _replace(
            turn,
            [
                ModifyPreviousAction(
                    goal="list",
                    exhaustive=True,
                    presentation="numbered_list",
                    sort_by="student_name",
                    sort_direction="asc",
                    page="first",
                )
            ],
        )
    followup_name = extract_followup_name(lowered)
    if followup_name and (has_active_query or len(followup_name.split()) >= 2):
        file_id = catalog.default_people_file_id if catalog is not None else 49
        requested_fields = []
        if catalog is not None and file_id:
            requested_fields = [
                spec.semantic_field
                for spec in catalog.fields_for(file_id)
                if spec.semantic_field != "student_name"
            ]
        return _replace(
            turn,
            [
                QueryAction(
                    datasets=[file_id] if file_id else [49],
                    goal="list",
                    filters=[
                        FilterSpec(field="student_name", operator="CONTAINS", value=followup_name)
                    ],
                    requested_fields=requested_fields,
                    presentation="numbered_list",
                    sort_by="student_name",
                    sort_direction="asc",
                    limit=25,
                    offset=0,
                )
            ],
        )
    if not _NEXT.fullmatch(lowered) and not _PREV.fullmatch(lowered):
        turn = _reset_implicit_paging(turn)
    if turn.respond_actions() and all(
        item.intent == "general_conversation" for item in turn.respond_actions()
    ):
        return turn.model_copy(
            update={
                "final_response": "compiler_response",
                "unresolved": [],
                "confidence": max(turn.confidence, 0.9),
                "needs_explanation": False,
                "needs_inference": False,
                "needs_evidence": False,
            }
        )
    if not looks_like_research(lowered) and _bare_research_dump(turn):
        return _respond(turn, "general_conversation")
    if catalog is not None:
        # Last: a calculation request must not leave here as a page of rows.
        turn = apply_analytic_guards(turn, text, catalog)
    return turn


def trusted_fast_turn(
    text: str,
    *,
    has_active_query: bool,
    catalog: FieldCatalog | None = None,
    last_assistant_text: str | None = None,
) -> TurnPlan | None:
    """Resolve only high-confidence rule matches without a model call.

    The clarification seed is returned unchanged when no trusted rule matches,
    so ambiguous language still goes through the semantic compiler.
    """

    normalized = " ".join(text.strip().split())
    # These requests benefit from semantic compilation. In particular, synthesis
    # must retain its authorization validation, evidence retrieval, observability,
    # and streaming path instead of being replaced by a regex-only query.
    if _CAUSE_SUMMARY.search(normalized):
        return None
    seed = TurnPlan(
        normalized_request=normalized,
        actions=[ClarifyAction(question="Semantic compilation is required.")],
        final_response="compiler_response",
        confidence=0.0,
    )
    candidate = apply_trusted_overrides(
        seed,
        text,
        has_active_query=has_active_query,
        catalog=catalog,
        last_assistant_text=last_assistant_text,
    )
    if candidate.clarify_actions():
        return None
    query_actions = candidate.query_actions()
    if query_actions:
        if all(action.sample for action in query_actions):
            return candidate
        if len(query_actions) == 1 and _PURE_LIST_STUDENTS.fullmatch(normalized):
            return candidate
        followup_name = extract_followup_name(normalized)
        if followup_name and (has_active_query or len(followup_name.split()) >= 2):
            return candidate
        return None
    return candidate


def _apply_random_sample(
    turn: TurnPlan,
    text: str,
    *,
    catalog: FieldCatalog | None,
) -> TurnPlan | None:
    queries = turn.query_actions()
    default_id = catalog.default_people_file_id if catalog is not None else 49
    if queries:
        base = queries[0]
        datasets = list(base.datasets) or ([default_id] if default_id else [])
        filters = list(base.filters)
        filter_groups = [group.model_copy(deep=True) for group in base.filter_groups]
        filter_logic = base.filter_logic
        fields = list(base.requested_fields)
    else:
        datasets = [default_id] if default_id else []
        filters = []
        filter_groups = []
        filter_logic = "and"
        fields = []
    if not datasets:
        return None

    if _KNOWN_BOTH_DATES.search(text):
        known_filters = [
            FilterSpec(field=field, operator="IS_KNOWN")
            for field in ("admitted_date", "discharged_date")
        ]
        if filter_groups:
            updated_groups: list[FilterGroupSpec] = []
            for group in filter_groups:
                members = list(group.filters)
                for required in known_filters:
                    if not any(
                        item.field == required.field and item.operator == required.operator
                        for item in members
                    ):
                        members.append(required)
                updated_groups.append(FilterGroupSpec(filters=members))
            filter_groups = updated_groups
        else:
            for required in known_filters:
                if not any(
                    item.field == required.field and item.operator == required.operator
                    for item in filters
                ):
                    filters.append(required)
        for field in ("admitted_date", "discharged_date"):
            if field not in fields:
                fields.append(field)
    if "student_name" not in fields:
        fields.insert(0, "student_name")

    return _replace(
        turn,
        [
            QueryAction(
                datasets=datasets,
                goal="list",
                filters=filters,
                filter_groups=filter_groups,
                filter_logic=filter_logic,
                sample=True,
                limit=_requested_sample_size(text),
                offset=0,
                exhaustive=False,
                presentation="table" if len(fields) > 1 else "numbered_list",
                requested_fields=fields,
            )
        ],
    )


def _enforce_cause_summary(turn: TurnPlan, text: str) -> TurnPlan | None:
    """Preserve the compiler's scoped query while enforcing summary semantics.

    Dataset and filter selection deliberately remain untouched. Validation can
    therefore still reject any compiler attempt to access an unauthorized file.
    """

    if not _CAUSE_SUMMARY.search(text) or not turn.query_actions():
        return None
    return turn.model_copy(
        update={
            "final_response": "llm_synthesis",
            "needs_evidence": True,
            "needs_explanation": True,
        }
    )


def _unavailable_catalog_field(text: str, *, catalog: FieldCatalog | None) -> str | None:
    match = _CATALOG_FIELD_REQUEST.search(text)
    if match is None or catalog is None:
        return None
    requested = {"communities": "community", "schools": "school"}[match.group(1).lower()]
    if any(catalog.resolve_field(dataset.file_id, requested) for dataset in catalog.datasets):
        return None
    available = [
        spec.human_label
        for spec in catalog.fields_for(catalog.default_people_file_id)
    ]
    label = requested.replace("_", " ").capitalize()
    return (
        f"{label} is not an available field in the authorized research records. "
        f"Available Student master list fields are: {', '.join(available)}."
    )


def _requested_sample_size(text: str) -> int:
    number = re.search(r"\b([1-9]|[1-4]\d|50)\b", text)
    if number:
        return int(number.group(1))
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            return value
    return 5


def _reset_implicit_paging(turn: TurnPlan) -> TurnPlan:
    changed = False
    actions = []
    for action in turn.actions:
        if isinstance(action, ModifyPreviousAction) and action.page in {"next", "previous"}:
            action = action.model_copy(update={"page": "first", "offset": 0})
            changed = True
        elif isinstance(action, QueryAction) and (action.offset or 0) > 0:
            action = action.model_copy(update={"offset": 0})
            changed = True
        actions.append(action)
    return _replace(turn, actions) if changed else turn


def _bare_research_dump(turn: TurnPlan) -> bool:
    """True when the compiler reused an unfiltered list/count for a non-research turn.

    A grouped or analytic query legitimately carries no filters — "which full names
    appear more than once" and "what communities are listed" are whole-dataset
    computations, not conversational filler — so they are never a bare dump.
    """
    if turn.modify_actions() or turn.verify_actions() or turn.reference_actions():
        return False
    queries = turn.query_actions()
    if not queries:
        return False
    return all(
        not item.filters
        and not item.filter_groups
        and not item.denominator_filters
        and not item.search_text
        and not item.compare
        and not item.aggregate
        and not item.group_by
        and item.goal not in ANALYTIC_GOALS
        for item in queries
    )


def _offers_next_page(last_assistant_text: str) -> bool:
    last = (last_assistant_text or "").lower()
    return "say \"next\"" in last or "say next" in last or "showing " in last


def _extract_result_window(text: str) -> ResultWindow | None:
    match = _RESULT_WINDOW.search(text)
    if match is None:
        return None
    size = int(match.group("size"))
    if size < 1 or size > MAX_WINDOW_SIZE:
        return None
    anchor = "end" if match.group("anchor").lower() in {"last", "final"} else "start"
    return ResultWindow(anchor=anchor, size=size, cursor=0)


# NIA's own single-candidate name offer (name_correction.ask_text), matched on the lowercased
# last assistant message. A list of several candidates needs the user to say which.
_OFFERED_ONE_NAME = re.compile(
    r'^no record matches "(?P<searched>.+?)"\. did you mean (?!one of these)(?P<offered>.+)\?$',
    re.DOTALL,
)


def accepts_offered_name(text: str, last_assistant_text: str | None) -> bool:
    """Whether a reply takes up NIA's offer of the one close name.

    After "No record matches "Alexander knegs". Did you mean Alexander KNAGGS (Manitoba Vital
    Stats)?", "yes" answered "Sure.", and the same question asked again was planned afresh,
    read "Manitoba Vital Stats" as another list, and refused. Both mean the researcher wants
    that record. The offer is NIA's own sentence and is registered as a listed result, so
    either reply resolves like "the first one".
    """
    match = _OFFERED_ONE_NAME.search((last_assistant_text or "").strip().lower())
    if match is None:
        return False
    lowered = " ".join((text or "").casefold().split())
    searched = match.group("searched").strip()
    return is_bare_affirmation(lowered) or bool(searched and searched in lowered)


# "I didn't ask for students" names a field but is a complaint, not a query.
# These forms cannot be a research request, so they outrank the research
# heuristic that would otherwise run a brand new search for the named noun.
_EXPLICIT_COMPLAINT = re.compile(
    r"\bi (?:did not|didn't|didnt) ask\b|\bthat(?:'s| is) not what i (?:asked|wanted)\b",
    re.IGNORECASE,
)


def _apply_result_window(
    turn: TurnPlan,
    window: ResultWindow,
    *,
    has_active_query: bool,
) -> TurnPlan | None:
    page_size = min(window.size, 50)
    data_actions = [*turn.query_actions(), *turn.modify_actions()]
    if data_actions:
        actions = []
        for action in turn.actions:
            if isinstance(action, QueryAction):
                actions.append(
                    action.model_copy(
                        update={
                            "goal": "list",
                            "window": window,
                            "limit": page_size,
                            "offset": 0,
                            "exhaustive": True,
                            "presentation": action.presentation or "numbered_list",
                        }
                    )
                )
            elif isinstance(action, ModifyPreviousAction):
                actions.append(
                    action.model_copy(
                        update={
                            "goal": "list",
                            "window": window,
                            "limit": page_size,
                            "offset": 0,
                            "exhaustive": True,
                            "presentation": action.presentation or "numbered_list",
                        }
                    )
                )
            else:
                actions.append(action)
        return _replace(turn, actions)
    if has_active_query:
        return _replace(
            turn,
            [
                ModifyPreviousAction(
                    goal="list",
                    window=window,
                    limit=page_size,
                    offset=0,
                    exhaustive=True,
                    presentation="numbered_list",
                )
            ],
        )
    return None


def _extract_requested_fields(text: str) -> list[str]:
    match = _FIELD_PROJECTION.search(text)
    if match is None:
        return []
    raw = re.sub(r"^(?:the|a|an)\s+", "", match.group("fields").strip())
    parts = [
        re.sub(r"^(?:the|a|an)\s+", "", item.strip(" ,"))
        for item in re.split(r"\s*(?:,|\band\b)\s*", raw)
        if item.strip(" ,")
    ]
    if len(parts) > 1 and parts[0].lower().startswith("date of "):
        parts = [
            item if " " in item else f"date of {item}"
            for item in parts
        ]
    return list(dict.fromkeys(parts))


def _apply_requested_fields(
    turn: TurnPlan,
    requested_fields: list[str],
    *,
    has_active_query: bool,
    catalog: FieldCatalog,
) -> TurnPlan | None:
    data_actions = [*turn.query_actions(), *turn.modify_actions()]
    if data_actions:
        actions = []
        for action in turn.actions:
            if isinstance(action, QueryAction):
                actions.append(
                    action.model_copy(
                        update={
                            "goal": "list",
                            "requested_fields": requested_fields,
                            "presentation": "table",
                        }
                    )
                )
            elif isinstance(action, ModifyPreviousAction):
                actions.append(
                    action.model_copy(
                        update={
                            "goal": "list",
                            "requested_fields": requested_fields,
                            "presentation": "table",
                        }
                    )
                )
            else:
                actions.append(action)
        return _replace(turn, actions)
    if has_active_query:
        return _replace(
            turn,
            [
                ModifyPreviousAction(
                    goal="list",
                    requested_fields=requested_fields,
                    presentation="table",
                )
            ],
        )
    return None


def _all_public_fields(catalog: FieldCatalog) -> list[str]:
    if not catalog.datasets:
        return []
    # The turn service supplies a single-dataset catalog, so this expansion is
    # schema-driven and cannot pull columns from the other list.
    return [item.semantic_field for item in catalog.fields_for(catalog.datasets[0].file_id)]


def _respond(turn: TurnPlan, intent: str, response: str = "") -> TurnPlan:
    return turn.model_copy(
        update={
            "actions": [ConversationAction(intent=intent, response=response)],
            "final_response": "compiler_response",
            "unresolved": [],
            "confidence": 1.0,
            "needs_explanation": False,
            "needs_inference": False,
            "needs_evidence": False,
        }
    )


def _replace(turn: TurnPlan, actions) -> TurnPlan:
    return turn.model_copy(
        update={
            "actions": actions,
            "final_response": "deterministic",
            "unresolved": [],
            "confidence": max(turn.confidence, 0.95),
            "needs_explanation": False,
            "needs_inference": False,
        }
    )

