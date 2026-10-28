import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.plan_schema import PlanOp
from app.planning.plan_validator import PlanValidationError
from app.planning.query_plan_builder import build_query_plan
from app.planning.turn_schema import (
    AggregateSpec,
    CompareBranch,
    ConversationAction,
    FilterSpec,
    QueryAction,
    TurnPlan,
)
from app.planning.turn_validator import MIN_TURN_CONFIDENCE, validate_turn_plan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=FakeReasoningProvider()
    )


COUNT_S_PARAPHRASES = (
    "How many students start with S?",
    "how many of the kids got names beginning in s",
    "the S names, what's the total?",
    "could you give me the number where their name starts S",
    "uh so students — names with S up front — how many?",
    "Count students whose names start with S.",
)


def test_count_paraphrases_are_semantically_equivalent():
    service = _service()
    results = [asyncio.run(service.answer(_scope(), question)) for question in COUNT_S_PARAPHRASES]
    # Wording that the catalog fully accounts for stays on the free deterministic
    # path. Colloquial wording ("kids") introduces a word the catalog does not know,
    # so the coverage guard hands it to the compiler rather than let a fast path
    # answer a question it only partly parsed. Either way the answer is the same.
    for result, question in zip(results, COUNT_S_PARAPHRASES, strict=True):
        assert result.status == "answered", question
        assert result.reasoning_calls <= 1, question
        assert result.plan is not None, question
        assert result.plan.op_names()[:3] == ["USE_CURRENT_VERSION", "FILTER", "COUNT"], question
        filt = result.plan.step_map()["matches"].where[0]
        assert filt.field == "student_name", question
        assert filt.operator == "STARTS_WITH", question
        assert filt.value == "S", question
        assert result.facts[0]["value"] == 2, question


def test_identity_paraphrases():
    service = _service()
    questions = ("What's your name?", "what is ur name", "hey what do people call you")
    for question, calls in zip(questions, (0, 1, 1), strict=True):
        result = asyncio.run(service.answer(_scope(), question))
        assert result.status == "answered"
        assert result.reasoning_calls == calls
        assert result.plan is None
        assert "Nia" in result.answer


def test_audio_check_paraphrases():
    service = _service()
    for question in ("Can you hear me?", "you hearing me okay?", "am I coming through"):
        result = asyncio.run(service.answer(_scope(), question, mode="voice"))
        assert result.status == "answered"
        assert result.reasoning_calls == 1
        assert "hear" in result.answer.lower() or "received" in result.answer.lower()


def test_self_correction_and_year_followup():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "How many students were admitted before 1920?"))
    assert first.status == "answered"
    second = asyncio.run(
        service.answer(_scope(), "What about 1910?", conversation_id=first.conversation_id)
    )
    assert second.status == "answered"
    assert second.plan is not None
    pred = second.plan.step_map()["matches"].where[0]
    assert pred.field == "admitted_date"
    assert pred.value == 1910
    third = asyncio.run(
        service.answer(_scope(), "no, I meant after 1920", conversation_id=first.conversation_id)
    )
    assert third.plan is not None
    changed = third.plan.step_map()["matches"].where[0]
    assert changed.operator == "AFTER"
    assert changed.value == 1920


def test_old_students_clarifies():
    result = asyncio.run(_service().answer(_scope(), "Show me the old ones."))
    assert result.status == "clarification"
    assert result.plan is None
    assert "didn't catch" not in result.answer.lower()


def test_garden_river_notes_uses_synthesis():
    result = asyncio.run(
        _service().answer(_scope(), "Find Garden River students and summarize their notes.")
    )
    assert result.status == "answered"
    assert result.reasoning_calls == 2
    assert result.plan is not None
    assert "GET_EVIDENCE" in result.plan.op_names()


def test_count_and_why_uses_two_llm_calls():
    result = asyncio.run(
        _service().answer(_scope(), "How many were deceased and why might that matter?")
    )
    assert result.status == "answered"
    assert result.reasoning_calls == 2


def test_turn_validator_rejects_unauthorized_dataset():
    turn = TurnPlan(
        normalized_request="count private file",
        actions=[QueryAction(datasets=[93], goal="count")],
        final_response="deterministic",
        confidence=0.9,
    )
    try:
        validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    except PlanValidationError as exc:
        assert exc.code == "access_restricted"
    else:
        raise AssertionError("expected access_restricted")


def test_turn_validator_rejects_invented_field():
    turn = TurnPlan(
        normalized_request="count password",
        actions=[
            QueryAction(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="password_hash", operator="EQUALS", value="x")],
            )
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    try:
        validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    except PlanValidationError as exc:
        assert exc.code == "unknown_field"
    else:
        raise AssertionError("expected unknown_field")


def test_freeform_search_text_may_contain_sql_words():
    action = QueryAction(
        datasets=[49],
        goal="search",
        search_text="select committee",
    )
    turn = TurnPlan(
        normalized_request="Find references to the select committee.",
        actions=[action],
        final_response="deterministic",
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert validated.query_actions()[0].search_text == "select committee"
    plan = build_query_plan(validated, scope=_scope(), catalog=static_catalog())
    assert any(step.query == "select committee" for step in plan.steps)


def test_structural_field_names_still_reject_sql():
    try:
        FilterSpec(field="select users", operator="EQUALS", value="x")
    except Exception:
        return
    raise AssertionError("expected identifier rejection for SQL-like field")


def test_respond_requires_conversational_intent():
    try:
        TurnPlan(
            normalized_request="how many students",
            actions=[{"type": "respond", "response": "Most students came from Garden River."}],
            final_response="compiler_response",
            confidence=0.9,
        )
    except Exception:
        return
    raise AssertionError("expected schema rejection without conversational intent")


def test_research_looking_respond_uses_canned_greeting_not_the_claim():
    from app.planning.response_policy import canned_response

    turn = TurnPlan(
        normalized_request="hello",
        actions=[
            {
                "type": "respond",
                "intent": "greeting",
                "response": "Most students came from Garden River.",
            }
        ],
        final_response="compiler_response",
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    reply = canned_response(validated, mode="text")
    assert reply == "Hi! How can I help?"
    assert "Garden River" not in (reply or "")


def test_low_confidence_without_an_executable_action_clarifies():
    turn = TurnPlan(
        normalized_request="something",
        actions=[ConversationAction(intent="describe_available_data")],
        final_response="deterministic",
        confidence=0.01,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert validated.clarify_actions()
    assert validated.final_response == "compiler_response"
    assert validated.confidence < MIN_TURN_CONFIDENCE


def test_low_confidence_keeps_an_executable_query():
    """A hard question is answered, not deflected.

    Replacing every low-confidence turn with "could you say that another way"
    discarded working queries for exactly the questions the compiler finds hardest.
    """
    turn = TurnPlan(
        normalized_request="count students",
        actions=[QueryAction(datasets=[49], goal="count")],
        final_response="deterministic",
        confidence=0.01,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert not validated.clarify_actions()
    assert validated.query_actions()[0].goal == "count"


def test_unresolved_without_clarify_keeps_an_executable_query():
    turn = TurnPlan(
        normalized_request="the old ones",
        actions=[QueryAction(datasets=[49], goal="list")],
        final_response="deterministic",
        unresolved=["old"],
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert not validated.clarify_actions()
    assert validated.query_actions()


def test_operator_is_checked_on_every_selected_dataset():
    turn = TurnPlan(
        normalized_request="search names",
        actions=[
            QueryAction(
                datasets=[49, 91],
                goal="search",
                filters=[FilterSpec(field="student_name", operator="GREATER_THAN", value=10)],
            )
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    try:
        validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    except PlanValidationError as exc:
        assert exc.code == "operator"
    else:
        raise AssertionError("expected operator rejection on file 91")


def test_independent_query_actions_are_not_anded():
    turn = TurnPlan(
        normalized_request="count Garden River and list S names",
        actions=[
            QueryAction(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
            ),
            QueryAction(
                datasets=[49],
                goal="list",
                filters=[FilterSpec(field="student_name", operator="STARTS_WITH", value="S")],
            ),
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    plan = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    filters = [step for step in plan.steps if step.op is PlanOp.FILTER]
    assert len(filters) == 2
    fields = {step.where[0].field for step in filters}
    assert fields == {"community", "student_name"}
    assert "COUNT" in plan.op_names()
    assert "PROJECT" in plan.op_names()
    anded = any(
        {pred.field for pred in step.where} >= {"community", "student_name"} for step in filters
    )
    assert anded is False


def test_compare_emits_independent_branches_and_compare_op():
    turn = TurnPlan(
        normalized_request="compare Garden River and Sault Ste. Marie",
        actions=[
            QueryAction(
                datasets=[49],
                goal="compare",
                compare=[
                    CompareBranch(
                        label="garden",
                        filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
                    ),
                    CompareBranch(
                        label="sault",
                        filters=[FilterSpec(field="community", operator="EQUALS", value="Sault Ste. Marie")],
                    ),
                ],
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    plan = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    assert plan.op_names().count("FILTER") == 2
    assert plan.op_names().count("COUNT") == 2
    assert "COMPARE" in plan.op_names()
    assert "PROJECT" not in plan.op_names()


def test_aggregate_emits_requested_function():
    turn = TurnPlan(
        normalized_request="earliest admission",
        actions=[
            QueryAction(
                datasets=[49],
                goal="aggregate",
                aggregate=AggregateSpec(function="min", field="admitted_date"),
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    plan = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    assert "MIN" in plan.op_names()
    assert "COUNT" not in plan.op_names()


def test_quote_and_provenance_emit_dedicated_ops():
    quote = build_query_plan(
        TurnPlan(
            normalized_request="quote the notes",
            actions=[QueryAction(datasets=[49], goal="quote", search_text="illness")],
            final_response="deterministic",
            confidence=0.95,
        ),
        scope=_scope(),
        catalog=static_catalog(),
    )
    assert "GET_QUOTE" in quote.op_names()
    provenance = build_query_plan(
        TurnPlan(
            normalized_request="where did this come from",
            actions=[QueryAction(datasets=[49], goal="provenance")],
            final_response="deterministic",
            confidence=0.95,
        ),
        scope=_scope(),
        catalog=static_catalog(),
    )
    assert "GET_PROVENANCE" in provenance.op_names()


def test_list_group_by_is_presentation_sort_not_sql_group_by():
    plan = build_query_plan(
        TurnPlan(
            normalized_request="list students by community",
            actions=[QueryAction(datasets=[49], goal="list", group_by=["community"])],
            final_response="deterministic",
            confidence=0.95,
        ),
        scope=_scope(),
        catalog=static_catalog(),
    )
    assert "GROUP_BY" not in plan.op_names()
    assert "SORT" in plan.op_names()
    assert "community" in plan.step_map()["result"].fields


def test_followup_without_active_query_becomes_a_first_class_query():
    """A refinement with nothing to refine still carries a usable question.

    Failing the turn discarded conditions the compiler had already understood; the
    conditions now stand on their own as a new query.
    """
    turn = TurnPlan(
        normalized_request="what about 1910",
        actions=[{"type": "modify_previous", "changes": [{"field": "admitted_date", "operator": "YEAR_EQUALS", "value": 1910}]}],
        final_response="deterministic",
        confidence=0.9,
    )
    validated = validate_turn_plan(
        turn, scope=_scope(), catalog=static_catalog(), has_active_query=False
    )
    assert not validated.modify_actions()
    query = validated.query_actions()[0]
    assert [(item.field, item.operator) for item in query.filters] == [
        ("admitted_date", "YEAR_EQUALS")
    ]


def test_followup_with_nothing_to_carry_asks_for_the_records():
    turn = TurnPlan(
        normalized_request="show me those",
        actions=[{"type": "reference", "selector": "first"}],
        final_response="deterministic",
        confidence=0.9,
    )
    validated = validate_turn_plan(
        turn, scope=_scope(), catalog=static_catalog(), has_active_query=False
    )
    assert validated.clarify_actions()


def test_spoken_multi_action_does_not_and_filters():
    result = asyncio.run(
        _service().answer(
            _scope(),
            "Count Garden River students and list students whose names start with S.",
        )
    )
    assert result.status == "answered"
    assert result.plan is not None
    filters = [step for step in result.plan.steps if step.op.value == "FILTER"]
    assert len(filters) == 2
    assert {step.where[0].field for step in filters} == {"community", "student_name"}
    assert "COUNT" in result.plan.op_names()
    assert "PROJECT" in result.plan.op_names()
    assert "4" in result.answer
    assert "Samuel A" in result.answer
    assert "Sarah B" in result.answer
    assert "from Garden River whose names start with S" not in result.answer
    assert "Garden River: 4" in result.answer
    assert "start with S" in result.answer


def test_multi_action_memory_keeps_separate_frames():
    from app.data_gateway.memory import MemoryDataGateway
    from app.execution.turn import AssistantTurnService
    from app.llm.fake import FakeReasoningProvider
    from app.memory.store import InMemoryMemoryStore

    store = InMemoryMemoryStore()
    service = AssistantTurnService(MemoryDataGateway(), store=store, reasoner=FakeReasoningProvider())
    result = asyncio.run(
        service.answer(
            _scope(),
            "Count Garden River students and list students whose names start with S.",
        )
    )
    record = asyncio.run(store.get(_scope(), result.conversation_id or ""))
    assert record is not None
    assert record.active_frame_key == "master"
    assert "master-a0" in record.frames
    assert "master-a1" in record.frames
    count_filters = {item.field for item in record.frames["master-a0"].query.filters}
    list_filters = {item.field for item in record.frames["master-a1"].query.filters}
    assert count_filters == {"community"}
    assert list_filters == {"student_name"}
    assert record.frames["master"].query.filters == record.frames["master-a1"].query.filters
    assert record.last_action_keys == ["master-a0", "master-a1"]


def test_synthesis_packet_keeps_action_branches():
    from app.data_gateway.memory import MemoryDataGateway
    from app.execution.executor import QueryExecutor
    from app.execution.turn import _packet_for_synthesis

    turn = TurnPlan(
        normalized_request="count Garden River and list S names",
        actions=[
            QueryAction(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")],
            ),
            QueryAction(
                datasets=[49],
                goal="list",
                filters=[FilterSpec(field="student_name", operator="STARTS_WITH", value="S")],
            ),
        ],
        final_response="llm_synthesis",
        needs_explanation=True,
        confidence=0.95,
    )
    plan = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    execution = asyncio.run(QueryExecutor(MemoryDataGateway(), static_catalog()).execute(plan, _scope()))
    packet = _packet_for_synthesis("explain both", execution)
    assert len(packet.branches) == 2
    assert packet.branches[0].action_id == "a0"
    assert packet.branches[0].goal == "count"
    assert any("a0.count=" in fact for fact in packet.branches[0].facts)
    assert packet.branches[1].action_id == "a1"
    assert packet.branches[1].goal == "list"
    assert any("community" in fact for fact in packet.branches[0].facts)
    assert not any("student_name" in fact and "community" in fact for fact in packet.facts)


def test_repeat_aggregate_uses_query_plan_builder():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "What is the earliest admission date?"))
    assert first.status == "answered"
    assert first.plan is not None
    assert "MIN" in first.plan.op_names()
    second = asyncio.run(
        service.answer(_scope(), "repeat the aggregate action", conversation_id=first.conversation_id)
    )
    assert second.status == "answered"
    assert second.plan is not None
    assert "MIN" in second.plan.op_names()
    assert "COUNT" not in second.plan.op_names()
    assert second.plan.planner_type == "semantic_compiler"


def test_repeat_comparison_uses_query_plan_builder():
    service = _service()
    first = asyncio.run(
        service.answer(_scope(), "Compare Garden River and Sault Ste. Marie students.")
    )
    assert first.plan is not None
    assert "COMPARE" in first.plan.op_names()
    second = asyncio.run(
        service.answer(_scope(), "repeat the comparison", conversation_id=first.conversation_id)
    )
    assert second.status == "answered"
    assert second.plan is not None
    assert "COMPARE" in second.plan.op_names()
    assert second.plan.op_names().count("FILTER") == 2


def test_quote_the_second_request_again():
    service = _service()
    first = asyncio.run(
        service.answer(_scope(), "Count Garden River students and quote notes about illness.")
    )
    assert first.plan is not None
    assert "GET_QUOTE" in first.plan.op_names()
    second = asyncio.run(
        service.answer(_scope(), "quote the second request again", conversation_id=first.conversation_id)
    )
    assert second.status == "answered"
    assert second.plan is not None
    assert "GET_QUOTE" in second.plan.op_names()
    assert "COUNT" not in second.plan.op_names()


def test_repeat_provenance_for_first_request():
    service = _service()
    first = asyncio.run(service.answer(_scope(), "Show provenance for students starting with S."))
    assert first.plan is not None
    assert "GET_PROVENANCE" in first.plan.op_names()
    second = asyncio.run(
        service.answer(
            _scope(),
            "show the provenance for the first request",
            conversation_id=first.conversation_id,
        )
    )
    assert second.status == "answered"
    assert second.plan is not None
    assert "GET_PROVENANCE" in second.plan.op_names()


def test_stored_source_action_round_trips_compare():
    from app.memory.types import QueryState
    from app.planning.query_plan_builder import query_action_from_state
    from app.planning.turn_schema import CompareBranch

    source = QueryAction(
        datasets=[49],
        goal="compare",
        compare=[
            CompareBranch(
                filters=[FilterSpec(field="community", operator="EQUALS", value="Garden River")]
            ),
            CompareBranch(
                filters=[FilterSpec(field="community", operator="EQUALS", value="Sault Ste. Marie")]
            ),
        ],
    )
    state = QueryState(file_ids=(49,), goal="compare", source_action=source)
    replayed = query_action_from_state(state)
    plan = build_query_plan(
        TurnPlan(
            normalized_request="repeat",
            actions=[replayed],
            final_response="deterministic",
            confidence=1.0,
        ),
        scope=_scope(),
        catalog=static_catalog(),
    )
    assert "COMPARE" in plan.op_names()


def test_query_plan_builder_is_deterministic():
    turn = TurnPlan(
        normalized_request="count names starting with S",
        actions=[
            QueryAction(
                datasets=[49],
                goal="count",
                filters=[FilterSpec(field="student_name", operator="STARTS_WITH", value="S")],
            )
        ],
        final_response="deterministic",
        confidence=0.99,
    )
    first = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    second = build_query_plan(turn, scope=_scope(), catalog=static_catalog())
    assert first.model_dump() == second.model_dump()
    assert first.op_names() == ["USE_CURRENT_VERSION", "FILTER", "COUNT"]
    assert first.planner_type == "semantic_compiler"


def test_acknowledgement_ignores_model_prose():
    from app.planning.response_policy import acknowledgement, canned_response

    turn = TurnPlan(
        normalized_request="yeah, list the S names",
        actions=[
            {
                "type": "respond",
                "intent": "acknowledgement",
                "response": "Most students came from Garden River.",
            }
        ],
        final_response="compiler_response",
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    assert acknowledgement(validated) == "Sure."
    assert canned_response(validated, mode="text") == "Sure."
    assert "Garden River" not in (acknowledgement(validated) or "")


def test_sum_on_student_name_is_rejected():
    turn = TurnPlan(
        normalized_request="sum names",
        actions=[
            QueryAction(
                datasets=[49],
                goal="aggregate",
                aggregate=AggregateSpec(function="sum", field="student_name"),
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    try:
        validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    except PlanValidationError as exc:
        assert exc.code == "operator"
    else:
        raise AssertionError("expected operator rejection for SUM(student_name)")


def test_min_on_community_and_deceased_is_rejected():
    for field in ("community", "deceased_status"):
        turn = TurnPlan(
            normalized_request=f"min {field}",
            actions=[
                QueryAction(
                    datasets=[49],
                    goal="aggregate",
                    aggregate=AggregateSpec(function="min", field=field),
                )
            ],
            final_response="deterministic",
            confidence=0.95,
        )
        try:
            validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
        except PlanValidationError as exc:
            assert exc.code == "operator", field
        else:
            raise AssertionError(f"expected operator rejection for MIN({field})")


def test_min_on_admitted_date_is_allowed():
    turn = TurnPlan(
        normalized_request="earliest admission",
        actions=[
            QueryAction(
                datasets=[49],
                goal="aggregate",
                aggregate=AggregateSpec(function="min", field="admitted_date"),
            )
        ],
        final_response="deterministic",
        confidence=0.95,
    )
    validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())


def test_general_conversation_allows_non_research_prose():
    turn = TurnPlan(
        normalized_request="Explain what SQL DELETE means.",
        actions=[
            {
                "type": "respond",
                "intent": "general_conversation",
                "response": "DELETE removes rows from a table.",
            }
        ],
        final_response="compiler_response",
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    from app.planning.response_policy import canned_response

    assert canned_response(validated, mode="text") == "DELETE removes rows from a table."


def test_general_conversation_beside_a_query_is_dropped():
    turn = TurnPlan(
        normalized_request="explain and count",
        actions=[
            {
                "type": "respond",
                "intent": "general_conversation",
                "response": "DELETE removes rows from a table.",
            },
            QueryAction(datasets=[49], goal="count"),
        ],
        final_response="deterministic",
        confidence=0.9,
    )
    validated = validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())

    # The research action is the answer; the chat aside is dropped rather than
    # failing the whole turn.
    assert not validated.respond_actions()
    assert validated.query_actions()


def test_general_conversation_rejects_research_claim():
    turn = TurnPlan(
        normalized_request="explain the records",
        actions=[
            {
                "type": "respond",
                "intent": "general_conversation",
                "response": "Most students came from Garden River.",
            }
        ],
        final_response="compiler_response",
        confidence=0.9,
    )
    try:
        validate_turn_plan(turn, scope=_scope(), catalog=static_catalog())
    except PlanValidationError as exc:
        assert exc.code == "research_fact_in_respond"
    else:
        raise AssertionError("expected research_fact_in_respond")
