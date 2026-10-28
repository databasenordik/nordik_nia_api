import ast
import asyncio
import sys

from repo_paths import BACKEND_ROOT, monorepo_file  # noqa: E402

sys.path.insert(0, str(BACKEND_ROOT))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.canonical_query import CanonicalQuery, CanonicalTurn, TurnType
from app.planning.catalog import static_catalog
from app.planning.router import plan_user_turn
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service(reasoner: FakeReasoningProvider | None = False) -> tuple[AssistantTurnService, FakeReasoningProvider | None]:
    provider = None if reasoner is None else (reasoner or FakeReasoningProvider())
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore(), reasoner=provider)
    return service, provider


def _layer(result, name: str) -> dict:
    layers = result.trace
    assert isinstance(layers, list), layers
    for item in layers:
        if item.get("layer") == name:
            return item
    raise AssertionError(f"missing layer {name}: {layers}")


def test_hello_uses_fast_path_and_skips_query_plan():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "Hello."))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert provider.normalize_calls == 0
    assert provider.plan_calls == 0
    assert "Hi" in result.answer or "Hello" in result.answer
    assert "didn't catch" not in result.answer.lower()
    assert _layer(result, "query_plan")["status"] == "SKIPPED"


def test_thanks_and_bye_use_compiler_response():
    service, provider = _service()
    thanks = asyncio.run(service.answer(_scope(), "thanks"))
    bye = asyncio.run(service.answer(_scope(), "bye"))
    assert thanks.status == "answered"
    assert thanks.answer == "You're welcome."
    assert bye.status == "answered"
    assert bye.answer == "Goodbye."
    assert thanks.reasoning_calls == 0
    assert bye.reasoning_calls == 0
    assert provider.normalize_calls == 0


def test_whats_your_name_uses_trusted_fast_path():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "What's your name?"))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert provider.normalize_calls == 0
    assert provider.plan_calls == 0
    assert result.planner_type == "trusted_fast_path"
    assert result.answer == "My name is Nia."
    assert "didn't catch" not in result.answer.lower()
    assert _layer(result, "deterministic_fast_path")["status"] == "matched"
    assert _layer(result, "semantic_compiler")["status"] == "skipped"
    assert _layer(result, "query_plan")["status"] == "SKIPPED"


def test_voice_identity_and_audio_check_are_program_replies():
    service, provider = _service()
    name = asyncio.run(service.answer(_scope(), "What's your name?", mode="voice"))
    hear = asyncio.run(service.answer(_scope(), "Can you hear me?", mode="voice"))
    assert name.answer == "I'm Nia."
    assert hear.answer == "Yes, I can hear you."
    assert name.reasoning_calls == 0
    assert hear.reasoning_calls == 1
    assert _layer(hear, "semantic_compiler")["output"]["actions"][0]["type"] == "respond"
    assert provider.normalize_calls == 1
    assert "didn't catch" not in hear.answer.lower()


def test_research_records_question_describes_authorized_catalog():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "What are in the research records?"))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert provider.normalize_calls == 0
    assert result.planner_type == "trusted_fast_path"
    assert "Student master list" in result.answer
    assert "Confirmed deaths" not in result.answer
    assert "Private Name" not in result.answer
    assert "Additional deaths" not in result.answer
    assert result.plan is None


def test_count_students_uses_deterministic_fast_path():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "Count students whose names start with S."))
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert result.planner_type == "deterministic_fast_path"
    assert provider.normalize_calls == 0
    assert provider.synthesis_calls == 0
    assert result.facts[0]["value"] == 2


def test_group_by_community_uses_interpreter_then_plan():
    service, provider = _service()
    result = asyncio.run(
        service.answer(
            _scope(),
            "Can you show me the students from each community and organize them by where they're from?",
        )
    )
    assert result.status == "answered"
    assert result.reasoning_calls == 1
    assert provider.normalize_calls == 1
    assert provider.plan_calls == 0
    assert result.planner_type == "semantic_compiler"
    assert result.plan is not None
    assert "GROUP_BY" not in result.plan.op_names()
    assert "SORT" in result.plan.op_names()
    assert "community" in result.plan.step_map()["result"].fields
    assert _layer(result, "semantic_compiler")["output"]["actions"][0]["type"] == "query"


def test_empty_transcript_is_unintelligible_not_a_language_failure():
    service, _provider = _service()
    result = asyncio.run(service.answer(_scope(), "   ", mode="voice"))
    assert result.status == "unintelligible"
    assert result.reasoning_calls == 0
    assert result.answer == "Sorry, I didn't catch that."


def test_valid_text_without_llm_does_not_claim_stt_failure():
    service = AssistantTurnService(MemoryDataGateway(), store=InMemoryMemoryStore())
    result = asyncio.run(service.answer(_scope(), "What's your name?"))
    assert result.reasoning_calls == 0
    assert result.status != "unintelligible"
    assert "didn't catch" not in result.answer.lower()
    assert result.planner_type == "trusted_fast_path"
    interp = _layer(result, "semantic_compiler")
    assert interp["status"] == "skipped"
    assert interp["reasoning_calls"] == 0
    assert "semantic_fallback" not in str(result.trace).lower()


def test_clarification_asks_a_real_question():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "This is unclear and ambiguous."))
    assert result.status == "clarification"
    assert result.reasoning_calls == 1
    assert provider.normalize_calls == 1
    assert "count" in result.answer.lower() or "list" in result.answer.lower()
    assert "didn't catch" not in result.answer.lower()


def test_canonical_turn_accepts_type_and_datasets_aliases():
    turn = CanonicalTurn.model_validate(
        {
            "type": "conversation",
            "intent": "assistant_identity",
            "answer_directly": True,
            "direct_answer": "My name is NIA.",
            "datasets": [49],
        }
    )
    assert turn is not None
    assert turn.turn_type is TurnType.CONVERSATION
    assert turn.file_ids == (49,)
    assert CanonicalTurn is CanonicalQuery


def test_name_question_uses_trusted_fast_path():
    route = asyncio.run(plan_user_turn("What's your name?", _scope(), static_catalog()))
    assert route.source == "trusted_fast_path"
    assert route.reasoning_calls == 0
    assert route.status == "conversational"
    assert route.compiler is None


def test_what_is_your_name_is_the_same_fast_path():
    service, provider = _service()
    result = asyncio.run(service.answer(_scope(), "What is your name?"))
    assert result.status == "answered"
    assert result.answer == "My name is Nia."
    assert provider.normalize_calls == 0
    assert result.planner_type == "trusted_fast_path"
    assert _layer(result, "semantic_compiler")["status"] == "skipped"


def test_backend_compose_does_not_blank_xai_key():
    text = monorepo_file("infra", "docker-compose.yml").read_text(
        encoding="utf-8"
    )
    backend = text.split("backend:")[1].split("livekit:")[0]
    assert "../.env" in backend
    assert "XAI_API_KEY: ${XAI_API_KEY:-}" not in backend


def test_voice_cli_silence_default_is_1400():
    source = monorepo_file("voice_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name != "add_argument":
                continue
            flags = [item.value for item in node.args if isinstance(item, ast.Constant)]
            if "--silence-ms" not in flags:
                continue
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    defaults.append(int(keyword.value.value))
    assert defaults == [1400]
    assert "silence_ms: int = 1400" in source
