import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.data_gateway.repository import _gateway_row
from app.execution.row_projection import semantic_value
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import DatasetSpec, FieldCatalog, static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.planning.response_policy import decide_final_response
from app.planning.turn_schema import QueryAction, TurnPlan
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def _service() -> AssistantTurnService:
    return AssistantTurnService(
        MemoryDataGateway(),
        store=InMemoryMemoryStore(),
        reasoner=FakeReasoningProvider(),
    )


def _ask(question: str, *, selected_file_id: int = 49):
    return asyncio.run(
        _service().answer(_scope(), question, selected_file_id=selected_file_id)
    )


def test_gateway_decodes_normalized_json_and_projects_typed_dates():
    payload = {
        "file_id": 49,
        "source_row_id": 1,
        "canonical_name": "Example Student",
        "row_data_normalized": json.dumps(
            {
                "canonical": {
                    "dates": {
                        "admitted": {
                            "iso": "1950-07-09",
                            "raw": "1950-07-09",
                            "kind": "exact_date",
                            "year": 1950,
                        }
                    },
                    "deceased_status": "yes",
                }
            }
        ),
    }

    decoded = _gateway_row(payload)

    assert semantic_value(decoded, "admitted_date", static_catalog()) == "1950-07-09"
    assert semantic_value(decoded, "deceased_status", static_catalog()) == "yes"


def test_natural_available_records_variant_uses_catalog():
    result = _ask("What are in the research records?")
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert result.planner_type == "trusted_fast_path"
    assert "Student master list" in result.answer
    assert "Confirmed deaths" not in result.answer


def test_community_list_and_count_use_exact_distinct_values():
    listed = _ask("List all communities")
    counted = _ask("How many communities do you have?")

    assert listed.reasoning_calls == 0
    assert "2 communities" in listed.answer
    assert "Garden River" in listed.answer
    assert "Sault Ste. Marie" in listed.answer
    assert counted.reasoning_calls == 0
    assert "2 communities" in counted.answer


def test_counts_each_authorized_dataset_separately():
    """A question about every list is run once per list and the answers merged.

    Nia still reads one dataset per execution; the branches never join.
    """
    result = _ask("How many records are in each authorized dataset?")
    assert result.status == "answered"
    assert "Student master list" in result.answer
    assert "Confirmed deaths" in result.answer


def test_database_catalog_label_supports_master_list_shorthand_without_aliases():
    catalog = FieldCatalog(
        datasets=(
            DatasetSpec(
                file_id=49,
                user_facing_label="Student master list",
                is_default_people_scope=True,
            ),
        ),
        default_people_file_id=49,
    )

    parsed = parse_deterministic(
        "How many students are in the master list?",
        AccessScope(principal_id="principal-researcher", allowed_file_ids=(49,)),
        catalog,
    )

    assert parsed.plan is not None
    assert parsed.plan.op_names() == ["USE_CURRENT_VERSION", "COUNT"]


def test_cause_summary_is_synthesized_and_evidence_backed():
    result = _ask(
        "Summarize the causes of death in the authorized records.",
        selected_file_id=91,
    )
    assert result.status == "answered"
    assert result.reasoning_calls == 2
    assert "cause" in result.answer.lower() or "Sarah B" in result.answer
    assert result.citations


def test_explicit_summary_intent_wins_over_list_retrieval_shape():
    turn = TurnPlan(
        normalized_request="summarize causes of death",
        actions=[QueryAction(datasets=[91], goal="list")],
        final_response="llm_synthesis",
        needs_evidence=True,
        needs_explanation=True,
        confidence=0.9,
    )

    assert decide_final_response(turn) == "llm_synthesis"


def test_named_record_followup_stays_in_authorized_records_and_projects_fields():
    result = _ask("Tell me more about Sarah B.")
    assert result.status == "answered"
    assert result.reasoning_calls == 0
    assert "Sarah B" in result.answer
    assert "Garden River" in result.answer
    assert "Deceased status: Yes" in result.answer
    assert "Twitter" not in result.answer


def test_unavailable_school_field_is_explained_in_research_context():
    result = _ask("What schools are there?")
    assert result.reasoning_calls == 0
    assert "not an available field" in result.answer
    assert "Student master list fields" in result.answer


def test_singing_request_gets_a_direct_capability_answer():
    result = _ask("Can you sing for me?")
    assert result.reasoning_calls == 0
    assert result.answer == "I can't sing, but I can speak answers in a voice session."
