import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.data_gateway.memory import MemoryDataGateway
from app.execution.turn import AssistantTurnService
from app.llm.fake import FakeReasoningProvider
from app.llm.schemas import PlannedQuery, PlannedStep
from app.memory.store import InMemoryMemoryStore
from app.planning.catalog import static_catalog
from app.planning.deterministic_parser import parse_deterministic
from app.planning.llm_planner import planned_query_to_plan
from app.planning.plan_validator import PlanValidationError
from app.security.access_scope import AccessScope


def _scope() -> AccessScope:
    return AccessScope(principal_id="principal-researcher", allowed_file_ids=(49, 91))


def test_scoped_catalog_hides_private_datasets():
    catalog = static_catalog().for_scope(_scope())
    assert [item.file_id for item in catalog.datasets] == [49, 91]
    assert catalog.resolve_dataset_ids("additional deaths") == []
    assert catalog.resolve_field(93, "student_name") is None


def test_all_datasets_beats_students_alias():
    parsed = parse_deterministic(
        "Across all datasets, how many students are there?",
        _scope(),
        static_catalog(),
    )
    assert parsed.plan is not None
    assert set(parsed.plan.scope.file_ids) == {49, 91}


def test_died_before_year_is_unplanned_without_death_date():
    parsed = parse_deterministic(
        "How many students died before 1900?",
        _scope(),
        static_catalog(),
    )
    assert parsed.plan is None
    # The deterministic tier abstains rather than guessing the wrong date field.
    # Abstention is an internal handoff to semantic planning, not a user error.
    assert parsed.status == "needs_planning"
    assert "death_date" in (parsed.detail or "")


def test_llm_plan_does_not_fallback_to_first_authorized_dataset():
    drafted = PlannedQuery(
        file_ids=[93],
        goals=["exact_count"],
        steps=[PlannedStep(id="result", op="COUNT", input="current_records")],
    )
    with pytest.raises(PlanValidationError) as exc:
        planned_query_to_plan(drafted, _scope())
    assert exc.value.code == "access_restricted"


def test_retrieval_plan_declares_union_and_dedupe():
    parsed = parse_deterministic(
        "Which records mention tuberculosis?",
        _scope(),
        static_catalog(),
    )
    assert parsed.plan is not None
    assert "UNION" in parsed.plan.op_names()
    assert "DEDUPLICATE" in parsed.plan.op_names()
    assert parsed.plan.step_map()["unique"].input == "fused"
    assert parsed.plan.step_map()["fused"].input == ["fts", "fuzzy"]


def test_inspected_registry_paths():
    catalog = static_catalog()
    name = catalog.resolve_field(49, "student_name")
    deceased = catalog.resolve_field(49, "deceased_status")
    notes = catalog.resolve_field(49, "notes")
    assert name is not None and name.canonical_json_path == "canonical.display_name"
    assert deceased is not None and deceased.canonical_json_path == "canonical.deceased_status"
    assert notes is not None and notes.canonical_json_path == "chat.narrative_bundle.notes"
    assert catalog.resolve_field(49, "school") is None
    assert catalog.resolve_field(49, "cause_of_death") is None


@pytest.mark.asyncio
async def test_unauthorized_llm_request_returns_access_restricted():
    class _Poison(FakeReasoningProvider):
        async def structured_output(self, *, system: str, user: str, response_model):
            self.normalize_calls += 1
            return response_model.model_validate(
                {
                    "normalized_request": "unauthorized dataset",
                    "actions": [
                        {
                            "type": "query",
                            "datasets": [93],
                            "goal": "search",
                        }
                    ],
                    "final_response": "llm_synthesis",
                    "needs_evidence": True,
                    "confidence": 0.95,
                }
            )

        async def plan_structured(self, *, system: str, user: str):
            self.plan_calls += 1
            return PlannedQuery(
                file_ids=[93],
                goals=["exact_count"],
                steps=[PlannedStep(id="result", op="COUNT", input="current_records")],
            )

    service = AssistantTurnService(
        MemoryDataGateway(),
        store=InMemoryMemoryStore(),
        reasoner=_Poison(),
    )
    result = await service.answer(
        _scope(),
        "Summarize what the records say about causes of death.",
        selected_file_id=91,
    )
    assert result.status == "access_restricted"
    assert result.plan is None
