from __future__ import annotations

import json
from typing import Protocol

from app.planning.canonical_query import CanonicalQuery
from app.planning.catalog import FieldCatalog
from app.security.access_scope import AccessScope

SEMANTIC_NORMALIZER_SYSTEM = """You are the Turn Interpreter for NIA, a research-records assistant.

Your job is NOT to query a database and NOT to write SQL. Translate one user utterance into the supplied CanonicalTurn / CanonicalQuery schema.

Rules:
- Preserve the user's meaning, including negation, corrections, comparisons, and uncertainty.
- Do not invent a dataset, field, filter, value, or operator.
- Use only datasets, semantic fields, and operators shown in the authorized catalog.
- Unauthorized datasets do not exist from your point of view.
- If the utterance is ordinary conversation (identity, hearing check, small talk, thanks, farewell, capabilities), set turn_type=conversation. Set intent to one of: assistant_identity, audio_check, greeting, thanks, farewell, capability. Set answer_directly=true. Put a short reply in direct_answer. Do not create a research query.
- If the user asks what records/datasets exist or what is in the research records, set turn_type=research, goal=describe_available_data, and file_ids to the authorized catalog ids. Do not invent row values.
- If the request is research that this small schema can represent, set turn_type=research (or follow_up), set goal to count/list/distinct/synthesis, and fill file_ids, filters, group_by, and search_text only from the catalog.
- If the request is ambiguous, set turn_type=clarification, list unresolved concepts, and provide one concise clarification_question. Never pretend you failed to hear the user.
- If the request is research but cannot be faithfully represented by this schema, set requires_full_planner=true. Do not squeeze it into a simpler query.
- canonical_text is a concise normalized statement of the intended request, not an answer.
- Never put database facts in direct_answer or response_text.
- Confidence reflects certainty about the interpretation, not confidence in an answer.
"""


class CanonicalStructuredCall(Protocol):
    async def __call__(
        self,
        *,
        system: str,
        user: str,
        response_model: type[CanonicalQuery],
    ) -> CanonicalQuery: ...


class SemanticNormalizerProtocol(Protocol):
    async def normalize(
        self,
        question: str,
        *,
        normalized_text: str,
        scope: AccessScope,
        catalog: FieldCatalog,
        memory_text: str = "",
    ) -> CanonicalQuery: ...


class LLMSemanticNormalizer:
    """One structured LLM call: natural utterance -> CanonicalQuery.

    ``structured_call`` is intentionally injected rather than hard-wiring an SDK.
    An XAIReasoningProvider adapter can pass its generic structured-output method.
    This keeps planning independent from provider implementation details.
    """

    def __init__(self, structured_call: CanonicalStructuredCall) -> None:
        self._structured_call = structured_call

    async def normalize(
        self,
        question: str,
        *,
        normalized_text: str,
        scope: AccessScope,
        catalog: FieldCatalog,
        memory_text: str = "",
    ) -> CanonicalQuery:
        scoped = catalog.for_scope(scope)
        user = semantic_normalizer_user_prompt(
            question=question,
            normalized_text=normalized_text,
            catalog=scoped,
            memory_text=memory_text,
        )
        result = await self._structured_call(
            system=SEMANTIC_NORMALIZER_SYSTEM,
            user=user,
            response_model=CanonicalQuery,
        )
        if isinstance(result, CanonicalQuery):
            return result
        return CanonicalQuery.model_validate(result)


def semantic_normalizer_user_prompt(
    *,
    question: str,
    normalized_text: str,
    catalog: FieldCatalog,
    memory_text: str = "",
) -> str:
    payload = {
        "user_utterance": question,
        "normalized_input": normalized_text,
        "conversation_memory": memory_text or "",
        "authorized_catalog": _catalog_payload(catalog),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _catalog_payload(catalog: FieldCatalog) -> dict:
    datasets = []
    for dataset in catalog.datasets:
        datasets.append(
            {
                "file_id": dataset.file_id,
                "label": dataset.user_facing_label,
                "aliases": list(dataset.aliases),
                "fields": [
                    {
                        "name": field.semantic_field,
                        "label": field.human_label,
                        "aliases": list(field.aliases),
                        "type": field.semantic_type,
                        "operators": list(field.allowed_operators),
                        "searchable": field.searchable,
                        "sortable": field.sortable,
                        "evidence_allowed": field.evidence_allowed,
                    }
                    for field in catalog.fields_for(dataset.file_id)
                ],
            }
        )
    return {"datasets": datasets, "default_people_file_id": catalog.default_people_file_id}
