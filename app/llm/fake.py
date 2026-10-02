from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app.llm.base import StructuredOutputResult
from app.llm.schemas import (
    MemoryUpdate,
    PlannedPredicate,
    PlannedQuery,
    PlannedStep,
    SynthesisAnswer,
)


class FakeReasoningProvider:
    """Deterministic stand-in so tests never call a paid API."""

    def __init__(self) -> None:
        self.plan_calls = 0
        self.synthesis_calls = 0
        self.stream_calls = 0
        self.normalize_calls = 0
        self.structured_calls = 0
        self.provider_name = "fake"
        self.model_name = "fake"
        self.structured_queue: list[Any] = []
        self.structured_log: list[dict[str, Any]] = []

    async def stream_answer(self, *, system: str, user: str) -> AsyncIterator[str]:
        self.stream_calls += 1
        if "Do not search the internet" in system:
            request = user.rsplit("User request:", 1)[-1].strip() or user
            yield f"General answer based on your request: {request[:160]}"
            return
        answer = await self.answer_structured(system=system, user=user)
        text = answer.answer
        split_at = text.find(". ")
        if split_at > 0:
            yield text[: split_at + 1]
            yield text[split_at + 1 :]
        else:
            yield text

    async def structured_output(self, *, system: str, user: str, response_model):
        result = await self.structured_output_with_metadata(
            system=system, user=user, response_model=response_model
        )
        return result.value

    async def structured_output_with_metadata(
        self, *, system: str, user: str, response_model
    ) -> StructuredOutputResult:
        """Test double for the semantic compiler and AI planner."""
        started = time.perf_counter()
        self.normalize_calls += 1
        self.structured_calls += 1
        self.structured_log.append(
            {
                "system": system[:80],
                "user": user[:240],
                "model": getattr(response_model, "__name__", ""),
            }
        )
        if self.structured_queue:
            item = self.structured_queue.pop(0)
            if isinstance(item, Exception):
                raise item
            if isinstance(item, response_model):
                value = item
            elif hasattr(item, "root") and isinstance(item, response_model):
                value = item
            else:
                value = response_model.model_validate(item)
        else:
            value = _default_structured(response_model, user)
        return StructuredOutputResult(
            value=value,
            provider=self.provider_name,
            model=self.model_name,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def plan_structured(self, *, system: str, user: str) -> PlannedQuery:
        self.plan_calls += 1
        lowered = user.lower()
        if "start" in lowered and "deceased" in lowered:
            return PlannedQuery(
                file_ids=[49],
                goals=["exact_count", "cause_summary"],
                needs_synthesis=True,
                steps=[
                    PlannedStep(id="scope_current", op="USE_CURRENT_VERSION", input="current_records"),
                    PlannedStep(
                        id="matches",
                        op="FILTER",
                        input="scope_current",
                        where=[
                            PlannedPredicate(field="student_name", operator="STARTS_WITH", value="S"),
                            PlannedPredicate(field="deceased_status", operator="IS_TRUE"),
                        ],
                    ),
                    PlannedStep(id="count", op="COUNT", input="matches"),
                    PlannedStep(
                        id="evidence",
                        op="GET_EVIDENCE",
                        input="matches",
                        fields=["student_name", "notes"],
                        limit=8,
                    ),
                ],
            )
        return PlannedQuery(
            file_ids=[49],
            goals=["cause_summary"],
            needs_synthesis=True,
            steps=[
                PlannedStep(id="scope_current", op="USE_CURRENT_VERSION", input="current_records"),
                PlannedStep(
                    id="matches",
                    op="FILTER",
                    input="scope_current",
                    where=[PlannedPredicate(field="deceased_status", operator="IS_TRUE")],
                ),
                PlannedStep(
                    id="evidence",
                    op="GET_EVIDENCE",
                    input="matches",
                    fields=["student_name", "notes"],
                    limit=8,
                ),
            ],
        )

    async def answer_structured(self, *, system: str, user: str) -> SynthesisAnswer:
        self.synthesis_calls += 1
        citations = _extract_source_ids(user)
        poisoned = [*citations, "file:93:v1:row:1108"]
        names = _extract_names(user)
        facts = _extract_facts(user)
        lead = facts[0] if facts else "According to the supplied records"
        body = ", ".join(names) if names else "the selected records"
        causes = _extract_causes(user)
        cause_text = f" Causes noted: {', '.join(causes)}." if causes else ""
        return SynthesisAnswer(
            answer=f"{lead}. Evidence refers to {body}.{cause_text}",
            citations=poisoned,
            inference="why" in user.lower() or "likely" in user.lower(),
            memory_update=MemoryUpdate(
                active_file_ids=[49, 93],
                active_topics=["death records"],
                last_source_ids=poisoned,
            ),
        )


def _default_structured(response_model, user: str):
    name = getattr(response_model, "__name__", "")
    if name == "TurnPlan":
        return response_model.model_validate(_turn_plan_payload(user))
    if name == "PlannedTurn":
        return response_model.model_validate(_planned_turn_payload(user))
    if name == "PlanReviewResponse":
        return response_model.model_validate(_review_payload(user))
    payload = _canonical_payload(user)
    return response_model.model_validate(payload)



def _review_payload(user: str) -> dict:
    """Produce a schema-faithful fake review from the candidate plan.

    Production review reconstructs from the question. The fake provider intentionally has
    no semantic model, so it mirrors the candidate requirements to exercise the trusted
    coverage checks without making tests depend on heuristic NLP in the test double.
    """
    marker = "CANDIDATE PLAN:\n"
    index = user.find(marker)
    candidate: dict[str, Any] = {}
    if index >= 0:
        tail = user[index + len(marker):]
        try:
            parsed, _ = json.JSONDecoder().raw_decode(tail)
            if isinstance(parsed, dict):
                candidate = parsed
        except json.JSONDecodeError:
            candidate = {}
    requirements = list(candidate.get("stated_requirements") or [])
    if not requirements:
        requirements = _stated_from_actions(list(candidate.get("actions") or []))
    # verify_previous has no query action but still has one semantic requirement.
    if not requirements:
        for action_index, action in enumerate(candidate.get("actions") or []):
            if action.get("type") == "verify_previous":
                requirements.append(
                    {
                        "kind": "option",
                        "action_index": action_index,
                        "text": "verify previous",
                        "option": "verify_previous",
                        "value": True,
                    }
                )
            elif action.get("type") == "modify_previous":
                for spec in action.get("changes") or []:
                    requirements.append(
                        {
                            "kind": "modify_edit",
                            "action_index": action_index,
                            "text": f"{spec.get('field')} {spec.get('operator')} {spec.get('value')}",
                            "edit": "filter",
                            "value": spec,
                        }
                    )
    # Review is skipped for conversational turns, so a missing requirement here indicates
    # a malformed test candidate rather than a legitimate complete data review.
    if not requirements:
        requirements = [
            {
                "kind": "goal",
                "action_index": 0,
                "text": "goal count",
                "goal": "count",
            }
        ]
    return {"verdict": "complete", "reconstructed_requirements": requirements}

def _planned_turn_payload(user: str) -> dict:
    """TurnPlan-shaped payload without dataset identifiers."""
    payload = _turn_plan_payload(user)
    actions = []
    for action in payload.get("actions") or []:
        item = dict(action)
        item.pop("datasets", None)
        if item.get("type") == "respond" and item.get("intent") == "capability":
            if "additional deaths" in user.lower() or "potential" in user.lower():
                item["intent"] = "dataset_not_selected"
                item["response"] = item.get("response") or ""
        actions.append(item)
    payload["actions"] = actions
    payload["stated_requirements"] = _stated_from_actions(actions)
    return payload


def _stated_from_actions(actions: list[dict]) -> list[dict]:
    requirements: list[dict] = []
    for index, action in enumerate(actions):
        if action.get("type") != "query":
            continue
        requirements.append(
            {
                "kind": "goal",
                "action_index": index,
                "text": f"goal {action.get('goal')}",
                "goal": action.get("goal"),
            }
        )
        for spec in action.get("filters") or []:
            requirements.append(
                {
                    "kind": "filter",
                    "action_index": index,
                    "text": f"{spec.get('field')} {spec.get('operator')} {spec.get('value')}",
                    "collection": "filters",
                    "field": spec.get("field"),
                    "operator": spec.get("operator"),
                    "value": spec.get("value"),
                }
            )
        if action.get("search_text"):
            requirements.append(
                {
                    "kind": "search_text",
                    "action_index": index,
                    "text": action["search_text"],
                    "search_text": action["search_text"],
                }
            )
        if action.get("limit") is not None:
            requirements.append(
                {
                    "kind": "numeric_slot",
                    "action_index": index,
                    "text": f"limit {action['limit']}",
                    "slot": "limit",
                    "value": action["limit"],
                }
            )
    return requirements


def _payload(user: str) -> dict:
    try:
        parsed = json.loads(user)
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _utterance(user: str) -> tuple[str, str, dict]:
    parsed = _payload(user)
    utterance = str(
        parsed.get("current_user_turn")
        or parsed.get("user_utterance")
        or parsed.get("normalized_input")
        or user
    )
    # Find the question by its label, not by its position. The real prompt puts the
    # stable catalog first so a provider can cache it, which leaves QUESTION on the last
    # line rather than the first.
    for line in utterance.splitlines():
        if line.startswith("QUESTION:"):
            spoken = line.split(":", 1)[1].strip()
            if spoken.startswith('"'):
                try:
                    decoded = json.loads(spoken)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, str):
                    spoken = decoded
            if spoken:
                utterance = spoken
            break
    mode = str(parsed.get("mode") or "text")
    return utterance, mode, parsed


def _catalog_ids(parsed: dict) -> list[int]:
    catalog = parsed.get("authorized_catalog") or {}
    datasets = catalog.get("datasets") or []
    ids = []
    for item in datasets:
        try:
            ids.append(int(item.get("file_id")))
        except (TypeError, ValueError, AttributeError):
            continue
    return ids


def _default_dataset(parsed: dict) -> int:
    catalog = parsed.get("authorized_catalog") or {}
    default = catalog.get("default_people_file_id")
    ids = _catalog_ids(parsed)
    if default and int(default) in ids:
        return int(default)
    return ids[0] if ids else 49


def _dataset_for_label(parsed: dict, text: str) -> int | None:
    catalog = parsed.get("authorized_catalog") or {}
    lowered = text.lower()
    for item in catalog.get("datasets") or []:
        labels = [str(item.get("label") or "")]
        labels.extend(str(alias) for alias in (item.get("aliases") or []))
        labels.append(f"file {item.get('file_id')}")
        if any(alias.lower() and alias.lower() in lowered for alias in labels):
            return int(item["file_id"])
    return None


def _turn_plan_payload(user: str) -> dict:
    utterance, mode, parsed = _utterance(user)
    text = utterance.lower()
    default_id = _default_dataset(parsed)
    authorized = set(_catalog_ids(parsed))
    active = parsed.get("active_query") or {}
    frames = parsed.get("topic_frames") or []
    voice = mode == "voice"

    if _is_identity(text):
        return _respond(
            "What is your name?",
            "I'm NIA." if voice else "My name is NIA.",
            0.99,
            intent="assistant_identity",
        )
    if _is_audio(text):
        reply = "Yes, I can hear you." if voice else "Yes, I received your message."
        return _respond("Can you hear me?", reply, 0.99, intent="audio_check")
    if re.fullmatch(r"(?:hello|hi|hey|hiya|yo)[\s.!?]*", text):
        return _respond("Hello.", "Hi!" if voice else "Hi! How can I help?", 0.99, intent="greeting")
    from app.planning.user_name import (
        extract_introduced_name,
        is_bare_affirmation,
        is_confusion,
        is_conversational_correction,
        is_speak_request,
        is_user_name_question,
    )

    introduced = extract_introduced_name(utterance)
    if introduced:
        return _respond(utterance, introduced, 0.98, intent="user_introduction")
    if is_user_name_question(text):
        return _respond(utterance, "", 0.96, intent="user_identity")
    if is_speak_request(text):
        return _respond(utterance, "__voice__", 0.97, intent="capability")
    if is_bare_affirmation(text):
        return _respond(utterance, "", 0.95, intent="acknowledgement")
    if is_confusion(text) or is_conversational_correction(text):
        return _respond(utterance, "", 0.95, intent="correction")
    if re.search(r"\b(?:thanks|thank you)\b", text) and len(text.split()) <= 4:
        return _respond("Thank you.", "You're welcome.", 0.99, intent="thanks")
    if re.fullmatch(r"(?:bye|goodbye|good bye)[\s.!?]*", text):
        return _respond("Goodbye.", "Goodbye.", 0.99, intent="farewell")
    if _is_catalog(text):
        return _respond(
            "describe authorized research records",
            "__describe_authorized_data__",
            0.95,
            intent="describe_available_data",
        )
    if any(token in text for token in ("unclear", "ambiguous", "old ones", "old students", "the old")):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "clarify",
                    "question": "Are you asking for a count, a list, or a description of the records?",
                }
            ],
            "final_response": "compiler_response",
            "unresolved": ["intended research question"],
            "confidence": 0.4,
        }

    named_dataset = _dataset_for_label(parsed, text)
    if "additional deaths" in text and (named_dataset is None or named_dataset not in authorized):
        return _respond(
            "count authorized records",
            "I can only search the authorized research records in this session.",
            0.86,
            intent="capability",
        )

    action_ref = _action_reference(text)
    if action_ref is not None:
        return action_ref

    ordinal = _ordinal_selector(text)
    if ordinal:
        return {
            "normalized_request": utterance,
            "actions": [{"type": "reference", "selector": ordinal, "target": "row"}],
            "final_response": "deterministic",
            "confidence": 0.92,
        }

    followup = _followup_plan(text, utterance, active, frames, default_id)
    if followup is not None:
        return followup

    if re.search(r"\bcompare\b", text) and "garden" in text:
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "compare",
                    "compare": [
                        {
                            "label": "garden",
                            "filters": [{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
                        },
                        {
                            "label": "sault",
                            "filters": [{"field": "community", "operator": "EQUALS", "value": "Sault Ste. Marie"}],
                        },
                    ],
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    if re.search(r"\b(?:earliest|minimum|min)\b", text) and re.search(r"\b(?:admit|admission|date)\b", text):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "aggregate",
                    "aggregate": {"function": "min", "field": "admitted_date"},
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.92,
        }
    if re.search(r"\bquote\b", text) and "request" not in text and " and quote" not in text:
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "quote",
                    "search_text": "illness",
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }
    if "provenance" in text and "request" not in text:
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "provenance",
                    "filters": [{"field": "student_name", "operator": "STARTS_WITH", "value": "S"}],
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }

    if ("each community" in text) or ("where they're from" in text) or (
        "organize" in text and "community" in text
    ) or ("group" in text and "community" in text):
        return {
            "normalized_request": "list students grouped by community",
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "list",
                    "group_by": ["community"],
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.88,
        }

    wants_synthesis = any(
        token in text
        for token in (
            "summarize",
            "summary",
            "explain",
            "why",
            "suggest",
            "interpret",
            "what do the records say",
            "what their notes",
            "causes of death",
        )
    )
    if "causes of death" in text or ("summarize" in text and "records" in text):
        return {
            "normalized_request": "summarize causes of death",
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "search",
                    "search_text": "cause of death",
                    "filters": [],
                }
            ],
            "final_response": "llm_synthesis",
            "needs_evidence": True,
            "needs_explanation": True,
            "confidence": 0.9,
        }

    if re.search(r"\band quote\b", text) and re.search(r"\b(?:count|how many)\b", text):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "count",
                    "filters": [{"field": "community", "operator": "EQUALS", "value": "Garden River"}],
                },
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "quote",
                    "search_text": "illness",
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }

    if re.search(r"\band list\b", text) and re.search(r"\b(?:count|how many)\b", text):
        count_filters = _extract_filters(re.split(r"\band list\b", text, maxsplit=1)[0])
        list_filters = _extract_filters(re.split(r"\band list\b", text, maxsplit=1)[1])
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "count",
                    "filters": count_filters,
                },
                {
                    "type": "query",
                    "datasets": [default_id],
                    "goal": "list",
                    "filters": list_filters,
                    "limit": 20,
                },
            ],
            "final_response": "deterministic",
            "confidence": 0.9,
        }

    filters = _extract_filters(text)
    search_text = _extract_search(text)
    datasets = [named_dataset] if named_dataset in authorized else [default_id]
    if named_dataset is None and re.search(r"\b(?:all datasets|each dataset|every dataset|across all)\b", text):
        datasets = [file_id for file_id in authorized] or [default_id]

    if wants_synthesis and ("how many" in text or "count" in text):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": datasets,
                    "goal": "count",
                    "filters": filters,
                    "search_text": search_text,
                }
            ],
            "final_response": "llm_synthesis",
            "needs_evidence": True,
            "needs_explanation": True,
            "confidence": 0.85,
        }

    if search_text or re.search(r"\b(?:mention|containing|search|find|look up|named|called)\b", text):
        goal = "search"
        if not search_text:
            search_text = utterance
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": datasets,
                    "goal": goal,
                    "filters": filters,
                    "search_text": search_text,
                }
            ],
            "final_response": "deterministic" if not wants_synthesis else "llm_synthesis",
            "needs_evidence": True,
            "needs_explanation": wants_synthesis,
            "confidence": 0.86,
        }

    if re.search(
        r"\b(?:how many|count|number of|the number|total|what's the total|whats the total)\b",
        text,
    ):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": datasets,
                    "goal": "count",
                    "filters": filters,
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.93,
        }

    if re.search(r"\b(?:list|show|which|who|enumerate)\b", text) or filters:
        limit = 25
        limit_match = re.search(r"\b(?:first|top|limit)\s+(\d{1,2})\b", text)
        if limit_match:
            limit = int(limit_match.group(1))
        exhaustive = bool(re.search(r"\ball\b|one by one|organized", text))
        presentation = (
            "numbered_list"
            if re.search(r"one by one|organized|alphabet|numbered", text) or exhaustive
            else None
        )
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "query",
                    "datasets": datasets,
                    "goal": "list",
                    "filters": filters,
                    "limit": limit,
                    "offset": 0,
                    "exhaustive": exhaustive,
                    "presentation": presentation,
                    "sort_by": "student_name" if exhaustive or presentation else None,
                    "sort_direction": "asc" if exhaustive or presentation else None,
                }
            ],
            "final_response": "deterministic" if not wants_synthesis else "llm_synthesis",
            "needs_evidence": wants_synthesis,
            "needs_explanation": wants_synthesis,
            "confidence": 0.9,
        }

    from app.planning.user_name import looks_like_research

    if not looks_like_research(text):
        if re.search(
            r"\b(?:why|how|what|who|where|when|write|explain|tell me|can you)\b",
            text,
        ) or "?" in text:
            return _respond(utterance, "", 0.9, intent="general_conversation")
    return {
        "normalized_request": utterance,
        "actions": [
            {
                "type": "clarify",
                "question": "Are you asking for a count, a list, or a description of the records?",
            }
        ],
        "final_response": "compiler_response",
        "unresolved": ["intended research question"],
        "confidence": 0.4,
    }


def _followup_plan(
    text: str,
    utterance: str,
    active: dict,
    frames: list,
    default_id: int,
) -> dict | None:
    going = re.search(r"going back to\s+(.+)", text)
    if going:
        frame = _find_frame(going.group(1), frames)
        if frame is not None:
            return {
                "normalized_request": utterance,
                "actions": [
                    {
                        "type": "modify_previous",
                        "changes": list(frame.get("filters") or [{"field": "student_name", "operator": "STARTS_WITH", "value": "S"}]),
                        "restore_frame": frame.get("frame_key") or frame.get("key") or "master",
                        "goal": frame.get("goal") or "count",
                    }
                ],
                "final_response": "deterministic",
                "confidence": 0.9,
            }
        if "master" in text:
            return {
                "normalized_request": utterance,
                "actions": [
                    {
                        "type": "query",
                        "datasets": [default_id],
                        "goal": "count",
                        "filters": [{"field": "student_name", "operator": "STARTS_WITH", "value": "S"}],
                    }
                ],
                "final_response": "deterministic",
                "confidence": 0.88,
            }

    if re.search(
        r"\b(?:more than that|that'?s wrong|still wrong|check again|are you sure|i believe there)\b",
        text,
    ):
        if active:
            return {
                "normalized_request": utterance,
                "actions": [{"type": "verify_previous"}],
                "final_response": "deterministic",
                "confidence": 0.93,
            }

    if active and re.fullmatch(
        r"(?:next(?:\s+\d+)?|keep going|show more|continue)[\s.!?]*",
        text,
    ):
        return {
            "normalized_request": utterance,
            "actions": [{"type": "modify_previous", "goal": "list", "page": "next"}],
            "final_response": "deterministic",
            "confidence": 0.95,
        }

    if active and re.search(
        r"\b(?:list (?:them|those|these)|one by one|organized|alphabet)\b",
        text,
    ):
        return {
            "normalized_request": utterance,
            "actions": [
                {
                    "type": "modify_previous",
                    "goal": "list",
                    "exhaustive": True,
                    "presentation": "numbered_list",
                    "sort_by": "student_name",
                    "sort_direction": "asc",
                    "page": "first",
                }
            ],
            "final_response": "deterministic",
            "confidence": 0.94,
        }

    if not active and not (
        re.match(r"^(?:what about|how about|only\b|actually|instead|rather)", text)
        or "those" in text
        or "exclude" in text
    ):
        return None
    if not active and not re.match(
        r"^(?:what about|how about|only\b|actually|instead|rather|which of those|exclude)",
        text,
    ):
        return None

    changes: list[dict] = []
    year = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", text)
    if year:
        operator = "BEFORE" if "before" in text else "AFTER" if "after" in text else "YEAR_EQUALS"
        field = "admitted_date"
        for item in active.get("filters") or []:
            if str(item.get("field", "")).endswith("_date"):
                field = item["field"]
                if operator == "YEAR_EQUALS" and item.get("operator") in {"BEFORE", "AFTER"}:
                    operator = item["operator"]
                break
        changes.append({"field": field, "operator": operator, "value": int(year.group(1))})
    if re.search(r"\bonly\b", text) or "garden river" in text:
        community = re.search(r"garden river|sault ste\.? marie", text)
        if community:
            label = "Garden River" if "garden" in community.group(0) else "Sault Ste. Marie"
            changes.append({"field": "community", "operator": "EQUALS", "value": label})
    if re.search(r"\bdeceased\b", text):
        changes.append({"field": "deceased_status", "operator": "IS_TRUE"})
    if re.search(r"unknown|missing", text) and re.search(r"discharge", text):
        changes.append({"field": "discharged_date", "operator": "IS_KNOWN"})
    letter = re.search(r"\bwith\s+([a-z])\b|\bletter\s+([a-z])\b", text)
    if letter:
        changes.append(
            {
                "field": "student_name",
                "operator": "STARTS_WITH",
                "value": (letter.group(1) or letter.group(2)).upper(),
            }
        )
    if not changes:
        return None
    return {
        "normalized_request": utterance,
        "actions": [{"type": "modify_previous", "changes": changes}],
        "final_response": "deterministic",
        "confidence": 0.9,
    }


def _find_frame(phrase: str, frames: list) -> dict | None:
    lowered = phrase.lower()
    for item in frames:
        if not isinstance(item, dict):
            continue
        blob = " ".join(
            str(item.get(key) or "")
            for key in ("frame_key", "key", "topic")
        ).lower()
        if any(token in blob or token in lowered for token in ("master", "confirmed", "student")):
            if "master" in lowered and ("master" in blob or item.get("datasets") == [49] or item.get("file_ids") == [49]):
                return item
            if "confirmed" in lowered and "confirmed" in blob:
                return item
    return None


def _extract_filters(text: str) -> list[dict]:
    filters: list[dict] = []
    prefix = re.search(
        r"(?:start|starts|starting|begin|begins|beginning|up front).{0,20}\b([a-z0-9])\b"
        r"|\b([a-z0-9])\s+(?:at the )?(?:beginning|start) of"
        r"|\bnames? (?:with|beginning in|beginning with)\s+([a-z0-9])"
        r"|\bthe ([a-z]) names\b",
        text,
    )
    if prefix:
        letter = next(group for group in prefix.groups() if group)
        filters.append({"field": "student_name", "operator": "STARTS_WITH", "value": letter.upper()})
    from app.retrieval.entity_resolver import resolve_entity

    community_match = re.search(
        r"\b(?:from|in|community)\s+([a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){0,3})",
        text,
    )
    if community_match:
        label = community_match.group(1).strip()
        resolved = resolve_entity(label)
        if resolved and not resolved.ambiguous:
            filters.append({"field": "community", "operator": "EQUALS", "value": resolved.canonical})
        elif "garden" in label:
            filters.append({"field": "community", "operator": "EQUALS", "value": "Garden River"})
    elif "garden river" in text:
        filters.append({"field": "community", "operator": "EQUALS", "value": "Garden River"})
    if re.search(r"\b(?:deceased|died|dead)\b", text):
        filters.append({"field": "deceased_status", "operator": "IS_TRUE"})
    year = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", text)
    if year and re.search(r"\b(?:admitted|admission|before|after|in)\b", text):
        operator = "BEFORE" if "before" in text else "AFTER" if "after" in text else "YEAR_EQUALS"
        field = "discharged_date" if re.search(r"\bdischarg", text) else "admitted_date"
        filters.append({"field": field, "operator": operator, "value": int(year.group(1))})
    return filters


def _extract_search(text: str) -> str | None:
    mention = re.search(
        r"\b(?:mention(?:s|ed)?|containing|contains|search(?:ing)? for)\s+['\"]?([^\"'.?]+)",
        text,
    )
    if mention:
        return mention.group(1).strip()
    named = re.search(
        r"\b(?:named|called|look up|lookup|find)\s+([a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){0,3})",
        text,
    )
    if named and "students" not in named.group(1):
        return named.group(1).strip()
    return None


def _respond(normalized: str, response: str, confidence: float, *, intent: str) -> dict:
    return {
        "normalized_request": normalized,
        "actions": [{"type": "respond", "intent": intent, "response": response}],
        "final_response": "compiler_response",
        "confidence": confidence,
    }


def _is_identity(text: str) -> bool:
    return any(
        token in text
        for token in (
            "your name",
            "ur name",
            "who are you",
            "who're you",
            "who are u",
            "what do people call you",
            "what do you call yourself",
        )
    )


def _is_audio(text: str) -> bool:
    return any(
        token in text
        for token in (
            "hear me",
            "hearing me",
            "coming through",
            "can you hear",
            "understand me",
            "you hearing",
            "am i coming",
        )
    )


def _action_reference(text: str) -> dict | None:
    if not re.search(
        r"\b(?:repeat|again|the comparison|the aggregate|"
        r"(?:first|second|last) request)\b",
        text,
    ):
        return None
    selector = "first"
    if re.search(r"\b(?:second|2nd)\b", text):
        selector = "second"
    elif re.search(r"\blast\b", text):
        selector = "last"
    elif re.search(r"\b(?:first|1st)\b", text):
        selector = "first"
    return {
        "normalized_request": text,
        "actions": [{"type": "reference", "selector": selector, "target": "action"}],
        "final_response": "deterministic",
        "confidence": 0.93,
    }


def _ordinal_selector(text: str) -> str | None:
    match = re.search(
        r"\b(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b",
        text,
    )
    if not match:
        return None
    if not re.search(r"\b(?:one|ones|that|those|result|record|them|it)\b", text) and "tell me about" not in text:
        return None
    raw = match.group(1)
    return {
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
        "4th": "fourth",
        "5th": "fifth",
    }.get(raw, raw)


def _is_catalog(text: str) -> bool:
    return any(
        token in text
        for token in (
            "what are in the research",
            "what is in the research",
            "what's in the research",
            "whats in the research",
            "what records",
            "what are the research records",
            "what's in the records",
            "what datasets",
            "what are the records",
        )
    )


def _canonical_payload(user: str) -> dict:
    utterance, _mode, _parsed = _utterance(user)
    text = utterance.lower()
    if _is_identity(text):
        return {
            "turn_type": "conversation",
            "intent": "assistant_identity",
            "answer_directly": True,
            "direct_answer": "My name is NIA.",
            "canonical_text": "ask assistant identity",
            "confidence": 0.95,
        }
    if _is_audio(text):
        return {
            "turn_type": "conversation",
            "intent": "audio_check",
            "answer_directly": True,
            "direct_answer": "Yes, I can hear you.",
            "canonical_text": "audio check",
            "confidence": 0.95,
        }
    return {
        "turn_type": "research",
        "requires_full_planner": True,
        "canonical_text": utterance[:200],
        "confidence": 0.5,
    }


def _extract_source_ids(user: str) -> list[str]:
    ids: list[str] = []
    token = "[file:"
    start = 0
    while True:
        index = user.find(token, start)
        if index < 0:
            break
        end = user.find("]", index)
        if end < 0:
            break
        ids.append(user[index + 1 : end])
        start = end + 1
    return ids


def _extract_names(user: str) -> list[str]:
    names: list[str] = []
    for line in user.splitlines():
        if "student_name:" in line:
            names.append(line.split(":", 1)[1].strip())
    return names


def _extract_causes(user: str) -> list[str]:
    causes: list[str] = []
    for line in user.splitlines():
        if "cause_of_death:" in line:
            causes.append(line.split(":", 1)[1].strip())
    return causes


def _extract_facts(user: str) -> list[str]:
    facts: list[str] = []
    capture = False
    for line in user.splitlines():
        if line.strip() == "COMPUTED FACTS":
            capture = True
            continue
        if line.strip() == "EVIDENCE":
            break
        if capture and line.startswith("- ") and line[2:] != "none":
            facts.append(line[2:])
    return facts
