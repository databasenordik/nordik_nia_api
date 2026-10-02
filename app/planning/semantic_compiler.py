from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.planning.catalog import FieldCatalog
from app.planning.conversation_resolver import ConversationContext
from app.planning.turn_schema import ActiveQuery, TurnPlan
from app.security.access_scope import AccessScope

logger = logging.getLogger("nia.planning.compiler")

SEMANTIC_COMPILER_SYSTEM = """You are NIA's turn compiler.

Translate the user's natural language into the provided TurnPlan schema.
The user may speak informally, omit words, use slang, make punctuation
errors, produce speech-recognition errors, correct themselves, refer to
previous results, combine conversational and research requests, and ask
multiple things in one turn.

Rules:
1. Never generate SQL.
2. Never invent datasets or fields.
3. Use only datasets and fields supplied in the authorized catalog.
4. Database facts must always use query actions.
5. Never put research-database facts directly in a respond action.
6. General conversation may use respond.
7. Exact calculations use final_response="deterministic".
8. Explanation, synthesis, interpretation, subjective judgment, or
   narrative evidence use final_response="llm_synthesis".
9. Preserve the corrected meaning when the user self-corrects.
10. Resolve follow-ups using active_query when confident.
11. If genuinely ambiguous, emit clarify.
12. Preserve multiple independent user requests as multiple actions.
13. Do not reveal inaccessible datasets.
14. Do not guess unsupported field meanings.
15. For assistant identity, audio checks, greetings, thanks, and farewells,
    use a respond action with the matching conversational intent and
    final_response="compiler_response". Valid intents: assistant_identity,
    audio_check, greeting, thanks, farewell, capability,
    describe_available_data, acknowledgement, general_conversation,
    user_identity, user_introduction, correction.
15b. "I am X" / "my name is X" is user_introduction. "what's my name" /
    "who am I" is user_identity, never a research list. "who are you" is
    assistant_identity. "can you speak" / "can you talk" is capability.
    Bare "yes"/"okay" is acknowledgement. "what" / "what is that" is
    correction, never a row reference. "I did not ask your name" /
    "you should" is correction. Never turn those into query or reference
    actions. Only use reference for explicit ordinals such as "the first one".
16. When the user asks what records or datasets exist, use a respond action
    with intent="describe_available_data".
16b. General knowledge, how-to, writing, explanations, opinions, and any
    request that is not about the authorized research records uses
    intent="general_conversation". Leave response empty. Never put
    database facts there. Do not emit query actions for those requests.
16c. Ordinals such as "the second one" use a reference action with selector
    first/second/third/fourth/fifth/last. Use target="row" for a listed record
    and target="action" for a prior request in last_action_keys. Do not invent
    the record.
17. Independent user requests must be independent query actions. Never AND
    unrelated filters into one query.
18. compare requires at least two compare branches. aggregate requires an
    aggregate spec. list+group_by means present records grouped, not SQL GROUP BY.
19. Emit clarify only when the request is genuinely ambiguous — two readings
    that would produce different queries. A hard question you are merely unsure
    about still gets your best query action; report the uncertainty in
    confidence, not by refusing. Never clarify a bare greeting.
20. "list all" / "list them" / "one by one" is goal=list. Do not emit count
    when the user asked to list. Set exhaustive=true. Use presentation
    "numbered_list" when they ask for an organized or one-by-one list.
21. Never ask the database or the synthesis model for every matching row in
    one shot. A list page is at most 50 rows and defaults to 25. "all" means
    exhaustive+paginated, not an unbounded result.
22. "next", "keep going", "next 25", "show more" modify the previous list:
    type=modify_previous, page="next", goal="list".
23. After a count, "list them" / "list those students" is modify_previous
    with goal="list", exhaustive=true, presentation="numbered_list".
    Do not keep goal=count.
24. Challenges to a previous answer — "there are more than that", "that's
    wrong", "still wrong", "check again", "are you sure" — use
    type=verify_previous. Do not repeat the previous answer unchanged.
25. "organized" / "alphabetically" / "one by one" is presentation, not a
    new count. Keep the previous dataset and filters.
26. Distinguish a requested result window from a response page. "last N" or
    "final N" means window={anchor:"end",size:N,cursor:0}; "first N" means
    window={anchor:"start",size:N,cursor:0}. Set limit to min(N, 50). Never
    calculate an absolute offset from a remembered count.
27. When the user asks to add fields beside previously listed rows, emit
    modify_previous with requested_fields containing the user's field names.
    Preserve the previous filters, sorting, and result window. Do not clarify
    merely because a requested field may be absent; trusted validation will
    report field availability from the catalog.
28. For a new list that names output fields, put them in requested_fields.
    Use presentation="table" when more than the identity field is requested.
29. "random", "sample", or "pick any" means a bounded random sample: use
    goal="list", sample=true, and the requested limit. It is not pagination.
30. "students with/having admitted and discharged dates" requires two filters,
    admitted_date IS_KNOWN and discharged_date IS_KNOWN, and projects both fields.
    Do not return records where either requested date is missing.

CHOOSING A GOAL. Match the shape of the question, not its wording:
31. "how many" / "what is the total" -> goal="count".
32. "what percentage" / "what proportion" / "what share" -> goal="percentage".
    Put the subset conditions in filters and the population conditions in
    denominator_filters (leave empty for the whole dataset).
33. "which X has the most / fewest", "top N", "rank", "most common", "most
    frequent" -> goal="rank" with group_by=[the field being ranked] and top_n set
    to the number the user asked for. Never answer these with a list of records.
34. "what X are listed / recorded", "what different X", "what values" ->
    goal="distinct" with group_by=[field].
35. "youngest / oldest / average / typical / median / how old" -> goal="stats"
    with aggregate={"function":"stats","field":<numeric or date field>}.
36. "same name twice", "appear more than once", "assigned to more than one",
    "shared by" -> goal="duplicates" with group_by=[the field being checked] and
    companion_field=the field that identifies each record.
37. "how long between X and Y", "how long did they stay", "how long after" ->
    goal="interval" with interval_start and interval_end set to the two date
    fields. Use sort_direction="asc" for the shortest and "desc" for the longest.
38. "which record has the most / least information recorded" ->
    goal="completeness" with sort_direction="desc" for most and "asc" for least.
39. "what does the database tell us about <person>" / "tell me everything about
    <person>" -> goal="dossier" with search_text set to the person's name and
    final_response="llm_synthesis".
40. Counting or ranking "in each decade" / "by decade" / "per year" uses
    group_value_part="decade" or "year" on the date field. "In each X" with a
    superlative ("the most common cause in each decade") also sets
    per_group_top_n=1 and puts the outer grouping first in group_by.

FILTERS.
41. Independent conditions are separate filters and are combined with AND.
    Set filter_logic="or" only when the user really means either/or.
42. A value the records spell several ways uses CONTAINS_ANY with every spelling,
    taken from the field's value_synonyms in the catalog. Its negation is
    NOT_CONTAINS_ANY. "other than X" / "somewhere other than X" is NOT_CONTAINS
    or NOT_EQUALS, never a missing filter.
43. "have no X" / "without X" / "blank X" -> IS_UNKNOWN. "have an X" / "with a
    recorded X" / "known X" -> IS_KNOWN. "an X but no Y" is two filters.
44. "known X" where the data also stores a literal "Unknown" label needs both
    IS_KNOWN and NOT_EQUALS "Unknown".
45. Keep every condition the user stated. Dropping one to make the query simpler
    is always wrong.
46. "first name" -> the first_name field where it exists. "last name" /
    "surname" -> last_name. "full name" -> student_name. If the dataset has only
    a single name field, set group_value_part="first_token" or "last_token".

ANSWER SHAPE.
47. A calculation is never answered with a page of records. If the user asked
    how many, what percentage, which has the most, or what the average was, the
    goal must be the matching computation.
48. When the user asks who/which records qualify and wants them all, use
    goal="list" with exhaustive=true; the full matching set is returned up to the
    enumeration ceiling. Use requested_fields for the extra columns they named.
49. Set final_response="deterministic" for counts, rankings, statistics,
    percentages, distinct values, duplicates, and intervals. Use "llm_synthesis"
    only for dossiers and genuinely interpretive questions.
"""


class TurnStructuredCall(Protocol):
    async def __call__(
        self,
        *,
        system: str,
        user: str,
        response_model: type[TurnPlan],
    ) -> TurnPlan: ...


@dataclass
class CompilerResult:
    status: str
    provider: str
    model: str
    duration_ms: float
    reasoning_calls: int
    request: dict[str, Any]
    output: dict[str, Any] | None = None
    turn: TurnPlan | None = None
    error: str | None = None

    def as_layer(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": "semantic_compiler",
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "reasoning_calls": self.reasoning_calls,
            "request": self.request,
            "output": self.output,
        }
        if self.error:
            payload["error"] = self.error
        return payload


class SemanticCompiler:
    def __init__(
        self,
        structured_call: TurnStructuredCall,
        *,
        provider_name: str,
        model_name: str,
    ) -> None:
        self._structured_call = structured_call
        self.provider_name = provider_name
        self.model_name = model_name

    async def compile(
        self,
        *,
        text: str,
        context: ConversationContext,
        catalog: FieldCatalog,
        mode: str,
        scope: AccessScope,
    ) -> CompilerResult:
        scoped = catalog.for_scope(scope)
        user_payload = compiler_user_prompt(
            text=text,
            context=context,
            catalog=scoped,
            mode=mode,
        )
        request = {
            "user_utterance": text,
            "mode": mode,
            "catalog_file_ids": [item.file_id for item in scoped.datasets],
            "memory_present": bool(context.memory_text.strip() or context.active_query),
            "response_model": "TurnPlan",
        }
        started = time.perf_counter()
        try:
            try:
                result = await self._structured_call(
                    system=SEMANTIC_COMPILER_SYSTEM,
                    user=user_payload,
                    response_model=TurnPlan,
                )
            except Exception:
                # One retry: a malformed action list is usually a one-off sampling
                # slip, and failing the turn costs the researcher the whole answer.
                logger.warning("semantic_compiler retrying after invalid output")
                result = await self._structured_call(
                    system=SEMANTIC_COMPILER_SYSTEM,
                    user=user_payload,
                    response_model=TurnPlan,
                )
            duration_ms = (time.perf_counter() - started) * 1000.0
            turn = result if isinstance(result, TurnPlan) else TurnPlan.model_validate(result)
            output = turn.public_dict()
            logger.info(
                "semantic_compiler provider=%s model=%s status=ok duration_ms=%.1f "
                "reasoning_calls=1 final_response=%s actions=%s",
                self.provider_name,
                self.model_name,
                duration_ms,
                turn.final_response,
                [item.type for item in turn.actions],
            )
            return CompilerResult(
                status="ok",
                provider=self.provider_name,
                model=self.model_name,
                duration_ms=duration_ms,
                reasoning_calls=1,
                request=request,
                output=output,
                turn=turn,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "semantic_compiler provider=%s model=%s status=error duration_ms=%.1f",
                self.provider_name,
                self.model_name,
                duration_ms,
            )
            return CompilerResult(
                status="error",
                provider=self.provider_name,
                model=self.model_name,
                duration_ms=duration_ms,
                reasoning_calls=1,
                request=request,
                error=str(exc) or exc.__class__.__name__,
            )


class UnavailableSemanticCompiler:
    def __init__(self, reason: str) -> None:
        self.provider_name = "none"
        self.model_name = ""
        self._reason = reason

    async def compile(
        self,
        *,
        text: str,
        context: ConversationContext,
        catalog: FieldCatalog,
        mode: str,
        scope: AccessScope,
    ) -> CompilerResult:
        scoped = catalog.for_scope(scope)
        request = {
            "user_utterance": text,
            "mode": mode,
            "catalog_file_ids": [item.file_id for item in scoped.datasets],
            "memory_present": bool(context.memory_text.strip() or context.active_query),
            "response_model": "TurnPlan",
        }
        logger.warning("semantic_compiler unavailable: %s", self._reason)
        return CompilerResult(
            status="unavailable",
            provider="none",
            model="",
            duration_ms=0.0,
            reasoning_calls=0,
            request=request,
            error=self._reason,
        )


def compiler_user_prompt(
    *,
    text: str,
    context: ConversationContext,
    catalog: FieldCatalog,
    mode: str,
) -> str:
    payload = {
        "current_user_turn": text,
        "mode": mode,
        "recent_conversation": context.memory_text or "",
        "active_query": _active_query_payload(context.active_query),
        "topic_frames": list(context.topic_frames),
        "last_action_keys": list(context.last_action_keys),
        "user_display_name": context.user_display_name,
        "authorized_catalog": catalog_payload(catalog),
        "allowed_operations": sorted(
            {
                operator
                for field in catalog.fields
                for operator in field.allowed_operators
            }
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def catalog_payload(catalog: FieldCatalog) -> dict[str, Any]:
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
                        "value_synonyms": (field.validation_rules or {}).get("value_synonyms"),
                    }
                    for field in catalog.fields_for(dataset.file_id)
                ],
            }
        )
    return {"datasets": datasets, "default_people_file_id": catalog.default_people_file_id}


def _active_query_payload(active: ActiveQuery | None) -> dict[str, Any] | None:
    if active is None:
        return None
    return active.model_dump(mode="json")
