from __future__ import annotations

import re
from dataclasses import dataclass

from app.planning.catalog import FieldCatalog, dataset_match_labels
from app.planning.conversation_resolver import ConversationContext, resolve_conversation_followup
from app.planning.input_normalizer import NormalizedTurn, normalize_user_turn
from app.planning.plan_schema import (
    FilterOperator,
    PlanOp,
    PlanScope,
    PlanStep,
    Predicate,
    QueryPlan,
)
from app.retrieval.entity_resolver import resolve_entity
from app.security.access_scope import AccessScope

# This parser is deliberately a high-confidence optimization, not the primary
# language-understanding layer. If meaning is incomplete or uncertain, it abstains
# and lets the semantic normalizer / constrained planner handle the natural turn.
SYNTHESIS_HINTS = re.compile(
    r"\b(summarize|summary|why|likely|seems|compare|comparison|explain|interpret|"
    r"tell\s+me\s+about|what\s+does|what\s+do\s+you\s+think|meaning)\b",
    re.IGNORECASE,
)
_COMPLEX_HINTS = re.compile(
    r"\b(versus|vs\.?|except|excluding|exclude|without|instead\s+of|either|both|"
    r"ratio|percentage|percent|difference\s+between|rank|relationship|correlation|"
    r"and\s+then|also|plus|whereas|while)\b|[;]",
    re.IGNORECASE,
)
_NEGATION_HINTS = re.compile(r"\b(?:not|never|neither|nor)\b", re.IGNORECASE)
_MULTI_ACTION_HINT = re.compile(
    r"\b(?:count|list|show|find|enumerate)\b[^.!?;]*\band\b[^.!?;]*"
    r"\b(?:count|list|show|find|enumerate)\b",
    re.IGNORECASE,
)
_COUNT_HINT = re.compile(
    r"\b(?:how\s+many|count|number\s+of|total(?:\s+number)?\s+of|"
    r"what(?:'s|\s+is)\s+the\s+(?:count|number)|give\s+me\s+the\s+(?:count|number))\b",
    re.IGNORECASE,
)
_LIST_HINT = re.compile(
    r"\b(?:list|show|which|who|enumerate|give\s+me|find|look\s+up|lookup)\b",
    re.IGNORECASE,
)
_PREFIX_PATTERNS = (
    re.compile(
        r"(?:names?|students?|ones)\b[^,.?!;]{0,45}?\b"
        r"(?:start|starts|starting|begin|begins|beginning)\b[\s,]*(?:(?:with|in)[\s,]+)?"
        r"(?:the\s+letter\s+)?([A-Za-z0-9])\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Za-z0-9])\s+(?:at\s+the\s+)?(?:beginning|start)\s+of\s+"
        r"(?:their\s+|the\s+)?names?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:beginning|starting|start)\s+(?:with\s+)?(?:the\s+letter\s+)?"
        r"([A-Za-z0-9])\b[^,.?!;]{0,25}?\bnames?\b",
        re.IGNORECASE,
    ),
)
_COMMUNITY = re.compile(
    r"\b(?:from|community(?:\s+of)?|(?:students?|records)\s+in)\s+"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,4}?)"
    r"(?=\s+(?:before|after|admitted|admission|discharged|discharge|deceased|died|"
    r"with|whose|that|who|count|how|show|list|and|or|top|first|limit)\b|[,.?!;]|$)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_LIMIT = re.compile(r"\b(?:first|top|limit)\s+(\d{1,2})\b", re.IGNORECASE)
_ADMITTED = re.compile(r"\b(?:admitted|admission)\b", re.IGNORECASE)
_DISCHARGED = re.compile(r"\b(?:discharged|discharge)\b", re.IGNORECASE)
_BEFORE = re.compile(r"\bbefore\b", re.IGNORECASE)
_AFTER = re.compile(r"\bafter\b", re.IGNORECASE)
_DECEASED = re.compile(r"\b(?:deceased|recorded\s+as\s+deceased|died|dead)\b", re.IGNORECASE)
_ALL_DATASETS = re.compile(
    r"\b(?:all\s+datasets|each\s+(?:authorized\s+)?dataset|every\s+dataset|across\s+all\s+datasets)\b",
    re.IGNORECASE,
)
_RECORDS_ONLY = re.compile(r"\brecords?\b", re.IGNORECASE)
_STUDENTS = re.compile(r"\b(?:students?|people|persons?)\b", re.IGNORECASE)
_NAMES = re.compile(r"\bnames?\b", re.IGNORECASE)
_MENTION = re.compile(
    r"\b(?:mention(?:s|ed)?|containing|contains|search(?:ing)?\s+for)\s+['\"]?"
    r"([^\"'.?]+?)(?=[,.?!;]|$)",
    re.IGNORECASE,
)
_NAMED = re.compile(
    r"\b(?:named|called|look\s+up|lookup|find)\s+"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3})(?=[,.?!;]|$)",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(?:hi|hello|hey|yo|hiya|thanks|thank\s+you|bye|goodbye|good\s+bye)"
    r"[\s.!?]*$",
    re.IGNORECASE,
)
_DISTINCT_FIELD = re.compile(
    r"\b(?:list|show|what|which|name|enumerate|all)\b[^?]*?\b"
    r"(communities|reserves|first\s+nations|schools|day\s+schools)\b",
    re.IGNORECASE,
)
_DISTINCT_COUNT_FIELD = re.compile(
    r"\b(?:how\s+many|count(?:\s+all)?|number\s+of)\s+"
    r"(communities|reserves|first\s+nations|schools|day\s+schools)\b",
    re.IGNORECASE,
)
_SCHEMA_DISTINCT_INTENT = re.compile(
    r"\b(?:different|distinct|unique|various)\b",
    re.IGNORECASE,
)
_SCHEMA_AGGREGATE_HINT = re.compile(
    r"\b(?:distribution|breakdown|most\s+(?:common|frequent)|least\s+(?:common|frequent)|"
    r"maximum|minimum|highest|lowest|largest|smallest|oldest|youngest|earliest|latest)\b",
    re.IGNORECASE,
)
_PLURAL_TO_FIELD = {
    "communities": "community",
    "reserves": "community",
    "first nations": "community",
    "schools": "school",
    "day schools": "school",
}

# Only genuinely grammatical/conversational words are ignored for coverage.
# Semantic words such as before/after/admitted/community/count are NOT stopwords;
# they count as understood only if a matching rule actually consumed them.
_SAFE_COVERAGE_WORDS = {
    "a", "an", "the", "of", "for", "to", "me", "my", "our", "please", "could", "can",
    "would", "will", "you", "tell", "give", "get", "i", "want", "like", "just", "kind", "sort",
    "that", "those", "these", "this", "whose", "who", "which", "ones", "their", "there", "are",
    "is", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
    "with", "from", "in", "on", "at", "by", "as", "about", "what", "whats", "and", "or", "then",
    "so", "basically", "maybe", "okay", "ok", "hey", "ya", "gimme", "lemme",
}
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")

GREETING_TEXT = (
    "Hello. I am NIA, your intelligent assistant. "
    "How can I help you? You can ask me questions about the records in the database."
)


@dataclass(frozen=True)
class ParseResult:
    plan: QueryPlan | None
    status: str
    detail: str
    reasoning_calls: int = 0
    confidence: float = 0.0
    coverage: float = 0.0
    unresolved_spans: tuple[str, ...] = ()
    normalized_text: str = ""
    requires_semantic_planner: bool = False


@dataclass(frozen=True)
class ConstraintExtraction:
    predicates: tuple[Predicate, ...] = ()
    retrieval: tuple[str, bool, bool, bool] | None = None
    error: str | None = None
    consumed_spans: tuple[tuple[int, int], ...] = ()


def parse_deterministic(
    question: str,
    scope: AccessScope,
    catalog: FieldCatalog,
    *,
    conversation: ConversationContext | None = None,
    normalized_turn: NormalizedTurn | None = None,
    min_coverage: float = 1.0,
) -> ParseResult:
    """High-confidence zero-model fast path.

    Natural language that is not completely covered is expected to fall through
    to semantic normalization. The user never has to rephrase to this grammar.
    """

    turn = normalized_turn or normalize_user_turn(question)
    text = turn.normalized_text
    if not text:
        return ParseResult(None, "empty", "question is empty", normalized_text=text)

    catalog = catalog.for_scope(scope)
    if _GREETING.match(text):
        return ParseResult(
            None,
            "greeting",
            GREETING_TEXT,
            confidence=1.0,
            coverage=1.0,
            normalized_text=text,
        )

    followup = resolve_conversation_followup(turn, conversation, scope, catalog)
    if followup and followup.plan is not None:
        return ParseResult(
            followup.plan,
            "planned",
            followup.detail,
            confidence=followup.confidence,
            coverage=1.0,
            normalized_text=text,
        )

    if (
        conversation is not None
        and conversation.active_query is not None
        and re.search(r"\b(?:those|these|them|of\s+those|which\s+of)\b", text, re.IGNORECASE)
    ):
        return _fallback(text, "referential follow-up requires semantic normalization")

    if turn.unresolved_correction:
        return _fallback(text, "self-correction needs semantic normalization")
    if needs_synthesis(text):
        return _fallback(text, "question requires synthesis or explanation", status="needs_reasoning")
    if _SCHEMA_AGGREGATE_HINT.search(text):
        return _fallback(text, "schema aggregate requires typed planning")
    if (
        _COMPLEX_HINTS.search(text)
        or _NEGATION_HINTS.search(text)
        or _MULTI_ACTION_HINT.search(text)
    ):
        return _fallback(text, "complex/negated request requires semantic normalization")

    file_ids, access_error = _resolve_file_ids(text, scope, catalog)
    if access_error:
        return ParseResult(None, "access_restricted", access_error, normalized_text=text)
    if not file_ids:
        return _fallback(text, "no authorized dataset could be selected")

    schema_distinct = _schema_distinct_field(text, tuple(file_ids), catalog)
    if schema_distinct is not None:
        scoped = tuple(
            file_id for file_id in file_ids if catalog.resolve_field(file_id, schema_distinct)
        )
        if _COUNT_HINT.search(text):
            plan = build_distinct_count_plan(file_ids=scoped, field=schema_distinct)
            detail = "count distinct schema field values"
        else:
            plan = build_distinct_plan(file_ids=scoped, field=schema_distinct, catalog=catalog)
            detail = "distinct schema field values"
        return ParseResult(
            plan,
            "planned",
            detail,
            confidence=0.995,
            coverage=1.0,
            normalized_text=text,
        )

    distinct_count_match = _DISTINCT_COUNT_FIELD.search(text)
    if distinct_count_match:
        plural = " ".join(distinct_count_match.group(1).lower().split())
        distinct_field = _PLURAL_TO_FIELD.get(plural)
        scoped = tuple(
            file_id for file_id in file_ids if distinct_field and catalog.resolve_field(file_id, distinct_field)
        )
        if scoped and distinct_field:
            spans = [distinct_count_match.span(), *_dataset_spans(text, catalog)]
            coverage, unresolved = _coverage(text, spans)
            if coverage >= min_coverage and not unresolved:
                return ParseResult(
                    build_distinct_count_plan(file_ids=scoped, field=distinct_field),
                    "planned",
                    "count distinct field values",
                    confidence=0.995,
                    coverage=coverage,
                    normalized_text=text,
                )
            return _fallback(text, "distinct count contains unresolved meaning", coverage, unresolved)

    distinct_match = _DISTINCT_FIELD.search(text)
    distinct_field = _distinct_field(text)
    if distinct_field and distinct_match and not (
        _STUDENTS.search(text) and _LIMIT.search(text)
    ):
        scoped = tuple(fid for fid in file_ids if catalog.resolve_field(fid, distinct_field))
        if not scoped:
            default_id = catalog.default_people_file_id
            if default_id in scope.allowed_file_ids and catalog.resolve_field(default_id, distinct_field):
                scoped = (default_id,)
        if scoped:
            spans = [distinct_match.span(), *_dataset_spans(text, catalog)]
            coverage, unresolved = _coverage(text, spans)
            if coverage >= min_coverage and not unresolved:
                return ParseResult(
                    build_distinct_plan(file_ids=scoped, field=distinct_field, catalog=catalog),
                    "planned",
                    "distinct field values",
                    confidence=0.995,
                    coverage=coverage,
                    normalized_text=text,
                )
            return _fallback(text, "distinct request contains unresolved meaning", coverage, unresolved)

    extraction = _constraints(text, catalog, file_ids)
    if extraction.error:
        return _fallback(text, extraction.error)

    predicates = list(extraction.predicates)
    retrieval = extraction.retrieval
    if predicates:
        needed_fields = {predicate.field for predicate in predicates}
        scoped = [
            file_id
            for file_id in file_ids
            if all(catalog.resolve_field(file_id, name) for name in needed_fields)
        ]
        if not scoped:
            return _fallback(text, "selected constraints are not valid on the selected dataset")
        file_ids = scoped

    count_match = _COUNT_HINT.search(text)
    list_match = _LIST_HINT.search(text)
    wants_count = bool(count_match)
    wants_list = bool(list_match) and not wants_count
    if retrieval and not wants_count:
        wants_list = True
    group_each = bool(_ALL_DATASETS.search(text) and wants_count and not predicates and not retrieval)

    if not wants_count and not wants_list:
        return _fallback(text, "request goal is not explicit enough for deterministic planning")
    if not predicates and not retrieval and not group_each:
        subject_present = bool(_STUDENTS.search(text) or _NAMES.search(text) or _RECORDS_ONLY.search(text))
        if not subject_present:
            return _fallback(text, "request depends on missing conversational context")

    limit_match = _LIMIT.search(text)
    limit = int(limit_match.group(1)) if limit_match else (20 if wants_list or retrieval else None)

    spans: list[tuple[int, int]] = list(extraction.consumed_spans)
    for match in (count_match, list_match, limit_match, _ALL_DATASETS.search(text), _STUDENTS.search(text), _RECORDS_ONLY.search(text), _NAMES.search(text)):
        if match:
            spans.append(match.span())
    spans.extend(_dataset_spans(text, catalog))
    coverage, unresolved = _coverage(text, spans)
    if coverage < min_coverage or unresolved:
        return _fallback(
            text,
            "deterministic interpretation left semantic content unresolved",
            coverage,
            unresolved,
        )

    goal = "exact_count" if wants_count else "list"
    plan = build_structured_plan(
        file_ids=tuple(file_ids),
        predicates=predicates,
        retrieval=retrieval,
        goal=goal,
        group_each=group_each,
        limit=limit,
        catalog=catalog,
    )
    return ParseResult(
        plan,
        "planned",
        "high-confidence deterministic plan",
        reasoning_calls=0,
        confidence=0.995,
        coverage=coverage,
        normalized_text=text,
    )


def needs_synthesis(text: str) -> bool:
    return bool(SYNTHESIS_HINTS.search(text))


def _fallback(
    text: str,
    detail: str,
    coverage: float = 0.0,
    unresolved: tuple[str, ...] = (),
    *,
    status: str = "needs_planning",
) -> ParseResult:
    return ParseResult(
        None,
        status,
        detail,
        confidence=0.0,
        coverage=coverage,
        unresolved_spans=unresolved,
        normalized_text=text,
        requires_semantic_planner=True,
    )


def _coverage(text: str, consumed_spans: list[tuple[int, int]]) -> tuple[float, tuple[str, ...]]:
    tokens = list(_TOKEN.finditer(text))
    meaningful = [
        token
        for token in tokens
        if token.group(0).lower().replace("'", "") not in _SAFE_COVERAGE_WORDS
    ]
    if not meaningful:
        return 1.0, ()

    unresolved: list[str] = []
    covered = 0
    for token in meaningful:
        if any(start <= token.start() and token.end() <= end for start, end in consumed_spans):
            covered += 1
        else:
            unresolved.append(token.group(0))
    return covered / len(meaningful), tuple(dict.fromkeys(unresolved))


# Nouns that merely label an already-named dataset ("the confirmed deaths
# dataset"). They are consumed only when they trail a matched dataset alias, so
# a bare "how many datasets are there?" still abstains instead of mis-planning.
_DATASET_QUALIFIER = re.compile(r"\s*(?:dataset|datasets|file|table|list)\b", re.IGNORECASE)

# "this list", "the dataset", "in these tables" - a self-reference to the data
# already in scope. It carries no extra constraint, so it must not count as
# unresolved meaning and block an otherwise complete deterministic plan.
_DATASET_SELF_REFERENCE = re.compile(
    r"\b(?:in|from|of|within)?\s*(?:this|that|these|those|the|current)\s+"
    r"(?:list|lists|dataset|datasets|data\s?set|table|tables|file|files|collection)\b",
    re.IGNORECASE,
)


def _dataset_spans(text: str, catalog: FieldCatalog) -> list[tuple[int, int]]:
    lowered = text.lower()
    spans: list[tuple[int, int]] = []
    for dataset in catalog.datasets:
        for alias in dataset_match_labels(dataset):
            alias = (alias or "").strip().lower()
            if not alias:
                continue
            start = 0
            while True:
                index = lowered.find(alias, start)
                if index < 0:
                    break
                end = index + len(alias)
                trailing = _DATASET_QUALIFIER.match(text, end)
                if trailing is not None:
                    end = trailing.end()
                spans.append((index, end))
                start = index + len(alias)
    spans.extend(match.span() for match in _DATASET_SELF_REFERENCE.finditer(text))
    return spans


def _distinct_field(text: str) -> str | None:
    match = _DISTINCT_FIELD.search(text)
    if not match or _COMMUNITY.search(text):
        return None
    plural = " ".join(match.group(1).lower().split())
    return _PLURAL_TO_FIELD.get(plural)


def _schema_distinct_field(
    text: str,
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> str | None:
    """Resolve explicit distinct/list-value requests from catalog aliases."""

    if _SCHEMA_DISTINCT_INTENT.search(text) is None:
        return None
    lowered = " ".join(text.lower().split())
    candidates: list[tuple[int, str]] = []
    for file_id in file_ids:
        for field in catalog.fields_for(file_id):
            if field.semantic_field == "student_name":
                continue
            if not field.aggregatable and field.semantic_type not in {"entity", "text", "date", "boolean"}:
                continue
            labels = {field.human_label, field.semantic_field.replace("_", " "), *field.aliases}
            for label in labels:
                normalized = " ".join(label.lower().split())
                variants = {normalized}
                if normalized.endswith("y"):
                    variants.add(f"{normalized[:-1]}ies")
                elif not normalized.endswith("s"):
                    variants.add(f"{normalized}s")
                for variant in variants:
                    if variant and re.search(rf"\b{re.escape(variant)}\b", lowered):
                        candidates.append((len(variant), field.semantic_field))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best = candidates[0][1]
    return best if all(item[1] == best or item[0] < candidates[0][0] for item in candidates) else None


def build_distinct_plan(
    *,
    file_ids: tuple[int, ...],
    field: str,
    catalog: FieldCatalog,
) -> QueryPlan:
    steps = [
        PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records"),
        PlanStep(id="result", op=PlanOp.GROUP_BY, input="scope_current", fields=[field]),
    ]
    return QueryPlan(
        scope=PlanScope(file_ids=tuple(file_ids), version_mode="current", authorized_only=True),
        goals=["distinct_values"],
        steps=steps,
        planner_type="deterministic",
    )


def build_distinct_count_plan(*, file_ids: tuple[int, ...], field: str) -> QueryPlan:
    return QueryPlan(
        scope=PlanScope(file_ids=file_ids, version_mode="current", authorized_only=True),
        goals=["count"],
        steps=[
            PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records"),
            PlanStep(
                id="result",
                op=PlanOp.COUNT_DISTINCT,
                input="scope_current",
                fields=[field],
            ),
        ],
        planner_type="deterministic",
    )


def build_structured_plan(
    *,
    file_ids: tuple[int, ...],
    predicates: list[Predicate],
    retrieval: tuple[str, bool, bool, bool] | None,
    goal: str,
    catalog: FieldCatalog,
    group_each: bool = False,
    limit: int | None = None,
) -> QueryPlan:
    wants_count = goal == "exact_count"
    steps: list[PlanStep] = [
        PlanStep(id="scope_current", op=PlanOp.USE_CURRENT_VERSION, input="current_records"),
    ]
    current: str | list[str] = "scope_current"
    if predicates:
        steps.append(
            PlanStep(
                id="matches",
                op=PlanOp.FILTER,
                input=current if isinstance(current, str) else current[0],
                where=predicates,
            )
        )
        current = "matches"
    if retrieval:
        query_text, want_exact, want_fts, want_fuzzy = retrieval
        retrieval_ids: list[str] = []
        if want_exact:
            steps.append(PlanStep(id="exact", op=PlanOp.EXACT_LOOKUP, input=current, query=query_text))
            retrieval_ids.append("exact")
        if want_fts:
            steps.append(PlanStep(id="fts", op=PlanOp.FULL_TEXT_SEARCH, input=current, query=query_text))
            retrieval_ids.append("fts")
        if want_fuzzy:
            steps.append(PlanStep(id="fuzzy", op=PlanOp.FUZZY_SEARCH, input=current, query=query_text))
            retrieval_ids.append("fuzzy")
        if len(retrieval_ids) > 1:
            steps.append(PlanStep(id="fused", op=PlanOp.UNION, input=retrieval_ids))
            steps.append(PlanStep(id="unique", op=PlanOp.DEDUPLICATE, input="fused"))
            current = "unique"
        else:
            current = retrieval_ids[0] if retrieval_ids else current
    if group_each:
        steps.append(PlanStep(id="by_dataset", op=PlanOp.GROUP_BY, input=current, fields=["file_id"]))
        steps.append(PlanStep(id="result", op=PlanOp.COUNT, input="by_dataset"))
    elif wants_count:
        steps.append(PlanStep(id="result", op=PlanOp.COUNT, input=current))
    else:
        if limit:
            steps.append(PlanStep(id="limited", op=PlanOp.LIMIT, input=current, limit=min(limit, 50)))
            current = "limited"
        project_fields = ["student_name"]
        if all(catalog.resolve_field(file_id, "community") for file_id in file_ids):
            project_fields.append("community")
        steps.append(PlanStep(id="result", op=PlanOp.PROJECT, input=current, fields=project_fields))
        if retrieval:
            evidence_fields = [
                name
                for name in ("student_name", "community", "cause_of_death", "notes")
                if all(catalog.resolve_field(file_id, name) for file_id in file_ids)
            ]
            if evidence_fields:
                steps.append(
                    PlanStep(
                        id="evidence",
                        op=PlanOp.GET_EVIDENCE,
                        input="result",
                        fields=evidence_fields,
                        limit=8,
                    )
                )
    return QueryPlan(
        scope=PlanScope(file_ids=tuple(file_ids), version_mode="current", authorized_only=True),
        goals=[goal],
        steps=steps,
        planner_type="deterministic",
    )


def _resolve_file_ids(
    text: str,
    scope: AccessScope,
    catalog: FieldCatalog,
) -> tuple[list[int], str | None]:
    if _ALL_DATASETS.search(text):
        return [file_id for file_id in scope.allowed_file_ids if catalog.dataset(file_id) is not None], None
    named = [file_id for file_id in catalog.resolve_dataset_ids(text) if file_id in scope.allowed_file_ids]
    if named:
        return named, None
    if _RECORDS_ONLY.search(text) and not _STUDENTS.search(text):
        return [file_id for file_id in scope.allowed_file_ids if catalog.dataset(file_id) is not None], None
    default_id = catalog.default_people_file_id
    if default_id in scope.allowed_file_ids and (_STUDENTS.search(text) or _COUNT_HINT.search(text)):
        return [default_id], None
    if _STUDENTS.search(text) and default_id not in scope.allowed_file_ids:
        return [], "the default student dataset is not in the current access scope"
    visible = [file_id for file_id in scope.allowed_file_ids if catalog.dataset(file_id) is not None]
    return visible[:1], None


def _constraints(
    text: str,
    catalog: FieldCatalog,
    file_ids: list[int],
) -> ConstraintExtraction:
    found: list[Predicate] = []
    retrieval: tuple[str, bool, bool, bool] | None = None
    spans: list[tuple[int, int]] = []

    prefix_match = next((pattern.search(text) for pattern in _PREFIX_PATTERNS if pattern.search(text)), None)
    if prefix_match:
        found.append(
            Predicate(
                field="student_name",
                operator=FilterOperator.STARTS_WITH,
                value=prefix_match.group(1).upper(),
            )
        )
        spans.append(prefix_match.span())

    community = _COMMUNITY.search(text)
    if community and community.group(1).lower() not in {"the", "each", "all"}:
        label = community.group(1).strip()
        resolved = resolve_entity(label)
        spans.append(community.span())
        if resolved and not resolved.ambiguous:
            found.append(Predicate(field="community", operator=FilterOperator.EQUALS, value=resolved.canonical))
        else:
            retrieval = (label, False, False, True)

    deceased = _DECEASED.search(text)
    if deceased:
        found.append(Predicate(field="deceased_status", operator=FilterOperator.IS_TRUE))
        spans.append(deceased.span())

    year = _YEAR.search(text)
    death_year = bool(year) and bool(deceased) and not _ADMITTED.search(text) and not _DISCHARGED.search(text)
    if death_year:
        has_death_date = all(catalog.resolve_field(file_id, "death_date") for file_id in file_ids)
        if not has_death_date:
            return ConstraintExtraction(
                predicates=tuple(found),
                retrieval=retrieval,
                error="no death_date field for this dataset; cannot apply a year filter to the wrong date",
                consumed_spans=tuple(spans),
            )
        operator = (
            FilterOperator.BEFORE
            if _BEFORE.search(text)
            else FilterOperator.AFTER
            if _AFTER.search(text)
            else FilterOperator.YEAR_EQUALS
        )
        found.append(Predicate(field="death_date", operator=operator, value=int(year.group(1))))
        spans.append(year.span())
        for match in (_BEFORE.search(text), _AFTER.search(text)):
            if match:
                spans.append(match.span())
    elif year and (_ADMITTED.search(text) or _DISCHARGED.search(text) or _BEFORE.search(text) or _AFTER.search(text)):
        admitted = _ADMITTED.search(text)
        discharged = _DISCHARGED.search(text)
        before = _BEFORE.search(text)
        after = _AFTER.search(text)
        if not admitted and not discharged:
            return ConstraintExtraction(
                predicates=tuple(found),
                retrieval=retrieval,
                error="a year relation was supplied without a high-confidence date field",
                consumed_spans=tuple(spans),
            )
        field = "discharged_date" if discharged else "admitted_date"
        operator = FilterOperator.BEFORE if before else FilterOperator.AFTER if after else FilterOperator.YEAR_EQUALS
        found.append(Predicate(field=field, operator=operator, value=int(year.group(1))))
        spans.append(year.span())
        for match in (admitted, discharged, before, after):
            if match:
                spans.append(match.span())

    mention = _MENTION.search(text)
    named = _NAMED.search(text)
    if mention:
        retrieval = (mention.group(1).strip(), False, True, True)
        spans.append(mention.span())
    elif named:
        retrieval = (named.group(1).strip(), True, False, True)
        spans.append(named.span())

    return ConstraintExtraction(
        predicates=tuple(found),
        retrieval=retrieval,
        consumed_spans=tuple(spans),
    )
