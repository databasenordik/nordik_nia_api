"""Deterministic corrections applied to every compiled turn.

Three guarantees must not depend on the language model getting it right:

1. A question that asks for a calculation is never answered with an arbitrary page
   of rows. "How many", "what percentage", and "which has the most" become count,
   percentage, and rank goals even when the compiler emitted goal="list".
2. A domain term the corpus spells several ways is matched against the whole family.
   The families live in the field registry as `validation_rules.value_synonyms`, so
   this module carries no dataset vocabulary of its own.
3. "First name" / "last name" reach a name part even on a dataset that stores only a
   single display-name field.
"""

from __future__ import annotations

import re

from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.plan_coverage import _label_pattern, normalize_question
from app.planning.turn_schema import FilterSpec, QueryAction, TurnPlan

_COUNT_SHAPE = re.compile(
    r"\bhow\s+many\b|\bnumber\s+of\b|\bcount\s+of\b|\btotal\s+number\b|\bhow\s+much\b",
    re.IGNORECASE,
)
_PERCENT_SHAPE = re.compile(
    r"\bwhat\s+percent(?:age)?\b|\bpercentage\s+of\b|\bwhat\s+(?:share|proportion)\b",
    re.IGNORECASE,
)
_SUPERLATIVE_SHAPE = re.compile(
    r"\b(?:most|largest|highest|greatest|biggest|fewest|least|smallest|lowest)\b"
    r"|\btop\s+\d+\b|\brank(?:ed|ing)?\b",
    re.IGNORECASE,
)
_STATS_SHAPE = re.compile(
    r"\b(?:average|mean|median|typical|youngest|oldest|earliest|latest"
    r"|longest|shortest|minimum|maximum|on\s+average)\b",
    re.IGNORECASE,
)
_DUPLICATE_SHAPE = re.compile(
    r"\b(?:more\s+than\s+once|duplicates?|duplicated|same\s+(?:full\s+)?name"
    r"|assigned\s+to\s+more\s+than|appears?\s+(?:more\s+than|twice)|shared\s+by)\b",
    re.IGNORECASE,
)
_EXPLICIT_LIST = re.compile(
    r"\blist\s+(?:them|those|all|every|each)\b|\bname\s+(?:them|all|every)\b"
    r"|\bone\s+by\s+one\b|\bshow\s+me\s+(?:them|the\s+list)\b",
    re.IGNORECASE,
)
# "the 20 communities with the highest number" puts the count before the noun,
# so a superlative question takes its first bare integer as the ranking depth.
# "what X are listed / recorded" wants the distinct values, and "in each X" wants
# a grouped breakdown; both were answered with a page of rows.
_DISTINCT_SHAPE = re.compile(
    r"\b(?:what|which)\b[^?]{0,40}\b(?:are|were)\s+(?:listed|recorded|present)\b"
    r"|\b(?:distinct|different|unique)\s+\w+\b",
    re.IGNORECASE,
)
_EACH_SHAPE = re.compile(
    r"\b(?:in|for|during|by)\s+each\b|\beach\s+(?:decade|year|community|cause|age)\b"
    r"|\bper\s+(?:decade|year|community|cause)\b|\bhow many\s+.{0,40}\beach\b",
    re.IGNORECASE,
)
# "how many X in each Y", "per decade", "at each age" all want a grouped
# breakdown; without a group they collapse into one meaningless total.
_BUCKET_WORDS = {
    "decade": "decade",
    "decades": "decade",
    "year": "year",
    "years": "year",
    "yearly": "year",
    "annually": "year",
}
_MISSING_BUCKET = re.compile(
    r"\bor (?:have |has )?(?:no|none|nothing|blank|missing)\b|\bnot recorded\b",
    re.IGNORECASE,
)
_TOP_N = re.compile(r"\b(\d{1,3})\b")
_FIRST_NAME = re.compile(r"\bfirst\s+names?\b|\bgiven\s+names?\b", re.IGNORECASE)
_LAST_NAME = re.compile(r"\blast\s+names?\b|\bsurnames?\b|\bfamily\s+names?\b", re.IGNORECASE)

_NEGATIVE_OPERATORS = {"NOT_CONTAINS": "NOT_CONTAINS_ANY", "NOT_EQUALS": "NOT_CONTAINS_ANY"}
_POSITIVE_OPERATORS = {"CONTAINS": "CONTAINS_ANY", "EQUALS": "CONTAINS_ANY"}


def is_analytic_question(question: str) -> bool:
    """Whether the question asks for a computed value rather than a page of records.

    The legacy deterministic and schema fast paths predate ranking, percentages,
    statistics, and duplicate detection, and they answer these shapes with a count or
    a first page. They must abstain so the compiler can plan the real computation.
    """
    return bool(
        _PERCENT_SHAPE.search(question)
        or _SUPERLATIVE_SHAPE.search(question)
        or _STATS_SHAPE.search(question)
        or _DUPLICATE_SHAPE.search(question)
        or _DISTINCT_SHAPE.search(question)
        or _EACH_SHAPE.search(question)
        or _MISSING_BUCKET.search(question)
    )


def apply_analytic_guards(turn: TurnPlan, question: str, catalog: FieldCatalog) -> TurnPlan:
    """Rewrite compiled query actions so a calculation stays a calculation."""
    if not turn.query_actions():
        return turn
    actions = [
        _guard_action(action, question, catalog) if isinstance(action, QueryAction) else action
        for action in turn.actions
    ]
    if actions == list(turn.actions):
        return turn
    return turn.model_copy(update={"actions": actions})


def _guard_action(action: QueryAction, question: str, catalog: FieldCatalog) -> QueryAction:
    updates: dict[str, object] = {}
    goal = action.goal

    if goal == "percentage" and not _PERCENT_SHAPE.search(question):
        # Nothing in the question asked for a share. Reporting one hides the count
        # the researcher actually asked for.
        goal = "rank" if action.group_by else "count"
    if _PERCENT_SHAPE.search(question) and goal in {"list", "count"}:
        goal = "percentage"
    elif _DUPLICATE_SHAPE.search(question) and action.group_by and goal in {"list", "count"}:
        goal = "duplicates"
    elif _SUPERLATIVE_SHAPE.search(question) and action.group_by and goal in {"list", "count"}:
        goal = "rank"
    elif _STATS_SHAPE.search(question) and goal == "list" and _stats_target(action, catalog):
        goal = "stats"
    elif (
        _COUNT_SHAPE.search(question)
        and goal == "list"
        and not _EXPLICIT_LIST.search(question)
        and not action.requested_fields
    ):
        goal = "count"
    if goal != action.goal:
        updates["goal"] = goal
        if goal == "stats" and action.aggregate is None:
            target = _stats_target(action, catalog)
            if target is not None:
                updates["aggregate"] = {"function": "stats", "field": target}

    if goal == "count" and action.group_by:
        # A count that names something to group by is a breakdown, and the count goal
        # would have discarded the grouping and returned one meaningless total.
        goal = "rank"
        updates["goal"] = goal
        if action.group_value_part is None:
            part = _bucket_part(question)
            if part is not None:
                updates["group_value_part"] = part

    if goal in {"count", "list"} and not action.group_by and (
        _EACH_SHAPE.search(question) or _MISSING_BUCKET.search(question)
    ):
        # "how many students were admitted during each decade" is a breakdown, not a
        # single total; find the field it is asking to break down by.
        grouping = _grouping_field(action, question, catalog)
        if grouping is not None:
            goal = "rank"
            updates["goal"] = goal
            updates["group_by"] = [grouping]
            part = _bucket_part(question)
            if part is not None:
                updates["group_value_part"] = part
            if _MISSING_BUCKET.search(question):
                updates["include_missing"] = True
            remaining = [item for item in action.filters if item.field != grouping]
            if len(remaining) != len(action.filters):
                # "deceased, not deceased, unknown, or none" lists the buckets; a
                # filter on that same field would leave exactly one of them.
                updates["filters"] = remaining

    if goal in {"rank", "distinct"} and action.top_n is None and _SUPERLATIVE_SHAPE.search(question):
        requested = _TOP_N.search(question)
        if requested is not None and 2 <= int(requested.group(1)) <= 500:
            updates["top_n"] = int(requested.group(1))

    updates.update(_name_part_update(action, question, catalog))
    updates.update(_band_number_update(action, question, catalog))

    filters = [_expand_synonyms(item, action.datasets, catalog) for item in action.filters]
    if filters != list(action.filters):
        updates["filters"] = filters
    updates.update(_merge_same_field_alternatives(action, updates))
    updates.update(_correct_presence_polarity(action, question, updates, catalog))

    updates.update(_breakdown_bucket_updates(action, question, updates))
    updates.update(_person_summary_updates(action, question, updates))
    updates.update(_drop_unresolvable_fields(action, updates, catalog))
    updates.update(_sanitize_goal_requirements(action, updates, question, catalog))
    return action.model_copy(update=updates) if updates else action

def _merge_same_field_alternatives(
    action: QueryAction,
    updates: dict[str, object],
) -> dict[str, object]:
    """Several positive matches on one field are alternatives, not a conjunction.

    "died of TB, consumption, phthisis or scrofula" arrives as four CONTAINS filters
    on cause_of_death. ANDed, no record can satisfy them and the answer is zero.
    """
    filters = list(updates.get("filters", action.filters) or [])
    if len(filters) < 2 or getattr(action, "filter_logic", "and") != "and":
        return {}
    positive = {"CONTAINS", "EQUALS", "CONTAINS_ANY"}
    by_field: dict[str, list[FilterSpec]] = {}
    for item in filters:
        if item.operator in positive:
            by_field.setdefault(item.field, []).append(item)
    merged: list[FilterSpec] = []
    consumed: set[int] = set()
    for field, group in by_field.items():
        if len(group) < 2:
            continue
        terms: list[str] = []
        for item in group:
            values = item.value if isinstance(item.value, list) else [item.value]
            for value in values:
                text = str(value).strip()
                if text and text not in terms:
                    terms.append(text)
            consumed.add(id(item))
        merged.append(
            FilterSpec(field=field, operator="CONTAINS_ANY", value=terms)
        )
    if not merged:
        return {}
    kept = [item for item in filters if id(item) not in consumed]
    return {"filters": [*kept, *merged]}


def _correct_presence_polarity(
    action: QueryAction,
    question: str,
    updates: dict[str, object],
    catalog: FieldCatalog,
) -> dict[str, object]:
    """Flip "has a recorded X" to "has no X" when the question said no.

    "students with no First Nation recorded" and "missing a student number" were
    compiled as IS_KNOWN, which answers the exact opposite question.
    """
    filters = list(updates.get("filters", action.filters) or [])
    if not filters:
        return {}
    lowered = normalize_question(question)
    rewritten: list[FilterSpec] = []
    changed = False
    for item in filters:
        if item.operator != "IS_KNOWN":
            rewritten.append(item)
            continue
        if _negated_before(lowered, item.field, action.datasets, catalog):
            rewritten.append(item.model_copy(update={"operator": "IS_UNKNOWN"}))
            changed = True
        else:
            rewritten.append(item)
    return {"filters": rewritten} if changed else {}


def _negated_before(
    lowered: str,
    field_name: str,
    datasets: list[int],
    catalog: FieldCatalog,
) -> bool:
    spec = _spec(field_name, datasets, catalog)
    if spec is None:
        return False
    labels = {spec.human_label, field_name.replace("_", " "), *spec.aliases}
    for label in labels:
        normalized = " ".join(label.lower().split())
        if len(normalized) < 4:
            continue
        for match in re.finditer(_label_pattern(normalized), lowered):
            window = lowered[max(0, match.start() - 30) : match.start()]
            if _NEGATION_CUE.search(window):
                return True
    return False


def _breakdown_bucket_updates(
    action: QueryAction,
    question: str,
    updates: dict[str, object],
) -> dict[str, object]:
    """A breakdown enumerates its buckets, so it must not be filtered to one of them.

    "deceased, not deceased, unknown, or no status" names four buckets of one field.
    Keeping a deceased_status filter alongside the grouping left a single bucket and
    reported it as the whole answer.
    """
    goal = str(updates.get("goal", action.goal))
    if goal not in {"rank", "distinct"}:
        return {}
    group_by = list(updates.get("group_by", action.group_by) or [])
    if not group_by or not _MISSING_BUCKET.search(question):
        return {}
    cleaned: dict[str, object] = {"include_missing": True}
    filters = list(updates.get("filters", action.filters) or [])
    remaining = [item for item in filters if item.field != group_by[0]]
    if len(remaining) != len(filters):
        cleaned["filters"] = remaining
    return cleaned


def _person_summary_updates(
    action: QueryAction,
    question: str,
    updates: dict[str, object],
) -> dict[str, object]:
    """Turn "what does the database tell us about <person>" into a dossier.

    A dossier projects every authorized field of the matching records, so the
    synthesis step sees the family, admission, death, and burial values instead of a
    four-field preview that reads as "the database has no records".
    """
    from app.execution.turn import is_person_summary

    if not is_person_summary(question):
        return {}
    goal = str(updates.get("goal", action.goal))
    if goal in {"count", "percentage", "rank", "stats", "duplicates", "interval"}:
        return {}
    search_text = action.search_text or _name_from_filters(action)
    if not search_text:
        return {}
    return {"goal": "dossier", "search_text": search_text, "filters": []}


def _name_from_filters(action: QueryAction) -> str | None:
    for item in action.filters:
        if item.field in {"student_name", "first_name", "last_name", "indigenous_name"}:
            if isinstance(item.value, str) and item.value.strip():
                return item.value.strip()
            if isinstance(item.value, list) and item.value:
                return str(item.value[0])
    return None

# Life-event order, used to put an interval's two dates the right way round.
# "died shortest after being admitted" names death first but measures from admission.
_NEGATION_CUE = re.compile(
    r"\b(?:no|not|without|missing|lack|lacks|lacking|blank|absent|never)\b", re.IGNORECASE
)
_SHORTEST_SHAPE = re.compile(
    r"\b(?:shortest|soonest|least|quickest|fewest|earliest)\b", re.IGNORECASE
)
_ELAPSED_SHAPE = re.compile(
    r"\bhow long\b|\b(?:days|years|months|time)\s+(?:between|after|before)\b"
    r"|\blongest\s+(?:period|time|amount)\b|\bshortest\s+(?:period|time|amount)\b"
    r"|\bhow much time\b|\bduration\b|\blived the longest\b|\bstayed\b",
    re.IGNORECASE,
)
_EVENT_ORDER = ("birth", "admission", "admitted", "discharge", "death", "burial", "registration")
# English morphology, not dataset vocabulary: the question says "died" where the
# field is named death, "buried" where the field is named burial.
_EVENT_WORDS = {
    "birth": ("birth", "born"),
    "admission": ("admission", "admitted", "admit", "entered", "enrolled"),
    "admitted": ("admission", "admitted", "admit", "entered", "enrolled"),
    "discharge": ("discharge", "discharged", "left"),
    "death": ("death", "died", "dying", "dies"),
    "burial": ("burial", "buried", "bury"),
    "registration": ("registration", "registered"),
}


def _event_rank(semantic_field: str) -> int | None:
    for index, event in enumerate(_EVENT_ORDER):
        if event in semantic_field:
            return index
    return None


def _mentions_event(question: str, semantic_field: str) -> bool:
    lowered = question.lower()
    for event, words in _EVENT_WORDS.items():
        if event not in semantic_field:
            continue
        if any(re.search(rf"\b{word}\w*\b", lowered) for word in words):
            return True
    return False


def _date_fields_by_rank(action: QueryAction, catalog: FieldCatalog) -> dict[int, str]:
    """One date field per life event, preferring the most directly named one."""
    datasets = list(action.datasets) or [catalog.default_people_file_id]
    best: dict[int, str] = {}
    for file_id in datasets:
        for spec in catalog.fields_for(file_id):
            if spec.semantic_type != "date":
                continue
            rank = _event_rank(spec.semantic_field)
            if rank is None:
                continue
            current = best.get(rank)
            if current is None or len(spec.semantic_field) < len(current):
                best[rank] = spec.semantic_field
    return best


def _infer_interval_fields(
    action: QueryAction,
    question: str,
    catalog: FieldCatalog,
) -> tuple[str, str] | None:
    """The two date fields an elapsed-time question is measuring between.

    Two named events bound the span directly. One named event is paired with the
    admission that precedes it, or with the death that follows it. With none named,
    "how long were they there" means admission to discharge.
    """
    by_rank = _date_fields_by_rank(action, catalog)
    if len(by_rank) < 2:
        return None
    named = sorted(rank for rank, name in by_rank.items() if _mentions_event(question, name))
    admission = next((rank for rank in sorted(by_rank) if rank in {1, 2}), None)
    death = next((rank for rank in sorted(by_rank) if rank >= 4), None)
    if len(named) >= 2:
        return by_rank[named[0]], by_rank[named[-1]]
    if len(named) == 1:
        only = named[0]
        early = admission if admission is not None else min(by_rank)
        partner = death if only <= early else admission
        if partner is None or partner == only:
            # Prefer the next event forward in the record's life; only look backwards
            # when nothing follows the one the question named.
            later = [rank for rank in sorted(by_rank) if rank > only]
            earlier = [rank for rank in sorted(by_rank, reverse=True) if rank < only]
            partner = later[0] if later else (earlier[0] if earlier else None)
        if partner is None:
            return None
        low, high = sorted((only, partner))
        return by_rank[low], by_rank[high]
    discharge = by_rank.get(3)
    if admission is not None and discharge is not None:
        return by_rank[admission], discharge
    ordered = sorted(by_rank)
    return by_rank[ordered[0]], by_rank[ordered[-1]]


def _sanitize_goal_requirements(
    action: QueryAction,
    updates: dict[str, object],
    question: str,
    catalog: FieldCatalog,
) -> dict[str, object]:
    """Fill in or stand down goals the compiler left underspecified.

    An aggregate with no field, or an interval with no dates, previously failed
    validation and turned the whole turn into "I wasn't able to answer that".
    """
    cleaned: dict[str, object] = {}
    goal = str(updates.get("goal", action.goal))
    aggregate = updates.get("aggregate", action.aggregate)
    function = None
    if isinstance(aggregate, dict):
        function = aggregate.get("function")
        agg_field = aggregate.get("field")
    elif aggregate is not None:
        function = aggregate.function
        agg_field = aggregate.field
    else:
        agg_field = None
    if function and function != "count" and not agg_field:
        target = _stats_target(action, catalog)
        if target is not None:
            cleaned["aggregate"] = {"function": function, "field": target}
        else:
            cleaned["aggregate"] = None
            if goal in {"aggregate", "stats"}:
                cleaned["goal"] = "count"
    if goal in {"aggregate", "stats", "count", "list"} and _ELAPSED_SHAPE.search(question):
        # "how long ...?" is an elapsed-time question whatever goal the compiler chose.
        if _infer_interval_fields(action, question, catalog) is not None:
            goal = "interval"
            cleaned["goal"] = goal
            cleaned.pop("aggregate", None)
    if goal == "interval":
        start = updates.get("interval_start", action.interval_start)
        end = updates.get("interval_end", action.interval_end)
        if not (start and end):
            inferred = _infer_interval_fields(action, question, catalog)
            if inferred is not None:
                cleaned["interval_start"], cleaned["interval_end"] = inferred
            else:
                cleaned["goal"] = "list"
    if str(cleaned.get("goal", goal)) == "interval":
        if action.sort_direction is None:
            cleaned["sort_direction"] = (
                "asc" if _SHORTEST_SHAPE.search(question) else "desc"
            )
        if action.interval_min_days is None:
            # A negative span is a data error, not the answer to "who stayed longest".
            # The count of impossible orderings is reported separately.
            cleaned["interval_min_days"] = 0.0
    return cleaned


def _drop_unresolvable_fields(
    action: QueryAction,
    updates: dict[str, object],
    catalog: FieldCatalog,
) -> dict[str, object]:
    """Discard field references that name no catalog field.

    The compiler sometimes writes sort_by="count" for a ranking, which is implicit:
    a ranking is already ordered by count. Rejecting the whole turn over it turned an
    otherwise correct plan into "I wasn't able to answer that one accurately".
    """
    cleaned: dict[str, object] = {}
    datasets = list(action.datasets)
    for name in ("sort_by", "companion_field", "interval_start", "interval_end"):
        value = updates.get(name, getattr(action, name))
        if isinstance(value, str) and value and _spec(value, datasets, catalog) is None:
            cleaned[name] = None
    group_by = list(updates.get("group_by", action.group_by) or [])
    kept = [name for name in group_by if _spec(name, datasets, catalog) is not None]
    if kept != group_by:
        cleaned["group_by"] = kept
    return cleaned


def _bucket_part(question: str) -> str | None:
    lowered = question.lower()
    for word, part in _BUCKET_WORDS.items():
        if re.search(rf"{re.escape(word)}", lowered):
            return part
    return None


def _grouping_field(action: QueryAction, question: str, catalog: FieldCatalog) -> str | None:
    """The aggregatable field a per-group question is asking to break down by."""
    from app.planning.plan_coverage import field_mentions

    datasets = tuple(action.datasets) or (catalog.default_people_file_id,)
    # A shorter label is enough here: the dataset is already chosen, and the question
    # is only naming which of its fields to break down by.
    mentioned = field_mentions(question, catalog, datasets)
    named = {item for values in mentioned.values() for item in values}
    filtered = {item.field for item in action.filters}
    candidates = [
        name
        for name in sorted(named)
        if (spec := _spec(name, list(datasets), catalog)) is not None and spec.aggregatable
    ]
    preferred = [name for name in candidates if name not in filtered]
    for pool in (preferred, candidates):
        if len(pool) == 1:
            return pool[0]
        if pool:
            dated = [
                name
                for name in pool
                if (spec := _spec(name, list(datasets), catalog)) is not None
                and spec.semantic_type == "date"
            ]
            if _bucket_part(question) is not None and len(dated) == 1:
                return dated[0]
            return pool[0]
    return None


def _stats_target(action: QueryAction, catalog: FieldCatalog) -> str | None:
    if action.aggregate is not None and action.aggregate.field:
        return action.aggregate.field
    for candidate in (*action.requested_fields, action.sort_by):
        if not candidate:
            continue
        spec = _spec(candidate, action.datasets, catalog)
        if spec is not None and spec.semantic_type in {"number", "date"}:
            return candidate
    return None


# Matches "without hashes", "without the hash numbers", "no band numbers",
# "remove the reserve numbers". Imported by trusted_overrides so the
# deterministic follow-up rule and the direct-phrasing guard stay identical.
_STRIP_BAND_NUMBER = re.compile(
    r"\b(?:without|no|drop|remove|exclude|ignore|minus|strip|omit)\b"
    r"[^.?!]{0,30}?\b(?:(?:hash(?:es)?|band|reserve|#)(?:\s*numbers?)?|numbers?)\b",
    re.IGNORECASE,
)


def _band_number_update(
    action: QueryAction,
    question: str,
    catalog: FieldCatalog,
) -> dict[str, object]:
    """Group entity labels without their band number when the user asks.

    "List the communities without the hash numbers" is a grouping change, not
    a filter: "Chisasibi #66" and "Chisasibi" are the same community. Without
    this the request has no expressible edit and is silently dropped.
    """
    if not action.group_by:
        return {}
    if _STRIP_BAND_NUMBER.search(question) is None:
        return {}
    spec = _spec(action.group_by[0], action.datasets, catalog)
    if spec is None or spec.semantic_type != "entity":
        # Only entity labels carry band numbers. Never reshape a date or
        # number grouping, where "without numbers" means something else.
        return {}
    if action.group_value_part not in (None, "raw"):
        return {}
    return {"group_value_part": "base_name"}


def _name_part_update(
    action: QueryAction,
    question: str,
    catalog: FieldCatalog,
) -> dict[str, object]:
    """Route first/last-name wording to a real field, or to a name part as a fallback."""
    if not action.group_by:
        return {}
    head = action.group_by[0]
    spec = _spec(head, action.datasets, catalog)
    if spec is None or spec.semantic_field not in {"student_name", "first_name", "last_name"}:
        return {}
    wants_first = bool(_FIRST_NAME.search(question))
    wants_last = bool(_LAST_NAME.search(question))
    if not wants_first and not wants_last:
        return {}
    wanted = "first_name" if wants_first else "last_name"
    if _spec(wanted, action.datasets, catalog) is not None:
        updates: dict[str, object] = {}
        if spec.semantic_field != wanted:
            updates["group_by"] = [wanted, *action.group_by[1:]]
        if action.group_value_part in {"first_token", "last_token"}:
            # The dataset stores the name part properly; slicing the display name
            # would split "Mary Jane" and inflate the count.
            updates["group_value_part"] = None
        return updates
    if action.group_value_part in (None, "raw"):
        return {"group_value_part": "first_token" if wants_first else "last_token"}
    return {}


def _expand_synonyms(
    spec: FilterSpec,
    datasets: list[int],
    catalog: FieldCatalog,
) -> FilterSpec:
    field = _spec(spec.field, datasets, catalog)
    if field is None:
        return spec
    families = field.validation_rules.get("value_synonyms") if field.validation_rules else None
    if not isinstance(families, dict):
        return spec
    if isinstance(spec.value, list):
        # The compiler already chose a multi-term operator; widen each term to its
        # full family so "tuberculosis" also covers consumption, phthisis, scrofula.
        widened: list[str] = []
        for item in spec.value:
            family = _family_for(families, str(item)) or [str(item)]
            widened.extend(term for term in family if term not in widened)
        if widened == [str(item) for item in spec.value]:
            return spec
        return spec.model_copy(update={"value": widened})
    if not isinstance(spec.value, str):
        return spec
    terms = _family_for(families, spec.value)
    if terms is None or len(terms) < 2:
        return spec
    if spec.operator in _POSITIVE_OPERATORS:
        return spec.model_copy(
            update={"operator": _POSITIVE_OPERATORS[spec.operator], "value": terms}
        )
    if spec.operator in _NEGATIVE_OPERATORS:
        return spec.model_copy(
            update={"operator": _NEGATIVE_OPERATORS[spec.operator], "value": terms}
        )
    return spec


def _family_for(families: dict, value: str) -> list[str] | None:
    needle = value.strip().casefold()
    if not needle:
        return None
    for key, members in families.items():
        if isinstance(members, list) and key.casefold() == needle:
            return [str(item) for item in members]
    for members in families.values():
        if isinstance(members, list) and any(str(item).casefold() == needle for item in members):
            return [str(item) for item in members]
    return None


def _spec(name: str, datasets: list[int], catalog: FieldCatalog) -> FieldSpec | None:
    for file_id in datasets or [catalog.default_people_file_id]:
        found = catalog.resolve_field(file_id, name)
        if found is not None:
            return found
    return None
