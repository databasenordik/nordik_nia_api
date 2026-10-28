"""Answering a question from the words a record's narrative actually uses.

Researchers ask about events a list records in prose: who was transferred, who ran away, who
was sent to a sanatorium. The planner turns those into a word match on one narrative field,
and two things then go wrong that a person reading the records would never let happen.

* The words are in the record, just not in the field the planner chose. On the Master List
  "transferred" is written only in Additional Information, so matching it against Notes
  answered "0 students" -- confidently, and wrongly. When a word match finds nothing, the
  list's other narrative fields are checked, and the answer says where the words were found.

* The word is used in more than one direction. Records say "transferred from" one school as
  often as "transferred to" another, and only the second is what "transferred to other
  schools" asks about. When the question pairs the word with a direction and the records use
  it with more than one, the count follows the question's own wording -- and says so, with
  the broader count beside it, so nothing is narrowed silently.

A count that rests on matched text is only as good as the match, so a short result also shows
the sentence each record matched on. The researcher can see what the number is made of --
including a record whose "transferred to" names a sanatorium rather than a school.

Nothing here decides what a question means. The planner chose the words; this only checks
them against what the records say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.execution.match_review import Reading, reading_summary
from app.execution.name_correction import is_name_field
from app.execution.roster import fetch_rows
from app.execution.row_projection import semantic_value
from app.planning.catalog import FieldCatalog, FieldSpec
from app.planning.plan_schema import FilterOperator, PlanOp, Predicate, QueryPlan
from app.security.access_scope import AccessScope
from app.value_normalization import json_object, semantic_scalar

TEXT_MATCH_OPERATORS = frozenset(
    {FilterOperator.CONTAINS, FilterOperator.CONTAINS_ANY, FilterOperator.FULL_TEXT_SEARCH}
)
# The words that give a movement its direction. A closed class of English rather than a
# vocabulary of this corpus: "transferred to" and "transferred from" are opposite events.
_DIRECTIONS = ("to", "from", "into", "onto", "towards", "toward", "back", "away", "out")
# A result this short is read record by record; a longer one is summarised, not dumped.
EXCERPT_LIMIT = 25
# Reading every match to see how a word is used is bounded; past this the count stands as asked.
_FOCUS_SCAN_LIMIT = 2000
_SNIPPET_CHARS = 220


@dataclass(frozen=True)
class Excerpt:
    name: str
    number: str
    text: str


@dataclass
class TextSearchRefinement:
    """The plan to run, and what to tell the researcher about how its words were matched."""

    plan: QueryPlan
    before: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    excerpts: list[Excerpt] = field(default_factory=list)
    number_label: str | None = None
    # The words the excerpts were matched on and the fields they were read from, so the parts
    # of the question the search did not check can be told apart from the parts it did.
    terms: list[str] = field(default_factory=list)
    field_labels: list[str] = field(default_factory=list)
    unchecked: str = ""
    readings: list[Reading] | None = None

    def records(self) -> list[tuple[str, str]]:
        """Each excerpt as plain text, for reading."""
        return [(item.name, _plain_text(item.text)) for item in self.excerpts]

    def compose(self, answer: str) -> str:
        parts = [" ".join(self.before), answer, " ".join(self.after)]
        if self.readings:
            parts.append(
                reading_summary(
                    self.unchecked, [item.name for item in self.excerpts], self.readings
                )
            )
        if self.excerpts:
            parts.append(
                "Each matching record, with the words it matched on:\n"
                + excerpt_table(
                    self.excerpts, number_label=self.number_label, readings=self.readings
                )
            )
        return "\n\n".join(part for part in parts if part and part.strip())


def narrative_fields(file_ids: tuple[int, ...], catalog: FieldCatalog) -> list[FieldSpec]:
    """Free-text fields a record's story is written in, in registry order."""
    found: dict[str, FieldSpec] = {}
    for file_id in file_ids:
        for spec in catalog.fields_for(file_id):
            if spec.semantic_field not in found and _is_narrative(spec, catalog):
                found[spec.semantic_field] = spec
    return list(found.values())


def _is_narrative(spec: FieldSpec, catalog: FieldCatalog) -> bool:
    # A field the registry lets a quote be taken from is prose; names, identifiers and
    # categories are not, even when they are stored as text.
    return (
        spec.semantic_type == "text"
        and "GET_QUOTE" in spec.allowed_operators
        and spec.exposure_policy != "internal"
        and not is_name_field(spec.semantic_field, (spec.file_id,), catalog)
    )


def _narrative_spec(
    field_name: str, file_ids: tuple[int, ...], catalog: FieldCatalog
) -> FieldSpec | None:
    for file_id in file_ids:
        spec = catalog.resolve_field(file_id, field_name)
        if spec is not None and _is_narrative(spec, catalog):
            return spec
    return None


def _text_leaf(
    predicate: Predicate, file_ids: tuple[int, ...], catalog: FieldCatalog
) -> FieldSpec | None:
    if predicate.is_group() or predicate.operator not in TEXT_MATCH_OPERATORS:
        return None
    return _narrative_spec(predicate.field, file_ids, catalog)


async def refine_text_search(
    gateway: Any,
    scope: AccessScope,
    plan: QueryPlan,
    catalog: FieldCatalog,
    question: str,
) -> TextSearchRefinement | None:
    """Check a plan's narrative word matches against the records before it runs.

    Returns None when the plan matches no narrative text, so every other question is
    untouched. Otherwise the returned plan is the one to execute -- identical to the input
    unless a match was moved to the field that holds the words, or narrowed to the direction
    the question asked about.
    """
    targets = [
        index
        for index, step in enumerate(plan.steps)
        if step.op is PlanOp.FILTER
        and any(
            _text_leaf(item, step.file_ids or plan.scope.file_ids, catalog) for item in step.where
        )
    ]
    if not targets:
        return None
    refinement = TextSearchRefinement(plan=plan)
    steps = list(plan.steps)
    for index in targets:
        step = steps[index]
        file_ids = step.file_ids or plan.scope.file_ids
        where = list(step.where)
        count = 0
        for position in range(len(where)):
            spec = _text_leaf(where[position], file_ids, catalog)
            if spec is None:
                continue
            terms = _terms(where[position].value)
            count = await gateway.count_records(scope, file_ids, where)
            if count == 0:
                widened = await _widen(gateway, scope, file_ids, where, position, spec, catalog)
                if widened is None:
                    others = [
                        item
                        for item in narrative_fields(file_ids, catalog)
                        if item.semantic_field != spec.semantic_field
                    ]
                    if others:
                        refinement.after.append(
                            f"I also checked {_join_labels(others)}: none of them mention "
                            f"{_quoted(terms)} either."
                        )
                    continue
                where[position], count, note = widened
                refinement.before.append(note)
            focused = await _focus(
                gateway, scope, file_ids, where, position, catalog, question, count
            )
            if focused is not None:
                where[position], count, note = focused
                refinement.before.append(note)
        if where != list(step.where):
            steps[index] = step.model_copy(update={"where": where})
        if 0 < count <= EXCERPT_LIMIT and _answers_with_a_number(step, plan):
            excerpts, number_label, needles, labels = await _excerpts(
                gateway, scope, file_ids, where, catalog, count
            )
            refinement.excerpts.extend(excerpts)
            refinement.number_label = refinement.number_label or number_label
            refinement.terms.extend(needles)
            refinement.field_labels.extend(labels)
    refinement.plan = plan.model_copy(update={"steps": steps})
    return refinement


async def _widen(
    gateway: Any,
    scope: AccessScope,
    file_ids: tuple[int, ...],
    where: list[Predicate],
    position: int,
    spec: FieldSpec,
    catalog: FieldCatalog,
) -> tuple[Predicate, int, str] | None:
    """Move a word match that found nothing to the narrative fields that hold the words."""
    leaf = where[position]
    operator = leaf.operator.value if leaf.operator is not None else ""
    siblings = [
        item
        for item in narrative_fields(file_ids, catalog)
        if item.semantic_field != spec.semantic_field and operator in item.allowed_operators
    ]
    if not siblings:
        return None

    def replaced(predicate: Predicate) -> list[Predicate]:
        return [*where[:position], predicate, *where[position + 1 :]]

    union = Predicate(
        op="or", items=[leaf.model_copy(update={"field": item.semantic_field}) for item in siblings]
    )
    if not await gateway.count_records(scope, file_ids, replaced(union)):
        return None
    found = [
        item
        for item in siblings
        if await gateway.count_records(
            scope, file_ids, replaced(leaf.model_copy(update={"field": item.semantic_field}))
        )
    ]
    if not found:
        return None
    replacement = (
        leaf.model_copy(update={"field": found[0].semantic_field})
        if len(found) == 1
        else Predicate(
            op="or",
            items=[leaf.model_copy(update={"field": item.semantic_field}) for item in found],
        )
    )
    count = await gateway.count_records(scope, file_ids, replaced(replacement))
    note = (
        f"No record has {_quoted(_terms(leaf.value))} in its {_lower(spec.human_label)}; "
        f"those words are written in {_join_labels(found)} instead, so I searched there."
    )
    return replacement, count, note


async def _focus(
    gateway: Any,
    scope: AccessScope,
    file_ids: tuple[int, ...],
    where: list[Predicate],
    position: int,
    catalog: FieldCatalog,
    question: str,
    total: int,
) -> tuple[Predicate, int, str] | None:
    """Follow the direction the question gives a word, when the records use it both ways."""
    predicate = where[position]
    leaves = predicate.leaves()
    terms = _terms([term for leaf in leaves for term in _terms(leaf.value)])
    if not terms or total <= 0 or total > _FOCUS_SCAN_LIMIT:
        return None
    pattern = _direction_pattern(terms)
    asked = {match.group(2).casefold(): match.group(0) for match in pattern.finditer(question)}
    if len(asked) != 1:
        return None
    direction, asked_phrase = next(iter(asked.items()))

    rows = await fetch_rows(gateway, scope, file_ids, where, total)
    records_by_direction: dict[str, int] = {}
    phrases: dict[str, dict[str, None]] = {}
    for row in rows:
        text = " ".join(_text_of(row, leaf.field, catalog) for leaf in leaves)
        used: set[str] = set()
        for match in pattern.finditer(text):
            way = match.group(2).casefold()
            used.add(way)
            phrases.setdefault(way, {})[f"{match.group(1).casefold()} {way}"] = None
        for way in used:
            records_by_direction[way] = records_by_direction.get(way, 0) + 1
    focused = records_by_direction.get(direction, 0)
    if not focused or len(records_by_direction) < 2 or focused >= total:
        return None
    wanted = list(phrases[direction])
    replacement = _phrase_predicate(leaves, wanted, file_ids, catalog)
    if replacement is None:
        return None
    count = await gateway.count_records(
        scope, file_ids, [*where[:position], replacement, *where[position + 1 :]]
    )
    if not count:
        # The database reads the phrase differently from this scan; answer as asked.
        return None
    others = sorted(
        ((way, number) for way, number in records_by_direction.items() if way != direction),
        key=lambda item: (-item[1], item[0]),
    )
    other_text = "; ".join(
        f'{number:,} say "{next(iter(phrases[way]))}"' for way, number in others
    )
    note = (
        f"{total:,} records mention {_quoted(terms)}. Your question says \"{asked_phrase}\", "
        f"so I counted the {count:,} that say {_quoted(wanted)} ({other_text})."
    )
    return replacement, count, note


def _phrase_predicate(
    leaves: list[Predicate],
    phrases: list[str],
    file_ids: tuple[int, ...],
    catalog: FieldCatalog,
) -> Predicate | None:
    items: list[Predicate] = []
    for leaf in leaves:
        spec = _narrative_spec(leaf.field, file_ids, catalog)
        if spec is None or "CONTAINS" not in spec.allowed_operators:
            return None
        if len(phrases) > 1 and "CONTAINS_ANY" in spec.allowed_operators:
            items.append(
                Predicate(field=leaf.field, operator=FilterOperator.CONTAINS_ANY, value=phrases)
            )
            continue
        items.extend(
            Predicate(field=leaf.field, operator=FilterOperator.CONTAINS, value=phrase)
            for phrase in phrases
        )
    if not items:
        return None
    return items[0] if len(items) == 1 else Predicate(op="or", items=items)


async def _excerpts(
    gateway: Any,
    scope: AccessScope,
    file_ids: tuple[int, ...],
    where: list[Predicate],
    catalog: FieldCatalog,
    count: int,
) -> tuple[list[Excerpt], str | None]:
    leaves = [
        leaf
        for predicate in where
        for leaf in predicate.leaves()
        if leaf.operator in TEXT_MATCH_OPERATORS and _narrative_spec(leaf.field, file_ids, catalog)
    ]
    needles = _terms([term for leaf in leaves for term in _terms(leaf.value)])
    rows = await gateway.list_records(
        scope, file_ids, where, limit=count, sort_field="student_name"
    )
    number_spec = next(
        (
            spec
            for file_id in file_ids
            if (spec := catalog.resolve_field(file_id, "student_number")) is not None
        ),
        None,
    )
    excerpts: list[Excerpt] = []
    for row in rows:
        snippet = next(
            (
                text
                for text in (_snippet(_text_of(row, leaf.field, catalog), needles) for leaf in leaves)
                if text
            ),
            "",
        )
        name = _plain(semantic_value(row, "student_name", catalog)) or _plain(row.get("canonical_name"))
        number = _as_recorded(row, number_spec, catalog) if number_spec else ""
        excerpts.append(Excerpt(name=name or "Unnamed record", number=number, text=snippet))
    has_numbers = number_spec is not None and any(item.number for item in excerpts)
    labels = list(
        dict.fromkeys(
            spec.human_label
            for leaf in leaves
            if (spec := _narrative_spec(leaf.field, file_ids, catalog)) is not None
        )
    )
    return (
        excerpts,
        (number_spec.human_label if has_numbers and number_spec else None),
        needles,
        labels,
    )


_FITS = {"yes": "Yes", "no": "No", "unclear": "Unclear"}


def excerpt_table(
    excerpts: list[Excerpt],
    *,
    number_label: str | None,
    readings: list[Reading] | None = None,
) -> str:
    header = [
        "Name",
        *([number_label] if number_label else []),
        "What the record says",
        *(["Fits the question"] if readings else []),
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for index, item in enumerate(excerpts):
        cells = [
            _cell(item.name),
            *([_cell(item.number or "Not recorded")] if number_label else []),
            item.text or "",
            *([_FITS[readings[index].fits]] if readings else []),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plain_text(text: str) -> str:
    return text.replace("**", "").replace("\\|", "|").strip("… ")


def _answers_with_a_number(step: Any, plan: QueryPlan) -> bool:
    goal = step.action_goal or (plan.goals[0] if plan.goals else "")
    return goal == "count"


def _direction_pattern(terms: list[str]) -> re.Pattern[str]:
    stems = sorted({re.escape(term.casefold()) for term in terms if term}, key=len, reverse=True)
    return re.compile(
        rf"\b((?:{'|'.join(stems)})\w*)\s+({'|'.join(_DIRECTIONS)})\b", re.IGNORECASE
    )


def _snippet(text: str, needles: list[str]) -> str:
    """The sentence a record matched on, with the matched words in bold."""
    flat = " ".join(str(text or "").split())
    if not flat:
        return ""
    lowered = flat.casefold()
    hits = [
        (lowered.find(needle.casefold()), needle)
        for needle in needles
        if needle and lowered.find(needle.casefold()) >= 0
    ]
    if not hits:
        return ""
    start, needle = min(hits, key=lambda item: (item[0], -len(item[1])))
    end = start + len(needle)
    left = max(flat.rfind(". ", 0, start), flat.rfind("; ", 0, start))
    left = 0 if left < 0 else left + 2
    stops = [index for index in (flat.find(". ", end), flat.find("; ", end)) if index >= 0]
    right = min(stops) + 1 if stops else len(flat)
    if right - left > _SNIPPET_CHARS:
        left = max(left, start - _SNIPPET_CHARS // 2)
        right = min(right, end + _SNIPPET_CHARS // 2)
    piece = flat[left:right]
    offset = start - left
    marked = (
        _cell(piece[:offset])
        + "**"
        + _cell(piece[offset : offset + len(needle)])
        + "**"
        + _cell(piece[offset + len(needle) :])
    )
    return ("…" if left > 0 else "") + marked.strip() + ("…" if right < len(flat) else "")


def _as_recorded(row: dict[str, Any], spec: FieldSpec, catalog: FieldCatalog) -> str:
    """A value as the list writes it -- "1959.660", not the normalized "1959-660"."""
    cells = json_object(row.get("row_data_normalized")).get("fields")
    if isinstance(cells, dict):
        for key in spec.raw_json_keys:
            text = _plain(semantic_scalar(cells.get(key)))
            if text:
                return text
    return _plain(semantic_value(row, spec.semantic_field, catalog))


def _text_of(row: dict[str, Any], field_name: str, catalog: FieldCatalog) -> str:
    value = semantic_value(row, field_name, catalog)
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def _terms(value: Any) -> list[str]:
    items = value if isinstance(value, list | tuple) else [value]
    found: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split())
        if text and text.casefold() not in {existing.casefold() for existing in found}:
            found.append(text)
    return found


def _quoted(terms: list[str]) -> str:
    return " or ".join(f'"{term}"' for term in terms)


def _lower(label: str) -> str:
    # Only a label with no other capitals is lowered, so proper nouns in a label survive.
    if len(label) > 1 and label[0].isupper() and not any(ch.isupper() for ch in label[1:]):
        return label[0].lower() + label[1:]
    return label


def _join_labels(specs: list[FieldSpec]) -> str:
    labels = [_lower(item.human_label) for item in specs]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _plain(value: Any) -> str:
    return " ".join(str(value).split()) if value not in (None, "") else ""


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")
