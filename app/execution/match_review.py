"""Reading each matched record against the part of a question a word search cannot check.

A word search found the records that say "transferred to" and counted 13. A tester reading the
same records counted 12: one of them names a sanatorium, and the question asked about "other
schools". No word match can draw that line. Reading can, and the tester drew it one record at a
time.

So when a short text-match answer lists its records and the question carries words the search
did not use, the model reads each quoted excerpt against those words and says, record by record,
whether it fits. The model classifies; code counts. The exact database count stays the answer,
and the reading is set beside it with a reason per record, so every judgement can be checked and
overruled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# What a counting question is built from rather than what it asks about. Plain English: none of
# these words comes from the records.
_SCAFFOLD = frozenset(
    """
    and but nor not for from with into onto than then also only just ever still
    are was were been being has have had does did can could would should will shall may might must
    how many much what which who whom whose when where why there here their them they its this that
    these those any some all each every either neither both
    number numbers count counts total totals please tell give find show list listed lists
    record records recorded student students child children people person persons kid kids
    entry entries row rows name names file files
    mention mentions mentioned mentioning say says said saying contain contains contained containing
    include includes included including note notes noted write writes written wrote refer refers
    referred reference references describe describes described state states stated appear appears
    appeared
    """.split()
)
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
READ_LIMIT = 25


class MatchVerdict(BaseModel):
    record: int = Field(ge=1, le=100)
    fits: Literal["yes", "no", "unclear"]
    reason: str = Field(default="", max_length=300)

    @field_validator("reason", mode="before")
    @classmethod
    def _clip_reason(cls, value: Any) -> Any:
        return value[:300] if isinstance(value, str) else value


class MatchReview(BaseModel):
    verdicts: list[MatchVerdict] = Field(default_factory=list, max_length=100)


@dataclass(frozen=True)
class Reading:
    fits: Literal["yes", "no", "unclear"]
    reason: str


MATCH_REVIEW_SYSTEM = (
    "You check research records for a researcher. Each record below was found by a word "
    "search, and you are given the part of the researcher's question that the search could "
    "not check. For every record, decide from its quoted text alone whether it fits that part: "
    '"yes" when the quoted text shows it fits, "no" when the quoted text shows it does not, and '
    '"unclear" when the quoted text does not say. You may rely on general knowledge of what '
    "words mean and what kind of place or institution a name refers to, but on nothing about "
    "the people beyond the quoted text. Give exactly one verdict for every record, using the "
    "record numbers given, each with a reason of at most twenty words that quotes the words "
    "you relied on."
)


def unchecked_words(question: str, searched: list[str], labels: list[str]) -> str:
    """The parts of a question the word search did not account for, in the order written.

    Question scaffolding ("how many ... were"), the searched words in any inflection, and the
    names of the list and the searched fields are set aside. What remains is what the search
    could not check -- "other schools" in "how many children were transferred to other
    schools?" -- with the searched words splitting it into separate parts.
    """
    searched_words = {word.casefold() for term in searched for word in _WORD.findall(term)}
    label_words = {word.casefold() for label in labels for word in _WORD.findall(label)}
    segments: list[list[tuple[str, bool]]] = [[]]
    for word in _WORD.findall(question or ""):
        folded = word.casefold()
        if any(_same_stem(folded, other) for other in searched_words):
            segments.append([])
            continue
        scaffold = len(folded) < 3 or folded in _SCAFFOLD or folded in label_words
        segments[-1].append((word, not scaffold))
    parts: list[str] = []
    for segment in segments:
        kept = [index for index, (_word, content) in enumerate(segment) if content]
        if kept:
            parts.append(" ".join(word for word, _content in segment[kept[0] : kept[-1] + 1]))
    return ", ".join(parts)


def _same_stem(word: str, other: str) -> bool:
    if word == other:
        return True
    shorter, longer = sorted((word, other), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


async def read_matches(
    reasoner: Any,
    question: str,
    unchecked: str,
    records: list[tuple[str, str]],
) -> list[Reading] | None:
    """One verdict per record, in record order, or None when the reading is not complete.

    A review that skips a record, numbers one twice or invents one is discarded whole: a
    partial reading would present a count of fits that nobody actually checked.
    """
    method = getattr(reasoner, "structured_output", None)
    if not callable(method) or not unchecked or not records or len(records) > READ_LIMIT:
        return None
    lines = [f'{index}. {name}: "{text}"' for index, (name, text) in enumerate(records, start=1)]
    user = (
        f"Researcher's question: {question}\n"
        f"Part the word search could not check: {unchecked}\n\n"
        "Records:\n" + "\n".join(lines)
    )
    review = await method(system=MATCH_REVIEW_SYSTEM, user=user, response_model=MatchReview)
    verdicts: dict[int, MatchVerdict] = {}
    for verdict in review.verdicts:
        if verdict.record > len(records) or verdict.record in verdicts:
            return None
        verdicts[verdict.record] = verdict
    if len(verdicts) != len(records):
        return None
    return [
        Reading(
            fits=verdicts[index].fits,
            reason=" ".join(verdicts[index].reason.split()).rstrip(" ."),
        )
        for index in range(1, len(records) + 1)
    ]


def reading_summary(unchecked: str, names: list[str], readings: list[Reading]) -> str:
    fits = sum(1 for item in readings if item.fits == "yes")
    misfits = [(name, item) for name, item in zip(names, readings, strict=False) if item.fits == "no"]
    unclear = sum(1 for item in readings if item.fits == "unclear")
    parts = [f"**{fits:,}** {'fits' if fits == 1 else 'fit'}"]
    if misfits:
        parts.append(f"**{len(misfits):,}** {'does' if len(misfits) == 1 else 'do'} not")
    if unclear:
        parts.append(
            f"**{unclear:,}** {'is' if unclear == 1 else 'are'} unclear from the words recorded"
        )
    text = f"Reading each record's words against “{unchecked}”: {_join(parts)}."
    if misfits:
        shown = "; ".join(
            f"{name} ({item.reason})" if item.reason else name for name, item in misfits[:5]
        )
        more = f"; and {len(misfits) - 5} more" if len(misfits) > 5 else ""
        text += f" Not fitting: {shown}{more}."
    return text + " That is a reading of the quoted words; the count above is exact."


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" and {items[-1]}"
