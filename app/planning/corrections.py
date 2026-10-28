"""Recognising a researcher pasting back what an answer should have been.

Testers check NIA against the records by hand, and when an answer is wrong they paste the
right one back: a sentence saying so and a table of what they found. Read as a new question,
that message is a wall of names and numbers with no request in it. The planner copied it into
its own fields until the schema refused the length, or restated its figures in a reply that
the response policy rightly blocked, so the researcher got "couldn't produce a usable plan"
for the most useful message they had written.

What such a message asks for is plain: check the earlier answer again. So it is recognised
here by its shape -- a corrective statement accompanied by a table -- and the turn service
asks the earlier question again rather than planning the paste.

The shape is deliberately narrow. A table alone could be a pasted list the researcher wants
looked up, and "should be" alone is ordinary conversation; together, after an answer, they
are a correction.
"""

from __future__ import annotations

import re

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# A markdown delimiter row: every cell hyphens, optionally colon-aligned, one hyphen minimum.
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)+\|?\s*$")
_CORRECTIVE = re.compile(
    r"\b(?:"
    r"(?:the\s+)?(?:correct|right|real|actual)\s+(?:answer|number|count)"
    r"|(?:answer|number|count|result)\s+should\s+(?:be|have\s+been)"
    r"|should\s+(?:be|have\s+been)\s*:"
    r"|i\s+(?:found|counted|checked)"
    r"|you\s+(?:missed|left\s+out|should\s+have)"
    r"|(?:that|this|it)\s+(?:is|was|'s)\s+(?:wrong|incorrect|not\s+right)"
    r")\b",
    re.IGNORECASE,
)
_MIN_LENGTH = 120


def is_pasted_correction(text: str) -> bool:
    """Whether a message pastes back a corrected answer rather than asking something new."""
    body = text or ""
    if len(body) < _MIN_LENGTH:
        return False
    lines = [line for line in body.splitlines() if line.strip()]
    table_rows = sum(1 for line in lines if _TABLE_ROW.match(line))
    has_table = table_rows >= 3 and any(_TABLE_RULE.match(line) for line in lines)
    return has_table and bool(_CORRECTIVE.search(body))
