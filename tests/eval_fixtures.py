"""Golden eval cases for plan validity, aggregates, citations, and follow-ups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expect_status: str
    expect_reasoning_calls: int
    expect_count: int | None = None
    expect_file_ids: tuple[int, ...] = (49,)
    selected_file_id: int = 49
    expect_plan_valid: bool = True
    allowed_file_ids: tuple[int, ...] = (49, 91)
    follow_up: str | None = None
    follow_up_count: int | None = None
    follow_up_frame: str | None = None
    synthesize: bool = False
    forbidden_citation_prefixes: tuple[str, ...] = ("file:93:", "file:94:")


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="count-starts-with-s",
        question="How many students have names starting with S?",
        expect_status="answered",
        expect_reasoning_calls=0,
        expect_count=2,
        expect_file_ids=(49,),
    ),
    EvalCase(
        id="count-1910-admitted",
        question="How many students were admitted in 1910?",
        expect_status="answered",
        expect_reasoning_calls=0,
        expect_count=1,
    ),
    EvalCase(
        id="deceased-follow-up",
        question="How many students have names starting with S?",
        expect_status="answered",
        expect_reasoning_calls=0,
        expect_count=2,
        follow_up="Which of those are recorded as deceased?",
        follow_up_count=1,
        follow_up_frame="master",
    ),
    EvalCase(
        id="synthesis-causes",
        question="Summarize what the records say about causes of death.",
        expect_status="answered",
        expect_reasoning_calls=2,
        expect_file_ids=(91,),
        selected_file_id=91,
        synthesize=True,
    ),
    EvalCase(
        id="private-files-invisible",
        question="How many students have names starting with S?",
        expect_status="answered",
        expect_reasoning_calls=0,
        expect_count=2,
        allowed_file_ids=(49, 91),
    ),
]


UNAUTHORIZED_SOURCES = ("file:93:", "file:94:", "users", "otps", "logs")
