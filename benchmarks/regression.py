"""Run a supplied question file through the production pipeline and score it.

The markdown file is the input, not a fixture baked into this module: questions and
expected answers are read from its table, so nothing here is tied to a particular
benchmark. Every question runs in a fresh conversation with no dataset pre-selected,
so dataset selection is part of what is measured.

    python -m benchmarks.regression docs/questions.md --live
    python -m benchmarks.regression docs/questions.md --offline --to 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import asyncpg

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.data_gateway.repository import AssistantDataGateway  # noqa: E402
from app.db.pool import close_pool, get_pool, init_pool  # noqa: E402
from app.execution.turn import AssistantTurnService, TurnResult  # noqa: E402
from app.llm.factory import get_reasoning_provider  # noqa: E402
from app.memory.store import InMemoryMemoryStore  # noqa: E402
from app.security.access_scope import AccessScope  # noqa: E402
from benchmarks.grading import Grade, grade, same_answer  # noqa: E402

_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")


@dataclass
class Question:
    number: int
    question: str
    expected: str
    previous_answer: str
    previous_verdict: str


def parse_question_file(path: Path) -> list[Question]:
    """Read a Number | Question | Answer | ... markdown table, or a numbered list."""
    text = path.read_text(encoding="utf-8-sig")
    rows: list[Question] = []
    for line in text.splitlines():
        match = _ROW.match(line.strip())
        if match is None:
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        rows.append(
            Question(
                number=int(cells[0]),
                question=cells[1],
                expected=cells[2],
                previous_answer=cells[3] if len(cells) > 3 else "",
                previous_verdict=_verdict(cells[4]) if len(cells) > 4 else "",
            )
        )
    if rows:
        return rows
    for match in re.finditer(r"(?m)^(\d+)\.\s+(.+)$", text):
        rows.append(
            Question(
                number=int(match.group(1)),
                question=" ".join(match.group(2).split()),
                expected="",
                previous_answer="",
                previous_verdict="",
            )
        )
    return rows


def _verdict(cell: str) -> str:
    return cell.replace("*", "").strip().lower()


def parse_paraphrases(path: Path | None) -> dict[int, list[str]]:
    """`12. reworded question` lines grouped under `## <number>` headings."""
    if path is None or not path.exists():
        return {}
    mapping: dict[int, list[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(\d+)\.\s+(.+?)\s*$", line)
        if match is None:
            continue
        mapping.setdefault(int(match.group(1)), []).append(match.group(2))
    return mapping


def parse_excluded(path: Path | None) -> dict[int, str]:
    """`12: reason` lines naming questions this build does not attempt."""
    if path is None or not path.exists():
        return {}
    excluded: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s*(\d+)\s*:\s*(.+?)\s*$", line)
        if match is not None:
            excluded[int(match.group(1))] = match.group(2)
    return excluded


async def _answer_once(
    service: AssistantTurnService,
    scope: AccessScope,
    question: str,
    timeout: float,
    selected_file_id: int | None = None,
) -> tuple[str, dict[str, object]]:
    """One question, one fresh conversation. AI mode requires selected_file_id."""
    result = await asyncio.wait_for(
        service.answer(scope, question, selected_file_id=selected_file_id),
        timeout=timeout,
    )
    meta = {
        "status": result.status,
        "error_code": result.error_code,
        "planner": result.planner_type,
        "reasoning_calls": result.reasoning_calls,
        "selected_file_id": result.selected_file_id,
        "planner_attempts": result.planner_attempts,
        "review_status": result.review_status,
        "planner_failure_stage": result.planner_failure_stage,
        "planner_provider": result.planner_provider,
        "planner_model": result.planner_model,
        "planner_usage": result.planner_usage,
        "detail": (result.detail or "")[:300],
        "ops": result.plan.op_names() if result.plan is not None else [],
    }
    return result.answer, meta, result


def load_suite(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or payload)
    for index, case in enumerate(cases, start=1):
        missing = [
            key
            for key in (
                "id",
                "source_question_id",
                "question",
                "selected_file_id",
                "category",
                "expected_outcome",
                "critical",
                "private",
            )
            if key not in case
        ]
        if missing:
            raise ValueError(f"suite case {index} is missing {', '.join(missing)}")
        if not case.get("expected_plan") and not str(
            case.get("expected_answer") or case.get("expected") or ""
        ).strip():
            raise ValueError(
                f"suite case {case['id']} needs expected_plan or an expected answer"
            )
        authorized = [int(value) for value in case.get("authorized_file_ids") or []]
        if authorized and int(case["selected_file_id"]) not in authorized:
            raise ValueError(f"suite case {case['id']} selects an unauthorized file")
    return cases


async def _preflight_private_cases(cases: list[dict], args: argparse.Namespace) -> None:
    private_ids = sorted(
        {int(case["selected_file_id"]) for case in cases if bool(case.get("private"))}
    )
    if not private_ids:
        return
    if not args.allow_private_files:
        raise SystemExit(
            "suite contains private cases; pass --allow-private-files after grants are seeded"
        )
    from app.config import get_settings

    connection = await asyncpg.connect(get_settings().assistant_migrator_database_url)
    try:
        rows = await connection.fetch(
            """
            SELECT file_id
            FROM assistant.access_grants
            WHERE principal_id = $1
              AND file_id = ANY($2::int[])
              AND grant_status = 'active'
            """,
            args.principal,
            private_ids,
        )
    finally:
        await connection.close()
    granted = {int(row["file_id"]) for row in rows}
    missing = sorted(set(private_ids) - granted)
    if missing:
        raise SystemExit(
            f"private-case preflight failed for principal {args.principal}: missing active grants {missing}"
        )


def _grade_case(
    item: Question,
    case: dict,
    answer: str,
    turn: TurnResult | None,
) -> Grade:
    contract = case.get("expected_plan")
    if contract:
        return _grade_plan_contract(turn, contract, case)
    if item.expected.strip():
        return grade(item.expected, answer)
    return Grade(
        verdict="fail",
        score=0.0,
        reasons=["suite case has no expected answer or expected_plan contract"],
    )


def _grade_plan_contract(turn: TurnResult | None, contract: dict, case: dict) -> Grade:
    reasons: list[str] = []
    expected_status = str(case.get("expected_status") or "answered")
    if turn is None:
        return Grade(verdict="fail", score=0.0, reasons=["turn result unavailable"])
    if turn.status != expected_status:
        reasons.append(f"status={turn.status}, expected {expected_status}")
    selected = int(case.get("selected_file_id") or 0)
    if turn.selected_file_id != selected:
        reasons.append(f"selected_file_id={turn.selected_file_id}, expected {selected}")
    if "suggested_file_id" in contract and turn.suggested_file_id != contract["suggested_file_id"]:
        reasons.append(
            f"suggested_file_id={turn.suggested_file_id}, expected {contract['suggested_file_id']}"
        )
    plan = turn.plan
    if contract.get("forbid_query"):
        if plan is not None:
            reasons.append("query plan was produced for a non-query outcome")
    elif plan is None:
        reasons.append("query plan missing")
    else:
        # A step with no explicit file_ids inherits the plan scope -- that is the
        # convention everywhere else in the codebase (see value_expansion, which reads
        # `step.file_ids or plan.scope.file_ids`). USE_CURRENT_VERSION is exactly such
        # a step, so requiring every step to carry the id failed every correctly
        # scoped plan on its own scope-establishing step.
        wrong_scope = [
            step.id
            for step in plan.steps
            if step.file_ids and tuple(step.file_ids) != (selected,)
        ]
        if tuple(plan.scope.file_ids) != (selected,) or wrong_scope:
            reasons.append(
                f"scope mismatch plan={list(plan.scope.file_ids)} steps={wrong_scope}"
            )
        ops = plan.op_names()
        for required in contract.get("required_ops") or []:
            if str(required).upper() not in ops:
                reasons.append(f"missing op {str(required).upper()}")
        predicates = [
            predicate
            for step in plan.steps
            for root in step.where
            for predicate in _leaf_predicates(root)
        ]
        for required in contract.get("predicates") or []:
            if not any(_predicate_matches(item, required) for item in predicates):
                options = required.get("any_of") or [required]
                reasons.append(
                    "missing predicate "
                    + " or ".join(
                        f"{item.get('field')} {item.get('operator') or '/'.join(item.get('operators') or [])} "
                        f"{item.get('value', '')}".strip()
                        for item in options
                    )
                )
        expected_groups = [str(item) for item in contract.get("group_by") or []]
        if expected_groups:
            actual_groups = [
                list(step.fields)
                for step in plan.steps
                if str(step.op.value) in {"GROUP_BY", "RANK"}
            ]
            if not any(group[: len(expected_groups)] == expected_groups for group in actual_groups):
                reasons.append(f"group_by={actual_groups}, expected {expected_groups}")
        expected_fields = set(contract.get("project_fields") or [])
        if expected_fields:
            projected = {
                field
                for step in plan.steps
                if str(step.op.value) == "PROJECT"
                for field in step.fields
            }
            missing = sorted(expected_fields - projected)
            if missing:
                reasons.append(f"missing projected fields {missing}")
        if contract.get("having_min_count") is not None:
            values = [
                step.having_min_count
                for step in plan.steps
                if str(step.op.value) in {"GROUP_BY", "RANK"}
            ]
            if int(contract["having_min_count"]) not in values:
                reasons.append(
                    f"having_min_count={values}, expected {contract['having_min_count']}"
                )
    return Grade(
        verdict="pass" if not reasons else "fail",
        score=1.0 if not reasons else 0.0,
        reasons=reasons,
    )


def _leaf_predicates(predicate):
    if predicate.is_group():
        for item in predicate.items:
            yield from _leaf_predicates(item)
        return
    yield predicate


def _predicate_matches(actual, expected: dict) -> bool:
    # The same condition has more than one faithful encoding: deceased is a boolean IS_TRUE in
    # the static catalog and a recorded "yes" value in the live field registry. A case lists
    # each encoding under any_of instead of being tied to one deployment's schema.
    if expected.get("any_of"):
        return any(_predicate_matches(actual, option) for option in expected["any_of"])
    if str(actual.field).casefold() != str(expected.get("field") or "").casefold():
        return False
    allowed = expected.get("operators") or [expected.get("operator")]
    if str(actual.operator or "").upper() not in {str(item).upper() for item in allowed if item}:
        return False
    if "value" not in expected and "values" not in expected:
        return True
    actual_value = actual.value
    wanted_values = expected.get("values")
    if wanted_values is None:
        wanted_values = [expected.get("value")]
    actual_values = actual_value if isinstance(actual_value, list) else [actual_value]
    return any(
        str(wanted).casefold() == str(item).casefold()
        for wanted in wanted_values
        for item in actual_values
    )


async def run(args: argparse.Namespace) -> int:
    import os

    from app.config import get_settings

    if args.planner_mode:
        os.environ["PLANNER_MODE"] = args.planner_mode
        get_settings.cache_clear()
    runs = args.runs if args.runs is not None else (3 if get_settings().planner_mode == "ai" else 1)

    if args.suite:
        suite_cases = [
            item
            for item in load_suite(args.suite)
            if args.file_id is None or int(item.get("selected_file_id") or 0) == args.file_id
        ]
        questions = [
            Question(
                number=int(item.get("source_question_id") or index),
                question=str(item["question"]),
                expected=str(item.get("expected_answer") or item.get("expected") or ""),
                previous_answer="",
                previous_verdict="",
            )
            for index, item in enumerate(suite_cases, start=1)
        ]
        suite_meta = suite_cases
    else:
        if args.question_file is None:
            raise SystemExit("question_file or --suite is required")
        questions = [
            item
            for item in parse_question_file(args.question_file)
            if args.start <= item.number <= args.end
        ]
        suite_meta = [{} for _ in questions]
    if not questions:
        raise SystemExit("no questions found in the selected range")
    excluded = parse_excluded(args.excluded)
    paraphrases = parse_paraphrases(args.paraphrases) if args.paraphrases else {}

    if args.suite:
        await _preflight_private_cases(suite_meta, args)
    await init_pool()
    reasoner = None if args.offline else get_reasoning_provider()
    if args.offline:
        from app.llm.fake import FakeReasoningProvider

        reasoner = FakeReasoningProvider()
    service = AssistantTurnService(
        AssistantDataGateway(get_pool()),
        store=InMemoryMemoryStore(),
        reasoner=reasoner,
    )
    file_ids = [int(item) for item in args.file_ids.split(",") if item.strip()]
    if args.file_id and args.file_id not in file_ids:
        file_ids.append(args.file_id)
    records: list[dict[str, object]] = []
    gate_failed = False
    try:
        for item, meta_case in zip(questions, suite_meta, strict=False):
            if item.number in excluded:
                records.append(
                    {
                        "number": item.number,
                        "question": item.question,
                        "verdict": "excluded",
                        "reason": excluded[item.number],
                        "previous_verdict": item.previous_verdict,
                    }
                )
                _emit(records[-1])
                continue
            selected = args.file_id or meta_case.get("selected_file_id")
            case_file_ids = tuple(
                int(value) for value in (meta_case.get("authorized_file_ids") or file_ids)
            )
            case_scope = AccessScope(
                principal_id=args.principal,
                allowed_file_ids=case_file_ids,
                can_use_private_files=bool(args.allow_private_files),
            )
            repeats = int(meta_case.get("repetitions") or runs)
            if meta_case.get("critical"):
                repeats = max(repeats, 5)
            run_verdicts: list[str] = []
            case_runs: list[dict[str, object]] = []
            last_record: dict[str, object] | None = None
            for attempt in range(repeats):
                turn = None
                outage_retries = 0
                while True:
                    try:
                        answer, meta, turn = await _answer_once(
                            service,
                            case_scope,
                            item.question,
                            args.timeout,
                            selected_file_id=selected,
                        )
                    except Exception as exc:  # keep the rest of the suite running
                        answer, meta = f"{type(exc).__name__}: {exc}", {"status": "runner_error"}
                    if (
                        meta.get("status") != "planner_unavailable"
                        or outage_retries >= _OUTAGE_RETRIES
                    ):
                        break
                    outage_retries += 1
                    await asyncio.sleep(_OUTAGE_BACKOFF_SECONDS * outage_retries)
                provider_outage = meta.get("status") == "planner_unavailable"
                scored = _grade_case(item, meta_case, answer, turn)
                if provider_outage:
                    # Infrastructure, not incorrectness. Kept in the record and counted
                    # separately so a real availability problem stays visible.
                    scored = Grade(
                        verdict="excluded",
                        score=0.0,
                        reasons=[
                            "planner_unavailable after "
                            f"{outage_retries} retries; provider outage, not a planning failure"
                        ],
                    )
                record: dict[str, object] = {
                    "number": item.number,
                    "question": item.question,
                    "expected": item.expected,
                    "answer": answer,
                    "verdict": scored.verdict,
                    "score": scored.score,
                    "reasons": scored.reasons,
                    "previous_verdict": item.previous_verdict,
                    "run": attempt + 1,
                    "critical": bool(meta_case.get("critical")),
                    "category": meta_case.get("category"),
                    "selected_file_id": selected,
                    **meta,
                }
                record["provider_outage"] = bool(provider_outage)
                record["outage_retries"] = outage_retries
                if meta.get("review_status") == "corrected" and turn is not None:
                    record["review_benefit"] = await _review_benefit(
                        service,
                        case_scope,
                        item,
                        meta_case,
                        selected,
                        args.timeout,
                        scored,
                        turn,
                    )
                run_verdicts.append(str(scored.verdict))
                case_runs.append(dict(record))
                last_record = record
            assert last_record is not None
            last_record["runs"] = case_runs
            last_record["run_verdicts"] = run_verdicts
            last_record["worst_verdict"] = _worst(run_verdicts)
            if paraphrases.get(item.number):
                last_record["paraphrases"] = await _check_paraphrases(
                    service,
                    case_scope,
                    paraphrases[item.number],
                    str(last_record["answer"]),
                    args.timeout,
                )
            records.append(last_record)
            _emit(last_record)
            if args.fail_on_gate and last_record.get("critical") and last_record["worst_verdict"] != "pass":
                gate_failed = True
            if args.fail_on_gate and last_record["verdict"] == "fail" and last_record.get("critical"):
                gate_failed = True
    finally:
        await close_pool()

    _write_reports(records, args)
    if args.fail_on_gate and (gate_failed or any(item.get("verdict") == "fail" and item.get("critical") for item in records)):
        return 1
    return 0


_OUTAGE_RETRIES = 2
_OUTAGE_BACKOFF_SECONDS = 3.0


def _worst(verdicts: list[str]) -> str:
    order = {"fail": 3, "partial": 2, "pass": 1, "excluded": 0}
    return max(verdicts, key=lambda item: order.get(item, 0))


async def _review_benefit(
    service: AssistantTurnService,
    scope: AccessScope,
    item: Question,
    case: dict,
    selected: int | None,
    timeout: float,
    final_score,
    turn,
) -> dict[str, object]:
    planned = getattr(turn, "original_planned", None)
    if planned is None or selected is None:
        return {"status": "skipped", "reason": "original planned turn unavailable"}
    try:
        original = await asyncio.wait_for(
            service.answer_from_planned(scope, planned, selected_file_id=selected, question=item.question),
            timeout=timeout,
        )
    except Exception as exc:
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    original_score = _grade_case(item, case, original.answer, original)
    final = final_score.verdict
    first = original_score.verdict
    if first != "pass" and final == "pass":
        kind = "benefit"
    elif first == "pass" and final != "pass":
        kind = "false_positive"
    else:
        kind = "neutral"
    return {
        "status": kind,
        "original_verdict": first,
        "final_verdict": final,
        "original_score": original_score.score,
        "final_score": final_score.score,
    }


async def _check_paraphrases(
    service: AssistantTurnService,
    scope: AccessScope,
    wordings: list[str],
    canonical_answer: str,
    timeout: float,
) -> list[dict[str, object]]:
    """A reworded question must produce the same structured answer, not a new one."""
    results: list[dict[str, object]] = []
    for wording in wordings:
        try:
            answer, _meta = await _answer_once(service, scope, wording, timeout)
        except Exception as exc:
            results.append({"question": wording, "consistent": False, "answer": str(exc)})
            continue
        results.append(
            {
                "question": wording,
                "consistent": same_answer(canonical_answer, answer),
                "answer": answer,
            }
        )
    return results


def _emit(record: dict[str, object]) -> None:
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _write_reports(records: list[dict[str, object]], args: argparse.Namespace) -> None:
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not args.report:
        return
    graded = [item for item in records if item["verdict"] != "excluded"]
    counts = {
        key: sum(1 for item in graded if item["verdict"] == key)
        for key in ("pass", "partial", "fail")
    }
    excluded = [item for item in records if item["verdict"] == "excluded"]
    before = sum(1 for item in graded if item.get("previous_verdict") == "pass")
    paraphrases = [item for record in records for item in record.get("paraphrases") or []]
    consistent = sum(1 for item in paraphrases if item.get("consistent"))
    lines = [
        "# Regression results",
        "",
        f"Mode: {'offline (no reasoning provider)' if args.offline else 'live'}  ",
        f"Questions graded: {len(graded)}  ",
        f"Excluded by scope: {len(excluded)}",
        "",
        "Before / after are counted over the graded questions only. The before column",
        "is the verdict recorded in the supplied question file.",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
        f"| pass | {before} | {counts['pass']} |",
        f"| partial | 0 | {counts['partial']} |",
        f"| fail | {len(graded) - before} | {counts['fail']} |",
        "",
        f"Planner mode: {getattr(args, 'planner_mode', None) or 'default'}  ",
        f"Mean score: {_mean(item.get('score') for item in graded)}  ",
        f"Review corrections: {sum(1 for item in graded if item.get('review_status') == 'corrected')}  ",
        f"Benefits / false positives: "
        f"{sum(1 for item in graded if (item.get('review_benefit') or {}).get('status') == 'benefit')} / "
        f"{sum(1 for item in graded if (item.get('review_benefit') or {}).get('status') == 'false_positive')}",
        "",
    ]
    if paraphrases:
        lines.extend(
            [
                f"Paraphrase self-consistency: {consistent}/{len(paraphrases)} rewordings "
                "produced the same structured answer as their canonical question.",
                "",
            ]
        )
    lines.extend(
        [
            "## Per question",
            "",
            "| # | Verdict | Before | Score | Question | Notes |",
            "|---:|:---:|:---:|---:|---|---|",
        ]
    )
    for item in records:
        note = "; ".join(item.get("reasons") or []) or item.get("reason", "")
        lines.append(
            "| {number} | {verdict} | {previous} | {score} | {question} | {note} |".format(
                number=item["number"],
                verdict=item["verdict"],
                previous=item.get("previous_verdict") or "-",
                score=item.get("score", ""),
                question=_cell(str(item["question"])),
                note=_cell(str(note)),
            )
        )
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(values) -> str:
    nums = [float(item) for item in values if isinstance(item, int | float)]
    if not nums:
        return "-"
    return f"{sum(nums) / len(nums):.3f}"


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "/")[:220]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question_file", type=Path, nargs="?")
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--to", dest="end", type=int, default=10_000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--offline", action="store_true", help="inject FakeReasoningProvider")
    parser.add_argument("--live", dest="offline", action="store_false")
    parser.add_argument("--planner-mode", choices=("legacy", "ai"), default=None)
    parser.add_argument("--runs", type=int, default=None, help="repeat each case; default 3 for AI, 1 for legacy")
    parser.add_argument("--suite", type=Path, default=None, help="scoped JSON manifest")
    parser.add_argument("--allow-private-files", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--principal", default="regression-suite")
    parser.add_argument("--file-ids", dest="file_ids", default="49,91")
    parser.add_argument("--file-id", dest="file_id", type=int, default=None, help="selected dataset")
    parser.add_argument("--paraphrases", type=Path, default=None)
    parser.add_argument("--excluded", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--json-out", dest="json_out", type=Path, default=None)
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="re-render the reports from a saved JSON result file without re-running",
    )
    parser.set_defaults(offline=False)
    args = parser.parse_args()
    if args.render is not None:
        records = json.loads(Path(args.render).read_text(encoding="utf-8"))
        _write_reports(records, args)
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
