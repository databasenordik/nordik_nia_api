"""Second independent database-grounded question matrix for NIA.

This suite intentionally uses different fields, values, phrasings, compound
filters, and sort/projection combinations from database_question_matrix.py.
The expected answers still come directly from current PostgreSQL rows before
the exact same questions are sent through AssistantTurnService.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import database_question_matrix as base

from app.data_gateway.repository import AssistantDataGateway
from app.db.pool import close_pool, get_pool, init_pool
from app.execution.turn import AssistantTurnService
from app.llm.factory import get_reasoning_provider
from app.memory.store import InMemoryMemoryStore
from app.security.access_scope import AccessScope

Case = base.Case
Condition = base.Condition


CASES: tuple[Case, ...] = (
    Case("N01", 49, "How many students have the first name John?", "count", conditions=(Condition("first_name", "equals", "John"),)),
    Case("N02", 49, "How many students have Williams as their last name?", "count", conditions=(Condition("last_name", "equals", "Williams"),)),
    Case("N03", 49, "How many unique last names appear in the master list?", "distinct_count", field="last_name"),
    Case("N04", 49, "Which first name occurs most frequently?", "mode", field="first_name"),
    Case("N05", 49, "How many master-list records have no first name recorded?", "count", conditions=(Condition("first_name", "missing"),)),
    Case("N06", 49, "How many students have a recorded middle name?", "count", conditions=(Condition("middle_names", "known"),)),
    Case("N07", 49, "How many distinct middle-name values are present?", "distinct_count", field="middle_names"),
    Case("N08", 49, "How many students have an Indigenous or spirit name recorded?", "count", conditions=(Condition("indigenous_name", "known"),)),
    Case("N09", 49, "How many records include parents' names?", "count", conditions=(Condition("parents_names", "known"),)),
    Case("N10", 49, "How many students have sibling information?", "count", conditions=(Condition("siblings", "known"),)),
    Case("N11", 49, "How many master-list records include a photo reference?", "count", conditions=(Condition("photos", "known"),)),
    Case("N12", 49, "How many students do not have a photo recorded?", "count", conditions=(Condition("photos", "missing"),)),
    Case("N13", 49, "How many records contain death details?", "count", conditions=(Condition("death_details", "known"),)),
    Case("N14", 49, "How many records contain additional information?", "count", conditions=(Condition("additional_information", "known"),)),
    Case("N15", 49, "How many master-list entries have no notes?", "count", conditions=(Condition("notes", "missing"),)),
    Case("N16", 49, "How many students have a mapping location?", "count", conditions=(Condition("mapping_location", "known"),)),
    Case("N17", 49, "How many records are missing latitude?", "count", conditions=(Condition("latitude", "missing"),)),
    Case("N18", 49, "How many students were admitted after 1960?", "count", conditions=(Condition("admitted_date", "after", 1960),)),
    Case("N19", 49, "How many students were discharged before 1940?", "count", conditions=(Condition("discharged_date", "before", 1940),)),
    Case("N20", 49, "How many birth dates fall between 1900 and 1910?", "count", conditions=(Condition("birth_date", "range", (1900, 1910)),)),
    Case("N21", 49, "How many recorded student ages are greater than 20?", "count", conditions=(Condition("age", "gt", 20),)),
    Case("N22", 49, "How many students have age 9 recorded?", "count", conditions=(Condition("age", "equals", 9),)),
    Case("N23", 49, "How many student names contain Williams?", "count", conditions=(Condition("student_name", "contains", "Williams"),)),
    Case("N24", 49, "List students whose names begin with Q, including student number.", "list", conditions=(Condition("student_name", "starts", "Q"),), field="student_name", projection=("student_name", "student_number"), limit=50),
    Case("N25", 49, "What is the number of distinct student numbers?", "distinct_count", field="student_number"),
    Case("N26", 49, "Which parents-name value is the most common?", "mode", field="parents_names"),
    Case("N27", 49, "Show the earliest 5 admissions with student name, admission date, and community.", "list", field="admitted_date", direction="asc", limit=5, projection=("student_name", "admitted_date", "community")),
    Case("N28", 49, "Show the latest 5 discharges with student name, discharge date, and community.", "list", field="discharged_date", direction="desc", limit=5, projection=("student_name", "discharged_date", "community")),
    Case("C01", 91, "Give me the frequency breakdown of death factors.", "distribution", field="death_factor", include_missing=True),
    Case("C02", 91, "How many unique death factors are recorded?", "distinct_count", field="death_factor"),
    Case("C03", 91, "What is the most frequent location of death?", "mode", field="location_of_death"),
    Case("C04", 91, "How many distinct death locations are present?", "distinct_count", field="location_of_death"),
    Case("C05", 91, "How many confirmed-death records identify a nation?", "count", conditions=(Condition("nation", "known"),)),
    Case("C06", 91, "Which nation is recorded most often?", "mode", field="nation"),
    Case("C07", 91, "How many confirmed-death entries include an Indigenous name?", "count", conditions=(Condition("indigenous_name", "known"),)),
    Case("C08", 91, "How many confirmed-death records are missing parents' names?", "count", conditions=(Condition("parents_names", "missing"),)),
    Case("C09", 91, "How many records include a reason for discharge?", "count", conditions=(Condition("reason_for_discharge", "known"),)),
    Case("C10", 91, "How many records have Died as the reason for discharge?", "count", conditions=(Condition("reason_for_discharge", "equals", "Died"),)),
    Case("C11", 91, "How many records specify an admission method?", "count", conditions=(Condition("admission_method", "known"),)),
    Case("C12", 91, "How many confirmed-death records have no documentation?", "count", conditions=(Condition("documentation", "missing"),)),
    Case("C13", 91, "How many records contain investigative notes?", "count", conditions=(Condition("investigative_notes", "known"),)),
    Case("C14", 91, "How many records include specific file notes?", "count", conditions=(Condition("specific_file_notes", "known"),)),
    Case("C15", 91, "How many records say information was shared with someone?", "count", conditions=(Condition("information_shared_with", "known"),)),
    Case("C16", 91, "How many confirmed-death records are missing other links?", "count", conditions=(Condition("other_links", "missing"),)),
    Case("C17", 91, "How many entries contain a death registration number?", "count", conditions=(Condition("death_registration_number", "known"),)),
    Case("C18", 91, "How many records have no death registration date?", "count", conditions=(Condition("death_registration_date", "missing"),)),
    Case("C19", 91, "How many records include census-year information?", "count", conditions=(Condition("census_year", "known"),)),
    Case("C20", 91, "How many recorded births occurred before 1880?", "count", conditions=(Condition("birth_date", "before", 1880),)),
    Case("C21", 91, "How many death dates fall between 1880 and 1889?", "count", conditions=(Condition("death_date", "range", (1880, 1889)),)),
    Case("C22", 91, "How many burial dates are recorded literally as Unknown?", "count", conditions=(Condition("burial_date", "equals", "Unknown"),)),
    Case("C23", 91, "How many confirmed-death records lack a place of burial?", "count", conditions=(Condition("place_of_burial", "missing"),)),
    Case("C24", 91, "How many different schools are represented in confirmed deaths?", "distinct_count", field="school"),
    Case("C25", 91, "How many records give Pneumonia as the exact cause of death?", "count", conditions=(Condition("cause_of_death", "equals", "Pneumonia"),)),
    Case("C26", 91, "How many female students were older than 12 at death?", "count", conditions=(Condition("gender", "equals", "F"), Condition("age_at_death", "gt", 12))),
    Case("C27", 91, "List the earliest 5 first-admission dates with student name and school.", "list", field="first_admitted_date", direction="asc", limit=5, projection=("student_name", "first_admitted_date", "school")),
    Case("C28", 91, "List the latest 5 birth dates with student name and gender.", "list", field="birth_date", direction="desc", limit=5, projection=("student_name", "birth_date", "gender")),
)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--ids", help="Comma-separated case IDs")
    args = parser.parse_args()
    selected = {item.strip().upper() for item in (args.ids or "").split(",") if item.strip()}
    cases = tuple(case for case in CASES if not selected or case.id in selected)
    oracle = await base.load_oracle()
    for file_id, labels in oracle.labels.items():
        for field_name, label in labels.items():
            base._LABELS[(file_id, field_name)] = label

    if args.expected_only:
        for case in cases:
            expected = oracle.expected(case)
            print(f"{case.id}\t{case.file_id}\t{case.question}\t{base.expected_summary(case, expected)}")
        return 0

    await init_pool()
    service = AssistantTurnService(
        AssistantDataGateway(get_pool()),
        store=InMemoryMemoryStore(),
        reasoner=get_reasoning_provider(),
    )
    scope = AccessScope(principal_id="benchmark-researcher-2", allowed_file_ids=(49, 91))
    failures = 0
    report: list[dict[str, object]] = []
    try:
        for index, case in enumerate(cases, start=1):
            expected = oracle.expected(case)
            result = await service.answer(scope, case.question, selected_file_id=case.file_id)
            issues = base.compare(case, expected, result)
            failures += bool(issues)
            status = "PASS" if not issues else "FAIL"
            report.append(
                {
                    "id": case.id,
                    "question": case.question,
                    "expected": base.expected_summary(case, expected),
                    "actual": result.answer,
                    "status": status,
                    "issues": issues,
                    "planner": result.planner_type,
                }
            )
            print(f"[{index:02d}/{len(cases)}] {case.id} {status}: {case.question}")
            print(f"  expected: {base.expected_summary(case, expected)}")
            print(f"  actual:   {result.answer[:700]}")
            if issues:
                print(f"  planner:  {result.planner_type}; ops={result.plan.op_names() if result.plan else []}")
                if result.plan:
                    filters = [item.model_dump(mode="json") for step in result.plan.steps for item in step.where]
                    print(f"  filters:  {filters}")
                for issue in issues:
                    print(f"  issue:    {issue}")
    finally:
        await close_pool()

    print(f"\nSUMMARY total={len(cases)} passed={len(cases) - failures} failed={failures}")
    print("FAILED_CASES " + ",".join(item["id"] for item in report if item["status"] == "FAIL"))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
