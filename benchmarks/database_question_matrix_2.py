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
    Case("N03", 49, "How many unique last names are recorded?", "distinct_count", field="last_name"),
    Case("N04", 49, "Which recorded first name is shared by the most students, and how many student records have that name?", "mode", field="first_name"),
    Case("N05", 49, "How many student records have no first name recorded?", "count", conditions=(Condition("first_name", "missing"),)),
    Case("N06", 49, "How many students have a recorded middle name?", "count", conditions=(Condition("middle_names", "known"),)),
    Case("N07", 49, "How many distinct recorded values appear in 'Middle Name', excluding missing or unknown values?", "distinct_count", field="middle_names"),
    Case("N08", 49, "How many students have an Indigenous or spirit name recorded?", "count", conditions=(Condition("indigenous_name", "known"),)),
    Case("N09", 49, "How many records include parents' names?", "count", conditions=(Condition("parents_names", "known"),)),
    Case("N10", 49, "How many student records have known sibling information in the 'Siblings' column?", "count", conditions=(Condition("siblings", "known"),)),
    Case("N11", 49, "How many student records have a photo reference recorded in the 'Photos' column?", "count", conditions=(Condition("photos", "known"),)),
    Case("N12", 49, "How many student records have no known photo reference in the 'Photos' column?", "count", conditions=(Condition("photos", "missing"),)),
    Case("N13", 49, "How many student records have known information recorded in 'Death details'?", "count", conditions=(Condition("death_details", "known"),)),
    Case("N14", 49, "How many student records have known information recorded in 'Additional Information'?", "count", conditions=(Condition("additional_information", "known"),)),
    Case("N15", 49, "How many student records have no known information in the 'Notes' column?", "count", conditions=(Condition("notes", "missing"),)),
    Case("N16", 49, "How many student records have a known location recorded in 'Mapping Location'?", "count", conditions=(Condition("mapping_location", "known"),)),
    Case("N17", 49, "How many student records are missing the latitude required for mapping in 'Lat'?", "count", conditions=(Condition("latitude", "missing"),)),
    Case("N18", 49, "How many students were admitted after 1960?", "count", conditions=(Condition("admitted_date", "after", 1960),)),
    Case("N19", 49, "How many students were discharged before 1940?", "count", conditions=(Condition("discharged_date", "before", 1940),)),
    Case("N20", 49, "How many students have a birth date from 1900 through 1910, including both years?", "count", conditions=(Condition("birth_date", "range", (1900, 1910)),)),
    Case("N21", 49, "How many student records have a numeric value greater than 20 years in the 'Age' column?", "count", conditions=(Condition("age", "gt", 20),)),
    Case("N22", 49, "How many student records have exactly '9' recorded in the 'Age' column?", "count", conditions=(Condition("age", "equals", 9),)),
    Case("N23", 49, "How many student records contain Williams anywhere in the full name formed from first, middle, and last name?", "count", conditions=(Condition("student_name", "contains", "Williams"),)),
    Case("N25", 49, "How many distinct student numbers are recorded, excluding missing or unknown values?", "distinct_count", field="student_number"),
    Case("N27", 49, "Show the earliest 5 admissions with student name, admission date, and community.", "list", field="admitted_date", direction="asc", limit=5, projection=("student_name", "admitted_date", "community")),
    Case("N28", 49, "Show the latest 5 discharges with student name, discharge date, and community.", "list", field="discharged_date", direction="desc", limit=5, projection=("student_name", "discharged_date", "community")),
    Case("C01", 91, "Show the number of student records in each category recorded in 'DEATH FACTOR'. Include a 'Not recorded' category for missing or unknown entries.", "distribution", field="death_factor", include_missing=True),
    Case("C02", 91, "How many distinct known categories are recorded in 'DEATH FACTOR'?", "distinct_count", field="death_factor"),
    Case("C03", 91, "Which category in 'LOCATION OF DEATH' has the most student records, and what is the count?", "mode", field="location_of_death"),
    Case("C04", 91, "How many distinct known categories are recorded in 'LOCATION OF DEATH'?", "distinct_count", field="location_of_death"),
    Case("C05", 91, "How many student records have a known nation recorded in 'NATION'?", "count", conditions=(Condition("nation", "known"),)),
    Case("C06", 91, "Which nation recorded in 'NATION' has the most student records, and how many records name it?", "mode", field="nation"),
    Case("C07", 91, "How many student records include an Indigenous name?", "count", conditions=(Condition("indigenous_name", "known"),)),
    Case("C08", 91, "How many student records are missing parents' names?", "count", conditions=(Condition("parents_names", "missing"),)),
    Case("C09", 91, "How many student records have a known reason recorded in 'REASON FOR DISCHARGE'?", "count", conditions=(Condition("reason_for_discharge", "known"),)),
    Case("C10", 91, "How many student records give exactly 'Died' as the recorded reason for discharge?", "count", conditions=(Condition("reason_for_discharge", "equals", "Died"),)),
    Case("C11", 91, "How many student records have known information about how admission occurred in 'ADMISSION METHOD'?", "count", conditions=(Condition("admission_method", "known"),)),
    Case("C12", 91, "How many student records have no known entry in 'DOCUMENTATION'?", "count", conditions=(Condition("documentation", "missing"),)),
    Case("C13", 91, "How many student records have known information in 'INVESTIGATIVE NOTES'?", "count", conditions=(Condition("investigative_notes", "known"),)),
    Case("C14", 91, "How many student records have known notes or references in 'SPECIFIC FILE NOTES/REFERENCES'?", "count", conditions=(Condition("specific_file_notes", "known"),)),
    Case("C15", 91, "How many student records have a known recipient recorded in 'INFORMATION SHARED WITH ?'?", "count", conditions=(Condition("information_shared_with", "known"),)),
    Case("C16", 91, "How many student records have no known reference in 'Other Links'?", "count", conditions=(Condition("other_links", "missing"),)),
    Case("C17", 91, "How many student records have a recorded death registration number?", "count", conditions=(Condition("death_registration_number", "known"),)),
    Case("C18", 91, "How many records have no death registration date?", "count", conditions=(Condition("death_registration_date", "missing"),)),
    Case("C19", 91, "How many student records have known census-year information in 'INDICATE CENSUS YEAR USED OR REQUIRED'?", "count", conditions=(Condition("census_year", "known"),)),
    Case("C20", 91, "How many students have a recorded birth date before 1880?", "count", conditions=(Condition("birth_date", "before", 1880),)),
    Case("C21", 91, "How many students have a recorded death date from 1880 through 1889, including both years?", "count", conditions=(Condition("death_date", "range", (1880, 1889)),)),
    Case("C22", 91, "How many students have an unknown burial date?", "count", conditions=(Condition("burial_date", "missing"),)),
    Case("C23", 91, "How many student records lack a place of burial?", "count", conditions=(Condition("place_of_burial", "missing"),)),
    Case("C24", 91, "How many distinct known school names are recorded in 'SCHOOL'?", "distinct_count", field="school"),
    Case("C25", 91, "How many student records give exactly 'Pneumonia' as the recorded cause of death?", "count", conditions=(Condition("cause_of_death", "equals", "Pneumonia"),)),
    Case("C26", 91, "How many female students were older than 12 at death?", "count", conditions=(Condition("gender", "equals", "F"), Condition("age_at_death", "gt", 12))),
    Case("C27", 91, "List the five earliest first admissions with each student's full name, first admission date, and school.", "list", field="first_admitted_date", direction="asc", limit=5, projection=("student_name", "first_admitted_date", "school")),
    Case("C28", 91, "List the five latest recorded births with each student's full name, birth date, and gender.", "list", field="birth_date", direction="desc", limit=5, projection=("student_name", "birth_date", "gender")),
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
