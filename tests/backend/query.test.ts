import { describe, expect, test } from "bun:test"
import { Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchQuery } from "@nia/backend/research/query"
import { ResearchValues } from "@nia/backend/research/values"
import { ResearchEvidence } from "@nia/backend/research/evidence"

const field = (name: string, type: string): Research.Field => ({
  name,
  label: name,
  type,
  operators: [
    "EQUALS",
    "NOT_EQUALS",
    "CONTAINS",
    "NOT_CONTAINS",
    "IS_KNOWN",
    "IS_UNKNOWN",
    "GREATER_THAN",
    "LESS_THAN",
    "YEAR_EQUALS",
    "BEFORE",
    "AFTER",
    "DATE_RANGE",
  ],
  aggregatable: type !== "text",
  sortable: true,
  quoteable: name === "notes",
  evidence: true,
})
const snapshot: Research.Snapshot = {
  binding: { database: "fixture", fileID: 49 },
  at: "2026-10-16T00:00:00Z",
  fingerprint: "current",
  dataset: {
    fileID: 49,
    label: "Master",
    version: 5,
    fields: [
      field("student_name", "text"),
      field("age", "number"),
      field("birth_date", "date"),
      field("death_date", "date"),
      field("cause", "entity"),
      field("notes", "text"),
    ],
  },
  records: [
    {
      sourceRowID: "1",
      values: {
        student_name: "A",
        age: "10",
        birth_date: "1900-02-01",
        death_date: "1910-02-01",
        cause: "TB",
        notes: "An exact recorded statement.",
      },
    },
    {
      sourceRowID: "2",
      values: { student_name: "B", age: "12", birth_date: "1900", death_date: "1912-03-01", cause: null },
    },
    {
      sourceRowID: "3",
      values: { student_name: "C", age: "10–14", birth_date: "1900-02-31", death_date: "1915-03-01", cause: "Other" },
    },
  ],
}

describe("strict source semantics", () => {
  test.each(["1900-02-31", "1944-09-31", "1900-13-01", "1900-02-29"])(
    "invalid date %s cannot acquire fabricated precision",
    (value) => {
      expect(ResearchValues.date(value).kind).toBe("invalid")
    },
  )
  test.each(["about 1900", "1900?", "between 1899 and 1901"])("uncertainty survives %s", (value) =>
    expect(ResearchValues.date(value).kind).toBe("uncertain"),
  )
  test("valid leap date, annotations, year-only, month-only and ranges", () => {
    expect(ResearchValues.date("2000-02-29").kind).toBe("exact")
    expect(ResearchValues.date("1902-09-14 (NCTR SOURCE)").kind).toBe("exact")
    expect(ResearchValues.date("1900").kind).toBe("year")
    expect(ResearchValues.date("1900-02").kind).toBe("month")
    expect(ResearchValues.date("1900-1902").kind).toBe("range")
  })
  test("ambiguous numeric formats and qualified dates never acquire exact precision", () => {
    expect(ResearchValues.date("02/03/1900").kind).toBe("uncertain")
    expect(ResearchValues.date("1900-02-01 (estimated)").kind).toBe("uncertain")
  })
  test.each(["10-14", "about 12", "1900/02/01", "12 and 14", "", null])("strict numeric parsing rejects %s", (value) =>
    expect(ResearchValues.number(value)).toBeNull(),
  )
  test("mean, median and tied modes include the complete eligible population", () => {
    expect(ResearchValues.stats([10, 12, 10, 12])).toMatchObject({
      count: 4,
      min: 10,
      max: 12,
      mean: 11,
      median: 11,
      modes: [10, 12],
      modeCount: 2,
    })
    expect(ResearchValues.stats([]).mean).toBeNull()
    expect(ResearchValues.stats([10, 12, 10, 12], [0, 0.5, 1])).toMatchObject({
      sum: 44,
      distinctCount: 2,
      standardDeviationPopulation: 1,
      percentiles: { p0: 10, p50: 11, p100: 12 },
    })
  })
})

describe("complete within-list queries", () => {
  test("an omitted filter counts every record; malformed empty objects fail schema validation", () => {
    expect(ResearchQuery.execute(snapshot, { operation: "count" }).total).toBe(3)
    expect(() => Schema.decodeUnknownSync(Research.Query)({ operation: "count", filter: {} })).toThrow()
  })
  test("pagination never changes the total", () => {
    const result = ResearchQuery.execute(snapshot, { operation: "list", fields: ["student_name"], limit: 1 })
    expect(result.total).toBe(3)
    expect(result.complete).toBe(false)
    expect(result.result).toEqual({ records: [{ sourceRowID: "1", values: { student_name: "A" } }] })
  })
  test("unknown does not prove a negative cause of death", () => {
    const filter: Research.Filter = { field: "cause", operator: "NOT_EQUALS", value: "TB" }
    expect(
      snapshot.records.filter((record) => ResearchQuery.matches(record, filter)).map((record) => record.sourceRowID),
    ).toEqual(["3"])
    expect(
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { op: "not", items: [{ field: "cause", operator: "EQUALS", value: "TB" }] },
      }).total,
    ).toBe(1)
  })
  test("exact statistics exclude numeric ranges and report the denominator", () => {
    const result = ResearchQuery.execute(snapshot, { operation: "stats", field: "age" })
    expect(result.total).toBe(3)
    expect(result.known).toBe(2)
    expect(result.missing).toBe(1)
    expect(result.result).toMatchObject({ mean: 11, median: 11 })
  })
  test("intervals exclude invalid and year-only dates", () => {
    const result = ResearchQuery.execute(snapshot, {
      operation: "interval",
      field: "birth_date",
      endField: "death_date",
    })
    expect(result.known).toBe(1)
    expect(result.result).toMatchObject({ count: 1, unit: "days" })
  })
  test("more than 500 eligible intervals are not truncated", () => {
    const population = {
      ...snapshot,
      records: Array.from({ length: 1001 }, (_, index) => ({ ...snapshot.records[0], sourceRowID: String(index) })),
    }
    expect(
      ResearchQuery.execute(population, { operation: "interval", field: "birth_date", endField: "death_date" }).known,
    ).toBe(1001)
  })
  test("exact-date filtering is finer than year-only comparisons", () => {
    expect(
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { field: "birth_date", operator: "BEFORE", value: "1900-03-01" },
      }).total,
    ).toBe(1)
  })
  test("NOT cannot convert a partly overlapping year into a definite date conclusion", () => {
    expect(
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { op: "not", items: [{ field: "birth_date", operator: "BEFORE", value: "1900-03-01" }] },
      }).total,
    ).toBe(0)
  })
  test("within-list cohort comparisons have explicit zero-denominator behavior", () => {
    expect(
      ResearchQuery.execute(snapshot, {
        operation: "compare",
        filter: { field: "cause", operator: "EQUALS", value: "TB" },
        other: { field: "cause", operator: "EQUALS", value: "not recorded" },
      }).result,
    ).toMatchObject({ left: 1, right: 0, leftPercentOfRight: null })
  })
  test("unknown fields, foreign source rows and unsupported operators fail explicitly", () => {
    expect(() => ResearchQuery.execute(snapshot, { operation: "list", fields: ["password"] })).toThrow()
    expect(() => ResearchQuery.execute(snapshot, { operation: "list", sourceRows: ["foreign-list-row"] })).toThrow()
    expect(() =>
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { field: "cause", operator: "YEAR_EQUALS", value: 1900 },
      }),
    ).toThrow()
    expect(() => ResearchQuery.execute(snapshot, { operation: "group", groupBy: ["notes"] })).toThrow()
  })
  test("empty/multiple-child NOT is rejected rather than silently broadening", () => {
    expect(() => ResearchQuery.execute(snapshot, { operation: "count", filter: { op: "not", items: [] } })).toThrow()
  })
  test("invalid/reversed date bounds and date parts on numeric groups fail", () => {
    expect(() =>
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { field: "birth_date", operator: "BEFORE", value: "1900-02-31" },
      }),
    ).toThrow()
    expect(() =>
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { field: "birth_date", operator: "DATE_RANGE", value: ["1910", "1900"] },
      }),
    ).toThrow()
    expect(() => ResearchQuery.execute(snapshot, { operation: "group", groupBy: ["age"], part: "year" })).toThrow()
  })
  test("query boundary rejects a list selector rather than silently accepting it", () => {
    expect(() =>
      Schema.decodeUnknownSync(Research.Query, { onExcessProperty: "error" })({ operation: "count", fileID: 94 }),
    ).toThrow()
  })
})

describe("answer evidence gate", () => {
  const query: Research.Query = { operation: "count" }
  const evidence: Research.Evidence = {
    ...ResearchQuery.execute(snapshot, query),
    id: "ev_current",
    sessionID: "ses_fixture",
    binding: snapshot.binding,
    version: 5,
    at: snapshot.at,
    fingerprint: snapshot.fingerprint,
    query,
  }
  test("renders deterministic values and citations", () => {
    expect(ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [evidence.id] }, [evidence]).text).toContain(
      "recordCount: 3",
    )
  })
  test("rejects cross-list evidence even for the same principal", () => {
    expect(() =>
      ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [evidence.id] }, [
        { ...evidence, binding: { ...snapshot.binding, fileID: 91 } },
      ]),
    ).toThrow()
  })
  test("rejects stale, fabricated or other-Session evidence", () => {
    expect(() =>
      ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [evidence.id] }, [{ ...evidence, version: 4 }]),
    ).toThrow()
    expect(() =>
      ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [evidence.id] }, [
        { ...evidence, result: { recordCount: 100 } },
      ]),
    ).toThrow()
    expect(() => ResearchEvidence.answer("ses_other", snapshot, { evidence: [evidence.id] }, [evidence])).toThrow()
  })
  test("unverified prose is not an answer input", () => {
    expect(() => ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [] }, [])).toThrow()
    expect(
      ResearchEvidence.answer("ses_fixture", snapshot, { evidence: [], reason: "out_of_scope" }, []).text,
    ).toContain("permanently selected list")
  })
  test("terminal escape sequences are removed", () =>
    expect(ResearchEvidence.safe("safe\u001b[31mtext\u0007")).toBe("safetext"))
})
