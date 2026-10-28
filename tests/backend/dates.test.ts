import { describe, expect, test } from "bun:test"
import { Research } from "@opencode/schema/research"
import { ResearchValues } from "@nia/backend/research/values"
import { ResearchQuery } from "@nia/backend/research/query"
import { ResearchEvidence } from "@nia/backend/research/evidence"

const dataset: Research.Dataset = {
  fileID: 49,
  label: "Synthetic dates",
  version: 5,
  fields: [
    "admitted_date",
    "discharged_date",
    "admitted_date_year",
    "admitted_date_month",
    "admitted_date_day",
    "student_name",
  ].map((name) => ({
    name,
    label: name,
    type: name.endsWith("_date") ? "date" : name.startsWith("admitted_date_") ? "number" : "text",
    operators: [
      "EQUALS",
      "NOT_EQUALS",
      "IN",
      "NOT_IN",
      "YEAR_EQUALS",
      "BEFORE",
      "AFTER",
      "DATE_RANGE",
      "GREATER_THAN",
      "LESS_THAN",
      "NUMBER_RANGE",
      "IS_KNOWN",
      "IS_UNKNOWN",
    ],
    aggregatable: true,
    sortable: true,
    quoteable: name.endsWith("_date"),
    evidence: !name.startsWith("admitted_date_"),
  })),
}
function snapshot(dates: readonly string[]): Research.Snapshot {
  return {
    binding: { database: "synthetic", fileID: 49 },
    dataset,
    at: "2026-10-16",
    fingerprint: "source",
    records: dates.map((date, index) => ({
      sourceRowID: String(index),
      values: {
        admitted_date: date,
        student_name: `Synthetic ${index}`,
        admitted_date_year: "9999",
        discharged_date: "1910-01-01",
      },
    })),
  }
}
function run(dates: readonly string[], query: Research.Query) {
  return ResearchQuery.execute(snapshot(dates), query)
}
const year = (value: number): Research.Predicate => ({ field: "admitted_date_year", operator: "EQUALS", value })

describe("source date components", () => {
  test.each([
    ["1900-01-01", "exact", "1900", "1", "1"],
    ["1900-06-00", "month", "1900", "6", null],
    ["1900-00-00", "year", "1900", null, null],
    ["0000-09-00", "partial", null, "9", null],
    ["0000-00-15", "partial", null, null, "15"],
    ["1900-00-31", "partial", "1900", null, "31"],
    ["0000-02-29", "partial", null, "2", "29"],
    ["0001-01-01", "exact", "1", "1", "1"],
    ["07.06.2019", "uncertain", "2019", null, null],
  ] as const)("%s retains only recorded components", (raw, kind, year, month, day) => {
    const date = ResearchValues.date(raw)
    expect(date.kind).toBe(kind)
    expect(date.raw).toBe(raw)
    expect(["year", "month", "day"].map((part) => ResearchValues.datePart(date, part))).toEqual([year, month, day])
  })
  test.each(["", " ", "0", "0000", "0000-00-00"])("%s is wholly unknown", (raw) => {
    expect(ResearchValues.date(raw).kind).toBe("unknown")
    expect(ResearchValues.dateKnown(ResearchValues.date(raw))).toBe(false)
  })
  test.each(["0000-02-30", "1900-02-29", "1900-13-00", "1900-00-32", "1900-09-31"])("%s is invalid", (raw) =>
    expect(ResearchValues.date(raw).kind).toBe("invalid"),
  )
  test("mixed operators are explicitly unsupported", () =>
    expect(ResearchValues.date("1900 & 1901 | 1902").limitation).toContain("Mixed"))
  test("legacy range and annotations remain supported", () => {
    expect(ResearchValues.date("1900-1902").kind).toBe("range")
    expect(ResearchValues.date("2000-02-29 (source)").kind).toBe("exact")
    expect(ResearchValues.date("2000-02-29 (source: archive)").kind).toBe("exact")
    expect(ResearchValues.date("1900-01-01 (estimated)").kind).toBe("uncertain")
    expect(ResearchValues.date("1900 & 1902 (estimated)").kind).toBe("uncertain")
  })
})

describe("date query certainty and units", () => {
  test("known years include zero month/day without trusting incorrect normalized components", () => {
    const result = run(["1961-00-00", "1961-09-00", "1960-01-01", "0000-09-00", "0"], {
      operation: "count",
      filter: { field: "admitted_date_year", operator: "GREATER_THAN", value: 1960 },
    })
    expect(result.total).toBe(2)
    expect(result.coverage).toMatchObject({ definite: 2, possible: 0, unknown: 2, excluded: 1 })
  })
  test("known month works without a recorded year", () =>
    expect(
      run(["0000-09-00"], {
        operation: "count",
        filter: { field: "admitted_date_month", operator: "EQUALS", value: 9 },
      }).total,
    ).toBe(1))
  test("alternatives are not continuous ranges", () => {
    expect(run(["1900 | 1902"], { operation: "count", filter: year(1901) }).coverage).toMatchObject({
      definite: 0,
      possible: 0,
      excluded: 1,
    })
    expect(run(["1900 | 1902"], { operation: "count", filter: year(1900) }).coverage).toMatchObject({
      definite: 0,
      possible: 1,
    })
    expect(
      run(["1900 | 1902"], {
        operation: "count",
        filter: { field: "admitted_date_year", operator: "GREATER_THAN", value: 1899 },
      }).total,
    ).toBe(1)
  })
  test("range is one event and inclusive at both endpoints", () => {
    const dates = ["1901-00-00 : 1911-00-00"]
    expect(
      run(dates, { operation: "count", unit: "events", eventField: "admitted_date", filter: year(1905) }).coverage,
    ).toMatchObject({ definite: 0, possible: 1 })
    expect(
      run(dates, {
        operation: "count",
        filter: { field: "admitted_date", operator: "DATE_RANGE", value: ["1901", "1911"] },
      }).total,
    ).toBe(1)
    expect(run(dates, { operation: "count", filter: year(1911) }).coverage?.possible).toBe(1)
    expect(run(dates, { operation: "count", filter: year(1912) }).coverage?.excluded).toBe(1)
  })
  test("shorthand means two alternatives, not an interval", () => {
    const dates = ["1909-09-04-06"]
    expect(
      run(dates, { operation: "count", filter: { field: "admitted_date_day", operator: "EQUALS", value: 5 } }).coverage
        ?.excluded,
    ).toBe(1)
    expect(run(dates, { operation: "count", filter: year(1909) }).total).toBe(1)
    expect(
      run(dates, { operation: "count", filter: { field: "admitted_date_day", operator: "EQUALS", value: 6 } }).coverage
        ?.possible,
    ).toBe(1)
  })
  test("separate events count once as a record and twice as events", () => {
    const dates = ["1900-01-01 & 1902-02-02", "1900 | 1902", "1900 : 1902"]
    expect(run(dates, { operation: "count" }).total).toBe(3)
    expect(run(dates, { operation: "count", unit: "events", eventField: "admitted_date" }).result).toMatchObject({
      eventCount: 4,
      recordCount: 3,
    })
    expect(
      run(dates, { operation: "count", unit: "events", eventField: "admitted_date", filter: year(1900) }).coverage,
    ).toMatchObject({ definite: 1, possible: 2, excluded: 1 })
    expect(() => run(dates, { operation: "count", unit: "events" })).toThrow("field")
  })
  test("conditions cannot combine components from different events or alternatives", () => {
    const filter: Research.Filter = {
      op: "and",
      items: [year(1900), { field: "admitted_date_month", operator: "EQUALS", value: 2 }],
    }
    expect(
      run(["1900-01-01 & 1902-02-02", "1900-01-01 | 1902-02-02"], { operation: "count", filter }).coverage?.excluded,
    ).toBe(2)
    expect(
      run(["1900-01-01 & 1902-02-02"], {
        operation: "count",
        filter: {
          op: "and",
          items: [
            { field: "student_name", operator: "EQUALS", value: "Synthetic 0" },
            { op: "and", items: filter.items },
          ],
        },
      }).total,
    ).toBe(0)
  })
  test("contradictory conditions cannot match an uncertain range", () => {
    expect(
      run(["1900 : 1902"], { operation: "count", filter: { op: "and", items: [year(1900), year(1902)] } }).coverage
        ?.excluded,
    ).toBe(1)
    expect(
      run(["1900-12-01 : 1901-02-01"], {
        operation: "count",
        filter: { op: "and", items: [year(1900), { field: "admitted_date_month", operator: "EQUALS", value: 1 }] },
      }).coverage?.excluded,
    ).toBe(1)
  })
  test("range Boolean filters preserve coverage and calendar correlations", () => {
    const dates = ["1900-12-01 : 1901-02-28"]
    expect(
      run(dates, { operation: "count", filter: { field: "admitted_date_month", operator: "EQUALS", value: 3 } })
        .coverage?.excluded,
    ).toBe(1)
    expect(run(dates, { operation: "count", filter: { op: "or", items: [year(1900), year(1901)] } }).total).toBe(1)
    expect(
      run(dates, {
        operation: "count",
        filter: {
          op: "or",
          items: [
            { op: "and", items: [year(1900), { field: "admitted_date_month", operator: "EQUALS", value: 12 }] },
            { op: "and", items: [year(1901), { field: "admitted_date_month", operator: "IN", value: [1, 2] }] },
          ],
        },
      }).total,
    ).toBe(1)
    expect(
      run(["1901-02-01 : 1901-02-28"], {
        operation: "count",
        filter: { field: "admitted_date_day", operator: "LESS_THAN", value: 29 },
      }).total,
    ).toBe(1)
    expect(
      run(["1900-01-01 : 1900-01-31"], {
        operation: "count",
        filter: { field: "admitted_date", operator: "AFTER", value: "1900-01-01" },
      }).coverage?.possible,
    ).toBe(1)
    expect(
      run(["1900 : 1902"], {
        operation: "count",
        filter: { field: "admitted_date_year", operator: "IN", value: [1900, 1901, 1902] },
      }).total,
    ).toBe(1)
  })
  test("a recorded day without a month still constrains possible calendar dates", () => {
    expect(
      run(["1900-00-31"], {
        operation: "count",
        filter: { field: "admitted_date", operator: "DATE_RANGE", value: ["1900-02-01", "1900-02-28"] },
      }).coverage?.excluded,
    ).toBe(1)
  })
  test("negation cannot turn uncertainty or missing data into a fact", () => {
    expect(
      run(["1900 | 1902", "0", "1900-13-00"], { operation: "count", filter: { op: "not", items: [year(1900)] } })
        .coverage,
    ).toMatchObject({ definite: 0, possible: 1, unknown: 1, uninterpretable: 1 })
    expect(
      run(["1900 & 1902"], {
        operation: "count",
        filter: { field: "admitted_date_year", operator: "NOT_EQUALS", value: 1900 },
      }).total,
    ).toBe(0)
  })
  test("completeness does not label invalid or unsupported dates as unrecorded", () => {
    const dates = ["0", "1900-00-00", "1900-02-29", "about 1900", "0000-09-00"]
    expect(
      run(dates, { operation: "count", filter: { field: "admitted_date", operator: "IS_UNKNOWN" } }).coverage,
    ).toMatchObject({ definite: 1, uninterpretable: 2 })
    expect(run(dates, { operation: "completeness", fields: ["admitted_date"] }).result).toMatchObject({
      fields: [{ field: "admitted_date", known: 2, unknown: 1, uninterpretable: 2, possible: 0, total: 5 }],
    })
  })
  test("source evidence still rejects prohibited derived fields with a useful correction", () =>
    expect(() => run(["1961-00-00"], { operation: "list", fields: ["admitted_date_year"] })).toThrow(
      "Fetch admitted_date instead",
    ))
  test("group counts separate definite and possible memberships, deduplicating records", () => {
    const result = run(["1900 & 1900 & 1902", "1900 | 1902", "1900 : 1902"], {
      operation: "group",
      groupBy: ["admitted_date"],
      part: "year",
    })
    expect(result.result).toMatchObject({
      groups: [
        { labels: ["1900"], count: 1, possibleCount: 2 },
        { labels: ["1902"], count: 1, possibleCount: 2 },
        { labels: ["1901"], count: 0, possibleCount: 1 },
      ],
    })
    expect(
      run(["1900 & 1900 & 1902"], {
        operation: "group",
        groupBy: ["admitted_date"],
        part: "year",
        unit: "events",
        eventField: "admitted_date",
      }).result,
    ).toMatchObject({
      groups: [
        { labels: ["1900"], count: 2 },
        { labels: ["1902"], count: 1 },
      ],
    })
  })
  test("grouped component combinations cannot manufacture events", () => {
    const result = run(["1900-01-01 & 1902-02-02"], {
      operation: "group",
      groupBy: ["admitted_date_year", "admitted_date_month"],
    })
    expect(result.result).toMatchObject({
      groupCount: 2,
      groups: [
        { labels: ["1900", "1"], count: 1 },
        { labels: ["1902", "2"], count: 1 },
      ],
    })
  })
  test("statistics exclude ambiguous components and preserve usable common components", () => {
    const dates = ["1900-00-00", "1900-09-04-06", "1900 | 1902", "1900 & 1902", "0"]
    expect(run(dates, { operation: "stats", field: "admitted_date", part: "year" }).result).toMatchObject({
      count: 2,
      mean: 1900,
      excludedCount: 3,
    })
    expect(run(dates, { operation: "interval", field: "admitted_date", endField: "discharged_date" }).known).toBe(0)
  })
  test("set difference does not treat a possible match as a definite absence", () => {
    const result = run(["1900 | 1902", "1900", "1903"], { operation: "set", set: "difference", other: year(1900) })
    expect(result.total).toBe(1)
    expect(result.coverage?.possible).toBe(1)
    expect(result.result).toMatchObject({ recordCount: 1, possibleCount: 1 })
  })
  test("old date semantics and tampered certainty cannot be saved as current evidence", () => {
    const source = snapshot(["1961-00-00"])
    const query: Research.Query = { operation: "count", filter: year(1961) }
    const evidence: Research.Evidence = {
      ...ResearchQuery.execute(source, query),
      id: "ev_source",
      sessionID: "ses_source",
      binding: source.binding,
      version: 5,
      at: source.at,
      fingerprint: source.fingerprint,
      query,
    }
    expect(() =>
      ResearchEvidence.answer("ses_source", source, { evidence: [evidence.id] }, [
        { ...evidence, semantics: undefined },
      ]),
    ).toThrow("stale")
    expect(() =>
      ResearchEvidence.answer("ses_source", source, { evidence: [evidence.id] }, [
        { ...evidence, coverage: { ...evidence.coverage!, possible: 99 } },
      ]),
    ).toThrow("match")
    const answer = ResearchEvidence.answer(
      "ses_source",
      source,
      {
        evidence: [evidence.id],
        quotes: [{ evidenceID: evidence.id, sourceRowID: "0", field: "admitted_date", text: "1961-00-00" }],
      },
      [evidence],
    )
    expect(answer.text).toContain("1961-00-00")
  })
  const calendarFilters: Research.Filter[] = [
    year(1900),
    year(1901),
    { field: "admitted_date_month", operator: "EQUALS", value: 1 },
    { field: "admitted_date_day", operator: "EQUALS", value: 31 },
    { field: "admitted_date", operator: "BEFORE", value: "1901-01-01" },
    { field: "admitted_date", operator: "AFTER", value: "1900-12-31" },
    { op: "or", items: [year(1900), year(1901)] },
    { op: "not", items: [year(1900)] },
    { op: "and", items: [year(1900), { field: "admitted_date_month", operator: "EQUALS", value: 1 }] },
    {
      op: "or",
      items: [
        { op: "and", items: [year(1900), { field: "admitted_date_month", operator: "EQUALS", value: 12 }] },
        year(1901),
      ],
    },
  ]
  const worlds = [
    { raw: "1900-12-30 : 1901-01-02", days: ["1900-12-30", "1900-12-31", "1901-01-01", "1901-01-02"], separate: false },
    { raw: "1900-12-30 | 1901-01-02", days: ["1900-12-30", "1901-01-02"], separate: false },
    { raw: "1900-12-30 & 1901-01-02", days: ["1900-12-30", "1901-01-02"], separate: true },
    { raw: "2000-02-28 : 2000-03-01", days: ["2000-02-28", "2000-02-29", "2000-03-01"], separate: false },
  ]
  test.each(worlds.flatMap((source) => calendarFilters.map((filter, index) => ({ ...source, filter, index }))))(
    "calendar oracle $raw case $index",
    (source) => {
      const truth = (iso: string, filter: Research.Filter): boolean => {
        if ("op" in filter)
          return filter.op === "not"
            ? !truth(iso, filter.items[0])
            : filter.op === "and"
              ? filter.items.every((child) => truth(iso, child))
              : filter.items.some((child) => truth(iso, child))
        const date = new Date(`${iso}T00:00:00Z`)
        if (filter.operator === "BEFORE") return date.getTime() < Date.parse(`${filter.value}T00:00:00Z`)
        if (filter.operator === "AFTER") return date.getTime() > Date.parse(`${filter.value}T00:00:00Z`)
        const value = filter.field.endsWith("_year")
          ? date.getUTCFullYear()
          : filter.field.endsWith("_month")
            ? date.getUTCMonth() + 1
            : date.getUTCDate()
        return value === filter.value
      }
      const candidates = source.separate ? [source.days] : source.days.map((day) => [day])
      const filter = source.filter
      const outcomes = candidates.map((days) =>
        "op" in filter && filter.op === "not"
          ? !days.some((day) => truth(day, filter.items[0]))
          : days.some((day) => truth(day, filter)),
      )
      const expected = outcomes.every(Boolean) ? "definite" : outcomes.some(Boolean) ? "possible" : "excluded"
      expect(run([source.raw], { operation: "count", filter: source.filter }).coverage?.[expected]).toBe(1)
    },
  )
  test("hidden source date permissions prevent use of unverified normalized components", () => {
    const source = snapshot(["1961-00-00"])
    const restricted = {
      ...source,
      dataset: { ...dataset, fields: dataset.fields.filter((field) => field.name !== "admitted_date") },
    }
    expect(() => ResearchQuery.execute(restricted, { operation: "count", filter: year(1961) })).toThrow("permissions")
    expect(() => ResearchQuery.execute(restricted, { operation: "group", groupBy: ["admitted_date_year"] })).toThrow(
      "permissions",
    )
  })
})
