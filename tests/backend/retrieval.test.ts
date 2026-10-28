import { describe, expect, test } from "bun:test"
import { Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchQuery } from "@nia/backend/research/query"

const field = (name: string, type = "text", evidence = true): Research.Field => ({
  name,
  label: name,
  type,
  evidence,
  quoteable: type === "text",
  aggregatable: type === "date",
  sortable: true,
  operators:
    type === "date" ? ["HAS_MULTIPLE_EVENTS", "YEAR_EQUALS", "IS_KNOWN"] : ["CONTAINS", "CONTAINS_ANY", "EQUALS"],
})
const snapshot: Research.Snapshot = {
  binding: { database: "fixture", fileID: 49 },
  at: "2026-10-17",
  fingerprint: "current",
  dataset: {
    fileID: 49,
    label: "Master",
    version: 5,
    fields: [
      field("student_name"),
      field("notes"),
      field("additional_information"),
      field("admitted_date", "date"),
      field("first_name", "text", false),
      field("private_notes", "text", false),
    ],
  },
  records: [
    {
      sourceRowID: "132545",
      values: {
        student_name: "Sophy Aundag",
        notes: null,
        additional_information: "Wawanosh Home. Transferred from Shingwauk.",
        admitted_date: "1878-05-27",
      },
    },
    {
      sourceRowID: "132432",
      values: {
        student_name: "Ernest Abetung",
        notes: "Archival reference",
        additional_information: "Transferred to Spanish IRS.",
        admitted_date: "1940-09-23",
      },
    },
    {
      sourceRowID: "133909",
      values: {
        student_name: "Leslie McCue",
        notes: null,
        additional_information: "To be Tansfered to Mohawk.",
        admitted_date: "1947-03-09",
      },
    },
    {
      sourceRowID: "132754",
      values: {
        student_name: "Gordon Murray Boissoneau",
        notes: "Was later sent to Spanish",
        additional_information: null,
        admitted_date: "1900-00-00 & 1902-00-00",
      },
    },
    {
      sourceRowID: "133325",
      values: {
        student_name: "Gifford Greenbird",
        notes: null,
        additional_information: "Transferred to Weston San.",
        admitted_date: "1900-00-00 | 1902-00-00",
      },
    },
    {
      sourceRowID: "unknown",
      values: { student_name: "Unknown", notes: null, additional_information: null, admitted_date: "0" },
    },
    {
      sourceRowID: "private",
      values: {
        student_name: "Private",
        notes: null,
        additional_information: null,
        private_notes: "Transferred to another school",
        admitted_date: "1900-00-00 : 1902-00-00",
      },
    },
  ],
}

describe("narrative retrieval coverage", () => {
  test("the failed notes-only recipe stays scoped rather than proving no transfers", () => {
    const result = ResearchQuery.execute(snapshot, {
      operation: "count",
      filter: { field: "notes", operator: "CONTAINS", value: "transfer" },
    })
    expect(result.total).toBe(0)
    expect(result.result).toMatchObject({
      textSearch: { searchedFields: ["notes"], unsearchedNarrativeFields: ["additional_information"] },
    })
  })

  test("search retrieves across every permitted narrative field, not private fields", () => {
    const query = ResearchQuery.search({ terms: ["transf", "tansfer", "sent to"], limit: 2 }, snapshot.dataset)
    const result = ResearchQuery.execute(snapshot, query)
    expect(query.fields).toEqual(["student_name", "notes", "additional_information"])
    expect(result.total).toBe(5)
    expect(result.complete).toBe(false)
    expect(result.result).toMatchObject({
      textSearch: { searchedFields: ["notes", "additional_information"], unsearchedNarrativeFields: [] },
    })
    // Plans and hospital moves are candidates too; the tool must not label keyword totals as verified transfers.
    expect(ResearchQuery.execute(snapshot, { ...query, limit: 10 }).total).toBe(5)
    expect(ResearchQuery.execute(snapshot, { ...query, offset: 2 }).total).toBe(5)
  })

  test("source hydration without projections includes the previously omitted transfer evidence", () => {
    const query: Research.Query = { operation: "list", sourceRows: ["132545"] }
    expect(ResearchQuery.fields(query, snapshot.dataset)).toContain("additional_information")
    expect(ResearchQuery.execute(snapshot, query).result).toMatchObject({
      records: [
        {
          sourceRowID: "132545",
          values: {
            student_name: "Sophy Aundag",
            additional_information: "Wawanosh Home. Transferred from Shingwauk.",
          },
        },
      ],
    })
    const narrow = { ...query, fields: ["notes"] }
    expect(ResearchQuery.execute(snapshot, narrow).result).toEqual({
      records: [{ sourceRowID: "132545", values: { notes: null } }],
    })
  })

  test("failed normalized name projection identifies the approved source field", () => {
    expect(() => ResearchQuery.execute(snapshot, { operation: "list", fields: ["first_name"] })).toThrow(
      "Fetch student_name",
    )
  })

  test("empty terms and lists without permitted narratives fail explicitly", () => {
    expect(() => ResearchQuery.search({ terms: [" "] }, snapshot.dataset)).toThrow("nonempty")
    expect(() =>
      ResearchQuery.search(
        { terms: ["transfer"] },
        { ...snapshot.dataset, fields: [field("private_notes", "text", false)] },
      ),
    ).toThrow("No permitted")
    expect(() =>
      Schema.decodeUnknownSync(Research.Query, { onExcessProperty: "error" })({
        ...ResearchQuery.search({ terms: ["transfer"] }, snapshot.dataset),
        fileID: 94,
      }),
    ).toThrow()
    expect(() =>
      ResearchQuery.execute(
        { ...snapshot, dataset: { ...snapshot.dataset, fields: [field("private_notes", "text", false)] } },
        { operation: "school_transfers" },
      ),
    ).toThrow("does not mean zero")
  })
})

describe("inspect multiple recorded admission events", () => {
  const filter: Research.Predicate = { field: "admitted_date", operator: "HAS_MULTIPLE_EVENTS" }
  test("separate events match; alternatives, ranges and unknown dates do not become two admissions", () => {
    const result = ResearchQuery.execute(snapshot, { operation: "count", filter })
    expect(result.total).toBe(1)
    expect(result.coverage?.unknown).toBe(1)
    expect(
      ResearchQuery.execute(snapshot, { operation: "list", filter, fields: ["admitted_date"] }).result,
    ).toMatchObject({ records: [{ sourceRowID: "132754", values: { admitted_date: "1900-00-00 & 1902-00-00" } }] })
  })
  test("the record-level predicate composes with year filters and negation", () => {
    expect(
      ResearchQuery.execute(snapshot, {
        operation: "count",
        filter: { op: "and", items: [filter, { field: "admitted_date", operator: "YEAR_EQUALS", value: 1902 }] },
      }).total,
    ).toBe(1)
    expect(ResearchQuery.execute(snapshot, { operation: "count", filter: { op: "not", items: [filter] } }).total).toBe(
      5,
    )
    expect(
      ResearchQuery.execute(snapshot, { operation: "count", filter, unit: "events", eventField: "admitted_date" })
        .total,
    ).toBe(2)
  })
})

test("an incoming transfer remains supported when the same record also contains a future outward plan", () => {
  const population: Research.Snapshot = {
    ...snapshot,
    records: [
      {
        sourceRowID: "dual",
        values: {
          student_name: "Example",
          additional_information: "Transferred from Chapleau IRS. To be transferred to Mohawk Institute.",
        },
      },
    ],
  }
  const any = ResearchQuery.execute(population, { operation: "school_transfers", transferScope: "any" })
  expect(any.result).toMatchObject({
    groupCountsComplete: true,
    groups: [expect.objectContaining({ category: "explicit_incoming_transfer", records: 1 })],
    cases: [
      expect.objectContaining({
        evidence: expect.arrayContaining([expect.objectContaining({ category: "planned_or_completion_uncertain" })]),
      }),
    ],
  })
  const outward = ResearchQuery.execute(population, { operation: "school_transfers", transferScope: "outward" })
  expect(outward.result).toMatchObject({
    explicitTransferEntries: 0,
    groups: [expect.objectContaining({ category: "planned_or_completion_uncertain", records: 1 })],
  })
})

test("outward coverage separates incoming transfers from unclassified source records", () => {
  const population: Research.Snapshot = {
    ...snapshot,
    records: [
      { sourceRowID: "incoming", values: { additional_information: "Transferred from Chapleau IRS." } },
      { sourceRowID: "outgoing", values: { additional_information: "Transferred to Spanish IRS." } },
      { sourceRowID: "unclassified", values: { additional_information: "No school history recorded here." } },
    ],
  }
  const outward = ResearchQuery.execute(population, { operation: "school_transfers" })
  expect(outward.total).toBe(1)
  expect(outward.coverage).toMatchObject({ population: 3, definite: 1, excluded: 1, unknown: 1 })
  expect(outward.result).toMatchObject({ outsideRequestedScopeRecords: 1, unclassifiedRecords: 1 })
  const any = ResearchQuery.execute(population, { operation: "school_transfers", transferScope: "any" })
  expect(any.total).toBe(2)
  expect(any.coverage).toMatchObject({ population: 3, definite: 2, excluded: 0, unknown: 1 })
})
