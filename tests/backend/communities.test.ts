import { expect, test } from "bun:test"
import { ResearchState } from "@nia/backend/research/state"

const snapshot = {
  binding: { database: "db", fileID: 49 as const, principal: "nia-runtime" },
  dataset: { fileID: 49 as const, label: "Master", version: 1, fields: [] },
  at: "2026-10-27T00:00:00Z",
  fingerprint: "f",
  records: [
    { sourceRowID: "1", values: { student_name: "A", community: "Garden River" } },
    { sourceRowID: "2", values: { student_name: "B", community: "BATCHEWANA" } },
    { sourceRowID: "3", values: { student_name: "C", community: "Thessalon" } },
    { sourceRowID: "4", values: { student_name: "D", community: null } },
    { sourceRowID: "5", values: { student_name: "E" } },
  ],
}

test("only records of the selected communities remain, matched without regard to case", () => {
  const limited = ResearchState.withinCommunities(snapshot, ["Batchewana", "Garden River"], true)
  expect(limited.records.map((record) => record.sourceRowID)).toEqual(["1", "2"])
  expect(limited.records[0]!.values.community).toBe("Garden River")
  expect(limited.fingerprint).toBe(snapshot.fingerprint)
})

test("the community field is dropped again when the caller did not ask for it", () => {
  const limited = ResearchState.withinCommunities(snapshot, ["Thessalon"], false)
  expect(limited.records).toEqual([{ sourceRowID: "3", values: { student_name: "C" } }])
})
