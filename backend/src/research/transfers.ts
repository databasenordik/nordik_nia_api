export * as ResearchTransfers from "./transfers.js"

import { Research } from "@opencode/schema/research"
import { ResearchValues } from "./values.js"

const school =
  /\b(?:school|i\.?r\.?s\.?|p\.?s\.?|h\.?s\.?|institute|institution|collegiate|wawanosh|shingwauk|spanish|sioux lookout|pelican lake|red lake mennonite|elkhorn|cecilia jeffr(?:ey|ies))\b/i
const medical = /\b(?:san\.|sanitorium|sanatorium|sanatarium|hospital)/i
const movement =
  /\b(?:transferred|transfered|transferd|tfrd\.?|tfd\.?|tansfered)\s+(to|from)\s+([^;\n]+?)(?=\.\s+(?:Note|Grade|Transferred|To|Religion|Condition|Total|After|Married)\b|;|\n|$)/gi

export function review(snapshot: Research.Snapshot, fields: readonly string[], query: Research.Query) {
  const parse = ResearchValues.dateCache(snapshot.dataset)
  const recognized = snapshot.records.flatMap((record) => {
    const found = fields.flatMap((field) => {
      const text = record.values[field] ?? ""
      const transfers = [...text.matchAll(movement)].map((match) => {
        const destination = match[2].slice(0, 220)
        const planned = /\bto be\s+(?:transferred|transfered|tansfered)\b/i.test(text)
        const year = /\b(?:\d{1,2}[./-]){2}(\d{4})\b|\b(\d{4})\b/.exec(destination)
        const admitted = ResearchValues.events(parse(record.values.admitted_date))
          .map((date) => ResearchValues.datePart(date, "year"))
          .filter((value): value is string => value !== null)
          .map(Number)
        const earlier =
          match[1].toLowerCase() === "to" &&
          year &&
          admitted.length &&
          Number(year[1] ?? year[2]) < Math.min(...admitted)
        const category =
          medical.test(destination) && !/\bschool\b/i.test(destination)
            ? "medical_move"
            : /\bagency\b/i.test(destination) || !school.test(destination)
              ? "school_unspecified"
              : match[1].toLowerCase() === "from" &&
                  /\bwawanosh\s+home\b/i.test(text) &&
                  /\bshingwauk\b/i.test(destination)
                ? "explicit_transfer_to_school"
                : match[1].toLowerCase() === "from"
                  ? "explicit_incoming_transfer"
                  : planned
                    ? "planned_or_completion_uncertain"
                    : /\b\d{4}\s*\?/.test(destination)
                      ? "transfer_date_uncertain"
                      : earlier
                        ? "earlier_school_transfer"
                        : "explicit_transfer_to_school"
        return {
          category,
          field,
          quote: text
            .slice(
              category === "explicit_transfer_to_school" && match[1].toLowerCase() === "from"
                ? Math.max(0, match.index - 40)
                : match.index,
              match.index + match[0].length,
            )
            .slice(0, 240),
        }
      })
      if (transfers.length) return transfers
      if (/\bday s(?:chool|\.)/i.test(text) && /\bto (?:be discharged|attend day school)\b/i.test(text))
        return [{ category: "proposed_day_school_discharge", field, quote: text.slice(0, 240) }]
      if (/\btransferring\b.{0,100}\bprivate home\b/i.test(text))
        return [{ category: "residential_placement_correspondence", field, quote: text.slice(0, 240) }]
      const moves = [
        ...text.matchAll(
          /\b(?:sent|sen|removed|went)\s+(?:lately\s+)?to\s+([^;\n]+?)(?=\.\s+[A-Z]|;|\n|$)|\bdischarged\b.{0,40}?\bto\s+(?:go to school|nursing school)[^;\n]*/gi,
        ),
      ].filter(
        (match) =>
          school.test(match[0]) &&
          !/\b(?:teach|staff)\b/i.test(match[0]) &&
          !(medical.test(match[0]) && !/\bschool\b/i.test(match[0])),
      )
      if (moves.length)
        return moves.map((match) => ({ category: "other_recorded_school_move", field, quote: match[0].slice(0, 240) }))
      if (/\bdischarged\b[\s.]*attending day school|\bleft shingwauk\b.{0,80}\battending\b/i.test(text))
        return [{ category: "later_attendance_only", field, quote: text.slice(0, 240) }]
      return []
    })
    if (!found.length) return []
    const priority = [
      "explicit_transfer_to_school",
      "earlier_school_transfer",
      ...((query.transferScope ?? "outward") === "any" ? ["explicit_incoming_transfer"] : []),
      "planned_or_completion_uncertain",
      "transfer_date_uncertain",
      "other_recorded_school_move",
      "proposed_day_school_discharge",
      "residential_placement_correspondence",
      "later_attendance_only",
      "explicit_incoming_transfer",
      "medical_move",
      "school_unspecified",
    ]
    const selected = priority.find((category) => found.some((item) => item.category === category))!
    return [
      {
        sourceRowID: record.sourceRowID,
        studentName: record.values.student_name ?? null,
        studentNumber: record.values.student_number ?? null,
        admittedDate: record.values.admitted_date ?? null,
        dischargedDate: record.values.discharged_date ?? null,
        category: selected,
        evidence: found,
        originExplicitlyShingwauk:
          selected === "explicit_transfer_to_school" &&
          fields.some(
            (field) =>
              /\bwawanosh\s+home\b/i.test(record.values[field] ?? "") &&
              /transferred from shingwauk/i.test(record.values[field] ?? ""),
          ),
        duplicateOf: null as string | null,
      },
    ]
  })
  const cases = recognized.filter(
    (item) =>
      query.transferScope === "any" ||
      !["explicit_incoming_transfer", "earlier_school_transfer", "school_unspecified"].includes(item.category),
  )
  const identities = new Map<string, string>()
  cases.forEach((item) => {
    if (!item.studentNumber || /-999$/.test(item.studentNumber)) return
    const date = parse(item.admittedDate)
    const key = JSON.stringify([
      item.studentNumber,
      item.studentName?.split(/[(/]/)[0].trim().toLowerCase(),
      date.year ?? ResearchValues.datePart(date, "year"),
      item.evidence.map((source) => source.quote),
    ])
    const prior = identities.get(key)
    if (prior) item.duplicateOf = prior
    if (!prior) identities.set(key, item.sourceRowID)
  })
  const retained = cases.filter((item) => !item.duplicateOf)
  const repeatedNames = Map.groupBy(retained, (item) => item.studentName?.trim().toLowerCase() ?? "")
  const groups = [...new Set(retained.map((item) => item.category))].map((category) => ({
    category,
    records: cases.filter((item) => item.category === category).length,
    casesAfterClearDuplicates: retained.filter((item) => item.category === category).length,
    sourceRows: retained.filter((item) => item.category === category).map((item) => item.sourceRowID),
  }))
  return {
    scope: query.transferScope ?? "outward",
    countingUnit:
      "Student entries, each assigned once to its strongest recognized category. A record may mention multiple transfers; these are not individual-event counts or certified unique children. Inspect all evidence on the case for secondary categories.",
    groupCountsComplete: true,
    paginationPurpose:
      "Group counts and sourceRows cover the whole permitted population. Do not page every case just to obtain counts. Pages are for inspecting individual source excerpts; hydrate specific source IDs when context is needed.",
    unresolvedSameNameGroups: [...repeatedNames.values()]
      .filter((items) => items.length > 1 && items[0].studentName)
      .map((items) => ({
        name: items[0].studentName,
        sourceRows: items.map((item) => item.sourceRowID),
        limitation: "Same name does not prove the same child; these entries were not merged.",
      })),
    explicitTransferEntries: retained.filter((item) => item.category === "explicit_transfer_to_school").length,
    explicitlyNamedShingwaukOrigins: retained.filter((item) => item.originExplicitlyShingwauk).length,
    originWarning:
      "A transfer-to-school statement does not establish a direct transfer FROM Shingwauk. Only originExplicitlyShingwauk cases explicitly name Shingwauk as origin; do not describe all explicitTransferEntries as direct transfers from Shingwauk. Other dated transfers may occur after discharge or after attending another school.",
    populationRecords: snapshot.records.length,
    recognizedCandidateRecords: cases.length,
    outsideRequestedScopeRecords: recognized.length - cases.length,
    unclassifiedRecords: snapshot.records.length - recognized.length,
    coverageInterpretation:
      "Candidate totals apply ONLY to the requested scope and include qualified/non-transfer categories. Outside-scope records can contain explicit incoming or earlier transfers. Unclassified records lack recognized patterns, not necessarily transfers. Neither group proves absence; never describe population minus candidates as children without transfers or without transfer wording.",
    groups,
    cases: cases.slice(query.offset ?? 0, (query.offset ?? 0) + (query.limit ?? 10)),
    limitation:
      "Conservative rules recognize recorded source wording, not all historical transfers. Categories are separate; never sum attendance, medical moves, residential placement, plans or uncertain dates into an unqualified completed-school-transfer count. Date uncertainty does not itself disprove the transfer. An outward statement does not always name Shingwauk as its origin. Earlier/incoming school histories are included only with scope any. Clear duplicates use matching name, student number, recorded admission year and source excerpt; remaining identities are unresolved, so these are not certified unique-child counts. The school-name vocabulary is limited; school_unspecified needs manual review. Quotes are exact excerpts, sometimes truncated; hydrate source rows for full context.",
  }
}
