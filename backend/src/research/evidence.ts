export * as ResearchEvidence from "./evidence.js"

import { Research } from "@opencode/schema/research"
import { ResearchQuery } from "./query.js"
import { ResearchValues } from "./values.js"

export function answer(
  sessionID: string,
  snapshot: Research.Snapshot,
  input: Research.AnswerInput,
  evidence: readonly Research.Evidence[],
): Research.Answer {
  if (evidence.length !== input.evidence.length || new Set(input.evidence).size !== evidence.length)
    throw new Error("Answer evidence identities must be unique and available")
  if (!evidence.length && !input.reason) throw new Error("Factual answers require query evidence")
  if (evidence.length && input.reason) throw new Error("A limitation response cannot include factual evidence")
  evidence.forEach((item, index) => {
    if (
      item.id !== input.evidence[index] ||
      item.sessionID !== sessionID ||
      item.binding.fileID !== snapshot.binding.fileID ||
      item.binding.database !== snapshot.binding.database
    )
      throw new Error("Answer evidence belongs to another Session or list")
    if (
      item.semantics !== ResearchValues.semantics ||
      item.version !== snapshot.dataset.version ||
      item.fingerprint !== snapshot.fingerprint
    )
      throw new Error("Evidence is stale; replay its query against the latest published version")
    const actual = ResearchQuery.execute(snapshot, item.query)
    if (
      JSON.stringify(actual.result) !== JSON.stringify(item.result) ||
      actual.total !== item.total ||
      actual.known !== item.known ||
      JSON.stringify(actual.coverage) !== JSON.stringify(item.coverage) ||
      actual.complete !== item.complete
    )
      throw new Error("Evidence does not match the current query result")
  })
  const quotes = (input.quotes ?? []).map((quote) => {
    const cited = evidence.find((item) => item.id === quote.evidenceID)
    const record = snapshot.records.find((item) => item.sourceRowID === quote.sourceRowID)
    const field = snapshot.dataset.fields.find((item) => item.name === quote.field)
    if (
      !cited ||
      !record ||
      !field?.quoteable ||
      !field.evidence ||
      !ResearchQuery.matches(record, cited.query.filter, snapshot.dataset) ||
      (cited.query.sourceRows && !cited.query.sourceRows.includes(record.sourceRowID)) ||
      !record.values[quote.field]?.includes(quote.text)
    )
      throw new Error("Quote is not an exact permitted excerpt of the cited cohort")
    return `> ${safe(quote.text)}\n\n[Q${evidence.indexOf(cited) + 1}] Source row ${quote.sourceRowID} · ${safe(field.label)}`
  })
  const text =
    input.reason === "out_of_scope"
      ? "This Session can answer only from its permanently selected list. Create a separate Session to ask about another list; cross-list answers are unavailable."
      : input.reason === "clarify"
        ? "Please clarify the person, cohort, time range or measure you want to examine within this Session's selected list."
        : input.reason === "unavailable"
          ? "I could not establish a supported answer from this list's latest available evidence. No other list or older version was used."
          : evidence
              .map((item, index) => {
                const coverage = item.coverage
                return `### ${item.query.operation} [Q${index + 1}]\n\n${render(item.result)}\n\nPopulation: ${coverage?.population ?? item.total} ${coverage?.unit ?? "records"}; definite: ${item.total}; additional possible: ${coverage?.possible ?? 0}; unknown: ${coverage?.unknown ?? 0}; uninterpretable: ${coverage?.uninterpretable ?? 0}. Eligible for calculation: ${item.known}; excluded from calculation: ${item.missing}. ${item.complete ? "Complete result." : "Preview only; totals cover the full cohort."}\n\n[Q${index + 1}] ${safe(snapshot.dataset.label)} · v${item.version} · ${safe(item.at)} · ${item.id}`
              })
              .concat(quotes)
              .join("\n\n")
  return { sessionID, binding: snapshot.binding, version: snapshot.dataset.version, at: snapshot.at, text, evidence }
}

export function safe(value: string) {
  // Database/user text must never execute terminal control sequences.
  return value
    .replace(/\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))?/g, "")
    .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f]/g, "")
}

function render(value: Research.Evidence["result"]): string {
  if (value === null) return "Not recorded / undefined."
  if (typeof value !== "object") return safe(String(value))
  if (Array.isArray(value)) return value.map(render).join("\n") || "No matching values."
  return (
    Object.entries(value)
      .map(
        ([key, item]) =>
          `${safe(key)}: ${item !== null && typeof item === "object" ? `\n${render(item)}` : render(item)}`,
      )
      .join("\n") || "No matching records."
  )
}
