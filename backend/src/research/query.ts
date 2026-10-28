export * as ResearchQuery from "./query.js"

import { Research } from "@opencode/schema/research"
import { ResearchValues } from "./values.js"
import { ResearchTemporal } from "./temporal.js"
import { ResearchTransfers } from "./transfers.js"

export const revision = ResearchValues.retrieval

export type Search = { terms: readonly string[]; limit?: number; offset?: number }

export function narratives(dataset: Research.Dataset) {
  return dataset.fields.filter(
    (field) =>
      field.evidence &&
      field.quoteable &&
      field.type === "text" &&
      field.name !== "student_name" &&
      field.operators.includes("CONTAINS_ANY"),
  )
}

export function search(input: Search, dataset: Research.Dataset): Research.Query {
  const terms = [...new Set(input.terms.map((term) => term.trim()).filter(Boolean))]
  if (!terms.length || terms.length > 8 || terms.some((term) => term.length > 80))
    throw new Error("Search requires 1-8 nonempty literal terms of at most 80 characters")
  const fields = narratives(dataset).map((field) => field.name)
  if (!fields.length) throw new Error("No permitted narrative fields are searchable in this list")
  return {
    operation: "list",
    filter: { op: "or", items: fields.map((field) => ({ field, operator: "CONTAINS_ANY", value: terms })) },
    fields: [
      ...(dataset.fields.some((field) => field.name === "student_name" && field.evidence) ? ["student_name"] : []),
      ...fields,
    ],
    limit: input.limit ?? 5,
    ...(input.offset !== undefined ? { offset: input.offset } : {}),
  }
}

function projection(query: Research.Query, dataset: Research.Dataset) {
  const exposed = dataset.fields.filter((field) => field.evidence).map((field) => field.name)
  if (query.fields) return query.fields
  if (query.operation === "completeness") return exposed
  if (query.sourceRows)
    return [
      ...new Set([
        "student_name",
        "student_number",
        "admitted_date",
        "discharged_date",
        ...narratives(dataset).map((field) => field.name),
      ]),
    ]
      .filter((name) => exposed.includes(name))
      .slice(0, 12)
  return exposed.slice(0, 4)
}

function textScope(query: Research.Query, dataset: Research.Dataset) {
  if (!query.filter && !query.other) return undefined
  const collect = (filter: Research.Filter): Research.Predicate[] =>
    "op" in filter ? filter.items.flatMap(collect) : [filter]
  const available = narratives(dataset).map((field) => field.name)
  const searched = [
    ...new Set(
      [...(query.filter ? collect(query.filter) : []), ...(query.other ? collect(query.other) : [])]
        .filter(
          (item) =>
            available.includes(item.field) &&
            ["CONTAINS", "NOT_CONTAINS", "CONTAINS_ANY", "NOT_CONTAINS_ANY", "EQUALS", "NOT_EQUALS"].includes(
              item.operator,
            ),
        )
        .map((item) => item.field),
    ),
  ]
  if (!searched.length) return undefined
  return {
    searchedFields: searched,
    unsearchedNarrativeFields: available.filter((field) => !searched.includes(field)),
    interpretation:
      "Literal text matches only. Review source wording before counting semantic events or unique people. Zero matches do not establish absence outside these fields or terms. Use research_search to search all permitted narrative fields; do not infer missing facts from an omitted field.",
  }
}

export function fields(query: Research.Query, dataset: Research.Dataset) {
  if (query.operation === "school_transfers")
    return [
      ...new Set([
        ...narratives(dataset).map((field) => field.name),
        ...["student_name", "student_number", "admitted_date", "discharged_date"].filter((name) =>
          dataset.fields.some((field) => field.name === name && field.evidence),
        ),
      ]),
    ]
  const filterFields = (filter: Research.Filter): string[] =>
    "op" in filter ? filter.items.flatMap(filterFields) : [filter.field]
  const projected = projection(query, dataset)
  return [
    ...new Set([
      ...(query.filter ? filterFields(query.filter) : []),
      ...(query.other ? filterFields(query.other) : []),
      ...(query.groupBy ?? []),
      ...(query.field ? [query.field] : []),
      ...(query.eventField ? [query.eventField] : []),
      ...(query.endField ? [query.endField] : []),
      ...(query.sort ? [query.sort] : []),
      ...(["list", "set", "completeness"].includes(query.operation) ? projected : []),
      ...(["list", "set"].includes(query.operation) && !query.sort ? ["student_name"] : []),
    ]),
  ]
}

export function validate(query: Research.Query, dataset: Research.Dataset) {
  if (query.operation === "school_transfers" && !narratives(dataset).length)
    throw new Error(
      "No permitted source narratives are available for transfer review; unavailable evidence does not mean zero transfers",
    )
  if (
    query.operation === "school_transfers" &&
    (query.filter || query.base || query.sourceRows || query.other || query.unit === "events")
  )
    throw new Error(
      "School transfer review uses the whole permitted current list; choose transferScope outward or any, then page with limit/offset",
    )
  const fields = new Map(dataset.fields.map((field) => [field.name, field]))
  const requireField = (name: string | undefined, capability?: "aggregatable" | "sortable" | "evidence") => {
    if (!name) throw new Error("This operation requires a field")
    const field = fields.get(name)
    if (
      field &&
      capability === "evidence" &&
      !field.evidence &&
      ["first_name", "last_name", "name_search"].includes(name)
    )
      throw new Error(
        `Field ${name} is for name filtering, not source evidence. Fetch student_name for the recorded full name; keep the other requested evidence fields. A failed fetch proves nothing about the person.`,
      )
    if (field && capability === "evidence" && !field.evidence && ResearchTemporal.field(name, dataset))
      throw new Error(
        `Field ${name} is for filtering/aggregation, not source evidence. Fetch ${ResearchTemporal.field(name, dataset)!.parent} instead; a failed fetch does not prove missing data.`,
      )
    if (!field || (capability && !field[capability]))
      throw new Error(`Field is unavailable for this operation: ${name}`)
    const dateComponent = /^(.*_date)_(year|month|day)$/.exec(name)
    if (dateComponent && !fields.has(dateComponent[1]))
      throw new Error(
        "The source date is unavailable under this list's field permissions; its components cannot be verified",
      )
    return field
  }
  const predicates: Research.Predicate[] = []
  const visit = (filter: Research.Filter, depth: number) => {
    if (depth > 6 || predicates.length >= 64) throw new Error("Filter exceeds the query budget")
    if ("op" in filter) {
      if (!filter.items.length || filter.items.length > 16 || (filter.op === "not" && filter.items.length !== 1))
        throw new Error("Logical filters need nonempty children; NOT accepts exactly one child")
      filter.items.forEach((item) => visit(item, depth + 1))
      return
    }
    const field = requireField(filter.field)
    if (!field.operators.includes(filter.operator))
      throw new Error(`Unsupported operator for ${filter.field}: ${filter.operator}`)
    if (
      ["YEAR_EQUALS", "BEFORE", "AFTER", "DATE_RANGE", "HAS_MULTIPLE_EVENTS"].includes(filter.operator) &&
      field.type !== "date"
    )
      throw new Error("Temporal predicates require a date field")
    if (["GREATER_THAN", "LESS_THAN", "NUMBER_RANGE"].includes(filter.operator) && field.type !== "number")
      throw new Error("Numeric predicates require a numeric field")
    if (
      !["IS_KNOWN", "IS_UNKNOWN", "IS_TRUE", "IS_FALSE", "HAS_MULTIPLE_EVENTS"].includes(filter.operator) &&
      filter.value === undefined
    )
      throw new Error("Filter value is required")
    if (
      ["IN", "NOT_IN", "CONTAINS_ANY", "NOT_CONTAINS_ANY", "NUMBER_RANGE", "DATE_RANGE"].includes(filter.operator) &&
      !Array.isArray(filter.value)
    )
      throw new Error("This filter requires an array value")
    if (
      ["NUMBER_RANGE", "DATE_RANGE"].includes(filter.operator) &&
      (!Array.isArray(filter.value) || filter.value.length !== 2)
    )
      throw new Error("Ranges require exactly two bounds")
    if (
      ["GREATER_THAN", "LESS_THAN", "YEAR_EQUALS"].includes(filter.operator) &&
      ResearchValues.number(String(filter.value)) === null
    )
      throw new Error("A valid numeric filter value is required")
    if (
      filter.operator === "NUMBER_RANGE" &&
      Array.isArray(filter.value) &&
      (filter.value.some((value) => ResearchValues.number(String(value)) === null) ||
        Number(filter.value[0]) > Number(filter.value[1]))
    )
      throw new Error("Numeric range bounds must be valid and ordered")
    if (["BEFORE", "AFTER", "DATE_RANGE"].includes(filter.operator)) {
      const values = Array.isArray(filter.value) ? filter.value : [filter.value]
      const bounds = values.map((value) => ResearchValues.date(String(value)))
      if (
        bounds.some((bound) => bound.lower === undefined || bound.upper === undefined) ||
        (bounds.length === 2 && (bounds[0].lower ?? 0) > (bounds[1].upper ?? 0))
      )
        throw new Error("Date filter bounds must be valid, unambiguous and ordered")
    }
    predicates.push(filter)
  }
  if (query.filter) visit(query.filter, 0)
  if (query.other) visit(query.other, 0)
  if (query.eventField && query.unit !== "events")
    throw new Error("eventField requires unit events; omit it for record counts")
  if (query.unit === "events") {
    if (requireField(query.eventField, "aggregatable").type !== "date")
      throw new Error("Event counts require a date eventField")
    if (["compare", "set", "interval", "completeness"].includes(query.operation))
      throw new Error("This operation counts records; query one event field with count, list, group or stats")
    if (query.operation === "stats" && query.field !== query.eventField)
      throw new Error("Event statistics must measure the selected eventField")
    if (query.groupBy?.some((name) => ResearchTemporal.field(name, dataset)?.parent !== query.eventField))
      throw new Error("Event grouping must use the selected eventField or its date components")
  }
  query.fields?.forEach((name) => requireField(name, "evidence"))
  query.groupBy?.forEach((name) => requireField(name, "aggregatable"))
  if (
    ["group", "duplicates"].includes(query.operation) &&
    query.part &&
    query.part !== "value" &&
    query.groupBy?.some((name) => requireField(name).type !== "date")
  )
    throw new Error("Date grouping parts require date fields")
  if (query.sort) requireField(query.sort, "sortable")
  if (["group", "duplicates"].includes(query.operation) && !query.groupBy?.length)
    throw new Error("Grouping fields are required")
  if (["stats", "interval"].includes(query.operation)) requireField(query.field, "aggregatable")
  if (
    query.operation === "stats" &&
    query.part !== undefined &&
    query.part !== "value" &&
    requireField(query.field).type !== "date"
  )
    throw new Error("Date parts require a date field")
  if (query.operation === "stats" && (query.part ?? "value") === "value" && requireField(query.field).type !== "number")
    throw new Error("Statistics require a numeric field or an explicit date part")
  if (
    query.operation === "interval" &&
    (requireField(query.field).type !== "date" || requireField(query.endField, "aggregatable").type !== "date")
  )
    throw new Error("Intervals require two date fields")
  if (["compare", "set"].includes(query.operation) && !query.other)
    throw new Error("A second within-list cohort is required")
  return query
}

export function matches(record: Research.Record, filter?: Research.Filter, dataset?: Research.Dataset): boolean {
  return evaluator(filter, dataset)(record) === "definite"
}

type Event = { field: string; date: ResearchValues.DateValue; index: number }
type Row = { record: Research.Record; event?: Event; match: ResearchTemporal.Match }

function evaluator(
  filter?: Research.Filter,
  dataset?: Research.Dataset,
  parse = ResearchValues.dateCache(),
): (record: Research.Record, event?: Event) => ResearchTemporal.Match {
  if (!filter) return () => "definite"
  if ("op" in filter && filter.op === "not") {
    const child = evaluator(filter.items[0], dataset, parse)
    return (record, event) => ResearchTemporal.negate(child(record, event))
  }
  if (!("op" in filter) && filter.operator === "HAS_MULTIPLE_EVENTS")
    return (record) => {
      const value = parse(record.values[filter.field])
      if (value.kind === "unknown") return "unknown"
      if (value.kind === "invalid" || value.kind === "uncertain") return "uninterpretable"
      return value.kind === "events" && value.dates!.length > 1 ? "definite" : "excluded"
    }
  const parents = ResearchTemporal.parents(filter, dataset)
  if (parents.length === 1 && parents[0]) {
    const cache = new Map<string, ResearchTemporal.Match>()
    return (record, event) => {
      const date = event?.field === parents[0] ? event.date : parse(record.values[parents[0]])
      const cached = cache.get(date.raw)
      if (cached) return cached
      const result = ResearchTemporal.test(date, filter, dataset)
      cache.set(date.raw, result)
      return result
    }
  }
  if ("op" in filter) {
    const flatten = (item: Research.Filter): Research.Filter[] =>
      "op" in item && item.op === "and" ? item.items.flatMap(flatten) : [item]
    const items = filter.op === "and" ? filter.items.flatMap(flatten) : filter.items
    const grouped = new Map<string, Research.Filter[]>()
    const children = items
      .flatMap((item) => {
        const names = ResearchTemporal.parents(item, dataset)
        if (filter.op === "and" && names.length === 1 && names[0] && !("op" in item && item.op === "not")) {
          grouped.set(names[0], [...(grouped.get(names[0]) ?? []), item])
          return []
        }
        return [evaluator(item, dataset, parse)]
      })
      .concat(
        [...grouped.values()].map((items) =>
          evaluator(items.length === 1 ? items[0] : { op: "and", items }, dataset, parse),
        ),
      )
    return (record, event) =>
      ResearchTemporal.combine(
        children.map((child) => child(record, event)),
        filter.op === "and" ? "and" : "or",
      )
  }
  return (record) => {
    const result = evaluate(record, filter)
    return result === true ? "definite" : result === false ? "excluded" : "unknown"
  }
}

function evaluate(record: Research.Record, filter: Research.Predicate): boolean | undefined {
  const value = record.values[filter.field] ?? null
  if (filter.operator === "IS_KNOWN") return ResearchValues.known(value)
  if (filter.operator === "IS_UNKNOWN") return !ResearchValues.known(value)
  // Missing information never proves a negative fact.
  if (!ResearchValues.known(value)) return undefined
  const folded = value?.trim().toLocaleLowerCase("en") ?? ""
  const expected = String(filter.value ?? "")
    .trim()
    .toLocaleLowerCase("en")
  const list = Array.isArray(filter.value)
    ? filter.value.map((item) => String(item).trim().toLocaleLowerCase("en"))
    : []
  switch (filter.operator) {
    case "EQUALS":
      return typeof filter.value === "number" ? ResearchValues.number(value) === filter.value : folded === expected
    case "NOT_EQUALS":
      return typeof filter.value === "number"
        ? ResearchValues.number(value) === null
          ? undefined
          : ResearchValues.number(value) !== filter.value
        : folded !== expected
    case "CONTAINS":
      return folded.includes(expected)
    case "NOT_CONTAINS":
      return !folded.includes(expected)
    case "STARTS_WITH":
      return folded.startsWith(expected)
    case "IN":
      return list.includes(folded)
    case "NOT_IN":
      return !list.includes(folded)
    case "CONTAINS_ANY":
      return list.some((item) => folded.includes(item))
    case "NOT_CONTAINS_ANY":
      return !list.some((item) => folded.includes(item))
    case "IS_TRUE":
      return ["true", "t", "1", "yes", "y", "deceased"].includes(folded)
    case "IS_FALSE":
      return ["false", "f", "0", "no", "n"].includes(folded)
    case "GREATER_THAN":
    case "LESS_THAN":
    case "NUMBER_RANGE": {
      const numeric = ResearchValues.number(value)
      if (numeric === null) return undefined
      if (filter.operator === "NUMBER_RANGE") return numeric >= Number(list[0]) && numeric <= Number(list[1])
      return filter.operator === "GREATER_THAN" ? numeric > Number(filter.value) : numeric < Number(filter.value)
    }
  }
}

export function execute(
  snapshot: Research.Snapshot,
  input: Research.Query,
): Omit<Research.Evidence, "id" | "sessionID" | "binding" | "version" | "at" | "fingerprint" | "query"> {
  const query = validate(input, snapshot.dataset)
  if (query.operation === "school_transfers") {
    const result = ResearchTransfers.review(
      snapshot,
      narratives(snapshot.dataset).map((field) => field.name),
      query,
    )
    return {
      total: result.recognizedCandidateRecords,
      known: result.recognizedCandidateRecords,
      missing: 0,
      complete: (query.offset ?? 0) === 0 && result.cases.length === result.recognizedCandidateRecords,
      semantics: ResearchValues.semantics,
      coverage: {
        unit: "records",
        population: snapshot.records.length,
        definite: result.recognizedCandidateRecords,
        possible: 0,
        unknown: result.unclassifiedRecords,
        uninterpretable: 0,
        excluded: result.outsideRequestedScopeRecords,
      },
      result,
    }
  }
  const scope = textScope(query, snapshot.dataset)
  const parse = ResearchValues.dateCache(snapshot.dataset)
  const match = evaluator(query.filter, snapshot.dataset, parse)
  const sourceRows = query.sourceRows && new Set(query.sourceRows)
  if (sourceRows && [...sourceRows].some((id) => !snapshot.records.some((record) => record.sourceRowID === id)))
    throw new Error("Source row is unavailable in this Session's selected list")
  const scoped = sourceRows ? snapshot.records.filter((record) => sourceRows.has(record.sourceRowID)) : snapshot.records
  // Counts need no materialized rows/pages. Keep the common full-population count constant-time.
  if (query.operation === "count" && query.unit !== "events") {
    const counts = {
      definite: query.filter ? 0 : scoped.length,
      possible: 0,
      unknown: 0,
      uninterpretable: 0,
      excluded: 0,
    }
    if (query.filter) scoped.forEach((record) => counts[match(record)]++)
    return {
      total: counts.definite,
      known: counts.definite,
      missing: 0,
      complete: true,
      semantics: ResearchValues.semantics,
      coverage: { unit: "records" as const, population: scoped.length, ...counts },
      result: { recordCount: counts.definite, ...(scope ? { textSearch: scope } : {}) },
    }
  }
  const leftPopulation: Row[] = scoped.flatMap((record) => {
    if (query.unit !== "events") return [{ record, match: match(record) }]
    return ResearchValues.events(parse(record.values[query.eventField!])).map((date, index) => {
      const event = { field: query.eventField!, date, index }
      return {
        record,
        event,
        match: ResearchValues.dateKnown(date)
          ? match(record, event)
          : date.kind === "unknown"
            ? "unknown"
            : "uninterpretable",
      }
    })
  })
  const otherMatch = query.other && evaluator(query.other, snapshot.dataset, parse)
  const population =
    query.operation !== "set"
      ? leftPopulation
      : leftPopulation.map((row) => ({
          ...row,
          match: ResearchTemporal.combine(
            [
              row.match,
              query.set === "difference" ? ResearchTemporal.negate(otherMatch!(row.record)) : otherMatch!(row.record),
            ],
            query.set === "union" ? "or" : "and",
          ),
        }))
  const records = population.filter((row) => row.match === "definite")
  const possible = population.filter((row) => row.match === "possible")
  const fields = projection(query, snapshot.dataset)
  const value = (row: Row, name: string, part = "value") => {
    const temporal = ResearchTemporal.field(name, snapshot.dataset)
    if (!temporal) return ResearchValues.display(row.record.values[name], part)
    const date = row.event?.field === temporal.parent ? row.event.date : parse(row.record.values[temporal.parent])
    const selected = temporal.part === "value" ? part : temporal.part
    return selected === "value" ? (row.record.values[name] ?? null) : ResearchTemporal.scalar(date, selected)
  }
  const dateFields = fields.filter((name) => ResearchTemporal.field(name, snapshot.dataset)?.part === "value")
  const page = (items: readonly Row[]) =>
    items.slice(query.offset ?? 0, (query.offset ?? 0) + (query.limit ?? 25)).map((row) => {
      const dates = dateFields.flatMap((field) => {
        const date = parse(row.record.values[field])
        return date.kind === "exact" ? [] : [[field, ResearchValues.interpretation(date)] as const]
      })
      return {
        sourceRowID: row.record.sourceRowID,
        values: Object.fromEntries(fields.map((field) => [field, row.record.values[field] ?? null])),
        ...(dates.length ? { dateInterpretations: Object.fromEntries(dates) } : {}),
        ...(row.event ? { eventIndex: row.event.index, eventDate: row.event.date.raw } : {}),
      }
    })
  const known = (row: Row, name: string, part = "value") => {
    const temporal = ResearchTemporal.field(name, snapshot.dataset)
    if (!temporal) return ResearchValues.known(row.record.values[name])
    const date = row.event?.field === temporal.parent ? row.event.date : parse(row.record.values[temporal.parent])
    const selected = temporal.part === "value" ? part : temporal.part
    const component = (item: ResearchValues.DateValue): boolean =>
      item.dates ? item.dates.some(component) : ResearchValues.partBounds(item, selected) !== undefined
    return selected === "value"
      ? ResearchValues.dateKnown(date)
      : date.kind === "range"
        ? ResearchValues.partBounds(date, selected) !== undefined
        : component(date)
  }
  const result = ((): { result: Record<string, Research.Evidence["result"]>; known: number; complete: boolean } => {
    if (query.operation === "count")
      return {
        result:
          query.unit === "events"
            ? {
                eventCount: records.length,
                recordCount: new Set(records.map((row) => row.record.sourceRowID)).size,
                eventField: query.eventField!,
              }
            : { recordCount: records.length },
        known: records.length,
        complete: true,
      }
    if (query.operation === "stats") {
      const values = records
        .map((row) => ResearchValues.number(value(row, query.field!, query.part)))
        .filter((item) => item !== null)
      return {
        result: { ...ResearchValues.stats(values, query.percentiles), excludedCount: records.length - values.length },
        known: values.length,
        complete: true,
      }
    }
    if (query.operation === "interval") {
      const values = records.flatMap((row) => {
        const start = parse(row.record.values[query.field!])
        const end = parse(row.record.values[query.endField!])
        return start.kind === "exact" && end.kind === "exact" ? [(end.lower! - start.lower!) / 86400000] : []
      })
      return {
        result: {
          ...ResearchValues.stats(values, query.percentiles),
          negativeCount: values.filter((value) => value < 0).length,
          unit: "days",
          precision: "exact valid dates only",
          excludedCount: records.length - values.length,
        },
        known: values.length,
        complete: true,
      }
    }
    if (query.operation === "group" || query.operation === "duplicates") {
      const groups = new Map<string, { labels: (string | null)[]; count: number; possibleCount: number }>()
      const ranges = new Set<string>()
      const included = records.filter(
        (row) => query.missing !== "exclude" || query.groupBy!.every((field) => known(row, field, query.part)),
      )
      const combinations = (items: readonly (readonly (string | null)[])[]): (string | null)[][] =>
        items.reduce<(string | null)[][]>(
          (rows, values) => rows.flatMap((row) => values.map((value) => [...row, value])),
          [[]],
        )
      const add = (row: Row, possibleRow: boolean) => {
        const dimensions = query.groupBy!.map((name) => {
          const temporal = ResearchTemporal.field(name, snapshot.dataset)
          if (!temporal)
            return { definite: [value(row, name, query.part)], possible: [] as string[], ranges: [] as string[] }
          const date = row.event?.field === temporal.parent ? row.event.date : parse(row.record.values[temporal.parent])
          return ResearchTemporal.labels(date, temporal.part === "value" ? (query.part ?? "value") : temporal.part)
        })
        dimensions.flatMap((dimension) => dimension.ranges).forEach((range) => ranges.add(range))
        if (
          dimensions.reduce((total, dimension) => total * (dimension.definite.length + dimension.possible.length), 1) >
          400
        )
          throw new Error(
            "Date grouping exceeds the interpretation budget; narrow the time range or group one component",
          )
        const definite = new Set(
          combinations(dimensions.map((item) => item.definite)).map((labels) =>
            JSON.stringify(labels.map((label) => label?.toLocaleLowerCase("en"))),
          ),
        )
        const candidates = combinations(dimensions.map((item) => [...new Set([...item.definite, ...item.possible])]))
        candidates.forEach((labels) => {
          const temporalGroups = new Map<string, Research.Filter[]>()
          query.groupBy!.forEach((name, index) => {
            const temporal = ResearchTemporal.field(name, snapshot.dataset)
            if (!temporal || labels[index] === null) return
            const part = temporal.part === "value" ? (query.part ?? "value") : temporal.part
            const predicate: Research.Predicate = {
              field: part === "value" ? temporal.parent : `${temporal.parent}_${part}`,
              operator: "EQUALS",
              value: labels[index],
            }
            temporalGroups.set(temporal.parent, [...(temporalGroups.get(temporal.parent) ?? []), predicate])
          })
          const memberships = [...temporalGroups].map(([parent, items]) =>
            ResearchTemporal.test(
              row.event?.field === parent ? row.event.date : parse(row.record.values[parent]),
              { op: "and", items },
              snapshot.dataset,
            ),
          )
          if (memberships.includes("excluded")) return
          const key = JSON.stringify(labels.map((label) => label?.toLocaleLowerCase("en")))
          const group = groups.get(key) ?? { labels, count: 0, possibleCount: 0 }
          const certain = !possibleRow && definite.has(key) && memberships.every((item) => item === "definite")
          if (certain) group.count++
          if (!certain) group.possibleCount++
          groups.set(key, group)
        })
      }
      included.forEach((row) => add(row, false))
      possible.forEach((row) => add(row, true))
      const sorted = [...groups.values()]
        .filter((group) => query.operation !== "duplicates" || group.count + group.possibleCount >= 2)
        .toSorted(
          (a, b) =>
            b.count - a.count ||
            b.possibleCount - a.possibleCount ||
            JSON.stringify(a.labels).localeCompare(JSON.stringify(b.labels)),
        )
      return {
        result: {
          groupCount: sorted.length,
          groups: sorted.slice(query.offset ?? 0, (query.offset ?? 0) + (query.limit ?? 25)),
          unit: query.unit ?? "records",
          overlappingGroups:
            sorted.reduce((total, group) => total + group.count + group.possibleCount, 0) >
            included.length + possible.length,
          possibleRanges: [...ranges],
        },
        known: included.length,
        complete: !ranges.size && (query.offset ?? 0) === 0 && sorted.length <= (query.limit ?? 25),
      }
    }
    if (query.operation === "completeness") {
      const selected =
        query.fields ?? snapshot.dataset.fields.filter((field) => field.evidence).map((field) => field.name)
      return {
        result: {
          fields: selected.map((field): Research.Evidence["result"] => {
            if (!ResearchTemporal.field(field, snapshot.dataset))
              return { field, known: records.filter((row) => known(row, field)).length, total: records.length }
            const classify = evaluator({ field, operator: "IS_KNOWN" }, snapshot.dataset, parse)
            const states = records.map((row) => classify(row.record, row.event))
            return {
              field,
              total: records.length,
              known: states.filter((state) => state === "definite").length,
              possible: states.filter((state) => state === "possible").length,
              unknown: states.filter((state) => state === "excluded").length,
              uninterpretable: states.filter((state) => state === "uninterpretable").length,
            }
          }),
        },
        known: records.length,
        complete: true,
      }
    }
    if (query.operation === "set")
      return {
        result: {
          recordCount: records.length,
          records: page(records),
          possibleCount: possible.length,
          possibleRecords: page(possible),
        },
        known: records.length,
        complete: (query.offset ?? 0) === 0 && Math.max(records.length, possible.length) <= (query.limit ?? 25),
      }
    if (query.operation === "compare") {
      const right = population.map((row) => ({ ...row, match: otherMatch!(row.record) }))
      const definiteRight = right.filter((row) => row.match === "definite")
      const both = population.filter((row, index) => row.match === "definite" && right[index].match === "definite")
      const uncertain = right.filter((row) => row.match === "possible").length
      const rightUnknown = right.filter((row) => row.match === "unknown").length
      const rightUninterpretable = right.filter((row) => row.match === "uninterpretable").length
      return {
        result: {
          left: records.length,
          right: definiteRight.length,
          intersection: both.length,
          leftPossible: possible.length,
          rightPossible: uncertain,
          rightUnknown,
          rightUninterpretable,
          intersectionPossible: population.filter(
            (row, index) => ResearchTemporal.combine([row.match, right[index].match], "and") === "possible",
          ).length,
          leftPercentOfRight:
            population.every((row) => row.match === "definite" || row.match === "excluded") &&
            !uncertain &&
            !rightUnknown &&
            !rightUninterpretable &&
            definiteRight.length
              ? Number(((records.length / definiteRight.length) * 100).toFixed(4))
              : null,
        },
        known: records.length,
        complete: true,
      }
    }
    const sortField = snapshot.dataset.fields.find((field) => field.name === query.sort)
    const sortValue = (row: Row) => {
      if (sortField?.type === "number") return ResearchValues.number(value(row, sortField.name))
      if (sortField?.type === "date") return parse(row.record.values[sortField.name]).lower ?? null
      return value(row, query.sort ?? "student_name")
    }
    const sorted = records.toSorted((a, b) => {
      const av = sortValue(a)
      const bv = sortValue(b)
      return (
        (av === null || bv === null
          ? ResearchValues.compare(av, bv)
          : ResearchValues.compare(av, bv) * (query.direction === "desc" ? -1 : 1)) ||
        a.record.sourceRowID.localeCompare(b.record.sourceRowID, "en", { numeric: true })
      )
    })
    return {
      result: {
        records: page(sorted),
        ...(sortField?.type === "date"
          ? {
              ordering:
                "Possible lower date bounds only; overlapping, alternative and unknown dates do not establish an exact chronological order",
            }
          : {}),
        ...(possible.length ? { possibleCount: possible.length, possibleRecords: page(possible) } : {}),
      },
      known: records.length,
      complete: (query.offset ?? 0) === 0 && Math.max(records.length, possible.length) <= (query.limit ?? 25),
    }
  })()
  return {
    total: records.length,
    known: result.known,
    missing: Math.max(0, records.length - result.known),
    complete: result.complete,
    semantics: ResearchValues.semantics,
    coverage: {
      unit: query.unit ?? "records",
      population: population.length,
      definite: records.length,
      possible: possible.length,
      unknown: population.filter((row) => row.match === "unknown").length,
      uninterpretable: population.filter((row) => row.match === "uninterpretable").length,
      excluded: population.filter((row) => row.match === "excluded").length,
    },
    result: { ...result.result, ...(scope ? { textSearch: scope } : {}) },
  }
}

export function resolve(snapshot: Research.Snapshot, text: string) {
  const query = text.toLocaleLowerCase("en").trim()
  if (!query || query.length > 200) throw new Error("Provide a name, identifier or short search phrase")
  const terms = query.split(/\s+/)
  const exposed = new Set(snapshot.dataset.fields.filter((field) => field.evidence).map((field) => field.name))
  return snapshot.records
    .map((record) => {
      const name = exposed.has("student_name") ? (record.values.student_name?.toLocaleLowerCase("en") ?? "") : ""
      const identifier = exposed.has("student_number")
        ? (record.values.student_number?.toLocaleLowerCase("en") ?? "")
        : ""
      const corpus = Object.entries(record.values)
        .filter(([field]) => exposed.has(field))
        .map((entry) => entry[1])
        .filter((value): value is string => value !== null)
        .join(" ")
        .toLocaleLowerCase("en")
      const overlap = terms.filter((term) => corpus.includes(term)).length / terms.length
      const distance = levenshtein(name, query)
      const score =
        name === query || identifier === query || record.sourceRowID === query
          ? 10
          : name.includes(query)
            ? 5
            : overlap + Math.max(0, 1 - distance / Math.max(name.length, query.length))
      return {
        sourceRowID: record.sourceRowID,
        name: exposed.has("student_name") ? (record.values.student_name ?? null) : null,
        score: Number(score.toFixed(4)),
      }
    })
    .filter((record) => record.score >= 0.75)
    .toSorted((a, b) => b.score - a.score || a.sourceRowID.localeCompare(b.sourceRowID, "en", { numeric: true }))
    .slice(0, 20)
}

function levenshtein(a: string, b: string) {
  const row = Array.from({ length: b.length + 1 }, (_, index) => index)
  for (let i = 1; i <= a.length; i++) {
    let previous = row[0]
    row[0] = i
    for (let j = 1; j <= b.length; j++) {
      const next = row[j]
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1))
      previous = next
    }
  }
  return row[b.length]
}
