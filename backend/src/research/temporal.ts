export * as ResearchTemporal from "./temporal.js"

import { Research } from "@opencode/schema/research"
import { ResearchValues } from "./values.js"

export type Match = "definite" | "possible" | "unknown" | "uninterpretable" | "excluded"
export type DateField = { parent: string; part: string }

export function field(name: string, dataset?: Research.Dataset): DateField | undefined {
  const derived = /^(.*_date)_(year|decade|month|day)$/.exec(name)
  if (derived && (!dataset || dataset.fields.some((item) => item.name === derived[1])))
    return { parent: derived[1], part: derived[2] }
  if (dataset?.fields.find((item) => item.name === name)?.type === "date" || name.endsWith("_date"))
    return { parent: name, part: "value" }
}

export function combine(values: readonly Match[], op: "and" | "or"): Match {
  if (op === "and" && values.includes("excluded")) return "excluded"
  if (op === "or" && values.includes("definite")) return "definite"
  if (op === "or" && values.includes("possible")) return "possible"
  if (values.includes("uninterpretable")) return "uninterpretable"
  if (values.includes("unknown")) return "unknown"
  if (values.includes("possible")) return "possible"
  return op === "and" ? "definite" : "excluded"
}

export function negate(value: Match): Match {
  return value === "definite" ? "excluded" : value === "excluded" ? "definite" : value
}

export function test(value: ResearchValues.DateValue, filter: Research.Filter, dataset?: Research.Dataset): Match {
  if (!("op" in filter) && ["NOT_EQUALS", "NOT_IN"].includes(filter.operator))
    return negate(test(value, { ...filter, operator: filter.operator === "NOT_EQUALS" ? "EQUALS" : "IN" }, dataset))
  if (value.kind === "events")
    return combine(
      value.dates!.map((item) => test(item, filter, dataset)),
      "or",
    )
  if (value.kind === "alternatives") {
    const choices = value.dates!.map((item) => test(item, filter, dataset))
    if (choices.every((item) => item === "definite")) return "definite"
    if (choices.every((item) => item === "excluded")) return "excluded"
    if (choices.some((item) => item === "definite" || item === "possible")) return "possible"
    return combine(choices, "or")
  }
  if (value.kind === "range" || (value.kind === "partial" && value.year !== undefined)) {
    const result = rangeMatch(value, filter, dataset)
    if (result) return result
  }
  if ("op" in filter) {
    const interval = intervalMatch(value, filter, dataset)
    if (interval) return interval
    const results = filter.items.map((item) => test(value, item, dataset))
    if (filter.op === "not") return negate(results[0])
    if (filter.op === "and" && contradictory(value, filter, dataset)) return "excluded"
    return combine(results, filter.op)
  }
  const part = filter.operator === "YEAR_EQUALS" ? "year" : (field(filter.field, dataset)?.part ?? "value")
  if (["IN", "NOT_IN"].includes(filter.operator) && Array.isArray(filter.value)) {
    const interval = intervalMatch(
      value,
      { op: "or", items: filter.value.map((candidate) => ({ ...filter, operator: "EQUALS", value: candidate })) },
      dataset,
    )
    if (interval) return filter.operator === "NOT_IN" ? negate(interval) : interval
    const choices = filter.value.map((candidate) =>
      test(value, { ...filter, operator: "EQUALS", value: candidate }, dataset),
    )
    const result = combine(choices, "or")
    return filter.operator === "NOT_IN" ? negate(result) : result
  }
  if (
    part === "value" &&
    ["CONTAINS", "NOT_CONTAINS", "STARTS_WITH", "CONTAINS_ANY", "NOT_CONTAINS_ANY"].includes(filter.operator)
  ) {
    const raw = value.raw.toLocaleLowerCase("en")
    const candidates = Array.isArray(filter.value) ? filter.value : [filter.value]
    const found = candidates.some((candidate) =>
      filter.operator === "STARTS_WITH"
        ? raw.startsWith(String(candidate).toLocaleLowerCase("en"))
        : raw.includes(String(candidate).toLocaleLowerCase("en")),
    )
    return found !== filter.operator.startsWith("NOT_") ? "definite" : "excluded"
  }
  const bounds =
    part === "value"
      ? value.lower !== undefined && value.upper !== undefined
        ? { lower: value.lower, upper: value.upper }
        : undefined
      : ResearchValues.partBounds(value, part)
  if (
    value.kind === "invalid" ||
    (value.kind === "uncertain" && (!ResearchValues.dateKnown(value) || (part !== "value" && !bounds)))
  )
    return "uninterpretable"
  if (filter.operator === "IS_KNOWN" || filter.operator === "IS_UNKNOWN") {
    const known =
      part === "value"
        ? ResearchValues.dateKnown(value)
        : value.kind === "range"
          ? value.dates!.every((item) => ResearchValues.datePart(item, part) !== null)
          : bounds !== undefined
    return known === (filter.operator === "IS_KNOWN") ? "definite" : "excluded"
  }
  if (!bounds) {
    if (value.kind === "uncertain") return "uninterpretable"
    return "unknown"
  }
  const constraint = limits(filter, part)
  if (!constraint) return "uninterpretable"
  const matches =
    bounds.lower >= constraint.lower && bounds.upper <= constraint.upper
      ? "definite"
      : bounds.upper < constraint.lower || bounds.lower > constraint.upper
        ? "excluded"
        : "possible"
  return ["NOT_EQUALS", "NOT_IN"].includes(filter.operator) ? negate(matches) : matches
}

type Interval = { lower: number; upper: number }

function intervalMatch(
  value: ResearchValues.DateValue,
  filter: Research.Filter,
  dataset?: Research.Dataset,
): Match | undefined {
  const leaves = (item: Research.Filter): Research.Predicate[] => ("op" in item ? item.items.flatMap(leaves) : [item])
  const predicates = leaves(filter)
  const parts = new Set(
    predicates.map((item) =>
      item.operator === "YEAR_EQUALS" ? "year" : (field(item.field, dataset)?.part ?? "value"),
    ),
  )
  if (parts.size !== 1) return undefined
  const part = [...parts][0]
  const domain =
    part === "value"
      ? value.lower !== undefined && value.upper !== undefined
        ? { lower: value.lower, upper: value.upper }
        : undefined
      : ResearchValues.partBounds(value, part)
  if (!domain) return undefined
  const step = part === "value" ? 86400000 : part === "decade" ? 10 : 1
  const merge = (items: Interval[]) =>
    items
      .filter((item) => item.lower <= item.upper)
      .toSorted((a, b) => a.lower - b.lower)
      .reduce<Interval[]>((result, item) => {
        const previous = result.at(-1)
        if (previous && item.lower <= previous.upper + step) previous.upper = Math.max(previous.upper, item.upper)
        if (!previous || item.lower > previous.upper + step) result.push({ ...item })
        if (result.length > 128)
          throw new Error("Date filter exceeds the interpretation budget; simplify its alternatives")
        return result
      }, [])
  const intersection = (left: Interval[], right: Interval[]) =>
    merge(
      left.flatMap((a) => right.map((b) => ({ lower: Math.max(a.lower, b.lower), upper: Math.min(a.upper, b.upper) }))),
    )
  const complement = (items: Interval[]) => {
    const sorted = merge(items)
    return merge([
      { lower: domain.lower, upper: (sorted[0]?.lower ?? domain.upper + step) - step },
      ...sorted.map((item, index) => ({
        lower: item.upper + step,
        upper: (sorted[index + 1]?.lower ?? domain.upper + step) - step,
      })),
    ])
  }
  const select = (item: Research.Filter): Interval[] | undefined => {
    if ("op" in item) {
      const children = item.items.map(select)
      if (children.some((child) => !child)) return undefined
      const selected = children.filter((child) => child !== undefined)
      if (item.op === "not") return complement(selected[0])
      if (item.op === "or") return merge(selected.flat())
      return selected.reduce(intersection, [domain])
    }
    const values = ["IN", "NOT_IN"].includes(item.operator) && Array.isArray(item.value) ? item.value : [item.value]
    const intervals = values.map((candidate) =>
      limits(
        { ...item, operator: ["IN", "NOT_IN"].includes(item.operator) ? "EQUALS" : item.operator, value: candidate },
        part,
      ),
    )
    if (intervals.some((interval) => !interval)) return undefined
    const selected = intersection(
      [domain],
      intervals
        .filter((interval) => interval !== undefined)
        .map((interval) =>
          part === "value"
            ? { lower: Math.ceil(interval.lower / step) * step, upper: Math.floor(interval.upper / step) * step }
            : interval,
        ),
    )
    return ["NOT_EQUALS", "NOT_IN"].includes(item.operator) ? complement(selected) : selected
  }
  const selected = select(filter)
  if (!selected) return undefined
  if (!selected.length) return "excluded"
  return selected.length === 1 && selected[0].lower <= domain.lower && selected[0].upper >= domain.upper
    ? "definite"
    : "possible"
}

function limits(filter: Research.Predicate, part: string) {
  if (part !== "value") {
    const values = Array.isArray(filter.value) ? filter.value.map(Number) : [Number(filter.value)]
    if (values.some((item) => !Number.isFinite(item))) return undefined
    if (filter.operator === "GREATER_THAN") return { lower: Math.floor(values[0]) + 1, upper: Infinity }
    if (filter.operator === "LESS_THAN") return { lower: -Infinity, upper: Math.ceil(values[0]) - 1 }
    if (filter.operator === "NUMBER_RANGE") return { lower: Math.ceil(values[0]), upper: Math.floor(values[1]) }
    if (["EQUALS", "NOT_EQUALS", "YEAR_EQUALS"].includes(filter.operator))
      return { lower: Math.ceil(values[0]), upper: Math.floor(values[0]) }
    return undefined
  }
  const values = Array.isArray(filter.value) ? filter.value : [filter.value]
  const start = ResearchValues.date(String(values[0]))
  const end = filter.operator === "DATE_RANGE" ? ResearchValues.date(String(values[1])) : start
  if (start.lower === undefined || end.upper === undefined) return undefined
  if (filter.operator === "BEFORE") return { lower: -Infinity, upper: start.lower - 1 }
  if (filter.operator === "AFTER") return { lower: end.upper + 1, upper: Infinity }
  if (["EQUALS", "NOT_EQUALS", "DATE_RANGE"].includes(filter.operator)) return { lower: start.lower, upper: end.upper }
}

function contradictory(value: ResearchValues.DateValue, filter: Research.Filter, dataset?: Research.Dataset): boolean {
  if (!("op" in filter) || filter.op !== "and") return false
  const constraints = new Map<string, { lower: number; upper: number }>()
  const flatten = (item: Research.Filter): Research.Predicate[] =>
    "op" in item ? (item.op === "and" ? item.items.flatMap(flatten) : []) : [item]
  const conflict = filter.items.flatMap(flatten).some((item) => {
    if (["NOT_EQUALS", "NOT_IN"].includes(item.operator)) return false
    const part = item.operator === "YEAR_EQUALS" ? "year" : (field(item.field, dataset)?.part ?? "value")
    const next = limits(item, part)
    if (!next) return false
    const previous = constraints.get(part)
    const combined = {
      lower: Math.max(next.lower, previous?.lower ?? -Infinity),
      upper: Math.min(next.upper, previous?.upper ?? Infinity),
    }
    constraints.set(part, combined)
    return combined.lower > combined.upper
  })
  return conflict || !feasible(value, constraints)
}

function feasible(value: ResearchValues.DateValue, constraints: Map<string, Interval>) {
  const component = (part: "year" | "month" | "day", lower: number, upper: number) => {
    const recorded = value.kind === "range" ? undefined : value[part]
    const constraint = constraints.get(part)
    return {
      lower: Math.max(recorded ?? lower, constraint?.lower ?? lower),
      upper: Math.min(recorded ?? upper, constraint?.upper ?? upper),
    }
  }
  const timeline = constraints.get("value")
  const lower = Math.max(value.lower ?? -Infinity, timeline?.lower ?? -Infinity)
  const upper = Math.min(value.upper ?? Infinity, timeline?.upper ?? Infinity)
  if (lower > upper) return false
  const year = component("year", 1, 9999)
  const month = component("month", 1, 12)
  const day = component("day", 1, 31)
  const first = Math.max(year.lower, Number.isFinite(lower) ? new Date(lower).getUTCFullYear() : 1)
  const last = Math.min(year.upper, Number.isFinite(upper) ? new Date(upper).getUTCFullYear() : 9999)
  if (first > last || month.lower > month.upper || day.lower > day.upper) return false
  // At most eight years reach the next Gregorian leap day, including a skipped century.
  // Check month boundaries, never expand a range into individual calendar days.
  return Array.from({ length: Math.min(9, last - first + 1) }, (_, index) => first + index).some((year) =>
    Array.from({ length: month.upper - month.lower + 1 }, (_, index) => month.lower + index).some((month) => {
      const boundary = Number.isFinite(lower) ? new Date(lower) : undefined
      const minimum =
        boundary?.getUTCFullYear() === year && boundary.getUTCMonth() + 1 === month
          ? Math.max(day.lower, boundary.getUTCDate() + (lower % 86400000 ? 1 : 0))
          : day.lower
      if (minimum > day.upper) return false
      const candidate = new Date(0)
      candidate.setUTCFullYear(year, month - 1, minimum)
      return candidate.getUTCMonth() + 1 === month && candidate.getTime() >= lower && candidate.getTime() <= upper
    }),
  )
}

/** Bounded interval constraints preserve correlations across year/month/day and Boolean branches. */
function rangeMatch(
  value: ResearchValues.DateValue,
  filter: Research.Filter,
  dataset?: Research.Dataset,
): Match | undefined {
  const intersect = (left: Map<string, Interval>, right: Map<string, Interval>) => {
    const result = new Map(left)
    right.forEach((interval, part) => {
      const previous = result.get(part)
      result.set(part, {
        lower: Math.max(previous?.lower ?? -Infinity, interval.lower),
        upper: Math.min(previous?.upper ?? Infinity, interval.upper),
      })
    })
    return result
  }
  const clauses = (item: Research.Filter, negative = false): Map<string, Interval>[] | undefined => {
    if ("op" in item) {
      if (item.op === "not") return clauses(item.items[0], !negative)
      const children = item.items.map((child) => clauses(child, negative))
      if (children.some((child) => !child)) return undefined
      const selected = children.filter((child) => child !== undefined)
      const union = (item.op === "or") !== negative
      const result = union
        ? selected.flat()
        : selected.reduce<Map<string, Interval>[]>(
            (left, right) => {
              if (left.length * right.length > 128)
                throw new Error("Date filter exceeds the interpretation budget; simplify its Boolean branches")
              return left.flatMap((a) => right.map((b) => intersect(a, b)))
            },
            [new Map()],
          )
      if (result.length > 128)
        throw new Error("Date filter exceeds the interpretation budget; simplify its Boolean branches")
      return result
    }
    if (["NOT_EQUALS", "NOT_IN"].includes(item.operator))
      return clauses({ ...item, operator: item.operator === "NOT_IN" ? "IN" : "EQUALS" }, !negative)
    const part = item.operator === "YEAR_EQUALS" ? "year" : (field(item.field, dataset)?.part ?? "value")
    if (["IS_KNOWN", "IS_UNKNOWN"].includes(item.operator)) return undefined
    if (item.operator === "IN" && Array.isArray(item.value))
      return clauses(
        { op: "or", items: item.value.map((candidate) => ({ ...item, operator: "EQUALS", value: candidate })) },
        negative,
      )
    const interval = limits(item, part)
    if (!interval) return undefined
    const selected =
      part === "decade"
        ? { lower: Math.ceil(interval.lower / 10) * 10, upper: Math.floor(interval.upper / 10) * 10 + 9 }
        : interval
    const intervals = negative
      ? [
          { lower: -Infinity, upper: selected.lower - 1 },
          { lower: selected.upper + 1, upper: Infinity },
        ]
      : [selected]
    return intervals.map((interval) => new Map([[part === "decade" ? "year" : part, interval]]))
  }
  const positive = clauses(filter)
  const negative = clauses(filter, true)
  if (!positive || !negative) return undefined
  if (!positive.some((clause) => feasible(value, clause))) return "excluded"
  return negative.some((clause) => feasible(value, clause)) ? "possible" : "definite"
}

export function parents(filter: Research.Filter, dataset?: Research.Dataset): string[] {
  return "op" in filter
    ? [...new Set(filter.items.flatMap((item) => parents(item, dataset)))]
    : filter.operator === "HAS_MULTIPLE_EVENTS"
      ? [""] // This predicate describes the whole record's expression, not one event's date.
      : [field(filter.field, dataset)?.parent ?? ""]
}

/** Only a single, unambiguous component contributes to an exact scalar statistic. */
export function scalar(value: ResearchValues.DateValue, part: string) {
  return ResearchValues.datePart(value, part)
}

export function labels(
  value: ResearchValues.DateValue,
  part: string,
): { definite: (string | null)[]; possible: string[]; ranges: string[] } {
  if (value.kind === "events") {
    const items = value.dates!.map((item) => labels(item, part))
    const definite = [...new Set(items.flatMap((item) => item.definite))]
    return {
      definite,
      possible: [...new Set(items.flatMap((item) => item.possible))].filter((item) => !definite.includes(item)),
      ranges: items.flatMap((item) => item.ranges),
    }
  }
  if (part === "value") {
    if (value.kind === "alternatives") {
      const choices = value.dates!.map((item) => item.raw)
      return choices.every((item) => item === choices[0])
        ? { definite: [choices[0]], possible: [], ranges: [] }
        : { definite: [], possible: [...new Set(choices)], ranges: [] }
    }
    if (value.kind === "range") return { definite: [], possible: [], ranges: [value.raw] }
    return { definite: [ResearchValues.dateKnown(value) ? value.raw : null], possible: [], ranges: [] }
  }
  const exact = scalar(value, part)
  if (exact !== null) return { definite: [exact], possible: [], ranges: [] }
  if (value.kind === "alternatives") {
    const items = value.dates!.map((item) => labels(item, part))
    return {
      definite: [],
      possible: [
        ...new Set(items.flatMap((item) => [...item.definite.filter((label) => label !== null), ...item.possible])),
      ],
      ranges: items.flatMap((item) => item.ranges),
    }
  }
  if (value.kind === "range") {
    const bounds = ResearchValues.partBounds(value, part)
    const step = part === "decade" ? 10 : 1
    if (bounds && (bounds.upper - bounds.lower) / step < 200)
      return {
        definite: [],
        possible: Array.from({ length: (bounds.upper - bounds.lower) / step + 1 }, (_, index) =>
          String(bounds.lower + index * step),
        ),
        ranges: [],
      }
    return { definite: [], possible: [], ranges: [value.raw] }
  }
  return { definite: [null], possible: [], ranges: [] }
}
