export * as ResearchValues from "./values.js"

export function number(value: string | null | undefined) {
  if (!value || !/^-?\d+(?:\.\d+)?(?:\s*(?:years?|yrs?))?$/i.test(value.trim())) return null
  const result = Number.parseFloat(value)
  return Number.isFinite(result) ? result : null
}

export const semantics = "dates-2"
export const retrieval = "retrieval-3"

export type DateValue = {
  kind:
    | "unknown"
    | "invalid"
    | "uncertain"
    | "exact"
    | "year"
    | "month"
    | "partial"
    | "range"
    | "alternatives"
    | "events"
  raw: string
  year?: number
  month?: number
  day?: number
  lower?: number
  upper?: number
  dates?: readonly DateValue[]
  limitation?: string
}

export function date(value: string | null | undefined): DateValue {
  const raw = value ?? ""
  const text = raw.trim()
  if (!text || /^0+(?:[-/.]0+)*$/.test(text)) return { kind: "unknown", raw }
  const iso = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text)
  if (iso) return components(raw, Number(iso[1]), Number(iso[2]), Number(iso[3]))
  if (
    /[?~]|\b(?:circa|about|approx(?:imate(?:ly)?)?|before|after|between|possibly|probably|maybe|uncertain|estimated)\b/i.test(
      text,
    )
  )
    return { kind: "uncertain", raw, limitation: "Qualified date; no exact date inferred" }
  const expression = text.replace(/\s*\([^)]*\)$/, "")
  const operators = ["&", "|", ":"].filter((operator) => expression.includes(operator))
  if (operators.length > 1) return { kind: "uncertain", raw, limitation: "Mixed date operators require clarification" }
  if (operators.length) {
    const dates = expression.split(operators[0]).map((item) => date(item.trim()))
    const kind = operators[0] === "&" ? "events" : operators[0] === "|" ? "alternatives" : "range"
    if (dates.length > 64 || (kind === "range" && dates.length !== 2))
      return { kind: "uncertain", raw, limitation: "Unsupported date expression" }
    if (kind !== "range") return { kind, raw, dates }
    if (dates.some((item) => item.lower === undefined || item.upper === undefined))
      return { kind: "uncertain", raw, limitation: "Range endpoints need an ordered, recorded year" }
    if (dates[0].lower! > dates[1].upper!) return { kind: "invalid", raw, limitation: "Reversed date range" }
    return { kind, raw, dates, lower: dates[0].lower, upper: dates[1].upper }
  }
  const shorthand = /^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})$/.exec(expression)
  if (shorthand)
    return {
      kind: "alternatives",
      raw,
      dates: [shorthand[3], shorthand[4]].map((day) => date(`${shorthand[1]}-${shorthand[2]}-${day}`)),
    }
  const exact = /^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:\s*\([^)]*\))?$/.exec(text)
  const reversed = /^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:\s*\([^)]*\))?$/.exec(text)
  if (reversed && Number(reversed[1]) <= 12 && Number(reversed[2]) <= 12 && reversed[1] !== reversed[2])
    return {
      kind: "uncertain",
      raw,
      year: Number(reversed[3]) || undefined,
      limitation: "Ambiguous day/month; recorded year retained",
    }
  if (exact || reversed) {
    const year = Number(exact?.[1] ?? reversed?.[3])
    const month = Number(exact?.[2] ?? reversed?.[2])
    const day = Number(exact?.[3] ?? reversed?.[1])
    return components(raw, year, month, day)
  }
  const month = /^(\d{4})[-/.](\d{1,2})$/.exec(text)
  if (month) {
    return components(raw, Number(month[1]), Number(month[2]), 0)
  }
  const year = /^(\d{4})(?:\s*\([^)]*\))?$/.exec(text)
  if (year) return components(raw, Number(year[1]), 0, 0)
  const range = /^(\d{4})\s*[-–/]\s*(\d{4})$/.exec(text)
  if (range && Number(range[1]) <= Number(range[2])) return { ...date(`${range[1]} : ${range[2]}`), raw }
  return { kind: "uncertain", raw, limitation: "Unsupported date expression" }
}

function utc(year: number, month: number, day: number) {
  // Date.UTC maps years 0–99 into 1900–1999; a Gregorian cycle preserves the recorded year.
  return year >= 100 ? Date.UTC(year, month - 1, day) : Date.UTC(year + 400, month - 1, day) - 146097 * 86400000
}

function components(raw: string, year: number, month: number, day: number): DateValue {
  const leap = !year || (year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0))
  const last = month ? [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1] : 31
  if (month > 12 || day > last || day > 31)
    return { kind: "invalid", raw, limitation: "Invalid recorded calendar components" }
  const parts = { raw, year: year || undefined, month: month || undefined, day: day || undefined }
  if (!year && !month && !day) return { kind: "unknown", raw }
  if (!year) return { ...parts, kind: "partial" }
  return {
    ...parts,
    kind: month && day ? "exact" : month ? "month" : day ? "partial" : "year",
    lower: utc(year, month || 1, day || 1),
    upper: utc(year, month || 12, day || last),
  }
}

export function events(value: DateValue) {
  return value.kind === "events" ? value.dates! : [value]
}

export function dateKnown(value: DateValue): boolean {
  if (value.dates) return value.dates.some(dateKnown)
  return value.year !== undefined || value.month !== undefined || value.day !== undefined
}

export function partBounds(value: DateValue, part: string): { lower: number; upper: number } | undefined {
  if (part === "decade") {
    const year = partBounds(value, "year")
    return year && { lower: Math.floor(year.lower / 10) * 10, upper: Math.floor(year.upper / 10) * 10 }
  }
  if (value.kind === "range") {
    const start = value.dates![0]
    const end = value.dates![1]
    if (part === "year") return { lower: start.year!, upper: end.year! }
    if (part === "month")
      return start.year === end.year && start.month && end.month
        ? { lower: start.month, upper: end.month }
        : { lower: 1, upper: 12 }
    if (part === "day")
      return start.year === end.year && start.month && start.month === end.month && start.day && end.day
        ? { lower: start.day, upper: end.day }
        : { lower: 1, upper: 31 }
  }
  const component =
    part === "year" ? value.year : part === "month" ? value.month : part === "day" ? value.day : undefined
  return component === undefined ? undefined : { lower: component, upper: component }
}

export function datePart(value: DateValue, part: string): string | null {
  if (value.kind === "events" || value.kind === "alternatives") {
    const parts = value.dates!.map((item) => datePart(item, part))
    return parts[0] !== null && parts.every((item) => item === parts[0]) ? parts[0] : null
  }
  const bounds = partBounds(value, part)
  return bounds && bounds.lower === bounds.upper ? String(bounds.lower) : null
}

const dateCaches = new WeakMap<object, (value: string | null | undefined) => DateValue>()

export function dateCache(scope?: object) {
  const existing = scope && dateCaches.get(scope)
  if (existing) return existing
  const cache = new Map<string, DateValue>()
  const parse = (value: string | null | undefined) => {
    const key = value ?? ""
    const cached = cache.get(key)
    if (cached) return cached
    const parsed = date(value)
    cache.set(key, parsed)
    return parsed
  }
  // A decoded snapshot owns its dataset object; adapter derivation and execution share one parse pass.
  // Weak ownership releases source values with that snapshot; a new database read gets a new scope.
  if (scope) dateCaches.set(scope, parse)
  return parse
}

export type DateInterpretation = {
  kind: DateValue["kind"]
  year?: number
  month?: number
  day?: number
  dates?: DateInterpretation[]
  limitation?: string
}

export function interpretation(value: DateValue): DateInterpretation {
  return {
    kind: value.kind,
    ...(value.year ? { year: value.year } : {}),
    ...(value.month ? { month: value.month } : {}),
    ...(value.day ? { day: value.day } : {}),
    ...(value.dates ? { dates: value.dates.map(interpretation) } : {}),
    ...(value.limitation ? { limitation: value.limitation } : {}),
  }
}

export function known(value: string | null | undefined) {
  return value !== null && value !== undefined && value.trim() !== ""
}

export function display(value: string | null | undefined, part = "value") {
  if (part === "value") return known(value) ? (value?.trim() ?? null) : null
  return datePart(date(value), part)
}

export function compare(a: string | number | null, b: string | number | null) {
  if (a === b) return 0
  if (a === null) return 1
  if (b === null) return -1
  if (typeof a === "number" && typeof b === "number") return a - b
  return String(a).localeCompare(String(b), "en", { sensitivity: "base", numeric: true })
}

export function stats(values: readonly number[], percentiles: readonly number[] = [0.25, 0.75, 0.9]) {
  const sorted = values.toSorted((a, b) => a - b)
  const frequencies = new Map<number, number>()
  values.forEach((value) => frequencies.set(value, (frequencies.get(value) ?? 0) + 1))
  const highest = Math.max(0, ...frequencies.values())
  const sum = values.reduce((total, value) => total + value, 0)
  const mean = values.length ? sum / values.length : null
  const squares = mean === null ? 0 : values.reduce((total, value) => total + (value - mean) ** 2, 0)
  const modes = [...frequencies]
    .filter((entry) => entry[1] === highest)
    .map((entry) => entry[0])
    .toSorted((a, b) => a - b)
  return {
    count: values.length,
    sum: values.length ? sum : null,
    distinctCount: frequencies.size,
    min: sorted[0] ?? null,
    max: sorted.at(-1) ?? null,
    mean: values.length ? Number((values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(4)) : null,
    median: values.length
      ? (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2
      : null,
    modes: modes.slice(0, 100),
    modeValueCount: modes.length,
    modesComplete: modes.length <= 100,
    modeCount: highest,
    standardDeviationPopulation: values.length ? Number(Math.sqrt(squares / values.length).toFixed(4)) : null,
    standardDeviationSample: values.length > 1 ? Number(Math.sqrt(squares / (values.length - 1)).toFixed(4)) : null,
    percentiles: Object.fromEntries(
      percentiles.map((probability) => {
        const position = (sorted.length - 1) * probability
        const lower = sorted[Math.floor(position)]
        const upper = sorted[Math.ceil(position)]
        return [
          `p${probability * 100}`,
          sorted.length ? Number((lower + (upper - lower) * (position % 1)).toFixed(4)) : null,
        ]
      }),
    ),
    percentileMethod: "linear interpolation (PostgreSQL percentile_cont)",
  }
}
