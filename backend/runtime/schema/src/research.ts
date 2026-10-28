export * as Research from "./research.js"

import { Schema } from "effect"
import { optional } from "./schema.js"

export const FileID = Schema.Literals([49, 91, 93, 94]).annotate({ identifier: "Research.FileID" })
export type FileID = typeof FileID.Type

export const Binding = Schema.Struct({
  database: Schema.String.check(Schema.isMinLength(1)),
  fileID: FileID,
  principal: Schema.String.pipe(optional),
}).annotate({ identifier: "Research.Binding" })
export interface Binding extends Schema.Schema.Type<typeof Binding> {}

export const labels: { [ID in FileID]: string } = {
  49: "Student master list",
  91: "Confirmed deaths",
  93: "Additional Deaths",
  94: "Potential",
}

export const Operator = Schema.Literals([
  "EQUALS",
  "NOT_EQUALS",
  "CONTAINS",
  "NOT_CONTAINS",
  "STARTS_WITH",
  "IN",
  "NOT_IN",
  "IS_KNOWN",
  "IS_UNKNOWN",
  "GREATER_THAN",
  "LESS_THAN",
  "NUMBER_RANGE",
  "YEAR_EQUALS",
  "BEFORE",
  "AFTER",
  "DATE_RANGE",
  "IS_TRUE",
  "IS_FALSE",
  "CONTAINS_ANY",
  "NOT_CONTAINS_ANY",
  "HAS_MULTIPLE_EVENTS",
])
const Scalar = Schema.Union([Schema.String, Schema.Finite, Schema.Boolean, Schema.Null])
export const Predicate = Schema.Struct({
  field: Schema.String.check(Schema.isMinLength(1), Schema.isMaxLength(160)),
  operator: Operator,
  value: Schema.Union([Scalar, Schema.Array(Scalar).check(Schema.isMaxLength(100))]).pipe(optional),
})
export type Predicate = typeof Predicate.Type

export type Filter = Predicate | { readonly op: "and" | "or" | "not"; readonly items: readonly Filter[] }
export const Filter: Schema.Codec<Filter> = Schema.Union([
  Predicate,
  Schema.Struct({ op: Schema.Literals(["and", "or", "not"]), items: Schema.Array(Schema.suspend(() => Filter)) }),
])

export const Part = Schema.Literals(["value", "year", "decade", "month", "day"])
export const Query = Schema.Struct({
  operation: Schema.Literals([
    "count",
    "list",
    "group",
    "stats",
    "interval",
    "duplicates",
    "completeness",
    "compare",
    "set",
    "school_transfers",
  ]),
  transferScope: Schema.Literals(["outward", "any"]).pipe(optional),
  unit: Schema.Literals(["records", "events"]).pipe(optional),
  eventField: Schema.String.pipe(optional),
  filter: Filter.pipe(optional),
  base: Schema.String.pipe(optional),
  sourceRows: Schema.Array(Schema.String).check(Schema.isMaxLength(100)).pipe(optional),
  other: Filter.pipe(optional),
  field: Schema.String.pipe(optional),
  percentiles: Schema.Array(Schema.Finite.check(Schema.isBetween({ minimum: 0, maximum: 1 })))
    .check(Schema.isMaxLength(10))
    .pipe(optional),
  part: Part.pipe(optional),
  groupBy: Schema.Array(Schema.String).check(Schema.isMaxLength(2)).pipe(optional),
  fields: Schema.Array(Schema.String).check(Schema.isMaxLength(12)).pipe(optional),
  endField: Schema.String.pipe(optional),
  sort: Schema.String.pipe(optional),
  direction: Schema.Literals(["asc", "desc"]).pipe(optional),
  missing: Schema.Literals(["separate", "exclude"]).pipe(optional),
  set: Schema.Literals(["intersection", "difference", "union"]).pipe(optional),
  limit: Schema.Int.check(Schema.isBetween({ minimum: 1, maximum: 100 })).pipe(optional),
  offset: Schema.Int.check(Schema.isBetween({ minimum: 0, maximum: 50000 })).pipe(optional),
}).annotate({ identifier: "Research.Query" })
export interface Query extends Schema.Schema.Type<typeof Query> {}

export const Field = Schema.Struct({
  name: Schema.String,
  label: Schema.String,
  type: Schema.String,
  operators: Schema.Array(Schema.String),
  aggregatable: Schema.Boolean,
  sortable: Schema.Boolean,
  quoteable: Schema.Boolean,
  evidence: Schema.Boolean,
})
export interface Field extends Schema.Schema.Type<typeof Field> {}

export const Dataset = Schema.Struct({
  fileID: FileID,
  label: Schema.String,
  version: Schema.Int,
  fields: Schema.Array(Field),
})
export interface Dataset extends Schema.Schema.Type<typeof Dataset> {}

export const Record = Schema.Struct({
  sourceRowID: Schema.String,
  values: Schema.Record(Schema.String, Schema.NullOr(Schema.String)),
})
export interface Record extends Schema.Schema.Type<typeof Record> {}

export const Snapshot = Schema.Struct({
  binding: Binding,
  dataset: Dataset,
  at: Schema.String,
  fingerprint: Schema.String,
  records: Schema.Array(Record),
})
export interface Snapshot extends Schema.Schema.Type<typeof Snapshot> {}

export const Evidence = Schema.Struct({
  id: Schema.String,
  sessionID: Schema.String,
  binding: Binding,
  version: Schema.Int,
  at: Schema.String,
  fingerprint: Schema.String,
  query: Query,
  total: Schema.Int,
  known: Schema.Int,
  missing: Schema.Int,
  complete: Schema.Boolean,
  semantics: Schema.String.pipe(optional),
  coverage: Schema.Struct({
    unit: Schema.Literals(["records", "events"]),
    population: Schema.Int,
    definite: Schema.Int,
    possible: Schema.Int,
    unknown: Schema.Int,
    uninterpretable: Schema.Int,
    excluded: Schema.Int,
  }).pipe(optional),
  result: Schema.Json,
})
export interface Evidence extends Schema.Schema.Type<typeof Evidence> {}

export const AnswerInput = Schema.Struct({
  evidence: Schema.Array(Schema.String).check(Schema.isMaxLength(8)),
  reason: Schema.Literals(["out_of_scope", "clarify", "unavailable"]).pipe(optional),
  quotes: Schema.Array(
    Schema.Struct({
      evidenceID: Schema.String,
      sourceRowID: Schema.String,
      field: Schema.String,
      text: Schema.String.check(Schema.isMinLength(1), Schema.isMaxLength(2000)),
    }),
  )
    .check(Schema.isMaxLength(8))
    .pipe(optional),
})
export type AnswerInput = typeof AnswerInput.Type

export const Answer = Schema.Struct({
  sessionID: Schema.String,
  binding: Binding,
  version: Schema.Int,
  at: Schema.String,
  text: Schema.String,
  evidence: Schema.Array(Evidence),
})
export interface Answer extends Schema.Schema.Type<typeof Answer> {}
