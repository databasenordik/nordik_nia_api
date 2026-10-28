export * as ResearchState from "./state.js"

import { Effect, Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { Session } from "@opencode/schema/session"
import { KV } from "@opencode/core/kv"
import { SessionStore } from "@opencode/core/session/store"
import { ResearchQuery } from "./query.js"
import { ResearchEvidence } from "./evidence.js"
import { ResearchValues } from "./values.js"
import type { ResearchDatabase } from "./database.js"

export const make = (connection: ResearchDatabase.Connection) =>
  Effect.gen(function* () {
    const store = yield* SessionStore.Service
    const kv = yield* KV.Service
    const currentEvidence = new Map<string, Research.Evidence[]>()
    const binding = Effect.fn("Research.binding")(function* (id: string) {
      const session = yield* store.get(Session.ID.make(id))
      if (
        !session?.research ||
        session.research.database !== connection.identity ||
        session.research.principal !== connection.principal
      )
        return yield* Effect.fail(new Error("Session has no valid permanent list binding for this database"))
      return session.research
    })
    // A Session limited to some communities (the gRPC caller's selection) keeps that limit with its
    // state, and every snapshot applies it, so no tool reads records outside the selection.
    const communities = Effect.fn("Research.communities")(function* (id: string) {
      const value = yield* kv.get(`research/${id}/communities`)
      return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []
    })
    const snapshot = Effect.fn("Research.snapshot")(function* (id: string, fields?: readonly string[]) {
      const selected = yield* binding(id)
      const limit = yield* communities(id)
      const requested = limit.length && fields ? [...new Set([...fields, "community"])] : fields
      const current = yield* Effect.tryPromise({
        try: (signal) => connection.snapshot(selected, signal, requested),
        catch: () =>
          new Error("Selected list unavailable, access denied, or current data incomplete; no fallback was used"),
      })
      if (!limit.length) return current
      return withinCommunities(current, limit, !fields || fields.includes("community"))
    })
    const catalog = Effect.fn("Research.catalog")(function* (id: string) {
      const selected = yield* binding(id)
      return yield* Effect.tryPromise({
        try: (signal) => connection.catalog(selected.fileID, signal),
        catch: () => new Error("Selected list unavailable or access denied"),
      })
    })
    const load = Effect.fn("Research.evidence")(function* (sessionID: string, id: string) {
      if (!/^ev_[0-9a-f-]{36}$/.test(id)) return yield* Effect.fail(new Error("Invalid evidence identity"))
      const value = yield* kv.get(`research/${sessionID}/${id}`)
      const parsed = Schema.decodeUnknownOption(Research.Evidence)(value)
      if (parsed._tag === "None") return yield* Effect.fail(new Error("Evidence does not belong to this Session"))
      const selected = yield* binding(sessionID)
      if (
        parsed.value.sessionID !== sessionID ||
        parsed.value.binding.database !== selected.database ||
        parsed.value.binding.fileID !== selected.fileID
      )
        return yield* Effect.fail(new Error("Evidence conflicts with the permanent Session binding"))
      return parsed.value
    })
    const run = Effect.fn("Research.query")(function* (
      sessionID: string,
      input: Research.Query | ResearchQuery.Search,
      remember = true,
    ) {
      const dataset = yield* catalog(sessionID)
      const request = "terms" in input ? ResearchQuery.search(input, dataset) : input
      const prior = request.base ? yield* load(sessionID, request.base) : undefined
      if (prior?.query.operation === "school_transfers")
        return yield* Effect.fail(
          new Error(
            "Transfer review separates different evidence categories. Select sourceRows from the requested group rather than using all candidates as a base cohort.",
          ),
        )
      const query = { ...prior?.query, ...request }
      delete query.base
      if (prior?.query.filter && request.filter)
        query.filter = { op: "and", items: [prior.query.filter, request.filter] }
      yield* Effect.try({ try: () => ResearchQuery.validate(query, dataset), catch: queryError })
      const current = yield* snapshot(sessionID, ResearchQuery.fields(query, dataset))
      const result = yield* Effect.try({ try: () => ResearchQuery.execute(current, query), catch: queryError })
      const evidence = Schema.decodeUnknownSync(Research.Evidence)({
        ...result,
        id: `ev_${crypto.randomUUID()}`,
        sessionID,
        binding: current.binding,
        version: current.dataset.version,
        at: current.at,
        fingerprint: current.fingerprint,
        query,
      })
      if (JSON.stringify(evidence).length > 24000)
        return yield* Effect.fail(new Error("Result exceeds the evidence budget; reduce page size or selected fields"))
      yield* kv.set(`research/${sessionID}/${evidence.id}`, Schema.encodeSync(Research.Evidence)(evidence))
      const previous = Schema.decodeUnknownOption(Schema.Array(Schema.String))(
        yield* kv.get(`research/${sessionID}/index`),
      )
      const index = [...(previous._tag === "Some" ? previous.value : []), evidence.id]
      yield* Effect.forEach(index.slice(0, -64), (id) => kv.remove(`research/${sessionID}/${id}`))
      yield* kv.set(`research/${sessionID}/index`, index.slice(-64))
      if (currentEvidence.has(sessionID))
        currentEvidence.set(sessionID, [...(currentEvidence.get(sessionID) ?? []), evidence].slice(-8))
      if (remember)
        yield* kv.set(`research/${sessionID}/frame`, {
          id: evidence.id,
          query: Schema.encodeSync(Research.Query)(query),
        })
      return evidence
    })
    const answer = Effect.fn("Research.answer")(function* (sessionID: string, input: Research.AnswerInput) {
      const evidence = yield* Effect.forEach(input.evidence, (id) => load(sessionID, id))
      const dataset = yield* catalog(sessionID)
      const current = yield* snapshot(sessionID, [
        ...new Set(
          evidence
            .flatMap((item) => ResearchQuery.fields(item.query, dataset))
            .concat((input.quotes ?? []).map((quote) => quote.field)),
        ),
      ])
      const result = yield* Effect.try({
        try: () => ResearchEvidence.answer(sessionID, current, input, evidence),
        catch: queryError,
      })
      yield* kv.set(`research/${sessionID}/answer`, Schema.encodeSync(Research.Answer)(result))
      return result
    })
    return {
      binding,
      catalog,
      snapshot,
      scope: (sessionID: string, values: readonly string[]) => kv.set(`research/${sessionID}/communities`, [...values]),
      load,
      run,
      answer,
      beginEvidence: (sessionID: string) => currentEvidence.set(sessionID, []),
      takeEvidence: (sessionID: string) => {
        const evidence = currentEvidence.get(sessionID) ?? []
        currentEvidence.delete(sessionID)
        return evidence
      },
      frame: (sessionID: string) => kv.get(`research/${sessionID}/frame`),
      observeCatalog: (sessionID: string, dataset: Research.Dataset) =>
        Effect.gen(function* () {
          const signature = JSON.stringify([ResearchValues.semantics, ResearchQuery.revision, dataset])
          const previous = yield* kv.get(`research/${sessionID}/policy`)
          yield* kv.set(`research/${sessionID}/policy`, signature)
          return previous !== undefined && previous !== signature
        }),
      clearAnswer: (sessionID: string) => kv.remove(`research/${sessionID}/answer`),
      clear: (sessionID: string) =>
        kv.scan({ prefix: `research/${sessionID}/`, limit: 1000 }).pipe(
          Effect.flatMap((result) => Effect.forEach(result.entries, (entry) => kv.remove(entry.key))),
          Effect.asVoid,
        ),
      latest: (sessionID: string) =>
        kv.get(`research/${sessionID}/answer`).pipe(
          Effect.map((value) => {
            const decoded = Schema.decodeUnknownOption(Research.Answer)(value)
            return decoded._tag === "Some" ? decoded.value : undefined
          }),
        ),
    }
  })

export type Interface = Effect.Success<ReturnType<typeof make>>

/** Keeps records whose community matches one of `communities`, ignoring case like the Python API's IN filter. */
export function withinCommunities(
  snapshot: Research.Snapshot,
  communities: readonly string[],
  keepField: boolean,
): Research.Snapshot {
  const wanted = new Set(communities.map((item) => item.toLowerCase()))
  return {
    ...snapshot,
    records: snapshot.records.flatMap((record) => {
      const community = record.values.community
      if (community === null || community === undefined || !wanted.has(community.toLowerCase())) return []
      if (keepField) return [record]
      return [{ ...record, values: Object.fromEntries(Object.entries(record.values).filter(([field]) => field !== "community")) }]
    }),
  }
}

function queryError(error: unknown) {
  return error instanceof Error ? error : new Error("Invalid research query; check catalog capabilities")
}
