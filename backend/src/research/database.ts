export * as ResearchDatabase from "./database.js"

import { SQL } from "bun"
import { Effect, Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchValues } from "./values.js"

export const Options = Schema.Struct({
  url: Schema.String.check(Schema.isMinLength(1)),
  principal: Schema.String.check(Schema.isMinLength(1)),
  private: Schema.Boolean,
  ssl: Schema.Literals(["disable", "verify-full"]),
})
export type Options = typeof Options.Type

const Identity = Schema.Struct({
  database: Schema.String,
  contract: Schema.Literal(1),
  dates: Schema.Literal(ResearchValues.semantics),
})
const Privileges = Schema.Struct({ unsafe: Schema.Boolean })
const decodeDataset = (value: unknown) => capabilities(Schema.decodeUnknownSync(Research.Dataset)(value))
const decodeSnapshot = (value: unknown) => {
  const snapshot = Schema.decodeUnknownSync(Research.Snapshot)(value)
  return { ...snapshot, dataset: capabilities(snapshot.dataset) }
}

function capabilities(dataset: Research.Dataset): Research.Dataset {
  return {
    ...dataset,
    fields: dataset.fields.map((field) => ({
      ...field,
      operators: [
        ...new Set([
          ...field.operators.filter((operator) => Research.Operator.literals.some((value) => value === operator)),
          ...(field.type === "date" ? ["HAS_MULTIPLE_EVENTS"] : []),
        ]),
      ],
      label:
        field.name === "additional_information"
          ? "Additional information: source narrative, including school history and transfers when recorded"
          : field.name === "notes"
            ? "Notes: search alongside Additional information and other narrative fields"
            : field.label,
    })),
  }
}

export class AccessDeniedError extends Error {
  constructor() {
    super("Selected list unavailable or access denied")
  }
}

export async function connect(input: Options) {
  const url = new URL(input.url)
  if (!["postgres:", "postgresql:"].includes(url.protocol))
    throw new Error("An explicit PostgreSQL profile is required")
  // Railway's *.railway.internal names resolve only inside its encrypted private network, so they
  // count as local; anything reachable from the internet still needs verify-full.
  const local = ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.hostname.endsWith(".railway.internal")
  if (input.ssl === "disable" && !local) throw new Error("Remote PostgreSQL profiles require verify-full TLS")
  // Own the pool in the host scope; never forward this profile into Session environment variables.
  const client = new SQL({ url: input.url, max: 4, connectionTimeout: 10, idleTimeout: 20, ssl: input.ssl })
  const query = async <A>(
    decode: (value: unknown) => A,
    name: "identity" | "catalog" | "snapshot",
    fileID?: Research.FileID,
    signal?: AbortSignal,
    fields?: readonly string[],
  ) => {
    return client
      .begin("READ ONLY ISOLATION LEVEL REPEATABLE READ", async (tx) => {
        await tx`SET LOCAL statement_timeout='10000ms'`
        await tx`SET LOCAL lock_timeout='500ms'`
        await tx`SET LOCAL idle_in_transaction_session_timeout='12000ms'`
        const pending =
          name === "identity"
            ? tx`SELECT research_api.identity() AS value`
            : name === "catalog"
              ? tx`SELECT research_api.catalog(${input.principal},${fileID},${input.private}) AS value`
              : tx`SELECT research_api.snapshot(${input.principal},${fileID},${input.private},${fields ? tx.array([...fields], "TEXT") : null}) AS value`
        const cancel = () => pending.cancel()
        signal?.throwIfAborted()
        signal?.addEventListener("abort", cancel, { once: true })
        try {
          const rows = await pending
          return decode(rows[0]?.value)
        } finally {
          signal?.removeEventListener("abort", cancel)
        }
      })
      .catch((error: unknown) => {
        if (error instanceof SQL.PostgresError && (error.errno === "42501" || error.code === "42501"))
          throw new AccessDeniedError()
        throw new Error("Research database request failed; verify availability, permissions and current normalization")
      })
  }
  const close = () => client.close()
  const initialize = async () => {
    const identity = await query(Schema.decodeUnknownSync(Identity), "identity")
    const rows = await client`SELECT (
      EXISTS(SELECT 1 FROM pg_roles WHERE rolname=current_user AND (rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls))
      OR EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname IN ('public','assistant','assistant_api','research_api') AND c.relkind IN ('r','p')
        AND (has_table_privilege(current_user,c.oid,'INSERT,UPDATE,DELETE,TRUNCATE')
          OR (n.nspname IN ('public','assistant') AND has_table_privilege(current_user,c.oid,'SELECT'))))
      OR has_table_privilege(current_user,'public.file_data','SELECT')
      OR has_table_privilege(current_user,'public.file_data_normalized','SELECT')
      OR has_table_privilege(current_user,'public.file','SELECT')
      OR has_schema_privilege(current_user,'public','CREATE')
      OR has_schema_privilege(current_user,'research_api','CREATE')
      OR has_schema_privilege(current_user,'assistant','CREATE')
      OR has_schema_privilege(current_user,'assistant_api','CREATE')
      OR pg_has_role(current_user,'pg_execute_server_program','MEMBER')
      OR pg_has_role(current_user,'pg_read_server_files','MEMBER')
      OR pg_has_role(current_user,'pg_write_server_files','MEMBER')
    ) AS unsafe`
    if (Schema.decodeUnknownSync(Privileges)(rows[0]).unsafe)
      throw new Error("Research connection has unsafe privileges; use the dedicated read-only role")
    return {
      identity: identity.database,
      principal: input.principal,
      catalog: (fileID: Research.FileID, signal?: AbortSignal) => query(decodeDataset, "catalog", fileID, signal),
      snapshot: async (binding: Research.Binding, signal?: AbortSignal, fields?: readonly string[]) => {
        if (binding.database !== identity.database) throw new Error("Session belongs to a different database")
        const columns = fields?.flatMap((field) => {
          const match = /^(.*_date)_(year|month|day)$/.exec(field)
          return match ? [field, match[1]] : [field]
        })
        const snapshot = await query(decodeSnapshot, "snapshot", binding.fileID, signal, columns)
        if (snapshot.binding.fileID !== binding.fileID || snapshot.binding.database !== binding.database)
          throw new Error("Database returned a conflicting Session binding")
        return derive(snapshot)
      },
      close,
    }
  }
  return initialize().catch(async () => {
    await close()
    // Database errors can contain a DSN, source SQL or private values. Keep that boundary redacted.
    throw new Error("Cannot qualify the research database connection; verify role, TLS and gateway installation")
  })
}

export type Connection = Awaited<ReturnType<typeof connect>>

export function acquire(options: Options) {
  return Effect.acquireRelease(
    Effect.tryPromise(() => connect(options)),
    (connection) => Effect.promise(connection.close),
  )
}

function derive(snapshot: Research.Snapshot): Research.Snapshot {
  const parse = ResearchValues.dateCache(snapshot.dataset)
  return {
    ...snapshot,
    fingerprint: `${ResearchValues.semantics}:${ResearchValues.retrieval}:${snapshot.fingerprint}`,
    records: snapshot.records.map((record) => ({
      ...record,
      values: Object.fromEntries(
        Object.entries(record.values).map(([field, value]) => {
          const match = /^(.*_date)_(year|month|day)$/.exec(field)
          if (!match) return [field, value]
          return [field, ResearchValues.datePart(parse(record.values[match[1]]), match[2])]
        }),
      ),
    })),
  }
}
