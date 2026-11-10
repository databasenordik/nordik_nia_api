export * as ResearchDatabase from "./database.js"

import { SQL } from "bun"
import { Effect, Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchValues } from "./values.js"

export const Options = Schema.Struct({
  url: Schema.String.check(Schema.isMinLength(1)),
  principal: Schema.String.check(Schema.isMinLength(1)),
  private: Schema.Boolean,
  ssl: Schema.Literals(["disable", "require", "verify-ca", "verify-full"]),
  max: Schema.optional(Schema.Int.check(Schema.isGreaterThanOrEqualTo(1))),
})
export type Options = typeof Options.Type

const Identity = Schema.Struct({
  database: Schema.String,
  contract: Schema.Literal(1),
  dates: Schema.Literal(ResearchValues.semantics),
})
const Privileges = Schema.Struct({ unsafe: Schema.Boolean })
/** A list the assistant offers: active in public.data_config and open to this principal. */
export const List = Schema.Struct({ fileID: Research.FileID, defaultPeople: Schema.Boolean, private: Schema.Boolean })
export interface List extends Schema.Schema.Type<typeof List> {}
const decodeLists = Schema.decodeUnknownSync(Schema.Array(List))
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

// Only fixed messages cross the startup boundary. Driver messages, SQL, detail, hint and causes can
// contain credentials or source values, including when the gateway raises an exception itself.
const qualificationReasons = new Map([
  ["28P01", "authentication failed; verify the ASSISTANT_DATABASE_URL username and password"],
  ["28000", "database login was rejected; verify the runtime role and database authentication rules"],
  ["3D000", "the configured database does not exist; verify ASSISTANT_DATABASE_URL"],
  ["42501", "permission denied; verify the runtime role's schema USAGE and gateway EXECUTE grants"],
  ...["3F000", "42883", "42P01", "42704"].map((code) =>
    [code, "required database objects are missing; install the research gateway and its prerequisite migrations"] as const,
  ),
  ...["ENOTFOUND", "EAI_AGAIN", "DNS_ENOTFOUND", "DNS_ESERVFAIL"].map((code) =>
    [code, "database hostname lookup failed; verify DNS and the database host in ASSISTANT_DATABASE_URL"] as const,
  ),
  ...["ECONNREFUSED", "ERR_POSTGRES_CONNECTION_REFUSED"].map((code) =>
    [code, "database connection refused; verify the database host, port and network access"] as const,
  ),
  ...["ETIMEDOUT", "ERR_POSTGRES_CONNECTION_TIMEOUT"].map((code) =>
    [code, "database connection timed out; verify routing, firewall rules and database availability"] as const,
  ),
  ...["ECONNRESET", "ERR_POSTGRES_CONNECTION_CLOSED", "ERR_POSTGRES_CONNECTION_FAILED", "57P03"].map((code) =>
    [code, "database connection unavailable or closed; verify database availability and network access"] as const,
  ),
  ...["ERR_POSTGRES_TLS_NOT_AVAILABLE", "ERR_POSTGRES_TLS_UPGRADE_FAILED"].map((code) =>
    [code, "database TLS negotiation failed; verify the server's TLS support and SSL configuration"] as const,
  ),
  ...[
    "DEPTH_ZERO_SELF_SIGNED_CERT",
    "SELF_SIGNED_CERT_IN_CHAIN",
    "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
    "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
    "CERT_HAS_EXPIRED",
    "ERR_TLS_CERT_ALTNAME_INVALID",
  ].map((code) =>
    [code, "database TLS certificate verification failed; verify the trusted CA, certificate and hostname"] as const,
  ),
  ["53300", "database connection limit reached; verify pool sizes and available database connections"],
])

function qualificationReason(error: unknown) {
  if (typeof error === "object" && error !== null) {
    const details = error as { errno?: unknown; code?: unknown }
    for (const code of [details.errno, details.code]) {
      const reason = typeof code === "string" ? qualificationReasons.get(code) : undefined
      if (reason) return reason
    }
  }
  return "verify database availability, role, TLS and gateway installation"
}

/**
 * Why the connection's role may not serve research; empty when it may. Covers the Python API's startup
 * self-test (app/db/privilege_selftest.py) and more. The API runtime role (assistant_runtime) keeps
 * conversations in the assistant schema, so rows there may be read and written. Everything else stays
 * refused: raw, public or released-record reads, writes in other schemas, TRUNCATE anywhere, schema
 * CREATE, role or RLS powers and server file access. Nothing restricted is read to find this out.
 */
export async function privileges(client: SQL, principal: string) {
  const failures: string[] = []
  const [role] = await client`SELECT (
      EXISTS(SELECT 1 FROM pg_roles WHERE rolname=current_user AND (rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls))
      OR EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname IN ('public','assistant','assistant_api','research_api') AND c.relkind IN ('r','p')
        AND (has_table_privilege(current_user,c.oid,'TRUNCATE')
          OR (n.nspname <> 'assistant' AND has_table_privilege(current_user,c.oid,'INSERT,UPDATE,DELETE'))
          OR (n.nspname = 'public' AND has_table_privilege(current_user,c.oid,'SELECT'))))
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
    ) AS unsafe,
    -- Released records are read only through the gateway functions, never from the views directly.
    COALESCE(has_table_privilege(current_user,to_regclass('assistant_api.v_current_records'),'SELECT'),false)
      OR COALESCE(has_table_privilege(current_user,to_regclass('assistant_api.v_current_raw_records'),'SELECT'),false)
      AS views,
    -- The Potential (94) release gate lives in those views; without it unreleased rows would be read.
    to_regprocedure('assistant_api.potential_access_revision()') IS NOT NULL
      AND (SELECT count(*) = 2 FROM pg_class WHERE oid = ANY(ARRAY[
        to_regclass('assistant_api.v_current_records'), to_regclass('assistant_api.v_current_raw_records')])
        AND 'security_barrier=true' = ANY(reloptions)) AS gate`
  if (Schema.decodeUnknownSync(Privileges)({ unsafe: role.unsafe }).unsafe)
    failures.push("the role has privileges beyond the assistant runtime role; use assistant_runtime")
  if (role.views) failures.push("the role can SELECT assistant_api.v_current_records or v_current_raw_records directly")
  if (!role.gate) failures.push("the Potential release gate is not installed; apply migrations through 073")
  if (failures.length) return failures
  const [leak] = await client`SELECT count(*)::integer AS leaked
    FROM jsonb_array_elements(research_api.lists(${principal}, false)) list WHERE (list->>'private')::boolean`
  if (leak.leaked) failures.push("private lists are listed without a private-files grant")
  return failures
}

// Railway's *.railway.internal names resolve only inside its encrypted private network, so they
// count as local; anything reachable from the internet needs TLS.
export function local(url: URL) {
  return ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.hostname.endsWith(".railway.internal")
}

export async function connect(input: Options) {
  const url = new URL(input.url)
  if (!["postgres:", "postgresql:"].includes(url.protocol))
    throw new Error("An explicit PostgreSQL profile is required")
  if (input.ssl === "disable" && !local(url)) throw new Error("Remote PostgreSQL connections require TLS")
  // Own the pool in the host scope; never forward this profile into Session environment variables.
  const client = new SQL({ url: input.url, max: input.max ?? 4, connectionTimeout: 10, idleTimeout: 20, ssl: input.ssl })
  const query = async <A>(
    decode: (value: unknown) => A,
    name: "identity" | "lists" | "catalog" | "snapshot",
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
            : name === "lists"
              ? tx`SELECT research_api.lists(${input.principal},${input.private}) AS value`
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
        // connect() owns the identity check and reports a safe startup diagnosis with its phase.
        if (name === "identity") throw error
        if (error instanceof SQL.PostgresError && (error.errno === "42501" || error.code === "42501"))
          throw new AccessDeniedError()
        throw new Error("Research database request failed; verify availability, permissions and current normalization")
      })
  }
  const close = () => client.close()
  let phase = "gateway identity"
  const initialize = async () => {
    const identity = await query((value) => {
      if (!Schema.is(Identity)(value))
        throw new SelfTestError(
          "Cannot qualify the research database connection at gateway identity: incompatible gateway contract or date semantics; update the research gateway installation",
        )
      return value
    }, "identity")
    phase = "privilege self-test"
    const failures = await privileges(client, input.principal)
    if (failures.length) throw new SelfTestError(`privilege self-test failed: ${failures.join("; ")}`)
    return {
      identity: identity.database,
      principal: input.principal,
      lists: (signal?: AbortSignal) => query(decodeLists, "lists", undefined, signal),
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
  return initialize().catch(async (error: unknown) => {
    // A cleanup error must not replace the safe diagnosis with a raw driver exception.
    await close().catch(() => {})
    // The self-test's findings are fixed text, so the operator sees them. Any other database error can
    // contain a DSN, source SQL or private values. Keep that boundary redacted.
    if (error instanceof SelfTestError) throw error
    throw new Error(`Cannot qualify the research database connection at ${phase}: ${qualificationReason(error)}`)
  })
}

class SelfTestError extends Error {}

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
