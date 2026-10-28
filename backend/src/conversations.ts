export * as Conversations from "./conversations"

import { randomUUID } from "node:crypto"
import { SQL } from "bun"
import type { ResearchDatabase } from "./research/database"

// Conversation history in the Python API's tables and format (assistant.conversations, .messages and
// .conversation_state, as of migration 074), so histories written by either service read the same in
// both. The SQL mirrors app/memory/postgres.py in nordik_nia_api.

/** A Nordik Drive user, through the service principal, on one list. */
export interface Scope {
  readonly principal: string
  readonly userID: number
  readonly fileID: number
}

export interface Conversation {
  readonly id: string
  /** The community selection the conversation's context belongs to, sorted. */
  readonly communities: readonly string[]
}

export interface Exchange {
  readonly question: string
  readonly answer: string
  readonly communities: readonly string[]
  readonly askedAt: Date
}

export interface History {
  readonly conversation_id: string | null
  readonly file_id: number
  readonly entries: readonly {
    readonly id: string
    readonly exchange_id: string
    readonly question_id: string
    readonly answer_id: string
    readonly question: string | null
    readonly answer: string | null
    readonly created_at: string
    readonly answered_at: string
    readonly communities: readonly string[]
  }[]
  readonly has_more: boolean
}

export class NotFound extends Error {
  constructor() {
    super("conversation not found")
  }
}

export class Invalid extends Error {}

export interface Store {
  /** The user's one conversation on this list, created on first use. A given ID must be that conversation. */
  readonly open: (scope: Scope, conversationID?: string) => Promise<Conversation>
  /** Moves the conversation to another community selection, dropping context built under the previous one. */
  readonly select: (scope: Scope, conversation: Conversation, communities: readonly string[]) => Promise<Conversation>
  /** Stores a question and its answer as one exchange and returns the exchange ID. */
  readonly record: (scope: Scope, conversation: Conversation, exchange: Exchange) => Promise<string>
  readonly history: (scope: Scope, conversationID: string | undefined, limit: number) => Promise<History>
  /** The latest exchanges under one community selection, oldest first: the context the Python API keeps. */
  readonly recent: (
    scope: Scope,
    conversationID: string,
    communities: readonly string[],
    limit: number,
  ) => Promise<{ question: string; answer: string }[]>
  /** One assistant.audit_events row, as the Python API writes for every question. */
  readonly audit: (
    scope: Scope,
    event: {
      readonly type: string
      readonly outcome: "allowed" | "denied"
      readonly conversationID: string
      readonly metadata: Record<string, unknown>
    },
  ) => Promise<void>
  readonly close: () => Promise<void>
}

// Context the Python API keeps per community selection; cleared when the selection changes, as it does.
const selectionContext = ["recent_turns", "frames", "result_sets", "rolling_summary", "active_frame_key", "last_action_keys"]

/** Normalizes a UUID the way Python's `str(UUID(value))` does, or refuses it. */
export function uuid(value: string) {
  const hex = value
    .trim()
    .toLowerCase()
    .replace(/^urn:uuid:/, "")
    .replace(/^\{(.*)\}$/, "$1")
    .replaceAll("-", "")
  if (!/^[0-9a-f]{32}$/.test(hex)) throw new Invalid("conversation_id must be a UUID")
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export async function postgres(options: {
  readonly url: string
  readonly ssl: ResearchDatabase.Options["ssl"]
  /** Pool size; 4 unless a test needs a single connection. */
  readonly max?: number
}): Promise<Store> {
  const sql = new SQL({ url: options.url, max: options.max ?? 4, connectionTimeout: 10, idleTimeout: 20, ssl: options.ssl })
  // Fail at startup, not on the first question, when the tables or the runtime role's grants are missing.
  await sql`SELECT 1 FROM assistant.conversations c JOIN assistant.messages m ON m.conversation_id = c.id
    JOIN assistant.conversation_state s ON s.conversation_id = c.id
    CROSS JOIN (SELECT 1 FROM assistant.audit_events LIMIT 0) audit LIMIT 0`.catch(async (error: unknown) => {
    await sql.close()
    throw error
  })
  const owned = (row: { principal_id: string; user_id: string | null; selected_file_id: number | null } | undefined, scope: Scope) =>
    row !== undefined &&
    row.principal_id === scope.principal &&
    row.user_id === String(scope.userID) &&
    row.selected_file_id === scope.fileID

  return {
    open: async (scope, requested) => {
      let id = requested ? uuid(requested) : undefined
      if (!id) {
        const [created] = await sql`
          INSERT INTO assistant.conversations (principal_id, user_id, selected_file_id)
          VALUES (${scope.principal}, ${String(scope.userID)}::bigint, ${scope.fileID}::integer)
          ON CONFLICT (user_id, file_id) WHERE user_id IS NOT NULL
          DO UPDATE SET user_id = EXCLUDED.user_id
          WHERE assistant.conversations.principal_id = EXCLUDED.principal_id
          RETURNING id::text AS id`
        if (!created) throw new NotFound()
        id = created.id as string
      }
      const [row] = await sql`
        SELECT c.principal_id, c.user_id::text AS user_id, c.selected_file_id,
               COALESCE(s.active_query_state->'community_filter', '[]'::jsonb) AS communities
        FROM assistant.conversations c
        LEFT JOIN assistant.conversation_state s ON s.conversation_id = c.id
        WHERE c.id = ${id}::uuid`
      if (!owned(row, scope)) throw new NotFound()
      return { id, communities: strings(json(row.communities)) }
    },

    select: async (scope, conversation, communities) => {
      const state = {
        id: conversation.id,
        principal_id: scope.principal,
        user_id: scope.userID,
        selected_file_id: scope.fileID,
        community_filter: [...communities],
      }
      // JSON goes in as text and is cast on the server: how Bun encodes a jsonb parameter itself depends on
      // whether statements are prepared (named statements, or unnamed ones behind a connection pooler).
      await sql`
        INSERT INTO assistant.conversation_state (conversation_id, rolling_summary, active_query_state, updated_at)
        VALUES (${conversation.id}::uuid, '', ${JSON.stringify(state)}::text::jsonb, now())
        ON CONFLICT (conversation_id) DO UPDATE SET
          rolling_summary = '',
          active_query_state = (assistant.conversation_state.active_query_state - ${sql.array(selectionContext, "TEXT")}::text[])
            || EXCLUDED.active_query_state,
          last_source_ids = '{}',
          updated_at = now()`
      return { id: conversation.id, communities: [...communities] }
    },

    record: (scope, conversation, exchange) =>
      sql.begin(async (tx) => {
        // Locking the conversation serializes exchanges, so turn numbers never collide.
        const [row] = await tx`
          SELECT principal_id, user_id::text AS user_id, selected_file_id
          FROM assistant.conversations WHERE id = ${conversation.id}::uuid FOR UPDATE`
        if (!owned(row, scope)) throw new NotFound()
        // Turns continue from both the stored messages and the Python API's own sequence counter.
        const [position] = await tx`
          SELECT GREATEST(
                   COALESCE((SELECT max(substring(turn_id from '[0-9]+')::bigint) FROM assistant.messages
                             WHERE conversation_id = ${conversation.id}::uuid AND turn_id ~ '^turn-[0-9]+$'), -1) + 1,
                   COALESCE((SELECT (active_query_state->>'message_sequence')::bigint FROM assistant.conversation_state
                             WHERE conversation_id = ${conversation.id}::uuid
                               AND active_query_state->>'message_sequence' ~ '^[0-9]+$'), 0))::integer AS next,
                 CASE WHEN ${scope.fileID}::integer = 94 THEN assistant_api.potential_access_revision() END AS revision`
        const next = Number(position.next)
        const revision = (position.revision ?? null) as string | null
        const question = randomUUID()
        const answer = randomUUID()
        const communities = tx.array([...exchange.communities], "TEXT")
        await tx`
          INSERT INTO assistant.messages
            (id, conversation_id, turn_id, role, text, mode, interrupted, created_at,
             exchange_id, communities, row_access_revision)
          VALUES
            (${question}::uuid, ${conversation.id}::uuid, ${`turn-${next}`}, 'user', ${exchange.question}, 'text', false,
             ${exchange.askedAt.toISOString()}::timestamptz, ${question}::uuid, ${communities}::text[], ${revision}),
            (${answer}::uuid, ${conversation.id}::uuid, ${`turn-${next + 1}`}, 'assistant', ${exchange.answer}, 'text', false,
             now(), ${question}::uuid, ${communities}::text[], ${revision})`
        await tx`
          UPDATE assistant.conversations
          SET title = COALESCE(title, ${Array.from(exchange.question).slice(0, 80).join("")}),
              current_mode = 'text', updated_at = now()
          WHERE id = ${conversation.id}::uuid`
        const state = {
          id: conversation.id,
          principal_id: scope.principal,
          user_id: scope.userID,
          selected_file_id: scope.fileID,
          community_filter: [...exchange.communities],
          message_sequence: next + 2,
          ...(revision === null ? {} : { row_access_revision: revision }),
        }
        await tx`
          INSERT INTO assistant.conversation_state (conversation_id, active_query_state, updated_at)
          VALUES (${conversation.id}::uuid, ${JSON.stringify(state)}::text::jsonb, now())
          ON CONFLICT (conversation_id) DO UPDATE SET
            active_query_state = assistant.conversation_state.active_query_state || EXCLUDED.active_query_state,
            updated_at = now()`
        return question
      }),

    history: async (scope, requested, limit) => {
      const id = requested ? uuid(requested) : null
      const [row] = await sql`
        SELECT id::text AS id FROM assistant.conversations
        WHERE principal_id = ${scope.principal} AND user_id = ${String(scope.userID)}::bigint
          AND file_id = ${scope.fileID}::integer AND (${id}::uuid IS NULL OR id = ${id}::uuid)`
      if (!row) {
        if (id) throw new NotFound()
        return { conversation_id: null, file_id: scope.fileID, entries: [], has_more: false }
      }
      // Timestamps are written as Python's datetime.isoformat() writes them, so both services agree.
      const rows = await sql`
        SELECT q.exchange_id::text AS exchange_id, q.id::text AS question_id, a.id::text AS answer_id,
               q.text AS question, a.text AS answer, q.communities,
               to_char(q.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')
                 || CASE WHEN extract(microseconds FROM q.created_at)::bigint % 1000000 = 0 THEN ''
                         ELSE to_char(q.created_at AT TIME ZONE 'UTC', '.US') END || '+00:00' AS created_at,
               to_char(a.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS')
                 || CASE WHEN extract(microseconds FROM a.created_at)::bigint % 1000000 = 0 THEN ''
                         ELSE to_char(a.created_at AT TIME ZONE 'UTC', '.US') END || '+00:00' AS answered_at
        FROM assistant.messages q
        JOIN assistant.messages a ON a.conversation_id = q.conversation_id
            AND a.exchange_id = q.exchange_id AND a.role = 'assistant'
        WHERE q.conversation_id = ${row.id}::uuid AND q.role = 'user'
          AND (${scope.fileID}::integer <> 94 OR a.row_access_revision = assistant_api.potential_access_revision())
        ORDER BY CASE WHEN a.turn_id ~ '^turn-[0-9]+$'
                      THEN substring(a.turn_id from '[0-9]+')::bigint ELSE 0 END DESC,
                 a.created_at DESC, a.id DESC
        LIMIT ${limit + 1}`
      const entries = [...rows]
        .slice(0, limit)
        .reverse()
        .map((entry) => ({
          id: entry.exchange_id as string,
          exchange_id: entry.exchange_id as string,
          question_id: entry.question_id as string,
          answer_id: entry.answer_id as string,
          question: entry.question as string | null,
          answer: entry.answer as string | null,
          created_at: entry.created_at as string,
          answered_at: entry.answered_at as string,
          communities: strings(entry.communities),
        }))
      return { conversation_id: row.id as string, file_id: scope.fileID, entries, has_more: rows.length > limit }
    },

    // The Python API's context filter: the same community selection, and on list 94 only answers given
    // under the current release.
    recent: async (scope, conversationID, communities, limit) => {
      const rows = await sql`
        SELECT q.text AS question, a.text AS answer
        FROM assistant.messages q
        JOIN assistant.messages a ON a.conversation_id = q.conversation_id
            AND a.exchange_id = q.exchange_id AND a.role = 'assistant'
        JOIN assistant.conversations c ON c.id = q.conversation_id
        WHERE q.conversation_id = ${uuid(conversationID)}::uuid AND q.role = 'user'
          AND c.principal_id = ${scope.principal} AND c.user_id = ${String(scope.userID)}::bigint
          AND q.communities = ${sql.array([...communities], "TEXT")}::text[]
          AND (${scope.fileID}::integer <> 94 OR a.row_access_revision = assistant_api.potential_access_revision())
        ORDER BY CASE WHEN a.turn_id ~ '^turn-[0-9]+$'
                      THEN substring(a.turn_id from '[0-9]+')::bigint ELSE 0 END DESC,
                 a.created_at DESC, a.id DESC
        LIMIT ${limit}`
      return [...rows].reverse().map((row) => ({ question: String(row.question ?? ""), answer: String(row.answer ?? "") }))
    },

    audit: async (scope, event) => {
      await sql`
        INSERT INTO assistant.audit_events
          (principal_id, event_type, conversation_id, dataset_ids, authorization_outcome, provider_metadata)
        VALUES (${scope.principal}, ${event.type}, ${event.conversationID}::uuid, ARRAY[${scope.fileID}::integer],
                ${event.outcome}, ${JSON.stringify(event.metadata)}::text::jsonb)`
    },

    close: () => sql.close(),
  }
}

function json(value: unknown): unknown {
  return typeof value === "string" ? JSON.parse(value) : value
}

function strings(value: unknown) {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []
}
