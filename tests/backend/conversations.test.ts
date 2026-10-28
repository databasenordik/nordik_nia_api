import { afterAll, beforeEach, expect, test } from "bun:test"
import { Conversations } from "@nia/backend/conversations"
import { postgres } from "../fixture/postgres"

const database = await postgres()
const store = await Conversations.postgres({ url: database.url, ssl: "disable", max: 1 })
afterAll(async () => {
  await store.close()
  await database[Symbol.asyncDispose]()
})
beforeEach(async () => {
  await database.reset()
  await database.release("release-1")
})

const scope = { principal: "nia-runtime", userID: 7, fileID: 49 }
const exchange = (question: string, answer: string, communities: string[] = [], askedAt = new Date()) => ({
  question,
  answer,
  communities,
  askedAt,
})

// ConversationStore.get from nordik_nia_api app/memory/postgres.py, verbatim.
const pythonGet = (id: string) => database.sql`
  WITH release AS MATERIALIZED (SELECT assistant_api.potential_access_revision() AS revision)
  SELECT c.id::text AS id, c.principal_id, c.user_id, c.selected_file_id,
         c.title,
         CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
              THEN '' ELSE s.rolling_summary END AS rolling_summary,
         CASE WHEN c.selected_file_id = 94 AND s.active_query_state->>'row_access_revision' IS DISTINCT FROM release.revision
              THEN s.active_query_state - ARRAY['rolling_summary','recent_turns','frames','result_sets','last_action_keys','active_frame_key']
              ELSE s.active_query_state END AS active_query_state,
         messages.stored_messages, release.revision AS current_access_revision
  FROM assistant.conversations c
  CROSS JOIN release
  LEFT JOIN assistant.conversation_state s ON s.conversation_id = c.id
  LEFT JOIN LATERAL (
      SELECT jsonb_agg(to_jsonb(recent) ORDER BY recent.sequence) AS stored_messages
      FROM (
          SELECT m.id::text AS message_id, m.exchange_id::text AS exchange_id,
                 m.communities, m.turn_id, m.role, m.text,
                 m.mode, m.interrupted, m.created_at,
                 CASE WHEN m.turn_id ~ '^turn-[0-9]+$'
                      THEN substring(m.turn_id from '[0-9]+')::integer
                      ELSE 0 END AS sequence
          FROM assistant.messages m
          WHERE m.conversation_id = c.id
            AND (c.selected_file_id IS DISTINCT FROM 94 OR m.role = 'user'
                 OR m.row_access_revision = release.revision)
            AND to_jsonb(m.communities) = COALESCE(s.active_query_state->'community_filter', '[]'::jsonb)
          ORDER BY sequence DESC, m.created_at DESC, m.id DESC
          LIMIT 12
      ) recent
  ) messages ON true
  WHERE c.id = ${id}::uuid`
const parse = (value: unknown) => (typeof value === "string" ? JSON.parse(value) : value)

test("a user has one conversation per list, created on first use", async () => {
  const first = await store.open(scope)
  expect(first.communities).toEqual([])
  expect(await store.open(scope)).toEqual(first)
  expect((await store.open({ ...scope, fileID: 91 })).id).not.toBe(first.id)
  expect((await store.open({ ...scope, userID: 8 })).id).not.toBe(first.id)
})

test("a conversation opens only for its own principal, user and list", async () => {
  const { id } = await store.open(scope)
  expect((await store.open(scope, `{${id.toUpperCase()}}`)).id).toBe(id)
  await expect(store.open({ ...scope, userID: 8 }, id)).rejects.toBeInstanceOf(Conversations.NotFound)
  await expect(store.open({ ...scope, principal: "other" }, id)).rejects.toBeInstanceOf(Conversations.NotFound)
  await expect(store.open({ ...scope, fileID: 91 }, id)).rejects.toBeInstanceOf(Conversations.NotFound)
  await expect(store.open(scope, crypto.randomUUID())).rejects.toBeInstanceOf(Conversations.NotFound)
  await expect(store.open(scope, "not-a-uuid")).rejects.toBeInstanceOf(Conversations.Invalid)
})

test("another principal's conversation for the same user and list is not taken over", async () => {
  await store.open({ ...scope, principal: "other" })
  await expect(store.open(scope)).rejects.toBeInstanceOf(Conversations.NotFound)
})

test("exchanges are stored as the Python API stores them", async () => {
  const conversation = await store.open(scope)
  const first = await store.record(scope, conversation, exchange("How many?", "84 records."))
  const second = await store.record(scope, conversation, exchange("And in 1920?", "3 records."))
  const rows = await database.sql`
    SELECT id::text AS id, turn_id, role, text, mode, interrupted, exchange_id::text AS exchange_id, row_access_revision
    FROM assistant.messages ORDER BY turn_id`
  expect(rows.map((row: Record<string, unknown>) => [row.turn_id, row.role, row.text, row.mode, row.interrupted])).toEqual([
    ["turn-0", "user", "How many?", "text", false],
    ["turn-1", "assistant", "84 records.", "text", false],
    ["turn-2", "user", "And in 1920?", "text", false],
    ["turn-3", "assistant", "3 records.", "text", false],
  ])
  // The question's own ID is the exchange ID, shared by its answer.
  expect(rows[0].id).toBe(first)
  expect(rows.map((row: Record<string, unknown>) => row.exchange_id)).toEqual([first, first, second, second])
  expect(rows.every((row: Record<string, unknown>) => row.row_access_revision === null)).toBe(true)
  const [row] = await pythonGet(conversation.id)
  expect(row.title).toBe("How many?")
  expect(parse(row.active_query_state)).toMatchObject({ message_sequence: 4, community_filter: [], user_id: 7 })
  expect(parse(row.stored_messages).map((message: { text: string }) => message.text)).toEqual([
    "How many?",
    "84 records.",
    "And in 1920?",
    "3 records.",
  ])
})

test("history returns the newest exchanges oldest first, in the Python API's shape", async () => {
  const conversation = await store.open(scope)
  await store.record(scope, conversation, exchange("one", "1", [], new Date("2026-10-27T10:00:00Z")))
  await store.record(scope, conversation, exchange("two", "2", [], new Date("2026-10-27T10:01:00.250Z")))
  await store.record(scope, conversation, exchange("three", "3", [], new Date("2026-10-27T10:02:00Z")))
  const page = await store.history(scope, undefined, 2)
  expect(page.conversation_id).toBe(conversation.id)
  expect(page.file_id).toBe(49)
  expect(page.has_more).toBe(true)
  expect(page.entries.map((entry) => entry.question)).toEqual(["two", "three"])
  const [entry] = page.entries
  expect(Object.keys(entry!)).toEqual([
    "id",
    "exchange_id",
    "question_id",
    "answer_id",
    "question",
    "answer",
    "created_at",
    "answered_at",
    "communities",
  ])
  expect(entry!.id).toBe(entry!.question_id)
  // datetime.isoformat(): microseconds only when present, and an explicit UTC offset.
  expect(entry!.created_at).toBe("2026-10-27T10:01:00.250000+00:00")
  expect(page.entries[1]!.created_at).toBe("2026-10-27T10:02:00+00:00")
  expect(entry!.answered_at).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{6})?\+00:00$/)
  expect((await store.history(scope, conversation.id, 10)).has_more).toBe(false)
})

test("history without a conversation is empty, and someone else's conversation is not found", async () => {
  expect(await store.history(scope, undefined, 10)).toEqual({ conversation_id: null, file_id: 49, entries: [], has_more: false })
  const { id } = await store.open({ ...scope, userID: 8 })
  await expect(store.history(scope, id, 10)).rejects.toBeInstanceOf(Conversations.NotFound)
  await expect(store.history(scope, "nope", 10)).rejects.toBeInstanceOf(Conversations.Invalid)
})

test("a new community selection keeps history but clears the context the Python API built", async () => {
  const conversation = await store.open(scope)
  await database.sql`
    INSERT INTO assistant.conversation_state (conversation_id, rolling_summary, active_query_state)
    VALUES (${conversation.id}::uuid, 'summary', ${{
      id: conversation.id,
      community_filter: [],
      frames: { a: {} },
      recent_turns: [{ text: "x" }],
      message_sequence: 6,
      title: "kept",
    }}::jsonb)`
  const selected = await store.select(scope, conversation, ["Batchewana", "Garden River"])
  expect(selected.communities).toEqual(["Batchewana", "Garden River"])
  expect((await store.open(scope, conversation.id)).communities).toEqual(["Batchewana", "Garden River"])
  await store.record(scope, selected, exchange("Who?", "Two people.", ["Batchewana", "Garden River"]))
  const [row] = await pythonGet(conversation.id)
  const state = parse(row.active_query_state)
  expect(state.frames).toBeUndefined()
  expect(state.recent_turns).toBeUndefined()
  expect(state.title).toBe("kept")
  expect(state.message_sequence).toBe(8)
  expect(row.rolling_summary).toBe("")
  // Turns continue after the Python API's own counter, and Python sees them under the new selection.
  const messages = parse(row.stored_messages)
  expect(messages.map((message: { turn_id: string }) => message.turn_id)).toEqual(["turn-6", "turn-7"])
  expect(messages[0].communities).toEqual(["Batchewana", "Garden River"])
  const history = await store.history(scope, undefined, 10)
  expect(history.entries[0]!.communities).toEqual(["Batchewana", "Garden River"])
})

test("answers on list 94 carry the access revision and leave history when it changes", async () => {
  const restricted = { ...scope, fileID: 94 }
  const conversation = await store.open(restricted)
  await store.record(restricted, conversation, exchange("Q", "A"))
  const revisions = await database.sql`SELECT DISTINCT row_access_revision FROM assistant.messages`
  expect(revisions.map((row: Record<string, unknown>) => row.row_access_revision)).toEqual(["release-1"])
  expect((await store.history(restricted, undefined, 10)).entries).toHaveLength(1)
  await database.release("release-2")
  expect((await store.history(restricted, undefined, 10)).entries).toHaveLength(0)
})

test("recent context is the latest exchanges under the same community selection, oldest first", async () => {
  const conversation = await store.open(scope)
  for (const question of ["one", "two", "three"]) await store.record(scope, conversation, exchange(question, `${question}!`))
  const selected = await store.select(scope, conversation, ["Batchewana"])
  await store.record(scope, selected, exchange("four", "four!", ["Batchewana"]))
  expect(await store.recent(scope, conversation.id, [], 2)).toEqual([
    { question: "two", answer: "two!" },
    { question: "three", answer: "three!" },
  ])
  expect(await store.recent(scope, conversation.id, ["Batchewana"], 6)).toEqual([{ question: "four", answer: "four!" }])
  expect(await store.recent({ ...scope, userID: 8 }, conversation.id, [], 6)).toEqual([])
  const restricted = { ...scope, fileID: 94 }
  const potential = await store.open(restricted)
  await store.record(restricted, potential, exchange("Q", "A"))
  await database.release("release-2")
  expect(await store.recent(restricted, potential.id, [], 6)).toEqual([])
})

test("audit events are written as the Python API writes them", async () => {
  const conversation = await store.open(scope)
  await store.audit(scope, {
    type: "query_completed",
    outcome: "allowed",
    conversationID: conversation.id,
    metadata: { planner_type: "research", mode: "text" },
  })
  const rows = await database.sql`
    SELECT principal_id, event_type, conversation_id::text AS conversation_id, dataset_ids,
           authorization_outcome, provider_metadata FROM assistant.audit_events`
  expect(rows.map((row: Record<string, unknown>) => ({ ...row, provider_metadata: parse(row.provider_metadata) }))).toEqual([
    {
      principal_id: "nia-runtime",
      event_type: "query_completed",
      conversation_id: conversation.id,
      dataset_ids: [49],
      authorization_outcome: "allowed",
      provider_metadata: { planner_type: "research", mode: "text" },
    },
  ])
})

test("concurrent exchanges in one conversation never share a turn", async () => {
  const conversation = await store.open(scope)
  await Promise.all(["a", "b", "c"].map((question) => store.record(scope, conversation, exchange(question, "ok"))))
  const turns = await database.sql`SELECT turn_id FROM assistant.messages`
  expect(new Set(turns.map((row: Record<string, unknown>) => row.turn_id)).size).toBe(6)
})
