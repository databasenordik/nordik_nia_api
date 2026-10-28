import { afterAll, beforeEach, expect, test } from "bun:test"
import * as grpc from "@grpc/grpc-js"
import { Conversations } from "@nia/backend/conversations"
import { NiaGrpc } from "@nia/backend/grpc"
import { postgres } from "../fixture/postgres"

// The gRPC service end to end: a real client and server on nia.proto, conversations in PostgreSQL,
// and a stand-in for the research engine.
const database = await postgres()
const store = await Conversations.postgres({ url: database.url, ssl: "disable", max: 1 })
const asked: Parameters<NiaGrpc.Engine["answer"]>[0][] = []
const contexts: (readonly { question: string; answer: string }[])[] = []
let failure: Error | undefined
const lists = new Map([
  [49, { fields: [{ name: "student_name" }, { name: "community" }] }],
  [91, { fields: [{ name: "student_name" }] }],
])
const engine: NiaGrpc.Engine = {
  dataset: async (fileID) => lists.get(fileID),
  answer: async (input) => {
    asked.push(input)
    contexts.push(await input.history())
    if (failure) throw failure
    return `Answer to ${input.question}`
  },
}
const server = await NiaGrpc.serve({ store, engine, principal: "nia-runtime", model: "grok-4.6", hostname: "127.0.0.1", port: 0 })
const client = new NiaGrpc.Assistant(`127.0.0.1:${server.port}`, grpc.credentials.createInsecure()) as unknown as Record<
  string,
  (request: object, callback: (error: grpc.ServiceError | null, response: Record<string, unknown>) => void) => void
>
const call = (method: string, request: object) =>
  new Promise<Record<string, unknown>>((resolve, reject) => client[method]!(request, (error, response) => (error ? reject(error) : resolve(response))))
const failed = (method: string, request: object) =>
  call(method, request).then(
    () => {
      throw new Error("the call succeeded")
    },
    (error: grpc.ServiceError) => ({ code: grpc.status[error.code], details: error.details }),
  )
const result = (response: Record<string, unknown>) => JSON.parse(String(response.result_json))

afterAll(async () => {
  ;(client as unknown as grpc.Client).close()
  await server.close()
  await store.close()
  await database[Symbol.asyncDispose]()
})
beforeEach(async () => {
  await database.reset()
  asked.length = 0
  contexts.length = 0
  failure = undefined
})

const query = { question: "How many students?", selected_file_id: 49, user_id: 7 }

test("Health reports the process is up", async () => {
  expect(await call("Health", {})).toEqual({ status: "ok" })
})

test("the deployment probe passes against a healthy service", async () => {
  const probe = Bun.spawn(
    [process.execPath, "run", require.resolve("@nia/backend/probe"), "--target", `127.0.0.1:${server.port}`, "--plaintext"],
    { stdout: "pipe", stderr: "pipe" },
  )
  expect(await probe.exited).toBe(0)
  expect(await new Response(probe.stdout).text()).toContain("NIA gRPC health: ok")
})

test("Query answers and returns the Python API's response, with the same result keys", async () => {
  const response = await call("Query", query)
  expect(response).toMatchObject({
    status: "answered",
    answer: "Answer to How many students?",
    selected_file_id: 49,
    error_code: "",
    detail: "",
  })
  expect(response.conversation_id).toMatch(/^[0-9a-f-]{36}$/)
  expect(response.exchange_id).toMatch(/^[0-9a-f-]{36}$/)
  const body = result(response)
  // app/api/assistant.py _serialize_turn, in order.
  expect(Object.keys(body)).toEqual([
    "status", "answer", "planner_type", "reasoning_calls", "facts", "rows", "list_results", "detail", "plan",
    "evidence", "methods_run", "parallel_step_peak", "critical_path_ms", "conversation_id", "exchange_id",
    "selected_file_id", "suggested_file_id", "error_code", "citations", "inference", "spoken_text", "trace",
    "debug_trace", "planner_attempts", "planner_retry_count", "review_attempts", "review_status",
    "planner_total_model_calls", "planner_failure_stage", "planner_provider", "planner_model", "planner_usage",
  ])
  expect(body).toMatchObject({ conversation_id: response.conversation_id, exchange_id: response.exchange_id, answer: response.answer })
})

test("a user's questions on one list continue one conversation", async () => {
  const first = await call("Query", query)
  const second = await call("Query", { ...query, question: "And in 1920?" })
  const explicit = await call("Query", { ...query, question: "Who?", conversation_id: first.conversation_id })
  expect(second.conversation_id).toBe(first.conversation_id)
  expect(explicit.conversation_id).toBe(first.conversation_id)
  expect(new Set(asked.map((item) => item.conversationID)).size).toBe(1)
  const other = await call("Query", { ...query, user_id: 8 })
  expect(other.conversation_id).not.toBe(first.conversation_id)
})

test("communities are trimmed, deduplicated and sorted, and reach the engine and history", async () => {
  await call("Query", { ...query, communities: ["  Garden River ", "Batchewana", "Garden River", ""] })
  expect(asked[0]!.communities).toEqual(["Batchewana", "Garden River"])
  const history = result(await call("History", { user_id: 7, selected_file_id: 49 }))
  expect(history.entries[0].communities).toEqual(["Batchewana", "Garden River"])
})

test("History returns the conversation's exchanges", async () => {
  expect(result(await call("History", { user_id: 7, selected_file_id: 49 }))).toEqual({
    conversation_id: null,
    file_id: 49,
    entries: [],
    has_more: false,
  })
  const first = await call("Query", query)
  await call("Query", { ...query, question: "And in 1920?" })
  const history = result(await call("History", { user_id: 7, selected_file_id: 49, limit: 1 }))
  expect(history.conversation_id).toBe(first.conversation_id)
  expect(history.has_more).toBe(true)
  expect(history.entries.map((entry: { question: string; answer: string }) => [entry.question, entry.answer])).toEqual([
    ["And in 1920?", "Answer to And in 1920?"],
  ])
  const named = result(await call("History", { user_id: 7, selected_file_id: 49, conversation_id: first.conversation_id }))
  expect(named.entries).toHaveLength(2)
})

test("invalid requests are refused as the Python API refuses them", async () => {
  const invalid = { code: "INVALID_ARGUMENT" }
  expect(await failed("Query", { ...query, user_id: 0 })).toEqual({ ...invalid, details: "user_id and selected_file_id must be positive" })
  expect(await failed("Query", { ...query, selected_file_id: 0 })).toMatchObject(invalid)
  expect(await failed("Query", { ...query, question: "" })).toMatchObject(invalid)
  expect(await failed("Query", { ...query, question: "x".repeat(4001) })).toMatchObject(invalid)
  expect(await failed("Query", { ...query, model: "m".repeat(101) })).toMatchObject(invalid)
  expect(await failed("Query", { ...query, communities: Array.from({ length: 101 }, (_, i) => `c${i}`) })).toEqual({
    ...invalid,
    details: "too many communities or an overlong community name",
  })
  expect(await failed("Query", { ...query, selected_file_id: 91, communities: ["Batchewana"] })).toEqual({
    ...invalid,
    details: "selected dataset does not support community filtering",
  })
  expect(await failed("Query", { ...query, conversation_id: "nope" })).toEqual({ ...invalid, details: "conversation_id must be a UUID" })
  expect(await failed("History", { user_id: 7, selected_file_id: 49, limit: 101 })).toEqual({
    ...invalid,
    details: "limit must be between 1 and 100",
  })
  expect(asked).toHaveLength(0)
})

test("unavailable lists and other users' conversations are not found", async () => {
  const notFound = { code: "NOT_FOUND", details: "conversation not found" }
  expect(await failed("Query", { ...query, selected_file_id: 93 })).toEqual(notFound)
  expect(await failed("History", { user_id: 7, selected_file_id: 93 })).toEqual(notFound)
  const theirs = await call("Query", { ...query, user_id: 8 })
  expect(await failed("Query", { ...query, conversation_id: theirs.conversation_id })).toEqual(notFound)
  expect(await failed("History", { user_id: 7, selected_file_id: 49, conversation_id: theirs.conversation_id })).toEqual(notFound)
})

test("a failed answer is a planner_unavailable reply, kept in history and audited, as in the Python API", async () => {
  failure = new Error("provider unavailable")
  const original = console.error
  console.error = () => {}
  const response = await call("Query", query).finally(() => {
    console.error = original
  })
  const text = "NIA's planning service is temporarily unavailable. Please try again in a moment."
  expect(response).toMatchObject({ status: "planner_unavailable", answer: text, error_code: "planner_unavailable", detail: "" })
  expect(result(response)).toMatchObject({ status: "planner_unavailable", error_code: "planner_unavailable", spoken_text: null })
  const history = result(await call("History", { user_id: 7, selected_file_id: 49 }))
  expect(history.entries.map((entry: { answer: string }) => entry.answer)).toEqual([text])
  const events = await database.sql`SELECT event_type, authorization_outcome FROM assistant.audit_events`
  expect([...events]).toEqual([{ event_type: "planner_unavailable", authorization_outcome: "allowed" }])
})

test("every answered question is audited", async () => {
  const response = await call("Query", query)
  const events = await database.sql`
    SELECT event_type, authorization_outcome, conversation_id::text AS conversation_id, dataset_ids, provider_metadata
    FROM assistant.audit_events`
  expect(events).toHaveLength(1)
  expect(events[0]).toMatchObject({
    event_type: "query_completed",
    authorization_outcome: "allowed",
    conversation_id: response.conversation_id,
    dataset_ids: [49],
  })
  const metadata = typeof events[0].provider_metadata === "string" ? JSON.parse(events[0].provider_metadata) : events[0].provider_metadata
  expect(metadata).toMatchObject({ planner_type: "research", mode: "text", planner_model: "grok-4.6" })
})

test("REASONING_MODEL answers every question, whatever model a request names", async () => {
  const response = await call("Query", { ...query, model: "grok-4.6-fast" })
  expect(result(response).planner_model).toBe("grok-4.6")
  expect(asked[0]).not.toHaveProperty("model")
})

test("the engine can read the conversation's recent exchanges under the current selection", async () => {
  await call("Query", { ...query, question: "one" })
  await call("Query", { ...query, question: "two" })
  await call("Query", { ...query, question: "three", communities: ["Batchewana"] })
  expect(contexts[1]).toEqual([{ question: "one", answer: "Answer to one" }])
  expect(contexts[2]).toEqual([])
})
