export * as NiaGrpc from "./grpc"

import path from "node:path"
import * as grpc from "@grpc/grpc-js"
import * as loader from "@grpc/proto-loader"
import { NiaConfig } from "./config"
import { Conversations } from "./conversations"
import type { ResearchHost } from "./host"

// nordik.nia.v1.Assistant, the contract the Nordik Drive Go adapter calls. backend/proto/nia.proto is
// a copy of nordik_nia_api's proto/nia.proto (itself mirrored in nordikdriveapi); keep them equal.
// Field names stay as declared and int64 values arrive as numbers.
const definition = loader.loadSync(path.join(import.meta.dirname, "../proto/nia.proto"), {
  keepCase: true,
  longs: Number,
  defaults: true,
})
const contract = grpc.loadPackageDefinition(definition) as unknown as {
  nordik: { nia: { v1: { Assistant: grpc.ServiceClientConstructor } } }
}
export const Assistant = contract.nordik.nia.v1.Assistant

// The Python API's reply when its model cannot be reached (app/planning/ai_planner.py).
const unavailable = "NIA's planning service is temporarily unavailable. Please try again in a moment."

type Exchange = { readonly question: string; readonly answer: string }

/** What the handlers need from the research engine; tests substitute a fake. */
export interface Engine {
  /** The list's catalog entry, or undefined when the service principal may not open it. */
  readonly dataset: (fileID: number) => Promise<{ readonly fields: readonly { readonly name: string }[] } | undefined>
  /** Answers one question in the conversation's research Session. */
  readonly answer: (input: {
    readonly conversationID: string
    readonly fileID: number
    readonly communities: readonly string[]
    readonly question: string
    /** The conversation's recent exchanges, read only when a new Session has to be started. */
    readonly history: () => Promise<readonly Exchange[]>
    readonly signal: AbortSignal
  }) => Promise<string>
}

type Host = Pick<ResearchHost.Interface, "dataset" | "get" | "create" | "prompt" | "interrupt">

/**
 * Adapts ResearchHost: one research Session per conversation and community selection.
 * `concurrency` bounds answers in progress (the Python API's model_max_concurrency_global).
 */
export function engine(host: Host, options: { readonly concurrency?: number } = {}): Engine {
  const slot = limit(options.concurrency ?? 16)
  const sessions = new Map<string, { readonly id: string; readonly key: string }>()
  const queues = new Map<string, Promise<unknown>>()
  return {
    dataset: host.dataset,
    answer: (input) => {
      // A Session answers one question at a time, so questions in one conversation wait their turn.
      const previous = queues.get(input.conversationID) ?? Promise.resolve()
      const run = previous
        .catch(() => undefined)
        .then(() =>
          slot(async () => {
            if (input.signal.aborted) throw new Error("cancelled")
            const key = JSON.stringify(input.communities)
            const known = sessions.get(input.conversationID)
            const reusable = known?.key === key && (await host.get(known.id).catch(() => undefined)) !== undefined
            let question = input.question
            let sessionID: string
            if (reusable) sessionID = known!.id
            else {
              sessionID = (await host.create(input.fileID, undefined, input.communities)).id
              question = withContext(await input.history(), input.question)
            }
            sessions.set(input.conversationID, { id: sessionID, key })
            // A cancelled call (the caller gave up or its deadline passed) interrupts the Session.
            const stop = () => void host.interrupt(sessionID).catch(() => undefined)
            input.signal.addEventListener("abort", stop, { once: true })
            try {
              return (await host.prompt(sessionID, question)).text
            } catch (error) {
              // An interrupted or failed Session may still be busy; the next question starts a fresh one.
              sessions.delete(input.conversationID)
              throw error
            } finally {
              input.signal.removeEventListener("abort", stop)
            }
          }),
        )
      queues.set(input.conversationID, run)
      return run.finally(() => {
        if (queues.get(input.conversationID) === run) queues.delete(input.conversationID)
      })
    },
  }
}

// A new Session (the first question on this instance, after a restart, or for another selection)
// starts from the conversation's last six exchanges, the context the Python API keeps.
export function withContext(history: readonly Exchange[], question: string) {
  if (!history.length) return question
  const clip = (text: string, size: number) => (text.length > size ? `${text.slice(0, size)}…` : text)
  let turns = history.slice(-6).map((item) => `User: ${clip(item.question, 600)}\nNia: ${clip(item.answer, 1500)}`)
  while (turns.length > 1 && turns.join("\n\n").length > 10_000) turns = turns.slice(1)
  return `Earlier in this conversation, for context only (check any fact again before stating it):\n\n${turns.join("\n\n")}\n\nCurrent question: ${question}`
}

function limit(size: number) {
  let active = 0
  const waiting: (() => void)[] = []
  return async <T>(task: () => Promise<T>) => {
    if (active < size) active++
    // A finishing task hands its place straight to the next waiter.
    else await new Promise<void>((resolve) => waiting.push(resolve))
    try {
      return await task()
    } finally {
      const next = waiting.shift()
      if (next) next()
      else active--
    }
  }
}

type QueryRequest = {
  question: string
  selected_file_id: number
  conversation_id: string
  model: string
  communities: string[]
  user_id: number
}
type HistoryRequest = { user_id: number; selected_file_id: number; conversation_id: string; limit: number }

// The Python API's TurnResult serialization, with the keys its planner fills left empty.
function turnResult(input: {
  status: string
  answer: string
  conversationID: string
  exchangeID: string
  fileID: number
  model: string
}) {
  const answered = input.status === "answered"
  return {
    status: input.status,
    answer: input.answer,
    planner_type: "research",
    reasoning_calls: 0,
    facts: [],
    rows: [],
    list_results: [],
    detail: null,
    plan: null,
    evidence: [],
    methods_run: [],
    parallel_step_peak: 0,
    critical_path_ms: 0,
    conversation_id: input.conversationID,
    exchange_id: input.exchangeID,
    selected_file_id: input.fileID,
    suggested_file_id: null,
    error_code: answered ? null : input.status,
    citations: [],
    inference: false,
    spoken_text: answered ? input.answer : null,
    trace: {},
    debug_trace: {},
    planner_attempts: 0,
    planner_retry_count: 0,
    review_attempts: 0,
    review_status: null,
    planner_total_model_calls: 0,
    planner_failure_stage: answered ? null : "engine",
    planner_provider: NiaConfig.provider,
    planner_model: input.model,
    planner_usage: {},
  }
}

/** The service implementation, with the Python API's validation, status codes and messages. */
export function handlers(input: {
  readonly store: Conversations.Store
  readonly engine: Engine
  readonly principal: string
  /** The model from backend/reasoning.json, reported in each result. */
  readonly model: string
}): grpc.UntypedServiceImplementation {
  const scope = (request: { user_id: number; selected_file_id: number }): Conversations.Scope => {
    if (!(request.user_id > 0) || !(request.selected_file_id > 0))
      throw new Conversations.Invalid("user_id and selected_file_id must be positive")
    return { principal: input.principal, userID: request.user_id, fileID: request.selected_file_id }
  }

  const query = async (request: QueryRequest, signal: AbortSignal) => {
    const length = Array.from(request.question).length
    if (length < 1 || length > 4000 || !request.question.trim())
      throw new Conversations.Invalid("question must be 1 to 4000 characters")
    // The model in backend/reasoning.json answers every question; a requested model is checked as the
    // Python API checks it, then not used.
    if (Array.from(request.model).length > 100) throw new Conversations.Invalid("model must be at most 100 characters")
    const model = input.model
    const current = scope(request)
    const communities = [...new Set(request.communities.map((item) => item.trim()).filter(Boolean))].sort()
    if (communities.length > 100 || communities.some((item) => Array.from(item).length > 200))
      throw new Conversations.Invalid("too many communities or an overlong community name")
    const dataset = await input.engine.dataset(current.fileID)
    if (communities.length && !dataset?.fields.some((field) => field.name === "community"))
      throw new Conversations.Invalid("selected dataset does not support community filtering")
    if (!dataset) throw new Conversations.NotFound()
    let conversation = await input.store.open(current, request.conversation_id || undefined)
    // Every exchange stays in the user's history, but context from another selection is not reused.
    if (JSON.stringify(conversation.communities) !== JSON.stringify(communities))
      conversation = await input.store.select(current, conversation, communities)
    const askedAt = new Date()
    const started = performance.now()
    // As in the Python API, a model failure is a reply the user sees and keeps in history, not an error.
    let status = "answered"
    let answer: string
    try {
      answer = await input.engine.answer({
        conversationID: conversation.id,
        fileID: current.fileID,
        communities,
        question: request.question,
        history: () => input.store.recent(current, conversation.id, communities, 6),
        signal,
      })
    } catch (error) {
      if (signal.aborted) throw error
      console.error("NIA gRPC query: the research engine failed", error)
      status = "planner_unavailable"
      answer = unavailable
    }
    if (signal.aborted) throw new Error("cancelled")
    const exchangeID = await input.store.record(current, conversation, {
      question: request.question,
      answer,
      communities,
      askedAt,
    })
    await input.store.audit(current, {
      type: status === "answered" ? "query_completed" : status,
      outcome: "allowed",
      conversationID: conversation.id,
      metadata: {
        planner_type: "research",
        mode: "text",
        planner_provider: NiaConfig.provider,
        planner_model: model,
        planner_failure_stage: status === "answered" ? null : "engine",
        communities: communities.length,
        e2e_ms: Math.round(performance.now() - started),
      },
    })
    const result = turnResult({ status, answer, conversationID: conversation.id, exchangeID, fileID: current.fileID, model })
    return {
      status,
      answer,
      conversation_id: conversation.id,
      selected_file_id: current.fileID,
      error_code: status === "answered" ? "" : status,
      detail: "",
      result_json: Buffer.from(JSON.stringify(result)),
      exchange_id: exchangeID,
    }
  }

  const history = async (request: HistoryRequest) => {
    const current = scope(request)
    const limit = request.limit || 10
    if (limit < 1 || limit > 100) throw new Conversations.Invalid("limit must be between 1 and 100")
    if (!(await input.engine.dataset(current.fileID))) throw new Conversations.NotFound()
    const result = await input.store.history(current, request.conversation_id || undefined, limit)
    return { result_json: Buffer.from(JSON.stringify(result)) }
  }

  const unary =
    <Request, Response>(name: string, handler: (request: Request, signal: AbortSignal) => Promise<Response>) =>
    (call: grpc.ServerUnaryCall<Request, Response>, callback: grpc.sendUnaryData<Response>) => {
      const controller = new AbortController()
      call.on("cancelled", () => controller.abort())
      handler(call.request, controller.signal).then(
        (response) => callback(null, response),
        (error: unknown) => {
          if (error instanceof Conversations.Invalid)
            return callback({ code: grpc.status.INVALID_ARGUMENT, details: error.message })
          if (error instanceof Conversations.NotFound)
            return callback({ code: grpc.status.NOT_FOUND, details: "conversation not found" })
          if (controller.signal.aborted) return callback({ code: grpc.status.CANCELLED, details: "cancelled" })
          console.error(`NIA gRPC ${name.toLowerCase()} failed`, error)
          callback({ code: grpc.status.INTERNAL, details: `NIA ${name.toLowerCase()} failed` })
        },
      )
    }

  return {
    Health: unary("Health", async () => ({ status: "ok" })),
    Query: unary("Query", query),
    History: unary("History", history),
  }
}

export async function serve(input: {
  readonly store: Conversations.Store
  readonly engine: Engine
  readonly principal: string
  readonly model: string
  readonly hostname: string
  readonly port: number
}) {
  const server = new grpc.Server()
  server.addService(Assistant.service, handlers(input))
  const port = await new Promise<number>((resolve, reject) =>
    server.bindAsync(`${input.hostname}:${input.port}`, grpc.ServerCredentials.createInsecure(), (error, bound) =>
      error ? reject(error) : resolve(bound),
    ),
  )
  return {
    port,
    close: () => new Promise<void>((resolve) => server.tryShutdown(() => resolve())),
  }
}
