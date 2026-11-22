export * as NiaHttp from "./http"

// HTTP API the Next.js frontend calls. Research answers go through ResearchHost.
import { randomBytes } from "node:crypto"
import type { NiaConfig } from "./config"
import type { NiaEngine } from "./engine"
import { HttpSession } from "./http-session"
import { NiaSpeech } from "./speech"

const speechStatus: Record<NiaSpeech.Reason, number> = { invalid: 400, rate_limited: 429, unavailable: 503 }
// Room for the longest text with every character escaped, and for the largest recording with its form.
const maxTextBody = 256 * 1024
const maxAudioBody = NiaSpeech.maxAudioBytes + 1024 * 1024
// Bun closes a connection after 10 idle seconds; synthesis and transcription send nothing until done.
const speechIdleSeconds = NiaSpeech.timeoutMs / 1000 + 10

type Chat = {
  id: string
  /** The signed-in account that started the chat; nobody else can list, read or continue it. */
  owner: string
  title: string
  fileID: number
  label: string
  sessionID: string
  turns: { role: string; text: string; citations?: string[] }[]
}

export function serve(input: {
  readonly config: NiaConfig.Config
  readonly auth: HttpSession.Config
  readonly engine: NiaEngine.Engine
  readonly speech: NiaSpeech.Speech
  readonly hostname: string
  readonly port: number
}) {
  const { config, auth } = input
  const host = input.engine.host
  // Chats live in memory for the life of the process, like the research Sessions they point to.
  const chats = new Map<string, Chat>()

  function tokenOf(request: Request) {
    const header = request.headers.get("cookie") ?? ""
    const match = header.match(new RegExp(`(?:^|;\\s*)${HttpSession.cookieName}=([^;]+)`))
    return match?.[1] ? decodeURIComponent(match[1]) : ""
  }

  function principal(request: Request) {
    return HttpSession.verify(auth, tokenOf(request))
  }

  function ownedChat(id: string | undefined, owner: string) {
    const chat = id ? chats.get(id) : undefined
    return chat?.owner === owner ? chat : undefined
  }

  function cors(request: Request) {
    const requestOrigin = request.headers.get("origin") ?? auth.origins[0]!
    return {
      "Access-Control-Allow-Origin": HttpSession.allowedOrigin(auth, requestOrigin) ? requestOrigin : auth.origins[0]!,
      "Access-Control-Allow-Credentials": "true",
      "Access-Control-Allow-Headers": "Content-Type, Accept",
      "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
      Vary: "Origin",
    }
  }

  function json(request: Request, body: unknown, status = 200, extra: Record<string, string> = {}) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json", ...cors(request), ...extra },
    })
  }

  async function sessionBody() {
    return {
      principal_id: config.principal,
      allowed_file_ids: (await host.lists()).map((list) => list.fileID),
      can_use_private_files: config.private,
      history_allowed: false,
    }
  }

  async function datasets() {
    const [lists, offered] = await Promise.all([host.catalog(), host.lists()])
    return {
      datasets: lists.map((dataset) => ({
        file_id: dataset.fileID,
        filename: dataset.label,
        description: null,
        current_version: dataset.version,
        private: false,
        assistant_enabled: true,
        user_facing_label: dataset.label,
        is_default_people_scope: offered.some((list) => list.fileID === dataset.fileID && list.defaultPeople),
        fields: dataset.fields.map((field) => ({
          semantic_field: field.name,
          human_label: field.label,
          semantic_type: field.type,
          allowed_operators: field.operators,
          aggregatable: field.aggregatable,
          sortable: field.sortable,
        })),
        supported_operations: ["count", "list", "group", "set", "completeness"],
        starter_prompts: [],
      })),
      models: [{ id: config.model, selected: true }],
    }
  }

  // nordikdriveapi's /api/chat contract: errors are `{ error }`, and speech is the audio's own bytes.
  async function speak(request: Request) {
    const bytes = await readBody(request, maxTextBody)
    if (!bytes) return json(request, { error: "request is too large" }, 413)
    const text = await spokenText(bytes, request.headers.get("content-type") ?? "")
    if (!text) return json(request, { error: "text is required" }, 400)
    const { data, mime } = await input.speech.speak(text, request.signal)
    return new Response(data, { headers: { "Content-Type": mime, "Cache-Control": "no-store", ...cors(request) } })
  }

  // Returns the words only; the caller sends them to /api/assistant/query like a typed question.
  async function transcribe(request: Request) {
    const bytes = await readBody(request, maxAudioBody)
    if (!bytes) return json(request, { error: "audio must be between 1 byte and 20 MiB" }, 413)
    const form = await formOf(bytes, request.headers.get("content-type") ?? "")
    const audio = form?.get("audio")
    if (!audio || typeof audio === "string") return json(request, { error: "audio is required" }, 400)
    // Bun takes audio.type from the file's extension, so it can be empty; the provider reads the bytes.
    const text = await input.speech.transcribe(new Uint8Array(await audio.arrayBuffer()), audio.type, request.signal)
    return json(request, { text })
  }

  return Bun.serve({
    hostname: input.hostname,
    port: input.port,
    async fetch(request, server) {
      const url = new URL(request.url)
      if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(request) })
      if (request.method === "GET" && url.pathname === "/health") return json(request, { status: "ok" })

      if (request.method === "POST" && url.pathname === "/api/auth/login") {
        const body = (await request.json().catch(() => null)) as { username?: string; password?: string } | null
        if (!body) return json(request, { detail: "invalid login request" }, 400)
        const username = HttpSession.login(auth, body.username?.trim() ?? "", body.password ?? "")
        if (!username) return json(request, { detail: "unauthorized" }, 401)
        return json(request, await sessionBody(), 200, {
          "Set-Cookie": HttpSession.cookie(auth, HttpSession.issue(auth, username)),
        })
      }

      if (request.method === "GET" && url.pathname === "/api/auth/session") {
        if (!principal(request)) return new Response(null, { status: 204, headers: cors(request) })
        return json(request, await sessionBody())
      }

      if (request.method === "POST" && url.pathname === "/api/auth/logout") {
        return json(request, { ok: true }, 200, { "Set-Cookie": HttpSession.cleared(auth) })
      }

      const user = principal(request)
      if (!user) return json(request, { detail: "unauthorized" }, 401)

      const voice =
        url.pathname === "/api/chat/tts" ? speak : url.pathname === "/api/chat/transcribe" ? transcribe : undefined
      if (voice && request.method === "POST") {
        server.timeout(request, speechIdleSeconds)
        try {
          return await voice(request)
        } catch (error) {
          // The caller has gone, so nobody reads this response.
          if (request.signal.aborted) return new Response(null, { status: 499 })
          if (error instanceof NiaSpeech.Failure) return json(request, { error: error.message }, speechStatus[error.reason])
          console.error("speech request failed", error)
          return json(request, { error: "speech failed" }, 500)
        }
      }

      if (request.method === "GET" && url.pathname === "/api/assistant/datasets") return json(request, await datasets())

      if (request.method === "GET" && url.pathname === "/api/assistant/conversations") {
        const conversations = [...chats.values()]
          .filter((chat) => chat.owner === user)
          .map((chat) => ({
            id: chat.id,
            title: chat.title,
            active_frame_key: null,
            preview: chat.turns.at(-1)?.text.slice(0, 160) ?? "",
            turn_count: chat.turns.length,
            selected_file_id: chat.fileID,
            selected_dataset_label: chat.label,
            legacy_unscoped: false,
          }))
        return json(request, { conversations })
      }

      const conversationMatch = url.pathname.match(/^\/api\/assistant\/conversations\/([^/]+)$/)
      if (conversationMatch && request.method === "GET") {
        const chat = ownedChat(conversationMatch[1], user)
        if (!chat) return json(request, { detail: "not found" }, 404)
        return json(request, {
          id: chat.id,
          title: chat.title,
          selected_file_id: chat.fileID,
          selected_dataset_label: chat.label,
          legacy_unscoped: false,
          // Evidence IDs stay on the server; without source fields the frontend's chips open empty panels.
          turns: chat.turns.map(({ role, text }) => ({ role, text })),
        })
      }
      if (conversationMatch && request.method === "PATCH") {
        const chat = ownedChat(conversationMatch[1], user)
        if (!chat) return json(request, { detail: "not found" }, 404)
        const body = (await request.json()) as { title?: string }
        chat.title = body.title?.trim() || chat.title
        return json(request, { ok: true })
      }
      if (conversationMatch && request.method === "DELETE") {
        if (ownedChat(conversationMatch[1], user)) chats.delete(conversationMatch[1])
        return json(request, { ok: true })
      }

      if (request.method === "POST" && url.pathname === "/api/assistant/query") {
        const body = (await request.json()) as { question?: string; conversation_id?: string; selected_file_id?: number }
        const question = body.question?.trim() ?? ""
        const fileID = body.selected_file_id
        if (!question || !fileID) return json(request, { detail: "question and selected_file_id are required" }, 400)
        const lists = await host.catalog()
        const dataset = lists.find((item) => item.fileID === fileID)
        if (!dataset) return json(request, { detail: "list unavailable" }, 403)
        const existing = ownedChat(body.conversation_id, user)
        const encoder = new TextEncoder()
        const controller = new AbortController()
        const signal = AbortSignal.any([request.signal, controller.signal])
        let ping: ReturnType<typeof setInterval> | undefined
        let open = true
        // The answer is produced while the stream is already open: plan_started goes out at once
        // and a comment line every few seconds keeps Bun's 10s idleTimeout from closing the socket.
        const stream = new ReadableStream({
          start(controller) {
            const write = (chunk: string) => {
              if (open) controller.enqueue(encoder.encode(chunk))
            }
            const send = (event: string, data: unknown) => write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
            const finish = () => {
              clearInterval(ping)
              if (!open) return
              open = false
              controller.close()
            }
            send("plan_started", { question })
            ping = setInterval(() => write(": ping\n\n"), 5000)
            ;(async () => {
              signal.throwIfAborted()
              const chat = existing ?? {
                id: `conv_${randomBytes(8).toString("hex")}`,
                owner: user,
                title: question.slice(0, 80),
                fileID,
                label: dataset.label,
                sessionID: (await host.create(fileID)).id,
                turns: [],
              }
              // A chat whose research Session is gone continues in a new Session.
              if (existing && !(await host.get(chat.sessionID).catch(() => undefined)))
                chat.sessionID = (await host.create(chat.fileID)).id
              chats.set(chat.id, chat)
              const askedAt = new Date()
              const started = performance.now()
              const answer = await host.prompt(
                chat.sessionID,
                question,
                (event) => {
                  if (event.type !== "text") return
                  if (event.reset) send("answer_reset", {})
                  send("answer_delta", { text: event.delta })
                },
                signal,
              )
              // One JSON line per query keeps answer times searchable in the platform log; the question stays out.
              console.log(
                `QUERY ${JSON.stringify({
                  at: askedAt.toISOString(),
                  transport: "http",
                  status: "answered",
                  model: config.model.slice(config.model.indexOf("/") + 1),
                  file_id: fileID,
                  conversation_id: chat.id,
                  model_ms: Math.round(performance.now() - started),
                })}`,
              )
              const citations = answer.evidence.map((item) => item.id)
              chat.turns.push({ role: "user", text: question }, { role: "assistant", text: answer.text, citations })
              // answer_completed carries the authoritative text; the frontend replaces the streamed draft with it.
              send("answer_completed", {
                status: "completed",
                answer: answer.text,
                planner_type: null,
                reasoning_calls: 0,
                facts: [],
                rows: [],
                citations: [],
                evidence: [],
                conversation_id: chat.id,
                selected_file_id: fileID,
              })
            })()
              // Closing without answer_completed makes the frontend report the failure.
              .catch((error) => {
                if (!signal.aborted) console.error("assistant query failed", error)
              })
              .finally(finish)
          },
          cancel() {
            open = false
            clearInterval(ping)
            controller.abort()
          },
        })
        return new Response(stream, {
          headers: { "Content-Type": "text/event-stream", ...cors(request) },
        })
      }

      if (request.method === "POST" && url.pathname === "/api/assistant/feedback") {
        const body = (await request.json().catch(() => null)) as {
          comment?: string
          reporter_name?: string
          conversation_id?: string | null
          tz_offset_minutes?: number
          attachments?: { filename?: string; content_type?: string; data_base64?: string }[]
        } | null
        if (!body) return json(request, { detail: "invalid report" }, 400)
        // Chats live only in memory, so the report carries the conversation it is about. One JSON line
        // per report keeps them searchable in the platform log; attachment bytes stay out of the log.
        const chat = ownedChat(body.conversation_id ?? undefined, user)
        console.log(
          `REPORT ${JSON.stringify({
            at: new Date().toISOString(),
            principal: user,
            reporter: body.reporter_name?.slice(0, 200) ?? "",
            comment: body.comment?.slice(0, 4000) ?? "",
            tz_offset_minutes: body.tz_offset_minutes,
            conversation: chat
              ? {
                  id: chat.id,
                  title: chat.title,
                  list: chat.label,
                  turns: chat.turns.slice(-20).map((turn) => ({ role: turn.role, text: turn.text.slice(0, 2000) })),
                }
              : (body.conversation_id ?? null),
            attachments: (body.attachments ?? []).map((item) => ({
              filename: item.filename,
              content_type: item.content_type,
              bytes: Math.floor(((item.data_base64?.length ?? 0) * 3) / 4),
            })),
          })}`,
        )
        return json(request, { ok: true })
      }
      return json(request, { detail: "not found" }, 404)
    },
  })
}

/** The request's body, or undefined as soon as it passes `limit`, however it was sent. */
async function readBody(request: Request, limit: number) {
  if (Number(request.headers.get("content-length")) > limit) return undefined
  const chunks: Uint8Array[] = []
  let size = 0
  const reader = request.body?.getReader()
  while (reader) {
    const { done, value } = await reader.read()
    if (done) break
    size += value.byteLength
    if (size > limit) {
      await reader.cancel()
      return undefined
    }
    chunks.push(value)
  }
  return Buffer.concat(chunks)
}

async function formOf(bytes: Buffer<ArrayBuffer>, type: string) {
  if (!/^(multipart\/form-data|application\/x-www-form-urlencoded)/i.test(type)) return undefined
  return new Response(bytes, { headers: { "Content-Type": type } }).formData().catch(() => undefined)
}

/** `text`, else `answer`, from a form or a JSON body, as the Go API read them. */
async function spokenText(bytes: Buffer<ArrayBuffer>, type: string) {
  const form = await formOf(bytes, type)
  let body: { text?: unknown; answer?: unknown } | null = null
  if (form) body = { text: form.get("text"), answer: form.get("answer") }
  else {
    try {
      body = JSON.parse(bytes.toString("utf8"))
    } catch {
      return ""
    }
  }
  const field = (value: unknown) => (typeof value === "string" ? value.trim() : "")
  return field(body?.text) || field(body?.answer)
}
