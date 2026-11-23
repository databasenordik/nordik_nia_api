import { expect, spyOn, test } from "bun:test"
import { NiaHttp } from "../../backend/src/http"
import { NiaConfig } from "@nia/backend/config"
import { HttpSession } from "@nia/backend/http-session"
import { NiaSpeech } from "../../backend/src/speech"

const config = NiaConfig.load({
  NIA_GRPC_PRINCIPAL_ID: "test",
  REASONING_BASE_URL: "https://unused.example/v1",
  REASONING_API_KEY: "unused",
  ASSISTANT_DATABASE_URL: "postgresql://test:test@localhost/test?sslmode=disable",
})
const auth = HttpSession.load({})

const silent: NiaSpeech.Speech = {
  speak: async () => ({ data: new Uint8Array([1, 2, 3]), mime: "audio/mpeg" }),
  transcribe: async () => "transcript",
}

function serve(
  prompt: (...args: unknown[]) => Promise<unknown>,
  lists = [{ fileID: 49, defaultPeople: true, private: false }],
  speech = silent,
) {
  return NiaHttp.serve({
    config,
    auth,
    speech,
    hostname: "127.0.0.1",
    port: 0,
    engine: {
      directory: "/tmp/unused",
      host: {
        lists: async () => lists,
        catalog: async () => lists.map((list) => ({ fileID: list.fileID, label: "Test list", version: 1, fields: [] })),
        create: async () => ({ id: "test-session" }),
        get: async () => ({ id: "test-session" }),
        prompt,
      } as unknown as Parameters<typeof NiaHttp.serve>[0]["engine"]["host"],
    },
  })
}

function query(server: ReturnType<typeof serve>, signal?: AbortSignal) {
  return fetch(new URL("/api/assistant/query", server.url), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Cookie: `${HttpSession.cookieName}=${HttpSession.issue(auth, "researcher")}`,
    },
    body: JSON.stringify({ question: "How many?", selected_file_id: 49 }),
    signal,
  })
}

test("HTTP queries keep streamed updates, completed answers and timing logs", async () => {
  const logs = spyOn(console, "log").mockImplementation(() => {})
  let promptSignal: AbortSignal | undefined
  const server = serve(async (...args) => {
    promptSignal = args[3] as AbortSignal
    const progress = args[2] as (event: { type: "text"; delta: string; reset?: boolean }) => void
    progress({ type: "text", delta: "Draft" })
    progress({ type: "text", delta: "Test answer", reset: true })
    return { text: "Test answer", evidence: [] }
  })
  try {
    const response = await query(server)
    expect(response.status).toBe(200)
    const events = await response.text()
    expect(events).toContain("event: plan_started")
    expect(events).toContain("event: answer_delta")
    expect(events).toContain('"text":"Draft"')
    expect(events).toContain("event: answer_reset")
    expect(events).toContain("event: answer_completed")
    expect(events).toContain('"answer":"Test answer"')
    expect(promptSignal).toBeInstanceOf(AbortSignal)
    expect(promptSignal?.aborted).toBe(false)
    expect(logs).toHaveBeenCalledTimes(1)
    const line = logs.mock.calls[0]![0] as string
    expect(line.startsWith("QUERY ")).toBe(true)
    const timing = JSON.parse(line.slice("QUERY ".length))
    expect(timing).toMatchObject({
      transport: "http",
      status: "answered",
      model: config.model.slice(config.model.indexOf("/") + 1),
      file_id: 49,
    })
    expect(timing.model_ms).toBeGreaterThanOrEqual(0)
    expect(line).not.toContain("How many?")
  } finally {
    await server.stop(true)
    logs.mockRestore()
  }
})

test("HTTP sessions and dataset defaults follow the lists offered by the gateway", async () => {
  const lists = [
    { fileID: 49, defaultPeople: false, private: false },
    { fileID: 121, defaultPeople: true, private: false },
  ]
  const server = serve(async () => ({ text: "Test answer", evidence: [] }), lists)
  const headers = { Cookie: `${HttpSession.cookieName}=${HttpSession.issue(auth, "researcher")}` }
  try {
    const session = await fetch(new URL("/api/auth/session", server.url), { headers })
    expect(session.status).toBe(200)
    expect(await session.json()).toMatchObject({ allowed_file_ids: [49, 121] })
    const response = await fetch(new URL("/api/assistant/datasets", server.url), { headers })
    expect(response.status).toBe(200)
    const body = (await response.json()) as { datasets: { file_id: number; is_default_people_scope: boolean }[] }
    expect(body.datasets.map((dataset) => [dataset.file_id, dataset.is_default_people_scope])).toEqual([
      [49, false],
      [121, true],
    ])
  } finally {
    await server.stop(true)
  }
})

test.each(["request abort", "response cancellation"])("HTTP %s aborts the provider prompt", async (mode) => {
  const started = Promise.withResolvers<AbortSignal>()
  const stopped = Promise.withResolvers<void>()
  const server = serve(async (...args) => {
    const signal = args[3] as AbortSignal
    started.resolve(signal)
    return new Promise((_, reject) => {
      const stop = () => {
        stopped.resolve()
        reject(signal.reason)
      }
      if (signal.aborted) stop()
      else signal.addEventListener("abort", stop, { once: true })
    })
  })
  const controller = new AbortController()
  try {
    const response = await query(server, controller.signal)
    const reader = response.body!.getReader()
    await reader.read()
    const signal = await started.promise
    expect(signal.aborted).toBe(false)
    if (mode === "request abort") controller.abort()
    else await reader.cancel()
    await stopped.promise
    expect(signal.aborted).toBe(true)
  } finally {
    controller.abort()
    await server.stop(true)
  }
})

function voiceServer(speech: Partial<NiaSpeech.Speech> = {}) {
  return serve(async () => ({}), undefined, { ...silent, ...speech })
}

function post(server: ReturnType<typeof serve>, route: string, init: RequestInit = {}, signedIn = true) {
  const headers = new Headers(init.headers)
  if (signedIn) headers.set("Cookie", `${HttpSession.cookieName}=${HttpSession.issue(auth, "researcher")}`)
  return fetch(new URL(`/api/chat/${route}`, server.url), { method: "POST", ...init, headers })
}

const asForm = (fields: Record<string, string>) => {
  const form = new FormData()
  for (const [name, value] of Object.entries(fields)) form.append(name, value)
  return { body: form }
}

test("both voice routes require sign-in and spend nothing without it", async () => {
  let used = 0
  const server = voiceServer({
    speak: async () => (used++, { data: new Uint8Array([1]), mime: "audio/mpeg" }),
    transcribe: async () => (used++, "x"),
  })
  try {
    for (const route of ["tts", "transcribe"]) expect((await post(server, route, asForm({ text: "Hi" }), false)).status).toBe(401)
    expect(used).toBe(0)
  } finally {
    await server.stop(true)
  }
})

test("tts takes text or answer, as a form or JSON, and returns the audio's bytes", async () => {
  const spoken: string[] = []
  const server = voiceServer({
    speak: async (text) => {
      spoken.push(text)
      return { data: new Uint8Array([0xff, 0xf3, 7]), mime: "audio/mpeg" }
    },
  })
  try {
    const inputs: RequestInit[] = [
      // What nordik-drive sends; the Go API ignores everything but the text.
      asForm({ text: "Form text", model: "gemini-2.5-flash-tts", voice: "Albenib", language: "en-US", style: "warm" }),
      asForm({ answer: "Form answer" }),
      asForm({ text: " ", answer: "Answer when text is blank" }),
      { body: new URLSearchParams({ text: "Urlencoded" }) },
      { body: JSON.stringify({ text: "JSON text" }), headers: { "Content-Type": "application/json" } },
      { body: JSON.stringify({ answer: "JSON answer" }), headers: { "Content-Type": "application/json" } },
    ]
    for (const init of inputs) {
      const response = await post(server, "tts", init)
      expect(response.status).toBe(200)
      expect(response.headers.get("content-type")).toBe("audio/mpeg")
      expect(response.headers.get("cache-control")).toBe("no-store")
      expect([...new Uint8Array(await response.arrayBuffer())]).toEqual([0xff, 0xf3, 7])
    }
    expect(spoken).toEqual(["Form text", "Form answer", "Answer when text is blank", "Urlencoded", "JSON text", "JSON answer"])
  } finally {
    await server.stop(true)
  }
})

test.each(["{}", "null", "not json", '{"text":42}', ""])("tts refuses %p as the Go API did", async (body) => {
  const server = voiceServer()
  try {
    const response = await post(server, "tts", { body, headers: { "Content-Type": "application/json" } })
    expect(response.status).toBe(400)
    expect((await response.json()) as unknown).toEqual({ error: "text is required" })
  } finally {
    await server.stop(true)
  }
})

test("the longest text fits even when every character is escaped", async () => {
  let spoken = ""
  const server = voiceServer({
    speak: async (text) => {
      spoken = text
      return { data: new Uint8Array([1]), mime: "audio/mpeg" }
    },
  })
  try {
    const body = `{"text":"${"\\ud83d\\ude42".repeat(16_000)}"}`
    expect((await post(server, "tts", { body })).status).toBe(200)
    expect(Array.from(spoken)).toHaveLength(16_000)
  } finally {
    await server.stop(true)
  }
})

test("transcribe returns the words of the uploaded recording", async () => {
  let heard: [number[], string] | undefined
  const server = voiceServer({
    transcribe: async (audio, mime) => {
      heard = [[...audio], mime]
      return "How many students?"
    },
  })
  try {
    const body = new FormData()
    body.append("audio", new Blob([new Uint8Array([4, 5, 6])], { type: "audio/webm" }), "q.webm")
    const response = await post(server, "transcribe", { body })
    expect(response.status).toBe(200)
    expect((await response.json()) as unknown).toEqual({ text: "How many students?" })
    // Bun's form parser takes the type from the file's extension...
    expect(heard).toEqual([[4, 5, 6], "video/webm"])
    // ...and gives none to a recorder's default name. The bytes still identify the container upstream.
    const unnamed = new FormData()
    unnamed.append("audio", new Blob([new Uint8Array([7])], { type: "audio/webm" }), "blob")
    expect((await post(server, "transcribe", { body: unnamed })).status).toBe(200)
    expect(heard).toEqual([[7], ""])
  } finally {
    await server.stop(true)
  }
})

test("transcribe refuses a request without a file, or with a broken form", async () => {
  const server = voiceServer()
  try {
    const broken = { body: "garbage", headers: { "Content-Type": "multipart/form-data; boundary=missing" } }
    for (const init of [asForm({ other: "x" }), asForm({ audio: "not a file" }), broken, { body: "{}" }])
      expect((await post(server, "transcribe", init)).status).toBe(400)
  } finally {
    await server.stop(true)
  }
})

test("oversized bodies are refused, with or without a declared length", async () => {
  let used = 0
  const server = voiceServer({
    speak: async () => (used++, { data: new Uint8Array([1]), mime: "audio/mpeg" }),
    transcribe: async () => (used++, "x"),
  })
  try {
    expect((await post(server, "tts", { body: "x".repeat(256 * 1024 + 1) })).status).toBe(413)
    const chunk = new Uint8Array(64 * 1024)
    let sent = 0
    const body = new ReadableStream({
      pull(controller) {
        if (++sent > 8) controller.close()
        else controller.enqueue(chunk)
      },
    })
    expect((await post(server, "tts", { body })).status).toBe(413)
    expect((await post(server, "transcribe", { body: new Uint8Array(22 * 1024 * 1024) })).status).toBe(413)
    expect(used).toBe(0)
  } finally {
    await server.stop(true)
  }
})

test.each([
  ["invalid", 400],
  ["rate_limited", 429],
  ["unavailable", 503],
] as const)("a %s speech failure is HTTP %i", async (reason, status) => {
  const server = voiceServer({
    speak: async () => {
      throw new NiaSpeech.Failure(reason, "safe message")
    },
  })
  try {
    const response = await post(server, "tts", asForm({ text: "Hello" }))
    expect(response.status).toBe(status)
    expect((await response.json()) as unknown).toEqual({ error: "safe message" })
  } finally {
    await server.stop(true)
  }
})

test("an unexpected failure does not reveal itself", async () => {
  const errors = spyOn(console, "error").mockImplementation(() => {})
  const server = voiceServer({
    speak: async () => {
      throw new Error("private detail")
    },
  })
  try {
    const response = await post(server, "tts", asForm({ text: "Hello" }))
    expect(response.status).toBe(500)
    expect((await response.json()) as unknown).toEqual({ error: "speech failed" })
  } finally {
    await server.stop(true)
    errors.mockRestore()
  }
})

test("synthesis slower than Bun's 10 second idle limit still returns", async () => {
  const server = voiceServer({
    speak: async () => {
      await Bun.sleep(16_000)
      return { data: new Uint8Array([1]), mime: "audio/mpeg" }
    },
  })
  try {
    expect((await post(server, "tts", asForm({ text: "A long answer" }))).status).toBe(200)
  } finally {
    await server.stop(true)
  }
}, 30_000)

test("a caller leaving cancels the speech request", async () => {
  let given: AbortSignal | undefined
  const server = voiceServer({
    speak: async (_, signal) => {
      given = signal
      await Bun.sleep(1000)
      signal?.throwIfAborted()
      return { data: new Uint8Array([1]), mime: "audio/mpeg" }
    },
  })
  try {
    await post(server, "tts", { ...asForm({ text: "Hello" }), signal: AbortSignal.timeout(150) }).catch(() => undefined)
    await Bun.sleep(300)
    expect(given?.aborted).toBe(true)
  } finally {
    await server.stop(true)
  }
})
