import { expect, spyOn, test } from "bun:test"
import { NiaHttp } from "../../backend/src/http"
import { NiaConfig } from "@nia/backend/config"
import { HttpSession } from "@nia/backend/http-session"

const config = NiaConfig.load({
  NIA_GRPC_PRINCIPAL_ID: "test",
  REASONING_BASE_URL: "https://unused.example/v1",
  REASONING_API_KEY: "unused",
  ASSISTANT_DATABASE_URL: "postgresql://test:test@localhost/test?sslmode=disable",
})
const auth = HttpSession.load({})

function serve(
  prompt: (...args: unknown[]) => Promise<unknown>,
  lists = [{ fileID: 49, defaultPeople: true, private: false }],
) {
  return NiaHttp.serve({
    config,
    auth,
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
