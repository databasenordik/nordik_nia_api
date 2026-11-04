import { expect, test } from "bun:test"
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

function serve(prompt: (...args: unknown[]) => Promise<unknown>) {
  return NiaHttp.serve({
    config,
    auth,
    hostname: "127.0.0.1",
    port: 0,
    engine: {
      directory: "/tmp/unused",
      host: {
        catalog: async () => [{ fileID: 49, label: "Test list", fields: [] }],
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

test("HTTP queries still deliver the completed answer", async () => {
  const server = serve(async () => ({ text: "Test answer", evidence: [] }))
  try {
    const response = await query(server)
    expect(response.status).toBe(200)
    const events = await response.text()
    expect(events).toContain("event: plan_started")
    expect(events).toContain("event: answer_completed")
    expect(events).toContain('"answer":"Test answer"')
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
