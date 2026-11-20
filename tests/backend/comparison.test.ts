import { afterEach, expect, test } from "bun:test"
import { Effect, Stream } from "effect"
import { HttpClientRequest, HttpClientResponse } from "effect/http"
import { captureProviderCalls, measureProviderHTTP, type ProviderCall } from "@opencode/core/session/comparison"

const original = process.env.CHAT_COMPARISON_UNTIL
afterEach(() => {
  if (original === undefined) delete process.env.CHAT_COMPARISON_UNTIL
  else process.env.CHAT_COMPARISON_UNTIL = original
})

test("provider capture works without a comparison window or after its expiry", async () => {
  for (const value of [undefined, "", "invalid", "2026-11-18T05:00:00Z"]) {
    if (value === undefined) delete process.env.CHAT_COMPARISON_UNTIL
    else process.env.CHAT_COMPARISON_UNTIL = value
    const calls: ProviderCall[] = []
    const release = captureProviderCalls("permanent", call => calls.push(call))
    try {
      expect(() => captureProviderCalls("permanent", () => {})).toThrow("Provider capture already active")
      const request = HttpClientRequest.post("https://provider.invalid")
      const response = await Effect.runPromise(measureProviderHTTP("permanent", "model")!(request, sent =>
        Effect.succeed(HttpClientResponse.fromWeb(sent, new Response("answer"))),
      ))
      await Effect.runPromise(Stream.runDrain(response.stream))
      expect(calls).toHaveLength(1)
      expect(calls[0]).toMatchObject({ model: "model", outcome: "ok" })
    } finally { release() }
    expect(measureProviderHTTP("permanent", "model")).toBeUndefined()
  }
  expect(measureProviderHTTP("inactive", "model")).toBeUndefined()
})

test("provider duration includes the response body, excludes tools and isolates sessions", async () => {
  const a: ProviderCall[] = [], b: ProviderCall[] = []
  const releaseA = captureProviderCalls("a", call => a.push(call))
  const releaseB = captureProviderCalls("b", call => b.push(call))
  try {
    const request = HttpClientRequest.post("https://provider.invalid")
    const handler = (sent: typeof request) => Effect.sync(() => HttpClientResponse.fromWeb(sent, new Response(new ReadableStream({
      async start(controller) { await Bun.sleep(30); controller.enqueue(new TextEncoder().encode("answer")); controller.close() },
    }))))
    // Stand-in for retrieval/tool work: must not be counted.
    await Bun.sleep(70)
    const response = await Effect.runPromise(measureProviderHTTP("a", "model-a")!(request, handler))
    await Effect.runPromise(Stream.runDrain(response.stream))
    expect(a).toHaveLength(1)
    expect(a[0]!.duration_ms).toBeGreaterThanOrEqual(20)
    expect(a[0]!.duration_ms).toBeLessThan(85)
    expect(a[0]!.outcome).toBe("ok")
    expect(b).toHaveLength(0)
    await Effect.runPromise(measureProviderHTTP("b", "model-b")!(request, () => Effect.fail(new Error("offline")))).catch(() => {})
    expect(b).toHaveLength(1)
    expect(b[0]!.outcome).toBe("error")
  } finally { releaseA(); releaseB() }
})
