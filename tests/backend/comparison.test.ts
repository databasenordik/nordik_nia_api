import { afterEach, expect, test } from "bun:test"
import { Effect, Stream } from "effect"
import { HttpClientRequest, HttpClientResponse } from "effect/http"
import { captureProviderCalls, comparisonEnabled, measureProviderHTTP, type ProviderCall } from "@opencode/core/session/comparison"

const original = process.env.CHAT_COMPARISON_UNTIL
afterEach(() => {
  if (original === undefined) delete process.env.CHAT_COMPARISON_UNTIL
  else process.env.CHAT_COMPARISON_UNTIL = original
})

test("comparison is opt-in and refuses expired and unbounded windows", () => {
  for (const value of ["", "invalid", "2020-01-01T00:00:00Z", "2099-01-01T00:00:00Z"]) {
    process.env.CHAT_COMPARISON_UNTIL = value
    expect(comparisonEnabled()).toBe(false)
    expect(measureProviderHTTP("inactive", "model")).toBeUndefined()
  }
})

test("provider duration includes the response body, excludes tools and isolates sessions", async () => {
  // Freeze only the gate clock; provider timings use the independent monotonic performance clock.
  const now = Date.now
  Date.now = () => Date.parse("2026-11-16T12:00:00Z")
  process.env.CHAT_COMPARISON_UNTIL = "2026-11-18T05:00:00Z"
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
  } finally { releaseA(); releaseB(); Date.now = now }
})
