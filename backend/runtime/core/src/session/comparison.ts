// Temporary, request-scoped provider measurements; remove after November 17.
import { Effect, Stream } from "effect"
import { HttpClientResponse } from "effect/http"
import type { HttpMiddleware } from "@opencode/ai/route"

export interface ProviderCall { model: string; duration_ms: number; outcome: string }
const captures = new Map<string, (call: ProviderCall) => void>()
export function comparisonEnabled() {
  const until = Date.parse(process.env.CHAT_COMPARISON_UNTIL ?? "")
  return Number.isFinite(until) && until <= Date.parse("2026-11-18T05:00:00Z") && Date.now() < until
}
export function captureProviderCalls(sessionID: string, record: (call: ProviderCall) => void) {
  if (!comparisonEnabled()) return () => {}
  if (captures.has(sessionID)) throw new Error("Provider capture already active for session")
  captures.set(sessionID, record)
  return () => { captures.delete(sessionID) }
}
// Start after hooks/preparation; finish only when the provider body ends, errors or is cancelled.
// Every actual HTTP attempt has its own sample. Tool execution and retry sleep are outside it.
export function measureProviderHTTP(sessionID: string, model: string, existing?: HttpMiddleware): HttpMiddleware {
  return (request, handler) => {
    const measured: typeof handler = (sent) => Effect.gen(function* () {
      const record = captures.get(sessionID)
      if (!record) return yield* handler(sent)
      const started = performance.now()
      let recorded = false
      const finish = (outcome: string) => {
        if (recorded) return
        recorded = true
        record({ model, duration_ms: performance.now() - started, outcome })
      }
      const response = yield* handler(sent).pipe(Effect.onError(() => Effect.sync(() => finish("error"))))
      let outcome = response.status >= 400 ? "error" : "incomplete"
      const stream = response.stream.pipe(
        Stream.concat(Stream.fromEffect(Effect.sync(() => { outcome = response.status >= 400 ? "error" : "ok" })).pipe(Stream.drain)),
        Stream.ensuring(Effect.sync(() => finish(outcome))),
      )
      const body = [204, 205, 304].includes(response.status) ? null : yield* Stream.toReadableStreamEffect(stream)
      if (!body) finish(response.status >= 400 ? "error" : "ok")
      return HttpClientResponse.fromWeb(sent, new Response(body, { status: response.status, headers: response.headers }))
    })
    return existing ? existing(request, measured) : measured(request)
  }
}
