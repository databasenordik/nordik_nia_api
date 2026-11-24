export * as NiaSpeech from "./speech"

// NIA owns spoken answers and transcription. reasoning.json selects Gemini or xAI for TTS;
// transcription continues to use {REASONING_BASE_URL}/stt.
import type { NiaConfig } from "./config"
import { create as geminiSpeech } from "./gemini-speech"
import type { Auth as GeminiAuth } from "./gemini-speech"

/** Longest text spoken at once, in characters: more than the model's longest (4,096-token) answer usually is. */
export const maxCharacters = 16_000
/** Largest recording, as the Go API accepted. */
export const maxAudioBytes = 20 * 1024 * 1024
/** Long enough for the longest text and recording; the transports allow a little more. */
export const timeoutMs = 120_000

export type Reason = "invalid" | "rate_limited" | "unavailable"

/** A failure whose message is fixed text, safe to show a caller: never the provider's reply. */
export class Failure extends Error {
  constructor(
    readonly reason: Reason,
    message: string,
  ) {
    super(message)
  }
}

export interface Speech {
  readonly speak: (text: string, signal?: AbortSignal) => Promise<{ data: Uint8Array<ArrayBuffer>; mime: string }>
  readonly transcribe: (audio: Uint8Array, mime: string, signal?: AbortSignal) => Promise<string>
}

const mimeTypes = { mp3: "audio/mpeg", wav: "audio/wav" } as const
// The provider recognises the container from its bytes; the name and type only describe the upload.
const extensions = new Map([
  ["audio/webm", "webm"],
  ["video/webm", "webm"],
  ["audio/ogg", "ogg"],
  ["audio/opus", "opus"],
  ["audio/wav", "wav"],
  ["audio/x-wav", "wav"],
  ["audio/mpeg", "mp3"],
  ["audio/mp4", "m4a"],
  ["video/mp4", "mp4"],
  ["audio/x-m4a", "m4a"],
  ["audio/aac", "aac"],
  ["audio/flac", "flac"],
])

export function create(input: {
  readonly baseURL: string
  /** Read at each call, as the engine does, so the key is never kept. */
  readonly apiKey: () => string | undefined
  /** Optional Gemini API key; otherwise use Vertex AI with application default credentials. */
  readonly geminiApiKey?: () => string | undefined
  readonly geminiAuth?: GeminiAuth
  readonly settings: NiaConfig.Speech
  readonly fetch?: (url: string, init: RequestInit) => Promise<Response>
  readonly timeoutMs?: number
}): Speech {
  const { tts, stt } = input.settings
  const base = input.baseURL.replace(/\/+$/, "")
  const send = input.fetch ?? fetch
  const gemini = tts.provider === "gemini"
    ? geminiSpeech({ settings: tts.gemini, apiKey: input.geminiApiKey, auth: input.geminiAuth })
    : undefined
  const xaiRequest = (operation: "tts" | "stt", body: BodyInit, headers: Record<string, string> = {}) => ({
    url: `${base}/${operation}`,
    body,
    headers: { ...headers, Authorization: `Bearer ${input.apiKey() ?? ""}` },
  })

  // One request and one log line. The line has sizes, time and status; the text, the transcript, the
  // audio and the key stay out of it. Nothing is retried: each attempt is billed.
  async function call<A>(options: {
    readonly operation: "speak" | "transcribe"
    readonly size: Record<string, number>
    readonly signal?: AbortSignal
    readonly provider?: "gemini" | "xai"
    readonly request: () => { readonly url: string; readonly body: BodyInit; readonly headers: Record<string, string> }
      | Promise<{ readonly url: string; readonly body: BodyInit; readonly headers: Record<string, string> }>
    readonly read: (response: Response) => Promise<A>
    readonly rejected: Failure
  }) {
    const { operation, signal } = options
    const started = performance.now()
    let status = "ok"
    let upstream: number | undefined
    try {
      signal?.throwIfAborted()
      const timeout = AbortSignal.timeout(input.timeoutMs ?? timeoutMs)
      const combined = signal ? AbortSignal.any([signal, timeout]) : timeout
      let abortRequest = () => {}
      const aborted = new Promise<never>((_, reject) => {
        abortRequest = () => reject(combined.reason)
        combined.addEventListener("abort", abortRequest, { once: true })
      })
      const { url, body, headers } = await Promise.race([Promise.resolve().then(options.request), aborted])
        .finally(() => combined.removeEventListener("abort", abortRequest))
      combined.throwIfAborted()
      const response = await send(url, {
        method: "POST",
        body,
        headers,
        // A redirect would carry the text or recording somewhere the operator did not configure.
        redirect: "error",
        signal: combined,
      })
      upstream = response.status
      if (response.ok) return await options.read(response)
      await response.body?.cancel().catch(() => undefined)
      if (response.status === 429) throw new Failure("rate_limited", "speech is busy; try again in a moment")
      if ([400, 413, 415, 422].includes(response.status)) throw options.rejected
      throw unavailable()
    } catch (error) {
      if (signal?.aborted) {
        status = "cancelled"
        throw signal.reason
      }
      // Anything else (network, timeout, unreadable reply) can quote the request, so it is replaced.
      const failure = error instanceof Failure ? error : unavailable()
      status = failure.reason
      throw failure
    } finally {
      console.log(
        `SPEECH ${JSON.stringify({
          at: new Date().toISOString(),
          operation,
          ...(options.provider ? { provider: options.provider } : {}),
          status,
          upstream_status: upstream,
          ...options.size,
          ms: Math.round(performance.now() - started),
        })}`,
      )
    }
  }

  return {
    speak: async (raw, signal) => {
      const written = raw.trim()
      const characters = Array.from(written).length
      if (characters > maxCharacters) throw new Failure("invalid", `text must be at most ${maxCharacters} characters`)
      if (gemini) {
        if (!written) throw new Failure("invalid", "text is required")
        return call({
          operation: "speak",
          provider: "gemini",
          size: { characters },
          signal,
          request: () => gemini.request(written),
          read: gemini.read,
          rejected: unavailable(),
        })
      }
      const text = plainSpeech(written)
      if (!text) throw new Failure("invalid", "text is required")
      const spoken = tts.delivery === "warm" ? warmDelivery(text) : text
      if (spoken.length > 60_000) throw new Failure("invalid", "speech formatting exceeds the supported length")
      return call({
        operation: "speak",
        size: { characters },
        signal,
        request: () => xaiRequest(
          "tts",
          JSON.stringify({
            text: spoken,
            voice_id: tts.voice,
            language: tts.language,
            speed: tts.speed,
            text_normalization: tts.text_normalization,
            optimize_streaming_latency: tts.optimize_streaming_latency,
            output_format: {
              codec: tts.codec,
              sample_rate: tts.sample_rate,
              ...(tts.codec === "mp3" ? { bit_rate: tts.bit_rate } : {}),
            },
            ...(Object.keys(tts.replace).length ? { replace: tts.replace } : {}),
          }),
          { "Content-Type": "application/json" },
        ),
        read: async (response) => {
          // A proxy can answer 200 with an error page; that must not be played as audio.
          const type = response.headers.get("content-type") ?? ""
          if (/json|^text\//i.test(type)) throw unavailable()
          const data = new Uint8Array(await response.arrayBuffer())
          if (!data.byteLength) throw unavailable()
          return { data, mime: mimeTypes[tts.codec] }
        },
        // The text was checked above, so a rejection means the provider refused the configured options.
        rejected: unavailable(),
      })
    },
    transcribe: async (audio, mime, signal) => {
      const bytes = audio.byteLength
      if (!bytes || bytes > maxAudioBytes) throw new Failure("invalid", "audio must be between 1 byte and 20 MiB")
      const type = mime.split(";")[0]!.trim().toLowerCase()
      const extension = extensions.get(type)
      return call({
        operation: "transcribe",
        size: { bytes },
        signal,
        request: () => {
          const form = new FormData()
          form.append("model", stt.model)
          if (stt.language !== "auto") form.append("language", stt.language)
          form.append("format", String(stt.format))
          form.append("filler_words", String(stt.filler_words))
          form.append("vad_threshold", String(stt.vad_threshold))
          for (const term of stt.keyterms) form.append("keyterm", term)
          // The file must come last: the provider may ignore any field after it.
          const file = new Blob([new Uint8Array(audio)], { type: extension ? type : "application/octet-stream" })
          form.append("file", file, extension ? `audio.${extension}` : "audio")
          return xaiRequest("stt", form)
        },
        read: async (response) => {
          const { text } = (await response.json()) as { text?: unknown }
          if (typeof text !== "string") throw unavailable()
          if (!text.trim()) throw new Failure("invalid", "no speech was detected")
          return text.trim()
        },
        rejected: new Failure("invalid", "audio could not be transcribed"),
      })
    },
  }
}

/**
 * The same words without the markup the provider acts on. It takes [x], (x), {x} and <x> for delivery
 * cues: tried live, it left out "[sic]" and "(pause)", said "(illegible)" as "legible", "[Smith or Smyth]"
 * as "Smith", "< 10" as "10" and "1903 | 1904" (one or the other) as a range. Records use all of these,
 * so an aside becomes a pause between commas, and a comparison or alternative becomes words.
 */
export function plainSpeech(text: string) {
  const pause = "\u0000"
  return (
    text
      // Single spaces from here on, so no pattern below scans a long run of them again and again.
      .replace(/[ \t]+/g, " ")
      // A comparison: < 10, >=5.
      .replace(/ ?< ?= ?(?=\d)/g, " at most ")
      .replace(/ ?> ?= ?(?=\d)/g, " at least ")
      .replace(/ ?< ?(?=\d)/g, " less than ")
      .replace(/ ?> ?(?=\d)/g, " more than ")
      // The records' notation for "exactly one of these".
      .replace(/(?<=[\p{L}\p{N}]) ?\| ?(?=[\p{L}\p{N}])/gu, " or ")
      // Letters supplied inside a word stay in it: Jo[h]nson, student(s), [J]ohn.
      .replace(/(?<=\p{L})[[(](\p{Ll}{1,3})[\])]/gu, "$1")
      .replace(/[[(](\p{L})[\])](?=\p{L})/gu, "$1")
      .replace(/(?: ?[[\](){}<>])+ ?/g, pause)
      .replace(/^\0|\0$/gm, "")
      .replace(/\0(?=[.,;:!?])/g, "")
      .replace(/([.,;:!?])\0/g, "$1 ")
      .replaceAll(pause, ", ")
      .trim()
  )
}

function unavailable() {
  return new Failure("unavailable", "speech is temporarily unavailable")
}

/** The selected Leo-Warm reader: only trusted tags, applied after plainSpeech sanitizes source notation. */
function warmDelivery(text: string) {
  const paragraphs = text.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean)
  return paragraphs.map((part, index) =>
    index === 0 || index === paragraphs.length - 1 ? `<soft>${part}</soft>` : part,
  ).join("\n\n[pause]\n\n")
}
