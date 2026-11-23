export * as NiaConfig from "./config"

import { ResearchDatabase } from "./research/database"
// The model, its effort and the speech options are part of the code, not the deployment:
// backend/reasoning.json is committed, and changing any of them is a commit and a release.
import file from "../reasoning.json"

/** The provider ID the configured REASONING_* endpoint is registered under. */
export const provider = "reasoning"

/** The contents of backend/reasoning.json. */
export const reasoning: { readonly model?: unknown; readonly effort?: unknown; readonly speech?: unknown } = file

/** reasoning.json's `speech` block, validated, under the names the provider uses. */
export interface Speech {
  readonly tts: {
    readonly provider: "xai" | "gemini"
    readonly gemini: {
      readonly model: string
      readonly voice: string
      readonly style: string
      readonly location: string
    }
    readonly voice: string
    readonly language: string
    readonly speed: number
    readonly text_normalization: boolean
    readonly optimize_streaming_latency: number
    readonly codec: "mp3" | "wav"
    readonly sample_rate: number
    /** Sent for mp3 only. */
    readonly bit_rate: number
    /** Phrase to the way it is said. */
    readonly replace: Readonly<Record<string, string>>
  }
  readonly stt: {
    readonly model: string
    /** "auto" lets the provider detect it. */
    readonly language: string
    readonly format: boolean
    readonly filler_words: boolean
    readonly vad_threshold: number
    readonly keyterms: readonly string[]
  }
}

export interface Config {
  readonly transport: "http" | "grpc"
  readonly production: boolean
  readonly principal: string
  readonly private: boolean
  /** `provider/model` reference for the engine: reasoning.json's model answers every question. */
  readonly model: string
  readonly baseURL: string
  /** Sent as reasoning_effort; undefined for a model that takes none. */
  readonly effort?: string
  /** Spoken-answer provider and transcription options. */
  readonly speech: Speech
  readonly database: { readonly url: string; readonly ssl: ResearchDatabase.Options["ssl"]; readonly max: number }
  /** Settings that were replaced by a default, for the operator to see at startup. */
  readonly warnings: readonly string[]
}

// The effort levels each model accepted on the configured endpoint, lowest first, as measured by the
// benchmark run in benchmark-results/xai-2026-10-17-rerun. The dated grok-4.20 models take none.
const modelEfforts: Readonly<Record<string, readonly string[]>> = {
  "grok-4.7": ["low", "medium", "high", "xhigh"],
  "grok-4.6": ["low", "medium", "high", "xhigh"],
  "grok-4.5": ["low", "medium", "high"],
  "grok-4.3": ["none", "low", "medium", "high"],
  "grok-4.20-0309-reasoning": [],
  "grok-4.20-0309-non-reasoning": [],
}
// For any other model: these names are passed on, and low (the lowest level every measured reasoning
// model accepts) is the default.
const knownEfforts = ["none", "minimal", "low", "medium", "high", "xhigh"]
// What xAI's /tts and /stt accept (docs.x.ai, November 2026). Only the codecs browsers can play are
// offered: pcm, mulaw and alaw are for telephony.
const voices = `altair ara atlas aurora carina castor celeste cosmo eve helios helix iris kepler leo liora
  lumen luna lux naksh orion perseus rex rigel sal sirius ursa zagan zenith`.split(/\s+/)
const spokenLanguages = `auto en ar-EG ar-SA ar-AE bn zh fr de hi id it ja ko pt-BR pt-PT ru es-MX es-ES
  tr vi`.split(/\s+/)
const heardLanguages = `auto ar ar-AE ar-EG ar-SA bs bg yue ca zh hr cs da nl en fil fi fr de el hi hu id it
  ja ko mk ms nb fa pl pt pt-BR pt-PT ro ru sk es es-ES es-MX sv th tr uk ur vi`.split(/\s+/)
// The languages whose spoken numbers the provider can write as digits (format: true).
const formattedLanguages = ["ar", "de", "en", "es", "fr", "ja", "pt", "ru", "sv", "vi", "zh"]
const codecs = ["mp3", "wav"] as const
const sampleRates = [8000, 16000, 22050, 24000, 44100, 48000]
const bitRates = [32000, 64000, 96000, 128000, 192000]
const transcribers = ["grok-voice-transcribe-2.0"]
const speechDefaults: Speech = {
  tts: {
    provider: "xai",
    gemini: {
      model: "gemini-2.5-flash-tts",
      voice: "Algenib",
      style: "empathetic and more human like",
      location: "global",
    },
    voice: "sal",
    language: "en",
    speed: 1,
    text_normalization: false,
    optimize_streaming_latency: 0,
    codec: "mp3",
    sample_rate: 24000,
    bit_rate: 64000, // clear for speech, and half the size of the provider's default
    replace: {},
  },
  stt: {
    model: "grok-voice-transcribe-2.0",
    language: "en",
    format: false,
    filler_words: false,
    vad_threshold: 0.5,
    keyterms: [],
  },
}
const sslModes = ["disable", "require", "verify-ca", "verify-full"] as const

/**
 * The effort sent for `model`: the requested level when the model accepts it, otherwise (unset or not
 * accepted) the model's lowest level. A model that takes no effort level gets none.
 */
export function effort(model: string, requested: unknown): { effort?: string; warning?: string } {
  const wanted = (typeof requested === "string" ? requested.trim().toLowerCase() : "") || undefined
  const levels = modelEfforts[model]
  if (levels?.length === 0)
    return wanted ? { warning: `reasoning.json effort "${wanted}" is ignored: ${model} takes no effort level` } : {}
  if (wanted && (levels ?? knownEfforts).includes(wanted)) return { effort: wanted }
  const lowest = levels?.[0] ?? "low"
  return wanted
    ? { effort: lowest, warning: `reasoning.json effort "${wanted}" is not accepted by ${model}; using ${lowest}` }
    : { effort: lowest }
}

/**
 * The speech options to use: each accepted setting, otherwise its default with a warning. An option
 * that is left out takes its default silently, as does a missing block.
 */
export function speech(value: unknown): { speech: Speech; warnings: string[] } {
  const warnings: string[] = []
  const object = (input: unknown) =>
    typeof input === "object" && input !== null && !Array.isArray(input) ? (input as Record<string, unknown>) : undefined
  const block = (input: unknown, name: string, known: object): Record<string, unknown> => {
    if (input === undefined) return {}
    const found = object(input)
    if (!found) {
      warnings.push(`reasoning.json ${name} must be an object; using its defaults`)
      return {}
    }
    for (const key of Object.keys(found))
      if (!Object.hasOwn(known, key)) warnings.push(`reasoning.json ${name}.${key} is not an option; ignoring it`)
    return found
  }
  const root = block(value, "speech", speechDefaults)
  const tts = block(root.tts, "speech.tts", speechDefaults.tts)
  const stt = block(root.stt, "speech.stt", speechDefaults.stt)
  const gemini = block(tts.gemini, "speech.tts.gemini", speechDefaults.tts.gemini)

  const pick = <T>(name: string, input: unknown, fallback: T, accept: (input: unknown) => T | undefined): T => {
    if (input === undefined) return fallback
    const accepted = accept(input)
    if (accepted !== undefined) return accepted
    // Word lists can hold people's names, so a warning shows only single values.
    const shown = typeof input === "object" ? "" : ` ${JSON.stringify(input)}`
    const used = typeof fallback === "object" ? "none" : String(fallback)
    warnings.push(`reasoning.json ${name}${shown} is not accepted; using ${used}`)
    return fallback
  }
  const word =
    <T extends string>(allowed: readonly T[]) =>
    (input: unknown) =>
      typeof input === "string" ? allowed.find((item) => item.toLowerCase() === input.trim().toLowerCase()) : undefined
  const among = (allowed: readonly number[]) => (input: unknown) => allowed.find((item) => item === input)
  const between = (min: number, max: number) => (input: unknown) =>
    typeof input === "number" && input >= min && input <= max ? input : undefined
  const boolean = (input: unknown) => (typeof input === "boolean" ? input : undefined)
  const text = (max: number, pattern?: RegExp) => (input: unknown) =>
    typeof input === "string" && input.trim() && input.length <= max && (!pattern || pattern.test(input.trim()))
      ? input.trim()
      : undefined
  // The provider refuses every request when one pronunciation entry is bad, so check the whole map here.
  const replacements = (input: unknown) => {
    const map = object(input)
    if (!map) return undefined
    const entries = Object.entries(map)
    const valid = entries.every(
      ([key, said]) =>
        key.trim() &&
        /^[\p{L}\p{M}\p{N}'’ ]+$/u.test(key) &&
        Array.from(key).length <= 100 &&
        typeof said === "string" &&
        Array.from(said).length <= 128,
    )
    const phrases = new Set(entries.map(([key]) => key.trim().replace(/\s+/g, " ").toLowerCase()))
    return valid && entries.length <= 200 && phrases.size === entries.length ? (map as Record<string, string>) : undefined
  }
  const terms = (input: unknown) =>
    Array.isArray(input) &&
    input.length <= 100 &&
    input.every((item) => typeof item === "string" && item.trim() && Array.from(item).length <= 50)
      ? (input as string[])
      : undefined

  const spoken = speechDefaults.tts
  const heard = speechDefaults.stt
  const language = pick("speech.stt.language", stt.language, heard.language, word(heardLanguages))
  let format = pick("speech.stt.format", stt.format, heard.format, boolean)
  if (format && !formattedLanguages.includes(language.split("-")[0]!)) {
    warnings.push(`reasoning.json speech.stt.format is not available for language "${language}"; using false`)
    format = false
  }
  return {
    speech: {
      tts: {
        provider: pick("speech.tts.provider", tts.provider, spoken.provider, word(["xai", "gemini"] as const)),
        gemini: {
          model: pick(
            "speech.tts.gemini.model", gemini.model, spoken.gemini.model,
            text(100, /^gemini-[a-z0-9.-]+-tts(?:-[a-z0-9.-]+)?$/),
          ),
          voice: pick("speech.tts.gemini.voice", gemini.voice, spoken.gemini.voice, text(80, /^[A-Za-z][A-Za-z0-9_-]*$/)),
          style: pick("speech.tts.gemini.style", gemini.style, spoken.gemini.style, text(2000)),
          location: pick("speech.tts.gemini.location", gemini.location, spoken.gemini.location, text(80, /^[a-z][a-z0-9-]*$/)),
        },
        voice: pick("speech.tts.voice", tts.voice, spoken.voice, word(voices)),
        language: pick("speech.tts.language", tts.language, spoken.language, word(spokenLanguages)),
        speed: pick("speech.tts.speed", tts.speed, spoken.speed, between(0.7, 1.5)),
        text_normalization: pick("speech.tts.text_normalization", tts.text_normalization, spoken.text_normalization, boolean),
        optimize_streaming_latency: pick(
          "speech.tts.optimize_streaming_latency",
          tts.optimize_streaming_latency,
          spoken.optimize_streaming_latency,
          among([0, 1, 2]),
        ),
        codec: pick("speech.tts.codec", tts.codec, spoken.codec, word(codecs)),
        sample_rate: pick("speech.tts.sample_rate", tts.sample_rate, spoken.sample_rate, among(sampleRates)),
        bit_rate: pick("speech.tts.bit_rate", tts.bit_rate, spoken.bit_rate, among(bitRates)),
        replace: pick("speech.tts.replace", tts.replace, spoken.replace, replacements),
      },
      stt: {
        model: pick("speech.stt.model", stt.model, heard.model, word(transcribers)),
        language,
        format,
        filler_words: pick("speech.stt.filler_words", stt.filler_words, heard.filler_words, boolean),
        vad_threshold: pick("speech.stt.vad_threshold", stt.vad_threshold, heard.vad_threshold, between(0, 1)),
        keyterms: pick("speech.stt.keyterms", stt.keyterms, heard.keyterms, terms),
      },
    },
    warnings,
  }
}

// The deployment's variables: NIA_TRANSPORT, APP_ENV, NIA_GRPC_PRINCIPAL_ID,
// NIA_GRPC_CAN_USE_PRIVATE_FILES, REASONING_BASE_URL, REASONING_API_KEY, ASSISTANT_DATABASE_URL
// and optional NIA_DB_POOL_MAX (per pool; both conversations and research have a pool),
// with the model, effort and speech options from backend/reasoning.json. Every problem is reported at once, before any
// connection is opened, and no message repeats a value (the database URL carries a password).
export function load(env: Readonly<Record<string, string | undefined>>, settings = reasoning): Config {
  const errors: string[] = []
  const transport = env.NIA_TRANSPORT?.trim() || "http"
  if (transport !== "http" && transport !== "grpc") errors.push("NIA_TRANSPORT must be http or grpc")
  const principal = env.NIA_GRPC_PRINCIPAL_ID?.trim() ?? ""
  if (!principal) errors.push("NIA_GRPC_PRINCIPAL_ID is required")
  const privateFiles = flag(env.NIA_GRPC_CAN_USE_PRIVATE_FILES)
  if (privateFiles === undefined) errors.push("NIA_GRPC_CAN_USE_PRIVATE_FILES must be true or false")
  const model = typeof settings.model === "string" ? settings.model.trim() : ""
  if (!model) errors.push("backend/reasoning.json must name a model")
  const baseURL = env.REASONING_BASE_URL?.trim() ?? ""
  if (!/^https?:\/\/\S+$/.test(baseURL)) errors.push("REASONING_BASE_URL must be an http(s) URL")
  if (!env.REASONING_API_KEY) errors.push("REASONING_API_KEY is required")
  const level = effort(model, settings.effort)
  const spoken = speech(settings.speech)
  const database = connection(env.ASSISTANT_DATABASE_URL, errors)
  const poolText = env.NIA_DB_POOL_MAX?.trim() || "4"
  const poolMax = Number(poolText)
  if (!/^[0-9]+$/.test(poolText) || !Number.isSafeInteger(poolMax) || poolMax < 1)
    errors.push("NIA_DB_POOL_MAX must be a positive safe integer")
  if (errors.length || !database || privateFiles === undefined)
    throw new Error(`invalid configuration: ${errors.join("; ")}`)
  return {
    transport: transport as Config["transport"],
    production: env.APP_ENV === "production",
    principal,
    private: privateFiles,
    model: `${provider}/${model}`,
    baseURL,
    effort: level.effort,
    speech: spoken.speech,
    database: { ...database, max: poolMax },
    warnings: [...(level.warning ? [level.warning] : []), ...spoken.warnings],
  }
}

/** Accepts the same spellings as the Python API's settings; unset means false, anything else undefined. */
export function flag(value: string | undefined) {
  const text = value?.trim().toLowerCase() ?? ""
  if (["", "0", "false", "f", "no", "n", "off"].includes(text)) return false
  if (["1", "true", "t", "yes", "y", "on"].includes(text)) return true
  return undefined
}

// TLS follows the URL's sslmode, as in libpq. Without one, a local database is reached directly and
// a remote one over TLS. Modes that may fall back to plain text (allow, prefer) are refused.
function connection(value: string | undefined, errors: string[]) {
  let url: URL
  try {
    url = new URL(value?.trim() ?? "")
  } catch {
    errors.push("ASSISTANT_DATABASE_URL must be a postgresql:// URL")
    return
  }
  if (!["postgres:", "postgresql:"].includes(url.protocol)) {
    errors.push("ASSISTANT_DATABASE_URL must be a postgresql:// URL")
    return
  }
  const local = ResearchDatabase.local(url)
  const mode = url.searchParams.get("sslmode") ?? (local ? "disable" : "require")
  const ssl = sslModes.find((item) => item === mode)
  if (!ssl) {
    errors.push(`ASSISTANT_DATABASE_URL sslmode must be one of ${sslModes.join(", ")}`)
    return
  }
  if (ssl === "disable" && !local) {
    errors.push("ASSISTANT_DATABASE_URL must use sslmode=require (or verify-ca, verify-full) for a remote database")
    return
  }
  url.searchParams.delete("sslmode")
  return { url: url.toString(), ssl }
}

/** ASSISTANT_DATABASE_URL alone, for tools that need the database but not the rest of the configuration. */
export function database(value: string | undefined) {
  const errors: string[] = []
  const result = connection(value, errors)
  if (!result) throw new Error(`invalid configuration: ${errors.join("; ")}`)
  return result
}
