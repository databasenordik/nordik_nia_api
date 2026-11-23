import { expect, test } from "bun:test"
import { SQL } from "bun"
import { NiaConfig } from "@nia/backend/config"

// The deployment's variables, with a stand-in password.
const deployed = {
  NIA_TRANSPORT: "grpc",
  APP_ENV: "production",
  NIA_GRPC_CAN_USE_PRIVATE_FILES: "true",
  REASONING_BASE_URL: "https://reasoning.example/v1",
  REASONING_API_KEY: "key",
  ASSISTANT_DATABASE_URL: "postgresql://assistant_runtime:s3cret%402024@34.11.220.37:5432/postgres?sslmode=require",
  NIA_GRPC_PRINCIPAL_ID: "nia-runtime",
}

test("the deployment's variables and the committed reasoning.json configure everything", () => {
  const config = NiaConfig.load(deployed, { model: "grok-4.6", effort: "low" })
  expect(config).toEqual({
    transport: "grpc",
    production: true,
    principal: "nia-runtime",
    private: true,
    model: "reasoning/grok-4.6",
    baseURL: "https://reasoning.example/v1",
    effort: "low",
    speech: NiaConfig.speech(undefined).speech,
    database: { url: "postgresql://assistant_runtime:s3cret%402024@34.11.220.37:5432/postgres", ssl: "require", max: 4 },
    warnings: [],
  })
  // The client decodes the password and encrypts the connection.
  const client = new SQL({ url: config.database.url, ssl: config.database.ssl }) as unknown as {
    options: { password: string; tls: unknown }
    close(): Promise<void>
  }
  expect(client.options.password).toBe("s3cret@2024")
  expect(Boolean(client.options.tls)).toBe(true)
  void client.close()
})

test("the committed reasoning.json names a model and an effort it accepts", () => {
  const config = NiaConfig.load(deployed)
  expect(typeof NiaConfig.reasoning.model).toBe("string")
  expect(config.model).toBe(`reasoning/${NiaConfig.reasoning.model}`)
  expect(config.warnings).toEqual([])
})

test("the deployment can bound each of its two database pools", () => {
  expect(NiaConfig.load({ ...deployed, NIA_DB_POOL_MAX: "1" }).database.max).toBe(1)
  expect(NiaConfig.load({ ...deployed, NIA_DB_POOL_MAX: " 2 " }).database.max).toBe(2)
  expect(NiaConfig.load({ ...deployed, NIA_DB_POOL_MAX: "" }).database.max).toBe(4)
})

test("invalid pool limits fail before startup without reflecting their values", () => {
  for (const value of ["0", "-1", "1.5", "Infinity", "1e2", "private-pool-value", "9007199254740992"]) {
    expect(() => NiaConfig.load({ ...deployed, NIA_DB_POOL_MAX: value })).toThrow(
      "NIA_DB_POOL_MAX must be a positive safe integer",
    )
  }
})

test("the model and effort are never read from the environment", () => {
  const config = NiaConfig.load(
    { ...deployed, REASONING_MODEL: "grok-4.5", REASONING_EFFORT: "high" },
    { model: "grok-4.6", effort: "low" },
  )
  expect([config.model, config.effort]).toEqual(["reasoning/grok-4.6", "low"])
  expect(() => NiaConfig.load(deployed, { effort: "low" })).toThrow(/reasoning\.json must name a model/)
})

test("an unset or unaccepted effort falls back to the model's lowest level, with a warning", () => {
  const load = (model: string, effort?: unknown) => {
    const config = NiaConfig.load(deployed, { model, effort })
    return [config.effort, config.warnings.length]
  }
  expect(load("grok-4.6", "medium")).toEqual(["medium", 0])
  expect(load("grok-4.6", "HIGH")).toEqual(["high", 0])
  expect(load("grok-4.6")).toEqual(["low", 0])
  expect(load("grok-4.6", "minimal")).toEqual(["low", 1])
  expect(load("grok-4.6", "maximum")).toEqual(["low", 1])
  expect(load("grok-4.5", "xhigh")).toEqual(["low", 1])
  expect(load("grok-4.3")).toEqual(["none", 0])
  // The dated grok-4.20 models take no effort parameter at all.
  expect(load("grok-4.20-0309-non-reasoning")).toEqual([undefined, 0])
  expect(load("grok-4.20-0309-reasoning", "low")).toEqual([undefined, 1])
  // A model the benchmark did not measure: any known level, else low.
  expect(load("another-model", "minimal")).toEqual(["minimal", 0])
  expect(load("another-model")).toEqual(["low", 0])
  expect(load("grok-4.6", 3)).toEqual(["low", 0])
  expect(NiaConfig.load(deployed, { model: "grok-4.6", effort: "maximum" }).warnings).toEqual([
    'reasoning.json effort "maximum" is not accepted by grok-4.6; using low',
  ])
})

test("a remote database is never reached without TLS", () => {
  for (const mode of ["disable", "allow", "prefer"]) {
    const url = deployed.ASSISTANT_DATABASE_URL.replace("sslmode=require", `sslmode=${mode}`)
    expect(() => NiaConfig.load({ ...deployed, ASSISTANT_DATABASE_URL: url })).toThrow(/sslmode/)
  }
  const unspecified = NiaConfig.load({ ...deployed, ASSISTANT_DATABASE_URL: deployed.ASSISTANT_DATABASE_URL.split("?")[0] })
  expect(unspecified.database.ssl).toBe("require")
  const verified = deployed.ASSISTANT_DATABASE_URL.replace("sslmode=require", "sslmode=verify-full")
  expect(NiaConfig.load({ ...deployed, ASSISTANT_DATABASE_URL: verified }).database.ssl).toBe("verify-full")
})

test("a local database may be reached directly", () => {
  const config = NiaConfig.load({ ...deployed, ASSISTANT_DATABASE_URL: "postgres://user:pass@127.0.0.1:5432/nia" })
  expect(config.database.ssl).toBe("disable")
})

test("every missing variable is reported at once, and no value is repeated", () => {
  const error = (() => {
    try {
      NiaConfig.load({ ASSISTANT_DATABASE_URL: "postgres://u:s3cret@db.example/x?sslmode=disable" })
    } catch (caught) {
      return String(caught)
    }
  })()
  for (const name of ["NIA_GRPC_PRINCIPAL_ID", "REASONING_BASE_URL", "REASONING_API_KEY", "ASSISTANT_DATABASE_URL"])
    expect(error).toContain(name)
  expect(error).not.toContain("s3cret")
})

test("flags and transport accept only known values", () => {
  expect(NiaConfig.load({ ...deployed, NIA_GRPC_CAN_USE_PRIVATE_FILES: "False" }).private).toBe(false)
  expect(NiaConfig.load({ ...deployed, NIA_GRPC_CAN_USE_PRIVATE_FILES: undefined }).private).toBe(false)
  expect(NiaConfig.load({ ...deployed, NIA_GRPC_CAN_USE_PRIVATE_FILES: "1" }).private).toBe(true)
  expect(() => NiaConfig.load({ ...deployed, NIA_GRPC_CAN_USE_PRIVATE_FILES: "maybe" })).toThrow(/PRIVATE_FILES/)
  expect(NiaConfig.load({ ...deployed, NIA_TRANSPORT: undefined }).transport).toBe("http")
  expect(() => NiaConfig.load({ ...deployed, NIA_TRANSPORT: "websocket" })).toThrow(/NIA_TRANSPORT/)
})

test("reasoning.json selects Gemini and retains the xAI speech settings", () => {
  const { speech, warnings } = NiaConfig.speech(NiaConfig.reasoning.speech)
  expect(warnings).toEqual([])
  expect(speech.tts.voice).toBe("sal")
  expect(speech.tts.provider).toBe("gemini")
  expect(speech.tts.gemini).toMatchObject({ model: "gemini-2.5-flash-tts", voice: "Algenib", style: "empathetic and more human like" })
  // Spelled out in the file, so each option can be found and changed there.
  expect(NiaConfig.reasoning.speech as unknown).toEqual(speech)
  expect(NiaConfig.load(deployed).speech).toEqual(speech)
})

test("the TTS provider can switch without changing either provider's settings", () => {
  const settings = NiaConfig.reasoning.speech as { tts: object }
  for (const provider of ["xai", "gemini"]) {
    const result = NiaConfig.speech({ ...settings, tts: { ...settings.tts, provider } })
    expect(result.warnings).toEqual([])
    expect(result.speech.tts.provider).toBe(provider)
    expect(result.speech.tts.gemini.voice).toBe("Algenib")
    expect(result.speech.tts.voice).toBe("sal")
  }
})

test("missing provider stays compatible with xAI, and malformed Gemini settings are reported", () => {
  expect(NiaConfig.speech({}).speech.tts.provider).toBe("xai")
  for (const gemini of [{ model: "../other" }, { voice: "" }, { style: "" }, { location: "evil.example/path" }]) {
    const result = NiaConfig.speech({ tts: { provider: "gemini", gemini } })
    expect(result.warnings).toHaveLength(1)
    expect(result.speech.tts.gemini).toEqual(NiaConfig.speech({}).speech.tts.gemini)
  }
})

test("a missing or partial speech block takes the defaults silently", () => {
  const defaults = NiaConfig.speech(undefined)
  expect(defaults.warnings).toEqual([])
  expect(NiaConfig.speech({})).toEqual(defaults)
  const partial = NiaConfig.speech({ tts: { speed: 0.9 } })
  expect(partial.warnings).toEqual([])
  expect(partial.speech.tts).toEqual({ ...defaults.speech.tts, speed: 0.9 })
})

test("speech words are case-insensitive and range edges are accepted", () => {
  const { speech, warnings } = NiaConfig.speech({
    tts: { voice: " LEO ", language: "PT-br", codec: "WAV", speed: 1.5 },
    stt: { language: "ES-mx", format: true, vad_threshold: 0 },
  })
  expect(warnings).toEqual([])
  expect(speech.tts).toMatchObject({ voice: "leo", language: "pt-BR", codec: "wav", speed: 1.5 })
  expect(speech.stt).toMatchObject({ language: "es-MX", format: true, vad_threshold: 0 })
})

test.each([
  ["tts", "voice", "algenib"],
  ["tts", "voice", 7],
  ["tts", "language", "xx"],
  ["tts", "speed", 0.69],
  ["tts", "speed", 1.51],
  ["tts", "speed", "1"],
  ["tts", "text_normalization", "true"],
  ["tts", "optimize_streaming_latency", 3],
  ["tts", "codec", "pcm"],
  ["tts", "sample_rate", 12345],
  ["tts", "bit_rate", 100],
  ["tts", "replace", []],
  ["tts", "replace", { "St. Mary": "saint mary" }],
  ["tts", "replace", { " ": "blank" }],
  ["tts", "replace", { Jane: 4 }],
  ["tts", "replace", { Jane: "a", jane: "b" }],
  ["tts", "replace", { Jane: "x".repeat(129) }],
  ["tts", "replace", { ["x".repeat(101)]: "x" }],
  ["tts", "replace", Object.fromEntries(Array.from({ length: 201 }, (_, index) => [`name ${index}`, "x"]))],
  ["stt", "model", "whisper"],
  ["stt", "language", "xx"],
  ["stt", "format", 1],
  ["stt", "filler_words", "no"],
  ["stt", "vad_threshold", 1.01],
  ["stt", "keyterms", "Jane"],
  ["stt", "keyterms", ["x".repeat(51)]],
  ["stt", "keyterms", Array(101).fill("term")],
] as const)("an invalid speech.%s.%s falls back with one warning (%p)", (group, option, value) => {
  const result = NiaConfig.speech({ [group]: { [option]: value } })
  expect(result.speech).toEqual(NiaConfig.speech(undefined).speech)
  expect(result.warnings).toHaveLength(1)
  expect(result.warnings[0]).toContain(`speech.${group}.${option}`)
})

test("formatting is off for a language the provider cannot format", () => {
  for (const language of ["auto", "hi"]) {
    const result = NiaConfig.speech({ stt: { language, format: true } })
    expect(result.speech.stt.format).toBe(false)
    expect(result.warnings).toHaveLength(1)
  }
})

test("misspelt options and malformed blocks are reported, and names in word lists are not", () => {
  expect(NiaConfig.speech({ tts: { voce: "ara" } }).warnings).toEqual([
    "reasoning.json speech.tts.voce is not an option; ignoring it",
  ])
  expect(NiaConfig.speech([]).warnings).toEqual(["reasoning.json speech must be an object; using its defaults"])
  expect(NiaConfig.speech({ stt: "fast" }).warnings).toHaveLength(1)
  const warning = NiaConfig.speech({ tts: { replace: { "Jane Example!": "jane" } } }).warnings.join()
  expect(warning).not.toContain("Jane")
})

test("the speech options are never read from the environment", () => {
  const config = NiaConfig.load({ ...deployed, TTS_VOICE: "ara", REASONING_VOICE: "ara" })
  expect(config.speech.tts.voice).toBe("sal")
})
