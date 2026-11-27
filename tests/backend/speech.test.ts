import { afterAll, afterEach, beforeEach, expect, spyOn, test } from "bun:test"
import { NiaConfig } from "../../backend/src/config"
import { NiaSpeech } from "../../backend/src/speech"

// A stand-in for the configured endpoint: it records each request and answers as `reply` says.
type Received = { path: string; method: string; headers: Headers; body: Buffer }
let received: Received[] = []
let reply: (request: Request) => Response | Promise<Response>
const provider = Bun.serve({
  port: 0,
  hostname: "127.0.0.1",
  async fetch(request) {
    const body = Buffer.from(await request.arrayBuffer())
    received.push({ path: new URL(request.url).pathname, method: request.method, headers: request.headers, body })
    return reply(request)
  },
})
afterAll(() => provider.stop(true))

const audio = () => new Response(new Uint8Array([0xff, 0xf3, 1, 2]), { headers: { "content-type": "audio/mpeg" } })
let logs: ReturnType<typeof spyOn<Console, "log">>
beforeEach(() => {
  received = []
  reply = audio
  logs = spyOn(console, "log").mockImplementation(() => {})
})
afterEach(() => {
  logs.mockRestore()
})

function speech(block: unknown = {}, options: Partial<Parameters<typeof NiaSpeech.create>[0]> = {}) {
  return NiaSpeech.create({
    baseURL: `http://127.0.0.1:${provider.port}/v1`,
    apiKey: () => "secret-key",
    settings: NiaConfig.speech(block).speech,
    ...options,
  })
}
const failure = async (work: Promise<unknown>) => {
  const error = await work.then(
    () => undefined,
    (caught: unknown) => caught,
  )
  expect(error).toBeInstanceOf(NiaSpeech.Failure)
  return error as NiaSpeech.Failure
}
const form = (request: Received) =>
  new Response(new Uint8Array(request.body), { headers: { "content-type": request.headers.get("content-type")! } }).formData()
const logged = () => logs.mock.calls.map((call) => String(call[0]))
const recording = new Uint8Array([1, 2, 3])

test("speech uses the base URL and key, with sal as the default voice", async () => {
  const result = await speech().speak("  Hello there.  ")
  expect([...result.data]).toEqual([0xff, 0xf3, 1, 2])
  expect(result.mime).toBe("audio/mpeg")
  expect(received).toHaveLength(1)
  expect([received[0]!.method, received[0]!.path]).toEqual(["POST", "/v1/tts"])
  expect(received[0]!.headers.get("authorization")).toBe("Bearer secret-key")
  expect(JSON.parse(received[0]!.body.toString())).toEqual({
    text: "Hello there.",
    voice_id: "sal",
    language: "en",
    speed: 1,
    text_normalization: false,
    optimize_streaming_latency: 0,
    output_format: { codec: "mp3", sample_rate: 24000, bit_rate: 64000 },
  })
})

test("a trailing slash on the base URL does not double up", async () => {
  await speech({}, { baseURL: `http://127.0.0.1:${provider.port}/v1/` }).speak("Hello")
  expect(received[0]!.path).toBe("/v1/tts")
})

test("the selected reader sends Ursa-Measured without adding delivery tags or prose instructions", async () => {
  await speech(NiaConfig.reasoning.speech).speak("Opening paragraph.\n\nDocumented details.\n\nClosing paragraph.")
  expect(JSON.parse(received[0]!.body.toString())).toEqual({
    text: "Opening paragraph.\n\nDocumented details.\n\nClosing paragraph.",
    voice_id: "ursa", language: "en", speed: 0.92, text_normalization: false,
    optimize_streaming_latency: 0,
    output_format: { codec: "mp3", sample_rate: 24000, bit_rate: 128000 },
  })
})

test("warm delivery preserves record qualifiers and prevents source speech tags from executing", async () => {
  await speech({ tts: { delivery: "warm" } }).speak("Jo[h]nson [sic], aged < 10, born 1903 | 1904.\n\nLiteral [cry] and <whisper> notation.")
  const sent = JSON.parse(received[0]!.body.toString()).text
  expect(sent).toBe("<soft>Johnson, sic, aged less than 10, born 1903 or 1904.</soft>\n\n[pause]\n\n<soft>Literal, cry, and, whisper, notation.</soft>")
  expect(sent).not.toContain("[cry]")
  expect(sent).not.toContain("<whisper>")
})

test("one paragraph is softened once and has no artificial pause", async () => {
  await speech({ tts: { delivery: "warm" } }).speak("The date is not recorded.")
  expect(JSON.parse(received[0]!.body.toString()).text).toBe("<soft>The date is not recorded.</soft>")
})

test("excessive paragraph markup is refused before reaching the provider", async () => {
  expect((await failure(speech({ tts: { delivery: "warm" } }).speak("a\n\n".repeat(5300)))).reason).toBe("invalid")
  expect(received).toHaveLength(0)
})

test("the key is read at each call", async () => {
  let key = "first"
  const voice = speech({}, { apiKey: () => key })
  await voice.speak("One")
  key = "second"
  await voice.speak("Two")
  expect(received.map((item) => item.headers.get("authorization"))).toEqual(["Bearer first", "Bearer second"])
})

test("every tts option reaches the provider", async () => {
  const tts = {
    voice: "ara",
    language: "fr",
    speed: 0.85,
    text_normalization: true,
    optimize_streaming_latency: 2,
    codec: "wav",
    sample_rate: 16000,
    bit_rate: 192000,
    replace: { Exampletown: "egg zam pull town" },
  }
  expect((await speech({ tts }).speak("Bonjour")).mime).toBe("audio/wav")
  expect(JSON.parse(received[0]!.body.toString())).toEqual({
    text: "Bonjour",
    voice_id: "ara",
    language: "fr",
    speed: 0.85,
    text_normalization: true,
    optimize_streaming_latency: 2,
    // bit_rate is for mp3 only.
    output_format: { codec: "wav", sample_rate: 16000 },
    replace: { Exampletown: "egg zam pull town" },
  })
})

test("every stt option reaches the provider, and the file is the last field", async () => {
  reply = () => Response.json({ text: "  How many students?  " })
  const stt = { language: "fr", format: true, filler_words: true, vad_threshold: 0.2, keyterms: ["Garden River", "Sampleton"] }
  expect(await speech({ stt }).transcribe(recording, "audio/webm;codecs=opus")).toBe("How many students?")
  expect(received[0]!.path).toBe("/v1/stt")
  expect(received[0]!.headers.get("authorization")).toBe("Bearer secret-key")
  const sent = [...(await form(received[0]!)).entries()]
  expect(sent.slice(0, -1) as [string, string][]).toEqual([
    ["model", "grok-voice-transcribe-2.0"],
    ["language", "fr"],
    ["format", "true"],
    ["filler_words", "true"],
    ["vad_threshold", "0.2"],
    ["keyterm", "Garden River"],
    ["keyterm", "Sampleton"],
  ])
  const [name, file] = sent.at(-1)! as unknown as [string, File]
  expect([name, file.name]).toEqual(["file", "audio.webm"])
  // Bun's form parser names types by extension, so the sent type is read from the bytes themselves.
  expect(received[0]!.body.toString("latin1")).toContain("Content-Type: audio/webm\r\n")
  expect([...new Uint8Array(await file.arrayBuffer())]).toEqual([1, 2, 3])
})

test("automatic language detection sends no language", async () => {
  reply = () => Response.json({ text: "hello" })
  await speech({ stt: { language: "auto" } }).transcribe(recording, "audio/ogg")
  expect((await form(received[0]!)).get("language")).toBeNull()
})

test("an unknown recording type is left for the provider to detect, without repeating it", async () => {
  reply = () => Response.json({ text: "hello" })
  await speech().transcribe(recording, "text/evil\r\nx-header: 1")
  const body = received[0]!.body.toString("latin1")
  expect(body).toContain('filename="audio"')
  expect(body).not.toContain("x-header")
})

test("a recording that is a view of a larger buffer sends only its own bytes", async () => {
  reply = () => Response.json({ text: "hello" })
  await speech().transcribe(Buffer.from([9, 9, 4, 5, 9]).subarray(2, 4), "audio/wav")
  const file = (await form(received[0]!)).get("file") as unknown as File
  expect([...new Uint8Array(await file.arrayBuffer())]).toEqual([4, 5])
})

test("text must be 1 to 16,000 characters, counted as characters", async () => {
  const voice = speech()
  expect((await failure(voice.speak(" \n "))).reason).toBe("invalid")
  expect((await failure(voice.speak("a".repeat(16_001)))).reason).toBe("invalid")
  expect(received).toHaveLength(0)
  await voice.speak("🙂".repeat(16_000))
  expect(received).toHaveLength(1)
})

test("a recording must be 1 byte to 20 MiB", async () => {
  const voice = speech()
  expect((await failure(voice.transcribe(new Uint8Array(0), "audio/webm"))).reason).toBe("invalid")
  expect((await failure(voice.transcribe(new Uint8Array(20 * 1024 * 1024 + 1), "audio/webm"))).reason).toBe("invalid")
  expect(received).toHaveLength(0)
  reply = () => Response.json({ text: "long" })
  expect(await voice.transcribe(new Uint8Array(20 * 1024 * 1024), "audio/webm")).toBe("long")
})

test("silence is refused rather than asked as an empty question", async () => {
  reply = () => Response.json({ text: "  " })
  const error = await failure(speech().transcribe(recording, "audio/webm"))
  expect([error.reason, error.message]).toEqual(["invalid", "no speech was detected"])
})

test.each([
  ["speak", 400, "unavailable"],
  ["speak", 401, "unavailable"],
  ["speak", 403, "unavailable"],
  ["speak", 404, "unavailable"],
  ["speak", 429, "rate_limited"],
  ["speak", 500, "unavailable"],
  ["speak", 503, "unavailable"],
  ["transcribe", 400, "invalid"],
  ["transcribe", 403, "unavailable"],
  ["transcribe", 413, "invalid"],
  ["transcribe", 415, "invalid"],
  ["transcribe", 429, "rate_limited"],
  ["transcribe", 503, "unavailable"],
] as const)("%s: provider status %i is %s, one attempt, without the provider's reply", async (operation, status, reason) => {
  reply = () => new Response('{"error":"private provider detail"}', { status })
  const voice = speech()
  const error = await failure(operation === "speak" ? voice.speak("Hello") : voice.transcribe(recording, "audio/webm"))
  expect(error.reason).toBe(reason)
  expect(error.message).not.toContain("private")
  expect(received).toHaveLength(1)
  expect(logged().join()).toContain(`"upstream_status":${status}`)
  expect(logged().join()).not.toContain("private")
})

test.each([
  ["a JSON error sent with 200", () => Response.json({ error: "oops" })],
  ["an HTML page sent with 200", () => new Response("<html>", { headers: { "content-type": "text/html" } })],
  ["an empty body", () => new Response(new Uint8Array(0), { headers: { "content-type": "audio/mpeg" } })],
] as const)("%s is not played as audio", async (_, response) => {
  reply = response
  expect((await failure(speech().speak("Hello"))).reason).toBe("unavailable")
})

test.each([
  ["not JSON", () => new Response("nope")],
  ["no text field", () => Response.json({ words: [] })],
] as const)("a transcript reply with %s is unavailable", async (_, response) => {
  reply = response
  expect((await failure(speech().transcribe(recording, "audio/webm"))).reason).toBe("unavailable")
})

test("a redirect is refused, so the key and text go nowhere else", async () => {
  reply = () => new Response(null, { status: 307, headers: { location: "/elsewhere" } })
  expect((await failure(speech().speak("Hello"))).reason).toBe("unavailable")
  expect(received.map((item) => item.path)).toEqual(["/v1/tts"])
})

test("an unreachable provider is unavailable", async () => {
  const fetch = async () => {
    throw new TypeError("connect failed to https://private-host")
  }
  const error = await failure(speech({}, { fetch }).speak("Hello"))
  expect([error.reason, error.message]).toEqual(["unavailable", "speech is temporarily unavailable"])
})

test("a provider that does not answer times out", async () => {
  reply = async () => {
    await Bun.sleep(500)
    return audio()
  }
  expect((await failure(speech({}, { timeoutMs: 50 }).speak("Hello"))).reason).toBe("unavailable")
})

test("a caller that gives up cancels the request and gets its own reason back", async () => {
  reply = async () => {
    await Bun.sleep(500)
    return audio()
  }
  const controller = new AbortController()
  const gone = new Error("client went away")
  const work = speech().speak("Hello", controller.signal)
  setTimeout(() => controller.abort(gone), 50)
  expect(await work.catch((error: unknown) => error)).toBe(gone)
  expect(logged().join()).toContain('"status":"cancelled"')
})

test("an already cancelled caller sends nothing", async () => {
  await speech()
    .speak("Hello", AbortSignal.abort(new Error("gone")))
    .catch(() => undefined)
  expect(received).toHaveLength(0)
})

test("the log has sizes and timing, never the text, transcript or key", async () => {
  await speech().speak("Private name Jane Example")
  reply = () => Response.json({ text: "Private transcript" })
  await speech().transcribe(recording, "audio/webm")
  const lines = logged().filter((line) => line.startsWith("SPEECH "))
  expect(lines).toHaveLength(2)
  expect(JSON.parse(lines[0]!.slice(7))).toMatchObject({ operation: "speak", status: "ok", upstream_status: 200, characters: 25 })
  expect(JSON.parse(lines[1]!.slice(7))).toMatchObject({ operation: "transcribe", status: "ok", bytes: 3 })
  for (const secret of ["Jane", "Private", "secret-key"]) expect(lines.join()).not.toContain(secret)
})

// Each of these was spoken wrongly by xAI when sent as written (tried live).
test.each([
  ["The surname is [illegible] on the card.", "The surname is, illegible, on the card."],
  ["The surname is (illegible) on the card.", "The surname is, illegible, on the card."],
  ["The surname is <illegible> on the card.", "The surname is, illegible, on the card."],
  ["The surname is {illegible} on the card.", "The surname is, illegible, on the card."],
  ["The surname is [[illegible]] on the card.", "The surname is, illegible, on the card."],
  ["The surname is Smith [or Smyth].", "The surname is Smith, or Smyth."],
  ["[Not recorded] appears here.", "Not recorded, appears here."],
  ["He left. (See notes) Then he returned.", "He left. See notes, Then he returned."],
  ["The word is (pause)", "The word is, pause"],
  ["One [a] [b] two", "One, a, b, two"],
  ["Line one\n[sic] line two", "Line one\nsic, line two"],
  ["Jo[h]nson and [J]ohn", "Johnson and John"],
  ["the student(s)", "the students"],
  ["Sampleton(Samplton)", "Sampleton, Samplton"],
  ["aged < 10", "aged less than 10"],
  ["aged <= 10", "aged at most 10"],
  ["aged > 12", "aged more than 12"],
  ["aged >=12", "aged at least 12"],
  ["born 1903 | 1904", "born 1903 or 1904"],
  ["Call the crisis line (1-866-925-4419) at any time.", "Call the crisis line, 1-866-925-4419, at any time."],
])("%p is spoken as %p", (written, spoken) => {
  expect(NiaSpeech.plainSpeech(written)).toBe(spoken)
})

test("ordinary text is sent unchanged", () => {
  for (const text of ["There are 1,073 records.", "Batchewana & Garden River", "From 1903-1907, aged 9.", "One\n\nTwo"])
    expect(NiaSpeech.plainSpeech(text)).toBe(text)
})

test("the provider receives the words, not the markup", async () => {
  await speech().speak("The surname is (illegible).")
  expect(JSON.parse(received[0]!.body.toString()).text).toBe("The surname is, illegible.")
})

test("text that is only markup is refused", async () => {
  expect((await failure(speech().speak("[] ()"))).reason).toBe("invalid")
  expect(received).toHaveLength(0)
})

test("long runs of spacing or markup do not block the server", () => {
  const started = performance.now()
  for (const unit of [" ", "\t", "[", " [", "< ", "a|"]) NiaSpeech.plainSpeech(`a${unit.repeat(100_000)}b`)
  expect(performance.now() - started).toBeLessThan(2000)
})

test("text over the limit is refused before any other work on it", async () => {
  const started = performance.now()
  expect((await failure(speech().speak(`a${" [".repeat(2_000_000)}`))).reason).toBe("invalid")
  expect(performance.now() - started).toBeLessThan(2000)
})
