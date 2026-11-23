import { expect, test } from "bun:test"
import { NiaConfig } from "../../backend/src/config"
import { NiaSpeech } from "../../backend/src/speech"
import { voicePrompt } from "../../backend/src/voice-prompt"

const pcm = Buffer.from([0, 0, 1, 0, 255, 127, 0, 128])
const audio = (data = pcm.toString("base64"), mimeType = "audio/L16;codec=pcm;rate=24000") =>
  Response.json({ candidates: [{ content: { parts: [{ inlineData: { data, mimeType } }] } }] })
const settings = () => NiaConfig.speech({ tts: { provider: "gemini" } }).speech

test("Gemini receives the old delivery prompt and unchanged answer, and returns a playable WAV", async () => {
  const requests: { url: string; init: RequestInit }[] = []
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => { throw new Error("xAI key must not be read for Gemini") },
    geminiApiKey: () => "gemini-test-key", settings: settings(),
    fetch: async (url, init) => { requests.push({ url, init }); return audio() },
  })
  const text = "There are 2,814 records. The surname is [illegible]."
  const result = await voice.speak(text)
  expect(requests).toHaveLength(1)
  expect(requests[0]!.url).toBe("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-tts:generateContent")
  const headers = new Headers(requests[0]!.init.headers)
  expect(headers.get("x-goog-api-key")).toBe("gemini-test-key")
  expect(headers.has("authorization")).toBe(false)
  expect(JSON.parse(String(requests[0]!.init.body))).toEqual({
    contents: [{ role: "user", parts: [{ text: voicePrompt("empathetic and more human like", text) }] }],
    generationConfig: { responseModalities: ["AUDIO"], speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: "Algenib" } } } },
  })
  const wav = Buffer.from(result.data)
  expect(result.mime).toBe("audio/wav")
  expect(wav.toString("ascii", 0, 4)).toBe("RIFF")
  expect(wav.toString("ascii", 8, 12)).toBe("WAVE")
  expect(wav.readUInt32LE(4)).toBe(wav.length - 8)
  expect(wav.readUInt32LE(24)).toBe(24000)
  expect(wav.readUInt16LE(22)).toBe(1)
  expect(wav.readUInt16LE(34)).toBe(16)
  expect(wav.subarray(44)).toEqual(pcm)
})

test("Vertex uses application default credentials and the detected project without an API key", async () => {
  let request: { url: string; init: RequestInit } | undefined
  const selected = settings()
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", settings: selected,
    geminiAuth: { getProjectId: async () => "fixture-project", getAccessToken: async () => "vertex-token" },
    fetch: async (url, init) => { request = { url, init }; return audio() },
  })
  await voice.speak("Read this.")
  expect(request!.url).toBe("https://aiplatform.googleapis.com/v1beta1/projects/fixture-project/locations/global/publishers/google/models/gemini-2.5-flash-tts:generateContent")
  expect(new Headers(request!.init.headers).get("authorization")).toBe("Bearer vertex-token")
  expect(new Headers(request!.init.headers).has("x-goog-api-key")).toBe(false)
})

test("a regional Vertex endpoint uses the selected region", async () => {
  const selected = NiaConfig.speech({ tts: { provider: "gemini", gemini: { location: "us-central1" } } }).speech
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", settings: selected,
    geminiAuth: { getProjectId: async () => "fixture-project", getAccessToken: async () => "token" },
    fetch: async (url) => { expect(new URL(url).hostname).toBe("us-central1-aiplatform.googleapis.com"); return audio() },
  })
  await voice.speak("Read this.")
})

test("transcription still uses xAI when Gemini is selected for spoken answers", async () => {
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", settings: settings(),
    fetch: async (url, init) => {
      expect(url).toBe("https://xai.invalid/v1/stt")
      expect(new Headers(init.headers).get("authorization")).toBe("Bearer xai-key")
      return Response.json({ text: "How many records?" })
    },
  })
  expect(await voice.transcribe(new Uint8Array([1, 2]), "audio/webm")).toBe("How many records?")
})

test.each([400, 403, 429, 503])("Gemini status %i returns a safe failure without fallback", async (status) => {
  let attempts = 0
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", geminiApiKey: () => "key", settings: settings(),
    fetch: async () => { attempts++; return Response.json({ error: "private detail" }, { status }) },
  })
  const error = await voice.speak("Read this.").catch((caught) => caught)
  expect(error).toBeInstanceOf(NiaSpeech.Failure)
  expect(error.reason).toBe(status === 429 ? "rate_limited" : "unavailable")
  expect(error.message).not.toContain("private")
  expect(attempts).toBe(1)
})

test.each([
  () => Response.json({ candidates: [] }),
  () => audio("not base64!"),
  () => audio(Buffer.from([1]).toString("base64")),
  () => audio(pcm.toString("base64"), "application/json"),
  () => audio(pcm.toString("base64"), "audio/L16;rate=1"),
])("malformed Gemini audio is unavailable", async (response) => {
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", geminiApiKey: () => "key", settings: settings(), fetch: async () => response(),
  })
  const error = await voice.speak("Read this.").catch((caught) => caught)
  expect(error).toBeInstanceOf(NiaSpeech.Failure)
  expect(error.reason).toBe("unavailable")
})

test("cancellation during credential lookup returns promptly without sending text", async () => {
  const controller = new AbortController()
  const gone = new Error("caller cancelled")
  const voice = NiaSpeech.create({
    baseURL: "https://xai.invalid/v1", apiKey: () => "xai-key", settings: settings(),
    geminiAuth: { getProjectId: () => new Promise(() => {}), getAccessToken: async () => "token" },
    fetch: async () => { throw new Error("must not send") },
  })
  const result = voice.speak("Read this.", controller.signal)
  setTimeout(() => controller.abort(gone), 10)
  expect(await result.catch((caught) => caught)).toBe(gone)
})
