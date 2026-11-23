import { GoogleAuth } from "google-auth-library"
import type { NiaConfig } from "./config"
import { voicePrompt } from "./voice-prompt"

export interface Auth {
  getProjectId(): Promise<string>
  getAccessToken(): Promise<string | null>
}

export function create(input: {
  settings: NiaConfig.Speech["tts"]["gemini"]
  apiKey?: () => string | undefined
  auth?: Auth
}) {
  let auth = input.auth
  return {
    request: async (text: string) => {
      const { model, voice, style, location } = input.settings
      const key = input.apiKey?.()
      let url: string
      let headers: Record<string, string>
      if (key) {
        url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`
        headers = { "x-goog-api-key": key }
      } else {
        auth ??= new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] })
        const project = await auth.getProjectId()
        const token = await auth.getAccessToken()
        if (!project || !token) throw new Error("Gemini credentials unavailable")
        const host = location === "global" ? "aiplatform.googleapis.com" : `${location}-aiplatform.googleapis.com`
        url = `https://${host}/v1beta1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}/publishers/google/models/${encodeURIComponent(model)}:generateContent`
        headers = { Authorization: `Bearer ${token}` }
      }
      return {
        url,
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ role: "user", parts: [{ text: voicePrompt(style, text) }] }],
          generationConfig: {
            responseModalities: ["AUDIO"],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: voice } } },
          },
        }),
      }
    },
    read: readAudio,
  }
}

async function readAudio(response: Response): Promise<{ data: Uint8Array<ArrayBuffer>; mime: string }> {
  const result = await response.json() as {
    candidates?: { content?: { parts?: { inlineData?: { data?: string; mimeType?: string } }[] } }[]
  }
  const audio = result.candidates?.flatMap((candidate) => candidate.content?.parts ?? [])
    .map((part) => part.inlineData).find((part) => part?.data)
  const encoded = audio?.data
  if (!encoded || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded) || encoded.length % 4 === 1)
    throw new Error("Gemini returned no valid audio")
  const pcm = Buffer.from(encoded, "base64")
  if (!pcm.length || pcm.length % 2 || !/^audio\/(?:L16|pcm)(?:;|$)/i.test(audio.mimeType ?? ""))
    throw new Error("Gemini returned no PCM audio")
  const sampleRate = Number(/\brate\s*=\s*(\d+)/i.exec(audio.mimeType ?? "")?.[1] ?? 24000)
  if (!Number.isInteger(sampleRate) || sampleRate < 8000 || sampleRate > 48000)
    throw new Error("Gemini returned an unsupported sample rate")
  const data = Buffer.alloc(44 + pcm.length)
  data.write("RIFF", 0)
  data.writeUInt32LE(36 + pcm.length, 4)
  data.write("WAVEfmt ", 8)
  data.writeUInt32LE(16, 16)
  data.writeUInt16LE(1, 20)
  data.writeUInt16LE(1, 22)
  data.writeUInt32LE(sampleRate, 24)
  data.writeUInt32LE(sampleRate * 2, 28)
  data.writeUInt16LE(2, 32)
  data.writeUInt16LE(16, 34)
  data.write("data", 36)
  data.writeUInt32LE(pcm.length, 40)
  pcm.copy(data, 44)
  return { data: new Uint8Array(data), mime: "audio/wav" }
}
