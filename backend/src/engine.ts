export * as NiaEngine from "./engine"

import { mkdtemp } from "node:fs/promises"
import os from "node:os"
import path from "node:path"
import { NiaConfig } from "./config"
import { ResearchHost } from "./host"

// The engine keeps its research Sessions in a private temporary directory, so they last as long as
// the process. Conversation history that must outlive it is kept by the gRPC transport in Postgres.
export async function start(config: NiaConfig.Config) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "nia-"))
  const model = config.model.slice(config.model.indexOf("/") + 1)
  const host = await ResearchHost.create({
    directory: path.join(directory, "state"),
    database: { ...config.database, principal: config.principal, private: config.private },
    model: config.model,
    allowedProviders: [NiaConfig.provider],
    providers: {
      [NiaConfig.provider]: {
        package: "aisdk:@ai-sdk/openai-compatible",
        // Resolved from the environment when the provider is called, so the key is never written down.
        settings: { baseURL: config.baseURL, apiKey: "{env:REASONING_API_KEY}" },
        models: {
          [model]: {
            capabilities: { tools: true },
            ...(config.effort ? { settings: { reasoningEffort: config.effort } } : {}),
            limit: { context: 32000, output: 4096 },
          },
        },
      },
    },
  })
  return { host, directory }
}

export type Engine = Awaited<ReturnType<typeof start>>
