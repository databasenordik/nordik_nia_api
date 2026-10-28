export * as NiaConfig from "./config"

import { ResearchDatabase } from "./research/database"
// The model and its effort are part of the code, not the deployment: backend/reasoning.json is
// committed, and changing either is a commit and a release.
import file from "../reasoning.json"

/** The provider ID the configured REASONING_* endpoint is registered under. */
export const provider = "reasoning"

/** The contents of backend/reasoning.json. */
export const reasoning: { readonly model?: unknown; readonly effort?: unknown } = file

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
  readonly database: { readonly url: string; readonly ssl: ResearchDatabase.Options["ssl"] }
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

// The deployment's variables: NIA_TRANSPORT, APP_ENV, NIA_GRPC_PRINCIPAL_ID,
// NIA_GRPC_CAN_USE_PRIVATE_FILES, REASONING_BASE_URL, REASONING_API_KEY and ASSISTANT_DATABASE_URL,
// with the model and effort from backend/reasoning.json. Every problem is reported at once, before any
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
  const database = connection(env.ASSISTANT_DATABASE_URL, errors)
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
    database,
    warnings: level.warning ? [level.warning] : [],
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
