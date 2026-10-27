export * as HttpSession from "./http-session"

import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto"

export interface Account {
  readonly username: string
  readonly password: string
}

export interface Config {
  readonly production: boolean
  readonly accounts: readonly Account[]
  readonly origins: readonly string[]
  readonly secret: string
  readonly ttlSeconds: number
  readonly secure: boolean
}

export const cookieName = "nia_session"

// Mirrors the Python API's validate_production_settings: with APP_ENV=production nothing falls back
// to a development default, and an unsafe configuration stops the process before it takes traffic.
export function load(env: Readonly<Record<string, string | undefined>>): Config {
  const production = env.APP_ENV === "production"
  const accounts = [
    {
      username: env.STANDALONE_DEMO_USERNAME?.trim() || "researcher",
      password: env.STANDALONE_DEMO_PASSWORD ?? (production ? "" : "devpassword"),
    },
    { username: "user", password: env.PUBLIC_TEST_PASSWORD ?? (production ? "" : "userpassword") },
  ].filter((account) => account.password)
  const origins = [
    ...new Set([env.FRONTEND_ORIGIN ?? "", ...(env.CORS_ORIGINS ?? "").split(",")].map((item) => item.trim())),
  ].filter(Boolean)
  const ttlSeconds = Number(env.JWT_TTL_SECONDS || 43200)
  const config = {
    production,
    accounts,
    origins: origins.length ? origins : ["http://localhost:3000"],
    // Sessions are signed rather than held in memory, so every instance accepts them and a restart
    // keeps them. Without a configured secret, development sessions last as long as the process.
    secret: env.JWT_SECRET || (production ? "" : randomBytes(32).toString("hex")),
    ttlSeconds,
    secure: production || env.AUTH_COOKIE_SECURE === "true",
  }
  const errors = [
    ...(Number.isInteger(ttlSeconds) && ttlSeconds > 0 ? [] : ["JWT_TTL_SECONDS must be a positive integer"]),
    ...(!production
      ? []
      : [
          ...(config.secret.length < 32 ? ["JWT_SECRET must be a unique secret of at least 32 characters"] : []),
          ...(env.STANDALONE_DEMO_PASSWORD ? [] : ["STANDALONE_DEMO_PASSWORD is required"]),
          ...(origins.includes("*") ? ["CORS_ORIGINS must not contain '*' when credentials are enabled"] : []),
        ]),
  ]
  if (errors.length) throw new Error(`unsafe configuration: ${errors.join("; ")}`)
  return config
}

export function login(config: Config, username: string, password: string) {
  return config.accounts.find((account) => account.username === username && same(account.password, password))
    ?.username
}

export function issue(config: Config, username: string, now = Date.now()) {
  const payload = Buffer.from(
    JSON.stringify({ sub: username, exp: Math.floor(now / 1000) + config.ttlSeconds }),
  ).toString("base64url")
  return `${payload}.${sign(config.secret, payload)}`
}

/** The signed-in username, or undefined for a missing, forged, expired or retired session. */
export function verify(config: Config, token: string, now = Date.now()) {
  const [payload, signature, ...rest] = token.split(".")
  if (!payload || !signature || rest.length || !same(signature, sign(config.secret, payload))) return
  const claims = (() => {
    try {
      return JSON.parse(Buffer.from(payload, "base64url").toString()) as { sub?: unknown; exp?: unknown }
    } catch {
      return undefined
    }
  })()
  if (typeof claims?.sub !== "string" || typeof claims.exp !== "number" || claims.exp * 1000 <= now) return
  // An account removed from the configuration loses its sessions at once.
  return config.accounts.some((account) => account.username === claims.sub) ? claims.sub : undefined
}

export function cookie(config: Config, token: string) {
  return `${cookieName}=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${config.ttlSeconds}${config.secure ? "; Secure" : ""}`
}

export function cleared(config: Config) {
  return `${cookieName}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0${config.secure ? "; Secure" : ""}`
}

export function allowedOrigin(config: Config, origin: string) {
  if (config.origins.includes(origin)) return true
  // Any local dev server port is accepted, but only outside production.
  return !config.production && (origin.startsWith("http://localhost:") || origin.startsWith("http://127.0.0.1:"))
}

function sign(secret: string, payload: string) {
  return createHmac("sha256", secret).update(payload).digest("base64url")
}

// Compares digests so neither the length nor the first differing byte of a secret shows in timing.
function same(left: string, right: string) {
  const digest = (value: string) => createHash("sha256").update(value).digest()
  return timingSafeEqual(digest(left), digest(right))
}
