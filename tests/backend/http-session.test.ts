import { describe, expect, test } from "bun:test"
import { HttpSession } from "@nia/backend/http-session"

const production = {
  APP_ENV: "production",
  JWT_SECRET: "s".repeat(32),
  STANDALONE_DEMO_PASSWORD: "researcher-password",
  CORS_ORIGINS: "https://nia.example",
}

describe("configuration", () => {
  test("development keeps the local accounts and accepts any loopback dev server", () => {
    const config = HttpSession.load({})
    expect(HttpSession.login(config, "researcher", "devpassword")).toBe("researcher")
    expect(HttpSession.login(config, "user", "userpassword")).toBe("user")
    expect(HttpSession.login(config, "researcher", "wrong")).toBeUndefined()
    expect(HttpSession.allowedOrigin(config, "http://localhost:3001")).toBe(true)
    expect(config.secure).toBe(false)
  })

  test("production refuses to start without a signing secret or the primary password", () => {
    expect(() => HttpSession.load({ APP_ENV: "production" })).toThrow(/JWT_SECRET.*STANDALONE_DEMO_PASSWORD/)
    expect(() => HttpSession.load({ ...production, JWT_SECRET: "short" })).toThrow(/JWT_SECRET/)
    expect(() => HttpSession.load({ ...production, CORS_ORIGINS: "*" })).toThrow(/CORS_ORIGINS/)
  })

  test("production has no default passwords and no loopback origins", () => {
    const config = HttpSession.load(production)
    expect(config.accounts.map((account) => account.username)).toEqual(["researcher"])
    expect(HttpSession.login(config, "user", "userpassword")).toBeUndefined()
    expect(HttpSession.allowedOrigin(config, "http://localhost:3000")).toBe(false)
    expect(HttpSession.allowedOrigin(config, "https://nia.example")).toBe(true)
    expect(HttpSession.cookie(config, "token")).toContain("; Secure")
  })

  test("FRONTEND_ORIGIN and CORS_ORIGINS are both honoured", () => {
    const config = HttpSession.load({ FRONTEND_ORIGIN: "https://a.example", CORS_ORIGINS: "https://b.example, https://a.example" })
    expect(config.origins).toEqual(["https://a.example", "https://b.example"])
  })

  test("the primary username follows STANDALONE_DEMO_USERNAME", () => {
    const config = HttpSession.load({ ...production, STANDALONE_DEMO_USERNAME: "archivist" })
    expect(HttpSession.login(config, "archivist", "researcher-password")).toBe("archivist")
  })

  test("an invalid session lifetime is rejected", () => {
    expect(() => HttpSession.load({ JWT_TTL_SECONDS: "soon" })).toThrow(/JWT_TTL_SECONDS/)
  })
})

describe("sessions", () => {
  const config = HttpSession.load(production)
  const now = Date.UTC(2026, 9, 27)

  test("a signed session is accepted by any instance sharing the secret", () => {
    const token = HttpSession.issue(config, "researcher", now)
    expect(HttpSession.verify(HttpSession.load(production), token, now)).toBe("researcher")
  })

  test("forged, foreign, expired and retired sessions are refused", () => {
    const token = HttpSession.issue(config, "researcher", now)
    const [, signature] = token.split(".")
    const forged = `${Buffer.from(JSON.stringify({ sub: "user", exp: now / 1000 + 60 })).toString("base64url")}.${signature}`
    expect(HttpSession.verify(config, forged, now)).toBeUndefined()
    expect(HttpSession.verify(HttpSession.load({ ...production, JWT_SECRET: "t".repeat(32) }), token, now)).toBeUndefined()
    expect(HttpSession.verify(config, token, now + config.ttlSeconds * 1000)).toBeUndefined()
    const retired = HttpSession.load({ ...production, STANDALONE_DEMO_USERNAME: "archivist" })
    expect(HttpSession.verify(retired, HttpSession.issue(retired, "researcher", now), now)).toBeUndefined()
    expect(HttpSession.verify(config, "", now)).toBeUndefined()
    expect(HttpSession.verify(config, "not.a.token", now)).toBeUndefined()
  })

  test("the cookie lasts as long as the session and logout clears it", () => {
    expect(HttpSession.cookie(config, "token")).toContain(`Max-Age=${config.ttlSeconds}`)
    expect(HttpSession.cleared(config)).toContain("Max-Age=0")
  })
})
