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
    database: { url: "postgresql://assistant_runtime:s3cret%402024@34.11.220.37:5432/postgres", ssl: "require" },
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
