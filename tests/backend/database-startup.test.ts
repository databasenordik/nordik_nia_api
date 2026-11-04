import { afterEach, beforeEach, expect, test } from "bun:test"
import { ResearchDatabase } from "@nia/backend/research/database"
import { postgres } from "../fixture/postgres"

let database: Awaited<ReturnType<typeof postgres>>
const privateValue = "private-database-value"

afterEach(() => database[Symbol.asyncDispose]())
beforeEach(async () => {
  database = await postgres("SELECT 1")
})

async function failure(url = database.url) {
  try {
    const connection = await ResearchDatabase.connect({
      url,
      ssl: "disable",
      principal: "nia-runtime",
      private: false,
    })
    await connection.close()
    throw new Error("The unqualified database unexpectedly passed startup")
  } catch (error) {
    expect(error).toBeInstanceOf(Error)
    const failure = error as Error
    expect(failure.message).toContain("Cannot qualify")
    expect(failure.cause).toBeUndefined()
    expect(failure.stack).not.toContain(privateValue)
    expect(failure.stack).not.toContain(database.url)
    expect(failure.stack).not.toContain(url)
    return failure.message
  }
}

test("a missing research schema identifies the gateway installation failure", async () => {
  expect(await failure()).toContain("at gateway identity: required database objects are missing")
})

test("a refused connection reports network guidance without including the connection URL", async () => {
  const listener = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } })
  const url = `postgres://user:${privateValue}@127.0.0.1:${listener.port}/${privateValue}`
  listener.stop(true)
  expect(await failure(url)).toContain("database connection refused; verify the database host, port and network access")
})

test("a missing identity function identifies the gateway installation failure", async () => {
  await database.sql`CREATE SCHEMA research_api`
  expect(await failure()).toContain("install the research gateway and its prerequisite migrations")
})

test.each([
  `'{"database":"${privateValue}","contract":1}'`,
  `'{"database":"${privateValue}","contract":2,"dates":"dates-2"}'`,
  `'{"database":"${privateValue}","contract":1,"dates":"dates-1"}'`,
  "NULL",
])("incompatible identity %s reports a fixed contract diagnosis", async (value) => {
  await database.sql.unsafe(`
    CREATE SCHEMA research_api;
    CREATE FUNCTION research_api.identity() RETURNS jsonb LANGUAGE sql AS $$ SELECT ${value}::jsonb $$;
  `)
  expect(await failure()).toContain("at gateway identity: incompatible gateway contract or date semantics")
})

test("a failure after a valid identity identifies the privilege self-test phase", async () => {
  await database.sql.unsafe(`
    CREATE SCHEMA research_api;
    CREATE FUNCTION research_api.identity() RETURNS jsonb LANGUAGE sql AS $$
      SELECT '{"database":"${privateValue}","contract":1,"dates":"dates-2"}'::jsonb
    $$;
    CREATE ROLE assistant_runtime;
    GRANT USAGE ON SCHEMA research_api TO assistant_runtime;
    SET ROLE assistant_runtime;
  `)
  expect(await failure()).toContain("at privilege self-test: required database objects are missing")
})

test.each([
  ["42501", "permission denied"],
  ["28P01", "authentication failed"],
  ["28000", "database login was rejected"],
  ["3D000", "the configured database does not exist"],
  ["53300", "database connection limit reached"],
] as const)("SQLSTATE %s reports fixed guidance without exposing server text", async (code, reason) => {
  await database.sql.unsafe(`
    CREATE SCHEMA research_api;
    CREATE FUNCTION research_api.identity() RETURNS jsonb LANGUAGE plpgsql AS $$
      BEGIN
        RAISE EXCEPTION '${privateValue}' USING ERRCODE='${code}',
          DETAIL='${privateValue}', HINT='${privateValue}';
      END
    $$;
  `)
  expect(await failure()).toContain(`at gateway identity: ${reason}`)
})

test("unclassified server exceptions keep messages, detail, hints and SQL private", async () => {
  await database.sql.unsafe(`
    CREATE SCHEMA research_api;
    CREATE FUNCTION research_api.identity() RETURNS jsonb LANGUAGE plpgsql AS $$
      BEGIN
        RAISE EXCEPTION '${privateValue}' USING DETAIL='${privateValue}', HINT='${privateValue}';
      END
    $$;
  `)
  expect(await failure()).toBe(
    "Cannot qualify the research database connection at gateway identity: verify database availability, role, TLS and gateway installation",
  )
})
