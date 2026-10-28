import { afterAll, expect, test } from "bun:test"
import { ResearchDatabase } from "@nia/backend/research/database"
import { postgres } from "../fixture/postgres"

// The startup self-test, run as a non-superuser assistant_runtime against the objects it inspects.
const database = await postgres(`
  CREATE SCHEMA assistant;
  CREATE SCHEMA assistant_api;
  CREATE SCHEMA research_api;
  CREATE TABLE public.users (id integer);
  CREATE TABLE public.file (id integer);
  CREATE TABLE public.file_data (id integer);
  CREATE TABLE public.file_data_normalized (id integer);
  CREATE TABLE assistant.conversations (id integer);
  CREATE TABLE assistant_api.datasets (file_id integer, private boolean);
  INSERT INTO assistant_api.datasets VALUES (49, false), (91, false), (93, true);
  CREATE VIEW assistant_api.v_current_records WITH (security_barrier=true) AS SELECT 1 AS x;
  CREATE VIEW assistant_api.v_current_raw_records WITH (security_barrier=true) AS SELECT 1 AS x;
  CREATE FUNCTION assistant_api.potential_access_revision() RETURNS text LANGUAGE sql AS $$ SELECT 'r' $$;
  CREATE FUNCTION assistant_api.list_datasets(p_principal text, p_ids integer[], p_private boolean)
    RETURNS TABLE(file_id integer, private boolean) LANGUAGE sql SECURITY DEFINER
    AS $$ SELECT file_id, private FROM assistant_api.datasets WHERE file_id = ANY(p_ids) AND (p_private OR NOT private) $$;
  CREATE ROLE assistant_runtime NOLOGIN;
  GRANT USAGE ON SCHEMA assistant, assistant_api, research_api TO assistant_runtime;
  GRANT SELECT, INSERT, UPDATE, DELETE ON assistant.conversations TO assistant_runtime;
`)
afterAll(() => database[Symbol.asyncDispose]())

// Runs the self-test as assistant_runtime after `setup` (as the owner), then undoes `setup`.
async function failures(setup = "", undo = "") {
  const connection = await database.sql.reserve()
  try {
    if (setup) await connection.unsafe(setup)
    await connection`SET ROLE assistant_runtime`
    return await ResearchDatabase.privileges(connection as never, "nia-runtime")
  } finally {
    await connection`RESET ROLE`
    if (undo) await connection.unsafe(undo)
    connection.release()
  }
}

test("the API runtime role, with its conversation tables, passes", async () => {
  expect(await failures()).toEqual([])
})

test("reading public or raw tables, or writing outside the assistant schema, is refused", async () => {
  const beyond = ["the role has privileges beyond the assistant runtime role; use assistant_runtime"]
  expect(await failures("GRANT SELECT ON public.users TO assistant_runtime", "REVOKE SELECT ON public.users FROM assistant_runtime")).toEqual(beyond)
  expect(
    await failures("GRANT INSERT ON assistant_api.datasets TO assistant_runtime", "REVOKE INSERT ON assistant_api.datasets FROM assistant_runtime"),
  ).toEqual(beyond)
  expect(
    await failures("GRANT TRUNCATE ON assistant.conversations TO assistant_runtime", "REVOKE TRUNCATE ON assistant.conversations FROM assistant_runtime"),
  ).toEqual(beyond)
  expect(await failures("GRANT CREATE ON SCHEMA assistant TO assistant_runtime", "REVOKE CREATE ON SCHEMA assistant FROM assistant_runtime")).toEqual(beyond)
})

test("direct reads of the released-record views are refused, as in the Python self-test", async () => {
  expect(
    await failures(
      "GRANT SELECT ON assistant_api.v_current_records TO assistant_runtime",
      "REVOKE SELECT ON assistant_api.v_current_records FROM assistant_runtime",
    ),
  ).toEqual(["the role can SELECT assistant_api.v_current_records or v_current_raw_records directly"])
})

test("a missing Potential release gate stops the service", async () => {
  expect(
    await failures(
      "ALTER VIEW assistant_api.v_current_raw_records RESET (security_barrier)",
      "ALTER VIEW assistant_api.v_current_raw_records SET (security_barrier=true)",
    ),
  ).toEqual(["the Potential release gate is not installed; apply migrations through 073"])
})

test("private lists listed without a grant stop the service", async () => {
  expect(
    await failures(
      `CREATE OR REPLACE FUNCTION assistant_api.list_datasets(p_principal text, p_ids integer[], p_private boolean)
         RETURNS TABLE(file_id integer, private boolean) LANGUAGE sql SECURITY DEFINER
         AS $$ SELECT file_id, private FROM assistant_api.datasets WHERE file_id = ANY(p_ids) $$`,
      `CREATE OR REPLACE FUNCTION assistant_api.list_datasets(p_principal text, p_ids integer[], p_private boolean)
         RETURNS TABLE(file_id integer, private boolean) LANGUAGE sql SECURITY DEFINER
         AS $$ SELECT file_id, private FROM assistant_api.datasets WHERE file_id = ANY(p_ids) AND (p_private OR NOT private) $$`,
    ),
  ).toEqual(["private lists are listed without a private-files grant"])
})
