import { afterAll, beforeAll, describe, expect, test } from "bun:test"
import { SQL } from "bun"
import { Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchDatabase } from "@nia/backend/research/database"
import { ResearchQuery } from "@nia/backend/research/query"

const profile = process.env.OPENCODE_RESEARCH_TEST_PROFILE
describe.skipIf(!profile)("PostgreSQL dump integration (isolated fixture)", () => {
  let connection: ResearchDatabase.Connection
  let options: ResearchDatabase.Options
  beforeAll(async () => {
    options = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
    connection = await ResearchDatabase.connect(options)
  })
  afterAll(async () => {
    await connection?.close()
  })

  test.each([
    [49, 5, 2814],
    [91, 5, 84],
    [93, 2, 29],
    [94, 2, 23],
  ] as const)("list %i publishes v%i with %i released records", async (fileID, version, count) => {
    const snapshot = await connection.snapshot({ database: connection.identity, fileID })
    expect(snapshot.dataset.version).toBe(version)
    expect(snapshot.records).toHaveLength(count)
    expect(snapshot.binding.fileID).toBe(fileID)
    expect(ResearchQuery.execute(snapshot, { operation: "count" }).total).toBe(count)
    expect(snapshot.dataset.fields.some((field) => field.name === "password")).toBe(false)
    expect(
      snapshot.records.every((record) =>
        Object.keys(record.values).every((key) => snapshot.dataset.fields.some((field) => field.name === key)),
      ),
    ).toBe(true)
  })

  test("private lists fail without matching active grants, regardless of the private flag", async () => {
    const denied = await ResearchDatabase.connect({ ...options, principal: "not-granted", private: true })
    await expect(denied.catalog(93)).rejects.toBeDefined()
    await expect(denied.catalog(94)).rejects.toBeDefined()
    expect((await denied.catalog(49)).version).toBe(5)
    await denied.close()
    const publicOnly = await ResearchDatabase.connect({ ...options, private: false })
    await expect(publicOnly.catalog(93)).rejects.toBeDefined()
    await publicOnly.close()
  })

  test("a conflicting database binding cannot be used", async () => {
    await expect(connection.snapshot({ database: "another-database", fileID: 49 })).rejects.toThrow(
      "different database",
    )
  })
  test("transfer search includes Additional information and hydrates the actual cited sources", async () => {
    const dataset = await connection.catalog(49)
    expect(
      dataset.fields.every((field) =>
        field.operators.every((operator) => Research.Operator.literals.some((item) => item === operator)),
      ),
    ).toBe(true)
    const query = ResearchQuery.search({ terms: ["transferred from Shingwauk"], limit: 25 }, dataset)
    const snapshot = await connection.snapshot(
      { database: connection.identity, fileID: 49 },
      undefined,
      ResearchQuery.fields(query, dataset),
    )
    const result = ResearchQuery.execute(snapshot, query)
    expect(result.total).toBe(10)
    expect(result.complete).toBe(true)
    for (const id of [
      "132545",
      "132570",
      "132849",
      "132852",
      "132936",
      "132977",
      "133521",
      "134058",
      "134464",
      "134465",
    ])
      expect(
        snapshot.records.find((row) => row.sourceRowID === id)?.values.additional_information?.toLowerCase(),
      ).toContain("transferred from shingwauk")
    const narrow = { operation: "count", filter: { field: "notes", operator: "CONTAINS", value: "transfer" } } as const
    expect(ResearchQuery.execute(snapshot, narrow).total).toBe(0)
    expect(ResearchQuery.execute(snapshot, narrow).result).toMatchObject({
      textSearch: { unsearchedNarrativeFields: expect.arrayContaining(["additional_information"]) },
    })
  })

  test("multiple-admission inspection operates on source expressions, not known-date counts", async () => {
    const dataset = await connection.catalog(49)
    expect(dataset.fields.find((field) => field.name === "admitted_date")?.operators).toContain("HAS_MULTIPLE_EVENTS")
    const query: Research.Query = {
      operation: "list",
      sourceRows: ["132545", "132570", "134465"],
      filter: { field: "admitted_date", operator: "HAS_MULTIPLE_EVENTS" },
      fields: ["student_name", "admitted_date", "additional_information"],
    }
    const snapshot = await connection.snapshot(
      { database: connection.identity, fileID: 49 },
      undefined,
      ResearchQuery.fields(query, dataset),
    )
    expect(ResearchQuery.execute(snapshot, query).total).toBe(2)
  })

  test("transfer review separates the independently reviewed 34 cases and four qualifications", async () => {
    const dataset = await connection.catalog(49)
    const query: Research.Query = { operation: "school_transfers", transferScope: "outward", limit: 100 }
    const snapshot = await connection.snapshot(
      { database: connection.identity, fileID: 49 },
      undefined,
      ResearchQuery.fields(query, dataset),
    )
    const result = ResearchQuery.execute(snapshot, query)
    expect(result.result).toMatchObject({
      explicitTransferEntries: 18,
      explicitlyNamedShingwaukOrigins: 10,
      groups: expect.arrayContaining([
        expect.objectContaining({
          category: "explicit_transfer_to_school",
          records: 18,
          sourceRows: expect.arrayContaining(["132737", "134077", "134230"]),
        }),
        expect.objectContaining({ category: "other_recorded_school_move", records: 10 }),
        expect.objectContaining({ category: "later_attendance_only", records: 5 }),
        expect.objectContaining({ category: "residential_placement_correspondence", records: 1 }),
        expect.objectContaining({ category: "planned_or_completion_uncertain", sourceRows: ["133909"] }),
        expect.objectContaining({ category: "transfer_date_uncertain", sourceRows: ["133847"] }),
        expect.objectContaining({
          category: "proposed_day_school_discharge",
          records: 3,
          casesAfterClearDuplicates: 2,
          sourceRows: ["135207", "135208"],
        }),
        expect.objectContaining({ category: "medical_move", records: 4 }),
      ]),
    })
    const all = ResearchQuery.execute(snapshot, { ...query, transferScope: "any" })
    expect(all.result).toMatchObject({
      groups: expect.arrayContaining([
        expect.objectContaining({ category: "earlier_school_transfer", records: 4, casesAfterClearDuplicates: 2 }),
        expect.objectContaining({
          category: "explicit_incoming_transfer",
          records: 154,
          casesAfterClearDuplicates: 152,
        }),
      ]),
    })
    const shape = Schema.Struct({
      cases: Schema.Array(
        Schema.Struct({
          sourceRowID: Schema.String,
          evidence: Schema.Array(Schema.Struct({ field: Schema.String, quote: Schema.String })),
        }),
      ),
    })
    for (const item of Schema.decodeUnknownSync(shape)(result.result).cases) {
      const row = snapshot.records.find((row) => row.sourceRowID === item.sourceRowID)
      for (const evidence of item.evidence) expect(row?.values[evidence.field]).toContain(evidence.quote)
    }
  })

  test.each([49, 91, 93, 94] as const)(
    "transfer review remains within list %i and approved projections",
    async (fileID) => {
      const dataset = await connection.catalog(fileID)
      const query: Research.Query = { operation: "school_transfers", transferScope: "any" }
      const fields = ResearchQuery.fields(query, dataset)
      expect(fields.every((name) => dataset.fields.some((field) => field.name === name && field.evidence))).toBe(true)
      const snapshot = await connection.snapshot({ database: connection.identity, fileID }, undefined, fields)
      if (!ResearchQuery.narratives(dataset).length) {
        expect(() => ResearchQuery.execute(snapshot, query)).toThrow("No permitted source narratives")
        return
      }
      const result = ResearchQuery.execute(snapshot, query)
      const shape = Schema.Struct({
        populationRecords: Schema.Number,
        cases: Schema.Array(Schema.Struct({ sourceRowID: Schema.String })),
      })
      const data = Schema.decodeUnknownSync(shape)(result.result)
      expect(data.populationRecords).toBe(snapshot.records.length)
      expect(
        data.cases.every((item) => snapshot.records.some((record) => record.sourceRowID === item.sourceRowID)),
      ).toBe(true)
    },
  )
  test("narrow projections preserve complete populations and revision identity", async () => {
    const binding = { database: connection.identity, fileID: 91 } as const
    const full = await connection.snapshot(binding)
    const narrow = await connection.snapshot(binding, undefined, ["death_date"])
    const count = await connection.snapshot(binding, undefined, [])
    expect(narrow.fingerprint).toBe(full.fingerprint)
    expect(count.fingerprint).toBe(full.fingerprint)
    expect(count.records).toHaveLength(84)
    expect(count.records.every((record) => Object.keys(record.values).length === 0)).toBe(true)
    expect(ResearchQuery.execute(narrow, { operation: "stats", field: "death_date", part: "year" })).toEqual(
      ResearchQuery.execute(full, { operation: "stats", field: "death_date", part: "year" }),
    )
    expect(ResearchQuery.fields({ operation: "completeness" }, full.dataset)).toHaveLength(
      full.dataset.fields.filter((field) => field.evidence).length,
    )
  })

  test("actual connection cannot write source data, self-grant or read base rows", async () => {
    const client = new SQL(options.url)
    await expect(client`SELECT * FROM public.file_data LIMIT 1`.execute()).rejects.toBeDefined()
    await expect(
      client`INSERT INTO assistant.access_grants(principal_id,file_id) VALUES ('attacker',94)`.execute(),
    ).rejects.toBeDefined()
    await expect(
      client`UPDATE assistant.dataset_policy SET enabled=true WHERE file_id=94`.execute(),
    ).rejects.toBeDefined()
    await expect(client`CREATE TABLE public.research_escape(id integer)`.execute()).rejects.toBeDefined()
    await expect(client`SELECT assistant.refresh_fill_rates()`.execute()).rejects.toBeDefined()
    await client.close()
  })
  test("published source examples retain unknown components, separate events, shorthand and ranges", async () => {
    const master = await connection.snapshot({ database: connection.identity, fileID: 49 }, undefined, [
      "birth_date",
      "birth_date_year",
      "birth_date_month",
      "admitted_date",
    ])
    expect(master.records.find((row) => row.sourceRowID === "134297")?.values).toMatchObject({
      birth_date: "0000-12-00",
      birth_date_year: null,
      birth_date_month: "12",
    })
    expect(
      ResearchQuery.execute(master, {
        operation: "count",
        sourceRows: ["132479"],
        unit: "events",
        eventField: "admitted_date",
      }).result,
    ).toMatchObject({ eventCount: 2, recordCount: 1 })
    const additional = await connection.snapshot({ database: connection.identity, fileID: 93 }, undefined, [
      "death_date",
    ])
    expect(additional.records.find((row) => row.sourceRowID === "135335")?.values.death_date).toBe("1909-09-04-06")
    expect(
      ResearchQuery.execute(additional, {
        operation: "count",
        sourceRows: ["135335"],
        filter: { field: "death_date", operator: "YEAR_EQUALS", value: 1909 },
      }).total,
    ).toBe(1)
    expect(
      ResearchQuery.execute(additional, {
        operation: "count",
        sourceRows: ["135351"],
        filter: { field: "death_date", operator: "YEAR_EQUALS", value: 1905 },
      }).coverage?.possible,
    ).toBe(1)
    const confirmed = await connection.snapshot({ database: connection.identity, fileID: 91 }, undefined, [
      "death_registration_date",
    ])
    expect(
      ResearchQuery.execute(confirmed, {
        operation: "count",
        sourceRows: ["135289"],
        unit: "events",
        eventField: "death_registration_date",
      }).result,
    ).toMatchObject({ eventCount: 2 })
  })

  test("unrestricted postgres credentials are rejected before research work", async () => {
    const unsafe = new URL(options.url)
    unsafe.username = "postgres"
    await expect(ResearchDatabase.connect({ ...options, url: unsafe.href })).rejects.toThrow("qualify")
  })

  describe.skipIf(process.env.OPENCODE_RESEARCH_TEST_MUTATIONS !== "true")(
    "explicitly enabled disposable-fixture mutations",
    () => {
      const admin = () => {
        const url = new URL(options.url)
        if (url.hostname !== "127.0.0.1" || url.username !== "research_fixture" || url.pathname !== "/postgres")
          throw new Error("Mutation tests require the isolated restore.sh fixture profile")
        url.username = "postgres"
        return new SQL(url.href)
      }

      test("an older gateway cannot silently return lossy date projections", async () => {
        const client = admin()
        const definition = Schema.decodeUnknownSync(Schema.Struct({ definition: Schema.String }))(
          (await client`SELECT pg_get_functiondef('research_api.identity()'::regprocedure) AS definition`)[0],
        ).definition
        try {
          await client`CREATE OR REPLACE FUNCTION research_api.identity() RETURNS jsonb LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,research_api,pg_temp AS $$ SELECT jsonb_build_object('database',id::text,'contract',contract) FROM research_api.installation $$`
          await expect(ResearchDatabase.connect(options)).rejects.toThrow("gateway installation")
        } finally {
          await client.unsafe(definition)
          await client.close()
        }
      })

      test("publication pointer wins over MAX(version); unrelated lists cannot change the selected evidence", async () => {
        const client = admin()
        const before = await connection.snapshot({ database: connection.identity, fileID: 49 })
        const previous = await client`SELECT min(version)::integer AS version FROM public.file_data WHERE file_id=49`
        const version = Schema.decodeUnknownSync(Schema.Struct({ version: Schema.Int }))(previous[0]).version
        try {
          await client`UPDATE public.file SET version=${version} WHERE id=49`
          const published = await connection.snapshot({ database: connection.identity, fileID: 49 })
          const oracle =
            await client`SELECT count(*)::integer AS count FROM public.file_data WHERE file_id=49 AND version=${version}`
          expect(published.dataset.version).toBe(version)
          expect(published.records.length).toBe(
            Schema.decodeUnknownSync(Schema.Struct({ count: Schema.Int }))(oracle[0]).count,
          )
          expect(published.fingerprint).not.toBe(before.fingerprint)
        } finally {
          await client`UPDATE public.file SET version=5 WHERE id=49`
        }
        const label = await client`SELECT user_facing_label FROM assistant.dataset_policy WHERE file_id=93`
        const original = Schema.decodeUnknownSync(Schema.Struct({ user_facing_label: Schema.String }))(
          label[0],
        ).user_facing_label
        try {
          await client`UPDATE assistant.dataset_policy SET user_facing_label='Unrelated fixture publication' WHERE file_id=93`
          expect((await connection.snapshot({ database: connection.identity, fileID: 49 })).fingerprint).toBe(
            before.fingerprint,
          )
        } finally {
          await client`UPDATE assistant.dataset_policy SET user_facing_label=${original} WHERE file_id=93`
          await client.close()
        }
      })

      test("expired grants, disabled lists and hidden fields take effect without restarting", async () => {
        const client = admin()
        try {
          await client`UPDATE assistant.access_grants SET expires_at=now()-interval '1 minute' WHERE principal_id=${options.principal} AND file_id=93`
          await expect(connection.catalog(93)).rejects.toBeInstanceOf(ResearchDatabase.AccessDeniedError)
        } finally {
          await client`UPDATE assistant.access_grants SET expires_at=NULL WHERE principal_id=${options.principal} AND file_id=93`
        }
        try {
          await client`UPDATE assistant.dataset_policy SET enabled=false WHERE file_id=94`
          await expect(connection.snapshot({ database: connection.identity, fileID: 94 })).rejects.toBeDefined()
        } finally {
          await client`UPDATE assistant.dataset_policy SET enabled=true WHERE file_id=94`
        }
        try {
          await client`UPDATE assistant.field_registry SET exposure_policy='internal' WHERE file_id=91 AND semantic_field='student_name'`
          const current = await connection.snapshot({ database: connection.identity, fileID: 91 })
          expect(current.dataset.fields.some((field) => field.name === "student_name")).toBe(false)
          expect(current.records.every((record) => !("student_name" in record.values))).toBe(true)
        } finally {
          await client`UPDATE assistant.field_registry SET exposure_policy='authorized' WHERE file_id=91 AND semantic_field='student_name'`
          await client.close()
        }
      })

      test("same-version source edits fail closed until normalization catches up", async () => {
        const client = admin()
        const row = Schema.decodeUnknownSync(Schema.Struct({ id: Schema.Int }))(
          (await client`SELECT min(id)::integer AS id FROM public.file_data WHERE file_id=91 AND version=5`)[0],
        )
        try {
          await client`UPDATE public.file_data_normalized SET source_updated_at=source_updated_at+interval '1 second' WHERE source_row_id=${row.id}`
          await expect(connection.snapshot({ database: connection.identity, fileID: 91 })).rejects.toThrow(
            "normalization",
          )
        } finally {
          await client`UPDATE public.file_data_normalized SET source_updated_at=source_updated_at-interval '1 second' WHERE source_row_id=${row.id}`
          await client.close()
        }
        expect((await connection.snapshot({ database: connection.identity, fileID: 91 })).records).toHaveLength(84)
      })
    },
  )
})
