import { afterAll, beforeAll, describe, expect, test } from "bun:test"
import { SQL } from "bun"
import { Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchDatabase } from "@nia/backend/research/database"
import { ResearchQuery } from "@nia/backend/research/query"

const profile = process.env.OPENCODE_RESEARCH_TEST_PROFILE
// The lists the restore.sh fixture holds; the oracle's own SQL below reads exactly these.
const fixtureLists = [49, 91, 93, 94] as const
// Admin reads are restricted to the explicitly enabled disposable-fixture lane.
describe.skipIf(!profile || process.env.OPENCODE_RESEARCH_TEST_MUTATIONS !== "true")(
  "independent SQL differential oracle",
  () => {
    let client: SQL
    let connection: ResearchDatabase.Connection
    const snapshots = new Map<Research.FileID, Research.Snapshot>()
    const names = new Map<Research.FileID, readonly string[]>()
    const Population = Schema.Struct({ file_id: Research.FileID, name: Schema.String })
    const Count = Schema.Struct({ count: Schema.Int })
    beforeAll(async () => {
      const options = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
      const url = new URL(options.url)
      if (url.hostname !== "127.0.0.1" || url.username !== "research_fixture" || url.pathname !== "/postgres")
        throw new Error("Oracle requires the isolated restore.sh fixture")
      url.username = "postgres"
      client = new SQL(url.href)
      connection = await ResearchDatabase.connect(options)
      // Independent base-table population and projection: no gateway functions or views.
      const rows = Schema.decodeUnknownSync(Schema.Array(Population))(
        await client`
      SELECT n.file_id, COALESCE(NULLIF(btrim(n.row_data_normalized #>> '{canonical,display_name}'),''),NULLIF(btrim(n.canonical_name),'')) AS name
      FROM public.file_data_normalized n JOIN public.file f ON f.id=n.file_id AND f.version=n.version
      JOIN public.file_data d ON d.id=n.source_row_id AND d.file_id=f.id AND d.version=f.version
      WHERE n.file_id IN (49,91,93,94) AND NOT COALESCE(f.is_delete,false) AND n.status='ready'
        AND (f.id<>94 OR lower(btrim(d.row_data->>'ACCESS')) IN ('true','yes','1'))
      ORDER BY n.source_row_id`,
      )
      for (const fileID of fixtureLists) {
        names.set(
          fileID,
          rows.filter((row) => row.file_id === fileID).map((row) => row.name),
        )
        snapshots.set(
          fileID,
          await connection.snapshot({ database: connection.identity, fileID }, undefined, ["student_name"]),
        )
      }
    })
    afterAll(async () => {
      await client?.close()
      await connection?.close()
    })
    const cases = fixtureLists.flatMap((fileID) =>
      Array.from({ length: 50 }, (_, index) => [fileID, index] as const),
    )
    test.each(cases)("list %i differential case %i", async (fileID, index) => {
      const population = names.get(fileID) ?? []
      const original = population[index % population.length]
      const value =
        index % 4 === 0 ? `${original} impossible-fixture-suffix` : index % 4 === 1 ? original.slice(0, 3) : original
      const operator = index % 4 === 1 ? "CONTAINS" : index % 4 === 2 ? "NOT_EQUALS" : "EQUALS"
      const rows = await client`
      WITH population AS (
        SELECT COALESCE(NULLIF(btrim(n.row_data_normalized #>> '{canonical,display_name}'),''),NULLIF(btrim(n.canonical_name),'')) AS name
        FROM public.file_data_normalized n JOIN public.file f ON f.id=n.file_id AND f.version=n.version
        JOIN public.file_data d ON d.id=n.source_row_id AND d.file_id=f.id AND d.version=f.version
        WHERE n.file_id=${fileID} AND NOT COALESCE(f.is_delete,false) AND n.status='ready'
          AND (${fileID}<>94 OR lower(btrim(d.row_data->>'ACCESS')) IN ('true','yes','1'))
      ) SELECT count(*)::integer AS count FROM population WHERE name IS NOT NULL AND CASE ${operator}
        WHEN 'CONTAINS' THEN strpos(lower(name),lower(${value}))>0
        WHEN 'NOT_EQUALS' THEN lower(name)<>lower(${value})
        ELSE lower(name)=lower(${value}) END`
      const snapshot = snapshots.get(fileID)
      if (!snapshot) throw new Error("Missing oracle snapshot")
      if (!snapshot.dataset.fields.find((field) => field.name === "student_name")?.operators.includes(operator)) {
        expect(() =>
          ResearchQuery.execute(snapshot, { operation: "count", filter: { field: "student_name", operator, value } }),
        ).toThrow("Unsupported operator")
        return
      }
      const result = ResearchQuery.execute(snapshot, {
        operation: index % 2 ? "list" : "count",
        fields: ["student_name"],
        filter: { field: "student_name", operator, value },
        limit: 1,
      })
      expect(result.total).toBe(Schema.decodeUnknownSync(Count)(rows[0]).count)
      expect(snapshot.records).toHaveLength(population.length)
    })
    test("all four lists preserve every registered original date cell", async () => {
      const cells = Schema.decodeUnknownSync(
        Schema.Array(
          Schema.Struct({
            file_id: Research.FileID,
            id: Schema.String,
            field: Schema.String,
            value: Schema.NullOr(Schema.String),
          }),
        ),
      )(
        await client`
        SELECT d.file_id,d.id::text AS id,f.semantic_field AS field,raw.value
        FROM public.file_data d JOIN public.file p ON p.id=d.file_id AND p.version=d.version
        JOIN assistant.field_registry f ON f.file_id=d.file_id AND f.exposure_policy='authorized' AND f.semantic_type='date'
        CROSS JOIN LATERAL (SELECT cell.value FROM unnest(f.raw_json_keys) WITH ORDINALITY k(key,ordinal)
          JOIN LATERAL jsonb_each_text(d.row_data) cell ON btrim(regexp_replace(cell.key,'[[:space:]]+',' ','g'))=btrim(regexp_replace(k.key,'[[:space:]]+',' ','g'))
          ORDER BY k.ordinal,(cell.key=k.key) DESC LIMIT 1) raw
        WHERE d.file_id IN(49,91,93,94) AND NOT COALESCE(p.is_delete,false)
          AND (p.id<>94 OR lower(btrim(d.row_data->>'ACCESS')) IN('true','yes','1'))`,
      )
      for (const fileID of fixtureLists) {
        const source = await connection.snapshot({ database: connection.identity, fileID }, undefined, [
          ...new Set(cells.filter((cell) => cell.file_id === fileID).map((cell) => cell.field)),
        ])
        const rows = new Map(source.records.map((row) => [row.sourceRowID, row]))
        cells
          .filter((cell) => cell.file_id === fileID)
          .forEach((cell) => expect(rows.get(cell.id)?.values[cell.field]).toBe(cell.value))
      }
      expect(cells).toHaveLength(9020)
    })
    test("admissions after 1960 agree with source-table recorded years, including zero components and 2019", async () => {
      const expected = await client`
        WITH source AS (SELECT d.row_data->>'Admitted' AS value FROM public.file_data d
          JOIN public.file f ON f.id=d.file_id AND f.version=d.version WHERE f.id=49 AND NOT COALESCE(f.is_delete,false)),
        years AS (SELECT CASE WHEN value ~ '^\\d{4}-\\d{2}-\\d{2}$' THEN left(value,4)::integer
          WHEN value ~ '^\\d{2}\\.\\d{2}\\.\\d{4}$' THEN right(value,4)::integer END AS year FROM source)
        SELECT count(*)::integer AS count FROM years WHERE year>1960`
      const source = await connection.snapshot({ database: connection.identity, fileID: 49 }, undefined, [
        "admitted_date_year",
        "admitted_date",
      ])
      const result = ResearchQuery.execute(source, {
        operation: "count",
        filter: { field: "admitted_date_year", operator: "GREATER_THAN", value: 1960 },
      })
      expect(result.total).toBe(Schema.decodeUnknownSync(Count)(expected[0]).count)
      expect(result.total).toBe(189)
      expect(source.records.find((row) => row.sourceRowID === "133931")?.values).toMatchObject({
        admitted_date: "1961-00-00",
        admitted_date_year: "1961",
      })
      expect(
        ResearchQuery.execute(source, {
          operation: "count",
          filter: { field: "admitted_date_year", operator: "EQUALS", value: 2019 },
        }).total,
      ).toBe(1)
    })
  },
)
