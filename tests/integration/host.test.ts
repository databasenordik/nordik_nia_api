import { describe, expect, test } from "bun:test"
import { Schema } from "effect"
import { Research } from "@opencode/schema/research"
import { ResearchHost } from "@nia/backend"
import { tmpdir } from "../fixture/tmpdir"
import { ensurePluginRuntime } from "@opencode/plugin/runtime"
import { ResearchDatabase } from "@nia/backend/research/database"

const profile = process.env.OPENCODE_RESEARCH_TEST_PROFILE
const Request = Schema.Struct({
  stream: Schema.Boolean,
  messages: Schema.Array(Schema.Struct({ role: Schema.String, content: Schema.NullOr(Schema.String) })),
  tools: Schema.Array(Schema.Struct({ function: Schema.Struct({ name: Schema.String }) })),
})

describe.skipIf(!profile)("standalone research host with real PostgreSQL and local provider", () => {
  test(
    "narrative search and default source hydration return the formerly omitted transfer evidence",
    async () => {
      await using directory = await tmpdir("opencode-research-narrative-")
      ensurePluginRuntime()
      const evidence: Research.Evidence[] = []
      let calls = 0
      const requests = [
        { name: "research_transfers", arguments: { scope: "outward", limit: 5 } },
        { name: "research_search", arguments: { terms: ["transferred from Shingwauk"], limit: 5 } },
        { name: "research_records", arguments: { sourceRows: ["132545", "134432"] } },
        {
          name: "research_query",
          arguments: {
            operation: "count",
            sourceRows: ["132545", "132570", "134465"],
            filter: { field: "admitted_date", operator: "HAS_MULTIPLE_EVENTS" },
          },
        },
      ]
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
          const body = Schema.decodeUnknownSync(Request)(await request.json())
          const latest = body.messages.filter((message) => message.role === "tool").at(-1)
          if (latest?.content) {
            const parsed = Schema.decodeUnknownOption(Schema.fromJsonString(Research.Evidence))(latest.content)
            if (parsed._tag === "Some") evidence.push(parsed.value)
            if (parsed._tag === "None") throw new Error(`Unexpected research tool response: ${latest.content}`)
          }
          const next = requests[calls++]
          const delta = next
            ? {
                tool_calls: [
                  {
                    index: 0,
                    id: `call_${calls}`,
                    type: "function",
                    function: { name: next.name, arguments: JSON.stringify(next.arguments) },
                  },
                ],
              }
            : {
                content:
                  "The source records describe transfers to Wawanosh and Mohawk. Multiple admissions are a separate fact.",
              }
          return new Response(
            `data: ${JSON.stringify({ id: `test_${calls}`, object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: null }] })}\n\ndata: ${JSON.stringify({ id: `test_${calls}`, object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: next ? "tool_calls" : "stop" }] })}\n\ndata: [DONE]\n\n`,
            { headers: { "Content-Type": "text/event-stream" } },
          )
        },
      })
      try {
        await using host = await ResearchHost.create({
          directory: directory.path,
          database: Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json()),
          model: "fixture/research-model",
          allowedProviders: ["fixture"],
          providers: {
            fixture: {
              package: "aisdk:@ai-sdk/openai-compatible",
              settings: { baseURL: `${server.url}v1`, apiKey: "fixture-only" },
              models: { "research-model": { capabilities: { tools: true }, limit: { context: 32000, output: 2048 } } },
            },
          },
        })
        const session = await host.create(49)
        const answer = await host.prompt(
          session.id,
          "Review transfers and multiple admission dates from the original source fields.",
        )
        expect(answer.text).toContain("Wawanosh")
        expect(calls).toBe(5)
        expect(evidence.map((item) => item.total)).toEqual([43, 10, 2, 2])
        expect(evidence[0].result).toMatchObject({ explicitTransferEntries: 18, explicitlyNamedShingwaukOrigins: 10 })
        expect(evidence[1].query.fields).toContain("additional_information")
        expect(evidence[2].result).toMatchObject({
          records: expect.arrayContaining([
            expect.objectContaining({
              sourceRowID: "132545",
              values: expect.objectContaining({
                student_name: "Sophy Aundag",
                additional_information: expect.stringContaining("Transferred from Shingwauk"),
              }),
            }),
            expect.objectContaining({
              sourceRowID: "134432",
              values: expect.objectContaining({
                student_name: "Earl Pinnance",
                additional_information: "Transferred to Mohawk Institute",
              }),
            }),
          ]),
        })
        expect(answer.evidence.every((item) => item.binding.fileID === 49)).toBe(true)
      } finally {
        server.stop(true)
      }
    },
    { timeout: 30000 },
  ) // Real PostgreSQL startup and four streamed production tool calls.

  test(
    "existing agent loop keeps research tools only and shows the model's closing reply",
    async () => {
      await using directory = await tmpdir("opencode-research-host-")
      ensurePluginRuntime()
      const tools: string[][] = []
      let calls = 0
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
          const body = Schema.decodeUnknownSync(Request)(await request.json())
          calls++
          tools.push(body.tools.map((tool) => tool.function.name))
          const results = body.messages
            .filter((message) => message.role === "tool")
            .flatMap((message) => {
              const parsed = Schema.decodeUnknownOption(Schema.fromJsonString(Schema.Json))(message.content)
              return parsed._tag === "Some" ? [parsed.value] : []
            })
          const accepted = results.some(
            (value) =>
              Schema.decodeUnknownOption(Schema.Struct({ accepted: Schema.Literal(true) }))(value)._tag === "Some",
          )
          const evidence = results
            .flatMap((value) => {
              const parsed = Schema.decodeUnknownOption(Research.Evidence)(value)
              return parsed._tag === "Some" ? [parsed.value] : []
            })
            .at(-1)
          const name = evidence ? "research_answer" : "research_query"
          const args = evidence ? { evidence: [evidence.id] } : { operation: "count" }
          const delta = accepted
            ? { content: "There are 84 records in this list." }
            : {
                tool_calls: [
                  {
                    index: 0,
                    id: `call_${calls}`,
                    type: "function",
                    function: { name, arguments: JSON.stringify(args) },
                  },
                ],
              }
          const chunk = (data: unknown) => `data: ${JSON.stringify(data)}\n\n`
          return new Response(
            chunk({
              id: `test_${calls}`,
              object: "chat.completion.chunk",
              choices: [{ index: 0, delta, finish_reason: null }],
            }) +
              chunk({
                id: `test_${calls}`,
                object: "chat.completion.chunk",
                choices: [{ index: 0, delta: {}, finish_reason: accepted ? "stop" : "tool_calls" }],
              }) +
              "data: [DONE]\n\n",
            { headers: { "content-type": "text/event-stream" } },
          )
        },
      })
      try {
        const database = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
        await using host = await ResearchHost.create({
          directory: directory.path,
          database,
          model: "fixture/research-model",
          allowedProviders: ["fixture"],
          providers: {
            fixture: {
              package: "aisdk:@ai-sdk/openai-compatible",
              settings: { baseURL: `${server.url}v1`, apiKey: "fixture-only" },
              models: { "research-model": { capabilities: { tools: true }, limit: { context: 32000, output: 2048 } } },
            },
          },
        })
        const session = await host.create(91)
        expect(session.research?.fileID).toBe(91)
        await expect(host.create(49, session.id)).rejects.toBeDefined()
        const result = await host.prompt(session.id, "How many records are in this list?")
        expect(result.text).toContain("84 records")
        expect(result.evidence.every((item) => item.binding.fileID === 91)).toBe(true)
        expect(calls).toBe(3)
        expect(
          tools.every((names) =>
            names.every((name) =>
              [
                "research_catalog",
                "research_resolve",
                "research_search",
                "research_transfers",
                "research_query",
                "research_records",
                "research_answer",
              ].includes(name),
            ),
          ),
        ).toBe(true)
        const evidence = await host.query(session.id, { operation: "count" })
        await expect(host.source("ses_wrong", evidence.id)).rejects.toBeDefined()
        await expect(
          host.query(
            session.id,
            Schema.decodeUnknownSync(Research.Query)({ operation: "list", sourceRows: ["foreign-row"] }),
          ),
        ).rejects.toBeDefined()
      } finally {
        server.stop(true)
      }
    },
    { timeout: 30000 },
  )
  test(
    "restart retains the list and cohort recipe; another principal cannot resume it",
    async () => {
      await using directory = await tmpdir("opencode-research-resume-")
      ensurePluginRuntime()
      const database = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
      const options = { directory: directory.path, database, model: "openai/gpt-4.1", allowedProviders: ["openai"] }
      const host = await ResearchHost.create(options)
      const session = await host.create(91)
      const previous = await host.query(session.id, { operation: "list", limit: 1 })
      await host.close()
      await using restarted = await ResearchHost.create(options)
      expect((await restarted.get(session.id)).research?.fileID).toBe(91)
      const refined = await restarted.query(session.id, { operation: "count", base: previous.id })
      expect(refined.total).toBe(84)
      expect((await restarted.source(session.id, previous.id)).id).toBe(previous.id)
      await using other = await ResearchHost.create({
        ...options,
        database: { ...database, principal: "another-principal" },
      })
      await expect(other.get(session.id)).rejects.toBeDefined()
      expect(await other.list()).toHaveLength(0)
      await expect(other.create(91, session.id)).rejects.toBeDefined()
      await expect(other.interrupt(session.id)).rejects.toBeDefined()
      await expect(other.remove(session.id)).rejects.toBeDefined()
      await restarted.remove(session.id)
      await expect(restarted.get(session.id)).rejects.toBeDefined()
      await expect(restarted.source(session.id, previous.id)).rejects.toBeDefined()
    },
    { timeout: 30000 },
  )
  test(
    "ordinary prose is shown without a research answer",
    async () => {
      await using directory = await tmpdir("opencode-research-repair-")
      ensurePluginRuntime()
      let calls = 0
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        fetch() {
          calls++
          return new Response(
            `data: ${JSON.stringify({ id: `chat_${calls}`, object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: "Hello! What would you like to explore?" }, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`,
            { headers: { "content-type": "text/event-stream" } },
          )
        },
      })
      try {
        const database = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
        await using host = await ResearchHost.create({
          directory: directory.path,
          database,
          model: "fixture/research-model",
          allowedProviders: ["fixture"],
          providers: {
            fixture: {
              package: "aisdk:@ai-sdk/openai-compatible",
              settings: { baseURL: `${server.url}v1`, apiKey: "fixture-only" },
              models: { "research-model": { capabilities: { tools: true }, limit: { context: 32000, output: 2048 } } },
            },
          },
        })
        const session = await host.create(91)
        const result = await host.prompt(session.id, "Hello")
        expect(result.text).toContain("Hello!")
        expect(result.evidence).toHaveLength(0)
        expect(calls).toBe(1)
      } finally {
        server.stop(true)
      }
    },
    { timeout: 30000 },
  )
  test(
    "date follow-ups retain natural conversation, recover actionable errors and cite unsaved query evidence",
    async () => {
      await using directory = await tmpdir("opencode-research-dates-")
      ensurePluginRuntime()
      const requests: (typeof Request.Type)[] = []
      const counts: number[] = []
      const server = Bun.serve({
        port: 0,
        hostname: "127.0.0.1",
        async fetch(request) {
          const body = Schema.decodeUnknownSync(Request)(await request.json())
          requests.push(body)
          const user = body.messages.findLastIndex((message) => message.role === "user")
          const current = body.messages.slice(user + 1)
          const evidence = current
            .filter((message) => message.role === "tool")
            .flatMap((message) => {
              const parsed = Schema.decodeUnknownOption(Schema.fromJsonString(Research.Evidence))(message.content)
              return parsed._tag === "Some" ? [parsed.value] : []
            })
            .at(-1)
          const count = body.messages[user].content?.includes("How many")
          const failed = !count && current.some((message) => message.role === "tool")
          const name = count ? "research_query" : "research_records"
          const args = count
            ? { operation: "count", filter: { field: "admitted_date_year", operator: "GREATER_THAN", value: 1960 } }
            : { sourceRows: ["133931"], fields: failed ? ["admitted_date"] : ["admitted_date_year", "admitted_date"] }
          const delta = evidence
            ? {
                content: count
                  ? `There are ${evidence.total} matching records. [${evidence.id}]`
                  : `The source says 1961-00-00: year 1961; month and day not recorded. He belongs in the after-1960 cohort. [${evidence.id}, source row 133931]`,
              }
            : {
                tool_calls: [
                  {
                    index: 0,
                    id: `date_${requests.length}`,
                    type: "function",
                    function: { name, arguments: JSON.stringify(args) },
                  },
                ],
              }
          return new Response(
            `data: ${JSON.stringify({ id: `date_${requests.length}`, object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason: evidence ? "stop" : "tool_calls" }] })}\n\ndata: [DONE]\n\n`,
            { headers: { "content-type": "text/event-stream" } },
          )
        },
      })
      try {
        const database = Schema.decodeUnknownSync(ResearchDatabase.Options)(await Bun.file(profile ?? "").json())
        await using host = await ResearchHost.create({
          directory: directory.path,
          database,
          model: "fixture/research-model",
          allowedProviders: ["fixture"],
          providers: {
            fixture: {
              package: "aisdk:@ai-sdk/openai-compatible",
              settings: { baseURL: `${server.url}v1`, apiKey: "fixture-only" },
              models: { "research-model": { capabilities: { tools: true }, limit: { context: 32000, output: 2048 } } },
            },
          },
        })
        const session = await host.create(49)
        const first = await host.prompt(session.id, "How many students were admitted after 1960?")
        counts.push(requests.length)
        expect(first.text).toContain("189")
        expect(first.evidence).toHaveLength(1)
        const followup = await host.prompt(session.id, "Why was Wallace McKay missing? Print his date exactly.")
        counts.push(requests.length - counts[0])
        expect(followup.text).toContain("1961-00-00")
        expect(followup.text).toContain("month and day not recorded")
        expect(followup.evidence).toHaveLength(1)
        expect(followup.evidence[0].result).toMatchObject({
          records: [
            {
              sourceRowID: "133931",
              values: { admitted_date: "1961-00-00" },
              dateInterpretations: { admitted_date: { kind: "year", year: 1961 } },
            },
          ],
        })
        expect((await host.source(session.id, followup.evidence[0].id)).id).toBe(followup.evidence[0].id)
        expect(counts).toEqual([2, 3])
        expect(
          requests.some((request) =>
            request.messages.some(
              (message) => message.role === "tool" && message.content?.includes("Fetch admitted_date instead"),
            ),
          ),
        ).toBe(true)
        expect(
          requests.some((request) =>
            request.messages.some(
              (message) => message.role === "assistant" && message.content?.includes("189 matching records"),
            ),
          ),
        ).toBe(true)
      } finally {
        server.stop(true)
      }
    },
    { timeout: 30000 },
  )
})
