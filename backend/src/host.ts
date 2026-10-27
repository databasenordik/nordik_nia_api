export * as ResearchHost from "./host"

import { mkdir } from "node:fs/promises"
import path from "node:path"
import { Effect, Fiber, Layer, Schema, Stream } from "effect"
import { Research } from "@opencode/schema/research"
import { Session } from "@opencode/schema/session"
import { SessionEvent } from "@opencode/schema/session-event"
import { ResearchResponse } from "./research-response.js"
import { Agent } from "@opencode/schema/agent"
import { Model } from "@opencode/schema/model"
import { AbsolutePath } from "@opencode/schema/schema"
import { Config } from "@opencode/core/config"
import { ConfigPluginSource } from "@opencode/core/config/plugin/source"
import { InstructionDiscovery } from "@opencode/core/instruction-discovery"
import { ResearchDatabase } from "./research/database"
import { ResearchState } from "./research/state"
import { ResearchPlugin } from "./plugin/research"
import { Global } from "@opencode/util/global"
import { SessionRestart } from "@opencode/core/session/execution/restart"
import { Bus } from "@opencode/core/bus"
import { EmbeddedHost } from "@opencode/sdk/embedded"

export interface Options {
  readonly directory: string
  readonly database: ResearchDatabase.Options
  readonly model: string
  readonly providers?: Readonly<Record<string, Schema.Json>>
  /** Server-owned provider approval, including private evidence. Never a prompt/client field. */
  readonly allowedProviders: readonly string[]
}

export type Progress =
  | { readonly type: "working" | "tool" | "finished"; readonly tool?: string }
  /** Live reply text. `reset` means earlier streamed text belongs to a finished step the final reply drops. */
  | { readonly type: "text"; readonly delta: string; readonly reset: boolean }

export async function create(options: Options) {
  const directory = path.resolve(options.directory)
  await mkdir(directory, { recursive: true, mode: 0o700 })
  const model = Model.Ref.parse(options.model)
  if (!options.allowedProviders.includes(model.providerID))
    throw new Error("Selected provider is not approved for research")
  const connection = await ResearchDatabase.connect(
    Schema.decodeUnknownSync(ResearchDatabase.Options)(options.database),
  )
  const start = async () => {
    const configuration = JSON.stringify({ snapshots: false, providers: options.providers ?? {}, model: options.model })
    const host = await Effect.runPromise(
      EmbeddedHost.create(
        {
          recover: false,
          app: { name: "research", version: "1.0.0", channel: "research" },
          database: { path: path.join(directory, "sessions.sqlite") },
          config: { project: false, claude: false, directory, content: configuration },
          fs: { filewatcher: false, fff: false },
          models: { fetch: false },
        },
        {
          overrides: [
            Global.node.replace(
              Global.layerWith({
                home: directory,
                data: directory,
                state: path.join(directory, "state"),
                config: directory,
                cache: path.join(directory, "cache"),
                log: path.join(directory, "log"),
                tmp: path.join(directory, "tmp"),
                bin: path.join(directory, "bin"),
                repos: path.join(directory, "repos"),
              }),
            ),
            Config.node.replace(
              Config.configured({
                project: false,
                global: false,
                system: false,
                claude: false,
                content: configuration,
              }),
            ),
            InstructionDiscovery.node.replace(InstructionDiscovery.configured({ project: false, global: false })),
            ConfigPluginSource.node.replace(
              Layer.succeed(ConfigPluginSource.Service, {
                operations: () => Effect.succeed([]),
                changes: () => Stream.empty,
              }),
            ),
          ],
        },
      ),
    )
    const finish = () => host.close().finally(connection.close)
    const initialize = async () => {
      const state = await host.runtime.runPromise(ResearchState.make(connection))
      await host.runtime.runPromise(host.plugins.register(ResearchPlugin.make(state, options.allowedProviders)))
      host.runtime.runFork(
        Effect.gen(function* () {
          const restart = yield* SessionRestart.Service
          yield* restart.resumeSuspendedSessions
        }),
      )
      const ownedSession = async (sessionID: string) => {
        const session = await host.runtime.runPromise(host.sessions.get(Session.ID.make(sessionID)))
        if (
          !session.research ||
          session.research.database !== connection.identity ||
          session.research.principal !== connection.principal
        )
          throw new Error("Session has no binding for this database/principal; create a new research Session")
        return { ...session, research: session.research }
      }
      const requireSession = async (sessionID: string) => {
        const session = await ownedSession(sessionID)
        await connection.catalog(session.research.fileID)
        return session
      }
      const active = new Set<string>()
      return {
        catalog: async () => {
          const results = await Promise.allSettled(
            Schema.Literals([49, 91, 93, 94]).literals.map((id) => connection.catalog(id)),
          )
          const failed = results.find(
            (result) => result.status === "rejected" && !(result.reason instanceof ResearchDatabase.AccessDeniedError),
          )
          if (failed?.status === "rejected")
            throw new Error("Research catalog unavailable; verify the database connection and gateway")
          return results.flatMap((result) => (result.status === "fulfilled" ? [result.value] : []))
        },
        create: async (fileID: Research.FileID, id?: string) => {
          const selected = Schema.decodeUnknownSync(Research.FileID)(fileID)
          const dataset = await connection.catalog(selected)
          return host.runtime.runPromise(
            host.sessions.create({
              id: id ? Session.ID.make(id) : undefined,
              research: { database: connection.identity, fileID: selected, principal: connection.principal },
              location: { directory: AbsolutePath.make(directory) },
              agent: Agent.ID.make("research"),
              model,
              title: dataset.label,
            }),
          )
        },
        get: requireSession,
        list: async () =>
          (await host.runtime.runPromise(host.sessions.list({}))).data.filter(
            (session) =>
              session.research?.database === connection.identity && session.research.principal === connection.principal,
          ),
        query: async (sessionID: string, input: Research.Query) => {
          await requireSession(sessionID)
          return host.runtime.runPromise(
            state.run(sessionID, Schema.decodeUnknownSync(Research.Query, { onExcessProperty: "error" })(input)),
          )
        },
        source: async (sessionID: string, evidenceID: string) => {
          await requireSession(sessionID)
          const evidence = await host.runtime.runPromise(state.load(sessionID, evidenceID))
          const snapshot = await host.runtime.runPromise(state.snapshot(sessionID, []))
          if (evidence.fingerprint !== snapshot.fingerprint || evidence.version !== snapshot.dataset.version)
            throw new Error("Source evidence is stale; replay the query against the latest publication")
          return evidence
        },
        prompt: async (sessionID: string, text: string, progress?: (event: Progress) => void) => {
          await requireSession(sessionID)
          if (active.has(sessionID)) throw new Error("This Session is already answering a question")
          if (!text.trim() || text.length > 16000) throw new Error("Enter a question of 1–16000 characters")
          active.add(sessionID)
          const run = async () => {
            progress?.({ type: "working" })
            const admitted = await host.runtime.runPromise(
              host.sessions.prompt({ sessionID: Session.ID.make(sessionID), text, resume: false }),
            )
            const watch = host.runtime.runFork(
              host.sessions.log({ sessionID: Session.ID.make(sessionID), follow: true }).pipe(
                Stream.runForEach((event) =>
                  Effect.sync(() => {
                    if (
                      event.type === "session.tool.input.started" &&
                      [
                        "research_catalog",
                        "research_resolve",
                        "research_search",
                        "research_transfers",
                        "research_query",
                        "research_records",
                        "research_answer",
                      ].includes(event.data.name)
                    )
                      progress?.({ type: "tool", tool: event.data.name })
                  }),
                ),
              ),
            )
            // Text deltas are ephemeral, so they never reach the durable log above; take them from the live bus.
            let boundary = false
            const live = host.runtime.runFork(
              Effect.gen(function* () {
                const bus = yield* Bus.Service
                yield* bus.subscribe([SessionEvent.Text.Delta, SessionEvent.Step.Ended]).pipe(
                  Stream.filter((event) => event.data.sessionID === sessionID),
                  Stream.runForEach((event) =>
                    Effect.sync(() => {
                      if (event.type === "session.step.ended") {
                        // Mirrors ResearchResponse.text: only length-truncated steps continue into the reply.
                        boundary = boundary || event.data.finish !== "length"
                        return
                      }
                      progress?.({ type: "text", delta: event.data.delta, reset: boundary })
                      boundary = false
                    }),
                  ),
                )
              }),
            )
            await host.runtime
              .runPromise(host.sessions.resume(Session.ID.make(sessionID)))
              .finally(() => host.runtime.runPromise(Fiber.interruptAll([watch, live])))
            const answer = await host.runtime.runPromise(state.latest(sessionID))
            const messages = await host.runtime.runPromise(
              host.sessions.messages({
                sessionID: Session.ID.make(sessionID),
                type: "assistant",
                order: "asc",
                cursor: { id: admitted.id, direction: "next" },
              }),
            )
            const spoken = ResearchResponse.text(messages)
            const currentEvidence = state.takeEvidence(sessionID)
            const evidence = answer?.evidence.length ? answer.evidence : currentEvidence
            const snapshot = await host.runtime.runPromise(state.snapshot(sessionID, []))
            const fresh = [...currentEvidence, ...(answer?.evidence ?? [])].every(
              (item) => item.fingerprint === snapshot.fingerprint && item.version === snapshot.dataset.version,
            )
            progress?.({ type: "finished" })
            if (evidence.length && !fresh)
              throw new Error("The selected list changed before delivery; ask again for a fresh answer")
            if (spoken) {
              const session = await requireSession(sessionID)
              const dataset = await connection.catalog(session.research.fileID)
              return {
                sessionID,
                binding: session.research,
                version: dataset.version,
                at: new Date().toISOString(),
                text: spoken,
                evidence,
              }
            }
            if (answer) return answer
            throw new Error("No reply was produced")
          }
          return run().finally(() => {
            state.takeEvidence(sessionID)
            active.delete(sessionID)
          })
        },
        interrupt: async (sessionID: string) => {
          await ownedSession(sessionID)
          await host.runtime.runPromise(host.sessions.interrupt(Session.ID.make(sessionID)))
        },
        remove: async (sessionID: string) => {
          await ownedSession(sessionID)
          await host.runtime.runPromise(host.sessions.remove(Session.ID.make(sessionID)))
          await host.runtime.runPromise(state.clear(sessionID))
        },
        close: finish,
        [Symbol.asyncDispose]: finish,
      }
    }
    return initialize().catch(async (error: unknown) => {
      await host.close()
      throw error
    })
  }
  return start().catch(async (error: unknown) => {
    await connection.close()
    throw error
  })
}

export type Interface = Awaited<ReturnType<typeof create>>
