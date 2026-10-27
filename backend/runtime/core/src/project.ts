export * as Project from "./project.js"

import { Context, Duration, Effect, Layer, Schedule, Schema } from "effect"
import { and, asc, desc, eq, gte, isNotNull, isNull, lt, lte, ne, sql } from "drizzle-orm"
import path from "path"
import { AbsolutePath } from "./schema.js"
import { Bus } from "./bus.js"
import { Database } from "./database/database.js"
import { Worktree } from "@opencode/schema/worktree"
import { FSUtil } from "@opencode/util/fs-util"
import { makeGlobalNode } from "@opencode/util/effect/app-node"
import { Hash } from "@opencode/util/hash"
import { ProjectSchema } from "./project/schema.js"
import { ProjectTable, upsertProject } from "./project/sql.js"
import { WorktreeTable } from "./worktree/sql.js"

export const ID = ProjectSchema.ID
export type ID = ProjectSchema.ID

export const Vcs = ProjectSchema.Vcs
export type Vcs = ProjectSchema.Vcs

export const Current = ProjectSchema.Current
export type Current = ProjectSchema.Current

export const Info = ProjectSchema.Info
export interface Info extends Schema.Schema.Type<typeof Info> {}

export const UpdateInput = ProjectSchema.UpdateInput
export type UpdateInput = ProjectSchema.UpdateInput

export class NotFoundError extends Schema.TaggedError<NotFoundError>()("Project.NotFoundError", {
  projectID: ID,
}) {}

export interface Resolved {
  readonly previous?: ID
  readonly id: ID
  readonly directory: AbsolutePath
  // This checkout's main directory; the stored project canonical may be another clone.
  readonly canonical: AbsolutePath
  readonly vcs?: Vcs
}

// A directory is its own project. Git and Mercurial metadata are not project boundaries.
export const root = Effect.fn("Project.root")(function* (_fs: FSUtil.Interface, _input: AbsolutePath) {
  return undefined
})

const ACTIVATE_INTERVAL = 60_000
const SWEEP_INTERVAL = Duration.hours(1)

export interface ListInput {
  /** Includes archived Projects when true. */
  readonly archived?: boolean
}

export interface Interface {
  readonly list: (input?: ListInput) => Effect.Effect<ReadonlyArray<Info>>
  readonly update: (input: UpdateInput) => Effect.Effect<Info, NotFoundError>
  /** Records Project activity for recency ordering, at most once per minute per Project. */
  readonly activate: (projectID: ID) => Effect.Effect<void>
  /** Resolves and persists the owning Project. */
  readonly resolve: (input: AbsolutePath) => Effect.Effect<Resolved>
  /** Archives Projects whose known directories are all gone; resolving a Project unarchives it. */
  readonly sweep: () => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/Project") {}

function fromRow(row: typeof ProjectTable.$inferSelect): Info {
  const icon =
    row.icon_url || row.icon_url_override || row.icon_color
      ? {
          url: row.icon_url ?? undefined,
          override: row.icon_url_override ?? undefined,
          color: row.icon_color ?? undefined,
        }
      : undefined
  return {
    id: row.id,
    canonical: row.worktree,
    vcs: row.vcs ?? undefined,
    name: row.name ?? undefined,
    icon,
    commands: row.commands ?? undefined,
    time: {
      created: row.time_created,
      updated: row.time_updated,
      active: row.time_active,
      archived: row.time_archived ?? undefined,
    },
    sandboxes: row.sandboxes,
  }
}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const fs = yield* FSUtil.Service
    const bus = yield* Bus.Service
    const db = (yield* Database.Service).db

    const announcing = new Set<string>()
    const persist = Effect.fnUntraced(function* (project: Resolved) {
      const previous = yield* db
        .select({ canonical: ProjectTable.worktree })
        .from(ProjectTable)
        .where(eq(ProjectTable.id, project.id))
        .get()
        .pipe(Effect.orDie)
      yield* upsertProject(db, project).pipe(Effect.orDie)
      const unarchived = yield* db
        .update(ProjectTable)
        .set({ time_archived: null, time_updated: sql`${ProjectTable.time_updated}` })
        .where(and(eq(ProjectTable.id, project.id), isNotNull(ProjectTable.time_archived)))
        .returning()
        .get()
        .pipe(Effect.orDie)
      // Clones share a project ID; only replace a canonical directory that is gone.
      const replaced =
        previous &&
        previous.canonical !== project.canonical &&
        !(yield* fs.exists(previous.canonical).pipe(Effect.orElseSucceed(() => true)))
          ? yield* db
              .update(ProjectTable)
              .set({ worktree: project.canonical })
              .where(eq(ProjectTable.id, project.id))
              .returning()
              .get()
              .pipe(Effect.orDie)
          : undefined
      const changed = replaced ?? unarchived
      if (changed) yield* bus.publish(ProjectSchema.Event.Updated, fromRow(changed))
      if (!project.vcs) return project
      const directories: Array<{ projectID: ID; directory: AbsolutePath; strategy?: string }> = [
        { projectID: project.id, directory: project.canonical },
      ]
      if (project.directory !== project.canonical)
        directories.push({
          projectID: project.id,
          directory: project.directory,
          strategy: project.vcs.type === "git" ? "git" : undefined,
        })
      // A missing directory row means this directory's resolution is a new durable
      // fact. The row insert commits atomically with the event, so a crash between
      // checks retries on the next resolve instead of stranding the announcement.
      // The in-flight set keeps concurrent resolves from publishing the same fact
      // twice.
      for (const item of directories) {
        const key = item.projectID + "\u0000" + item.directory
        if (announcing.has(key)) continue
        announcing.add(key)
        yield* Effect.gen(function* () {
          const stored = yield* db
            .select({ directory: WorktreeTable.directory })
            .from(WorktreeTable)
            .where(and(eq(WorktreeTable.project_id, item.projectID), eq(WorktreeTable.directory, item.directory)))
            .get()
            .pipe(Effect.orDie)
          if (stored) return
          const directory = AbsolutePath.make(yield* fs.resolve(item.directory))
          const markerless = yield* db
            .select({ id: ProjectTable.id, directory: ProjectTable.worktree })
            .from(ProjectTable)
            .where(
              and(
                isNull(ProjectTable.vcs),
                gte(ProjectTable.worktree, directory),
                lte(ProjectTable.worktree, AbsolutePath.make(directory + "\uffff")),
              ),
            )
            .all()
            .pipe(Effect.orDie)
          const adopted = yield* Effect.filter(markerless, (candidate) =>
            Effect.gen(function* () {
              if (candidate.id === item.projectID) return false
              if (!FSUtil.contains(directory, candidate.directory)) return false
              const found = yield* fs
                .up({ targets: [".git", ".hg"], start: candidate.directory, stop: directory, mode: "first" })
                .pipe(Effect.orElseSucceed(() => []))
              if (!found[0]) return false
              return (yield* fs.resolve(path.dirname(found[0]))) === directory
            }),
          )
          yield* bus.publish(
            Worktree.Event.Resolved,
            {
              projectID: item.projectID,
              directory: item.directory,
              previous: project.previous ?? ID.global,
              ...(adopted.length ? { adopted: adopted.map((candidate) => candidate.id) } : {}),
            },
            {
              commit: () =>
                db
                  .insert(WorktreeTable)
                  .values({ project_id: item.projectID, directory: item.directory, strategy: item.strategy })
                  .onConflictDoNothing()
                  .run()
                  .pipe(Effect.orDie, Effect.asVoid),
            },
          )
        }).pipe(Effect.ensuring(Effect.sync(() => announcing.delete(key))))
      }
      return project
    })

    const list = Effect.fn("Project.list")(function* (input?: ListInput) {
      const rows = yield* db
        .select()
        .from(ProjectTable)
        .where(input?.archived ? undefined : isNull(ProjectTable.time_archived))
        .orderBy(desc(ProjectTable.time_active), asc(ProjectTable.id))
        .all()
        .pipe(Effect.orDie)
      return rows.map(fromRow)
    })

    const update = Effect.fn("Project.update")(function* (input: UpdateInput) {
      const row = yield* db
        .update(ProjectTable)
        .set({
          worktree: input.canonical,
          name: input.name === undefined ? undefined : input.name || null,
          icon_url_override: input.icon?.override === undefined ? undefined : input.icon.override || null,
          icon_color: input.icon?.color === undefined ? undefined : input.icon.color || null,
          commands:
            input.commands?.start === undefined
              ? undefined
              : input.commands.start
                ? { start: input.commands.start }
                : null,
          time_updated: Date.now(),
        })
        .where(eq(ProjectTable.id, input.projectID))
        .returning()
        .get()
        .pipe(Effect.orDie)
      if (!row) return yield* new NotFoundError({ projectID: input.projectID })
      const project = fromRow(row)
      yield* bus.publish(ProjectSchema.Event.Updated, project)
      return project
    })

    const activated = new Map<ID, number>()
    const activate = Effect.fn("Project.activate")(function* (projectID: ID) {
      const now = Date.now()
      if (now - (activated.get(projectID) ?? 0) < ACTIVATE_INTERVAL) return
      activated.set(projectID, now)
      const row = yield* db
        .update(ProjectTable)
        // Activity is not a metadata edit, so keep time_updated out of the implicit bump.
        .set({ time_active: now, time_updated: sql`${ProjectTable.time_updated}` })
        .where(eq(ProjectTable.id, projectID))
        .returning()
        .get()
        .pipe(Effect.orDie)
      if (row) yield* bus.publish(ProjectSchema.Event.Updated, fromRow(row))
    })

    const resolve = Effect.fn("Project.resolve")(function* (input: AbsolutePath) {
      const directory = AbsolutePath.make(yield* fs.resolve(input))
      return yield* persist({
        id: ID.make(Hash.fast(`directory:${directory}`)),
        directory,
        canonical: directory,
        vcs: undefined,
      })
    })

    const sweep = Effect.fn("Project.sweep")(function* () {
      const cutoff = Date.now() - Duration.toMillis(SWEEP_INTERVAL)
      // Projects touched since the cutoff are left alone. Checking again in the update means a
      // resolve that moves the canonical directory mid-sweep bumps time_updated and wins.
      const idle = and(
        isNull(ProjectTable.time_archived),
        lt(ProjectTable.time_updated, cutoff),
        lt(ProjectTable.time_active, cutoff),
      )
      const projects = yield* db
        .select({ id: ProjectTable.id, canonical: ProjectTable.worktree })
        .from(ProjectTable)
        .where(and(idle, ne(ProjectTable.id, ID.global)))
        .all()
        .pipe(Effect.orDie)
      const worktrees = yield* db.select().from(WorktreeTable).all().pipe(Effect.orDie)
      // Treat unreadable paths as present so transient filesystem errors never archive a Project.
      const exists = (directory: AbsolutePath) => fs.exists(directory).pipe(Effect.orElseSucceed(() => true))
      yield* Effect.forEach(
        projects,
        Effect.fnUntraced(function* (project) {
          if (yield* exists(project.canonical)) return
          const known = worktrees.filter(
            (item) => item.project_id === project.id && item.directory !== project.canonical,
          )
          const surviving = yield* Effect.filter(known, (item) => exists(item.directory))
          // Linked worktrees carry a strategy; only another main checkout can become the canonical directory.
          const main = surviving.find((item) => !item.strategy)
          if (!main && surviving.length) return
          const row = yield* db
            .update(ProjectTable)
            .set(
              main
                ? { worktree: main.directory }
                : { time_archived: Date.now(), time_updated: sql`${ProjectTable.time_updated}` },
            )
            .where(and(eq(ProjectTable.id, project.id), idle))
            .returning()
            .get()
            .pipe(Effect.orDie)
          if (row) yield* bus.publish(ProjectSchema.Event.Updated, fromRow(row))
        }),
        { concurrency: 8, discard: true },
      )
    })

    return Service.of({ list, update, activate, resolve, sweep })
  }),
)

export const node = makeGlobalNode({
  service: Service,
  layer: layer,
  deps: [Bus.node, Database.node, FSUtil.node],
})

export const sweepNode = makeGlobalNode({
  name: "project-archive-sweep",
  layer: Layer.effectDiscard(
    Effect.gen(function* () {
      const project = yield* Service
      yield* project
        .sweep()
        .pipe(
          Effect.ignoreCause({ log: "Error", message: "Project archive sweep failed" }),
          Effect.repeat(Schedule.spaced(SWEEP_INTERVAL)),
          Effect.forkScoped,
        )
    }),
  ),
  deps: [node],
})
