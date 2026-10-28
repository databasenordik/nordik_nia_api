export * as Reference from "./reference.js"

import { makeLocationNode } from "@opencode/util/effect/app-node"
import { Context, Effect, Layer } from "effect"
import { Reference } from "@opencode/schema/reference"
import { Bus } from "./bus.js"
import { State } from "./state.js"

export const LocalSource = Reference.LocalSource
export type LocalSource = Reference.LocalSource

export const GitSource = Reference.GitSource
export type GitSource = Reference.GitSource

export const Source = Reference.Source
export type Source = Reference.Source

export { Event } from "@opencode/schema/reference"

export const Info = Reference.Info
export type Info = Reference.Info

type Data = {
  sources: Map<string, Source>
}

type Editor = {
  add(name: string, source: Source): void
  remove(name: string): void
  list(): readonly [string, Source][]
  get(name: string): Source | undefined
}

export interface Interface extends State.Transformable<Editor> {
  readonly list: () => Effect.Effect<Info[]>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/Reference") {}

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const bus = yield* Bus.Service
    const list = (): Info[] =>
      Array.from(state.get().sources).flatMap(([name, source]) => {
        const info = {
          name,
          source,
          ...(source.description === undefined ? {} : { description: source.description }),
          ...(source.hidden === undefined ? {} : { hidden: source.hidden }),
        }
        if (source.type === "local") return [Info.make({ ...info, path: source.path })]
        return []
      })
    const state = State.create<Data, Editor>({
      name: "reference",
      initial: () => ({ sources: new Map() }),
      editor: (editor) => ({
        add: (name, source) => editor.sources.set(name, source),
        remove: (name) => editor.sources.delete(name),
        list: () => Array.from(editor.sources),
        get: (name) => editor.sources.get(name),
      }),
      notify: () => bus.publish(Reference.Event.Updated, {}),
    })

    return Service.of({
      transform: state.transform,
      reload: state.reload,
      list: Effect.fn("Reference.list")(function* () {
        return list()
      }),
    })
  }),
)

export const node = makeLocationNode({
  service: Service,
  layer,
  deps: [Bus.node],
})
