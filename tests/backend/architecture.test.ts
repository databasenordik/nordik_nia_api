import { expect, test } from "bun:test"
import path from "node:path"
import { Schema } from "effect"

const root = path.resolve(import.meta.dirname, "../..")
const Manifest = Schema.Struct({
  name: Schema.String,
  dependencies: Schema.optional(Schema.Record(Schema.String, Schema.String)),
  devDependencies: Schema.optional(Schema.Record(Schema.String, Schema.String)),
})
const files = [
  ...Array.from(new Bun.Glob("backend/runtime/*/package.json").scanSync({ cwd: root })),
  "backend/package.json",
  "tests/package.json",
]
const manifests = await Promise.all(
  files.map(async (file) => Schema.decodeUnknownSync(Manifest)(await Bun.file(path.join(root, file)).json())),
)

test("application workspaces have no circular dependencies", () => {
  const graph = new Map(
    manifests.map((manifest) => [
      manifest.name,
      Object.keys({ ...manifest.dependencies, ...manifest.devDependencies }),
    ]),
  )
  const visited = new Set<string>()
  const visit = (name: string, trail: readonly string[]): void => {
    expect(trail, `Circular dependency through ${name}`).not.toContain(name)
    if (visited.has(name)) return
    graph.get(name)?.forEach((dependency) => visit(dependency, [...trail, name]))
    visited.add(name)
  }
  manifests.forEach((manifest) => visit(manifest.name, []))
})

// The Next.js frontend is a separate npm app that reaches the backend only over HTTP, so it has no manifest here.
test("backend, client and tests keep their dependency boundaries", () => {
  const backend = manifests.find((manifest) => manifest.name === "@nia/backend")!
  const client = manifests.find((manifest) => manifest.name === "@opencode/client")!
  expect(backend.dependencies).toHaveProperty("@opencode/sdk")
  expect(backend.dependencies).not.toHaveProperty("@nia/frontend")
  expect(backend.dependencies).not.toHaveProperty("@opencode/tui")
  expect(client.dependencies).not.toHaveProperty("@opencode/core")
  expect(client.dependencies).not.toHaveProperty("@opencode/server")
  manifests.forEach((manifest) => expect(manifest.dependencies ?? {}).not.toHaveProperty("@nia/tests"))
})
