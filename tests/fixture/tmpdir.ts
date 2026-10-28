import { mkdtemp, rm } from "node:fs/promises"
import os from "node:os"
import path from "node:path"

// A private temporary directory removed when the `await using` scope ends.
export async function tmpdir(prefix: string) {
  const directory = await mkdtemp(path.join(os.tmpdir(), prefix))
  return {
    path: directory,
    [Symbol.asyncDispose]: () => rm(directory, { recursive: true, force: true }),
  }
}
