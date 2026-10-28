import { expect, test } from "bun:test"
import { NiaGrpc } from "@nia/backend/grpc"

// NiaGrpc.engine over a stand-in for ResearchHost: Session reuse, context rebuilt from history,
// the concurrency limit, failures and cancellation.
function fakeHost() {
  const created: { id: string; communities: readonly string[] }[] = []
  const prompts: { sessionID: string; text: string }[] = []
  const interrupted: string[] = []
  const live = new Set<string>()
  const state = { delay: 0, fail: false, active: 0, peak: 0 }
  const host = {
    dataset: async () => ({ fields: [] }),
    get: async (id: string) => {
      if (!live.has(id)) throw new Error("gone")
      return { id }
    },
    create: async (_fileID: number, _id?: string, communities: readonly string[] = []) => {
      const id = `ses_${created.length}`
      created.push({ id, communities })
      live.add(id)
      return { id }
    },
    prompt: async (sessionID: string, text: string) => {
      prompts.push({ sessionID, text })
      if (state.fail) throw new Error("provider unavailable")
      state.active++
      state.peak = Math.max(state.peak, state.active)
      try {
        await Bun.sleep(state.delay)
        return { text: `answer ${prompts.length}` }
      } finally {
        state.active--
      }
    },
    interrupt: async (id: string) => {
      interrupted.push(id)
    },
  }
  return { host: host as unknown as Parameters<typeof NiaGrpc.engine>[0], created, prompts, interrupted, live, state }
}

const ask = (overrides: Partial<Parameters<NiaGrpc.Engine["answer"]>[0]> = {}) => ({
  conversationID: "c1",
  fileID: 49,
  communities: [] as string[],
  question: "How many?",
  history: async () => [],
  signal: new AbortController().signal,
  ...overrides,
})

test("a conversation keeps its Session, and a new Session starts from the stored history", async () => {
  const fake = fakeHost()
  const engine = NiaGrpc.engine(fake.host)
  let reads = 0
  const history = async () => {
    reads++
    return [{ question: "Who attended in 1920?", answer: "Three students." }]
  }
  expect(await engine.answer(ask({ question: "And in 1921?", history }))).toBe("answer 1")
  await engine.answer(ask({ question: "Which school?", history }))
  expect(fake.created).toEqual([{ id: "ses_0", communities: [] }])
  expect(reads).toBe(1)
  expect(fake.prompts[0]!.text).toContain("User: Who attended in 1920?\nNia: Three students.")
  expect(fake.prompts[0]!.text.endsWith("Current question: And in 1921?")).toBe(true)
  expect(fake.prompts[1]).toEqual({ sessionID: "ses_0", text: "Which school?" })
})

test("another selection or a lost Session starts a new Session", async () => {
  const fake = fakeHost()
  const engine = NiaGrpc.engine(fake.host)
  await engine.answer(ask())
  await engine.answer(ask({ communities: ["Batchewana"] }))
  await engine.answer(ask({ communities: ["Batchewana"] }))
  fake.live.delete("ses_1")
  await engine.answer(ask({ communities: ["Batchewana"] }))
  expect(fake.created.map((item) => item.communities)).toEqual([[], ["Batchewana"], ["Batchewana"]])
})

test("a failed answer leaves no Session behind, so the next question gets a fresh one", async () => {
  const fake = fakeHost()
  const engine = NiaGrpc.engine(fake.host)
  fake.state.fail = true
  await expect(engine.answer(ask())).rejects.toThrow(/provider unavailable/)
  fake.state.fail = false
  expect(await engine.answer(ask())).toBe("answer 2")
  expect(fake.prompts.map((prompt) => prompt.sessionID)).toEqual(["ses_0", "ses_1"])
})

test("no more answers run at once than the concurrency limit", async () => {
  const fake = fakeHost()
  fake.state.delay = 30
  const engine = NiaGrpc.engine(fake.host, { concurrency: 2 })
  await Promise.all(["a", "b", "c", "d", "e"].map((id) => engine.answer(ask({ conversationID: id }))))
  expect(fake.state.peak).toBe(2)
  expect(fake.prompts).toHaveLength(5)
})

test("questions in one conversation are answered one after another", async () => {
  const fake = fakeHost()
  fake.state.delay = 20
  const engine = NiaGrpc.engine(fake.host)
  await Promise.all([engine.answer(ask()), engine.answer(ask()), engine.answer(ask())])
  expect(fake.state.peak).toBe(1)
})

test("a cancelled call interrupts its Session", async () => {
  const fake = fakeHost()
  fake.state.delay = 50
  const engine = NiaGrpc.engine(fake.host)
  const controller = new AbortController()
  const answer = engine.answer(ask({ signal: controller.signal }))
  await Bun.sleep(10)
  controller.abort()
  await answer
  expect(fake.interrupted).toEqual(["ses_0"])
})

test("rebuilt context keeps at most six exchanges and about 10,000 characters", () => {
  const history = Array.from({ length: 8 }, (_, i) => ({ question: `q${i}`, answer: "a".repeat(5000) }))
  const text = NiaGrpc.withContext(history, "Now?")
  expect(text).not.toContain("User: q1\n")
  expect(text).toContain("User: q7\n")
  expect(text.length).toBeLessThan(11_000)
  expect(text).toContain(`${"a".repeat(1500)}…`)
  expect(NiaGrpc.withContext([], "Now?")).toBe("Now?")
})
