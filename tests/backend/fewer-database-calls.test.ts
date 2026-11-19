import { describe, expect, test } from "bun:test"
import { ResearchHost } from "@nia/backend"
import { ResearchDatabase } from "@nia/backend/research/database"

const current = (fingerprint: string, version: number) => async () => ({ fingerprint, dataset: { version } })
const counted = <A>(read: () => Promise<A>) => {
  const calls = { count: 0 }
  return { calls, read: () => (calls.count++, read()) }
}

describe("delivery freshness check", () => {
  test("a reply that cites nothing does not read the database", async () => {
    const { calls, read } = counted(current("f1", 5))
    expect(await ResearchHost.deliverable([], read)).toBe(true)
    expect(calls.count).toBe(0)
  })

  test("a reply that cites nothing is delivered even when the database is down", async () => {
    const down = async () => {
      throw new Error("database unavailable")
    }
    expect(await ResearchHost.deliverable([], down)).toBe(true)
  })

  test("cited evidence that matches the current publication is delivered", async () => {
    const { calls, read } = counted(current("f1", 5))
    expect(await ResearchHost.deliverable([{ fingerprint: "f1", version: 5 }], read)).toBe(true)
    expect(calls.count).toBe(1)
  })

  test("cited evidence from before a republish is refused", async () => {
    expect(await ResearchHost.deliverable([{ fingerprint: "f1", version: 5 }], current("f1", 6))).toBe(false)
  })

  test("cited evidence whose data changed is refused", async () => {
    expect(await ResearchHost.deliverable([{ fingerprint: "f1", version: 5 }], current("f2", 5))).toBe(false)
  })

  test("one stale item among fresh ones refuses the whole answer", async () => {
    const cited = [
      { fingerprint: "f2", version: 5 },
      { fingerprint: "f1", version: 5 },
    ]
    expect(await ResearchHost.deliverable(cited, current("f2", 5))).toBe(false)
  })

  test("cited evidence still fails when the database is down (unchanged behaviour)", async () => {
    const down = async () => {
      throw new Error("database unavailable")
    }
    await expect(ResearchHost.deliverable([{ fingerprint: "f1", version: 5 }], down)).rejects.toThrow(
      "database unavailable",
    )
  })
})

describe("catalog reuse", () => {
  const clock = () => {
    let now = 0
    return { now: () => now, advance: (ms: number) => (now += ms) }
  }

  test("repeat reads within the window share one database read", async () => {
    const time = clock()
    let reads = 0
    const catalog = ResearchDatabase.reuseFor(3000, async (id: number) => ({ id, version: ++reads }), time.now)
    const first = await catalog(49)
    time.advance(2999)
    expect(await catalog(49)).toBe(first)
    expect(reads).toBe(1)
  })

  test("a read is repeated once the window has passed", async () => {
    const time = clock()
    let reads = 0
    const catalog = ResearchDatabase.reuseFor(3000, async () => ++reads, time.now)
    await catalog(49)
    time.advance(3000)
    expect(await catalog(49)).toBe(2)
  })

  test("concurrent callers share one in-flight read", async () => {
    let reads = 0
    let release!: (value: string) => void
    const catalog = ResearchDatabase.reuseFor(3000, () => {
      reads++
      return new Promise<string>((resolve) => (release = resolve))
    })
    const both = Promise.all([catalog(49), catalog(49), catalog(49)])
    release("catalog")
    expect(await both).toEqual(["catalog", "catalog", "catalog"])
    expect(reads).toBe(1)
  })

  test("each list is cached separately", async () => {
    const reads: number[] = []
    const catalog = ResearchDatabase.reuseFor(3000, async (id: number) => (reads.push(id), id))
    expect(await catalog(49)).toBe(49)
    expect(await catalog(91)).toBe(91)
    expect(reads).toEqual([49, 91])
  })

  test("a failed read is never reused: the next caller asks the database again", async () => {
    let attempt = 0
    const catalog = ResearchDatabase.reuseFor(3000, async () => {
      if (++attempt === 1) throw new ResearchDatabase.AccessDeniedError()
      return "granted"
    })
    await expect(catalog(94)).rejects.toBeInstanceOf(ResearchDatabase.AccessDeniedError)
    expect(await catalog(94)).toBe("granted")
    expect(attempt).toBe(2)
  })

  test("callers waiting on a read that fails all see the failure, and it is not kept", async () => {
    let reads = 0
    let fail!: (error: Error) => void
    const catalog = ResearchDatabase.reuseFor(3000, () => {
      if (++reads > 1) return Promise.resolve("recovered")
      return new Promise<string>((_, reject) => (fail = reject))
    })
    const a = catalog(49), b = catalog(49)
    fail(new Error("timeout"))
    await expect(a).rejects.toThrow("timeout")
    await expect(b).rejects.toThrow("timeout")
    expect(await catalog(49)).toBe("recovered")
    expect(reads).toBe(2)
  })

  // Known limit of the change: catalog checks can see a revocation or republish up to the window late.
  test("EDGE: access revoked inside the window is still served from the catalog cache until it expires", async () => {
    const time = clock()
    let granted = true
    const catalog = ResearchDatabase.reuseFor(3000, async () => {
      if (!granted) throw new ResearchDatabase.AccessDeniedError()
      return "private list"
    }, time.now)
    await catalog(94)
    granted = false
    time.advance(1000)
    expect(await catalog(94)).toBe("private list")
    time.advance(2000)
    await expect(catalog(94)).rejects.toBeInstanceOf(ResearchDatabase.AccessDeniedError)
  })

  test("EDGE: a republish inside the window shows the old version until it expires", async () => {
    const time = clock()
    let version = 5
    const catalog = ResearchDatabase.reuseFor(3000, async () => ({ version }), time.now)
    expect((await catalog(49)).version).toBe(5)
    version = 6
    time.advance(2000)
    expect((await catalog(49)).version).toBe(5)
    time.advance(1000)
    expect((await catalog(49)).version).toBe(6)
  })
})

describe("abortable shared reads", () => {
  test("one caller aborting does not fail another caller of the same read", async () => {
    let release!: (value: string) => void
    const shared = new Promise<string>((resolve) => (release = resolve))
    const a = new AbortController()
    const first = ResearchDatabase.abortable(shared, a.signal)
    const second = ResearchDatabase.abortable(shared, new AbortController().signal)
    a.abort(new Error("client went away"))
    release("catalog")
    await expect(first).rejects.toThrow("client went away")
    expect(await second).toBe("catalog")
  })

  test("an already-aborted signal rejects immediately", async () => {
    const a = new AbortController()
    a.abort(new Error("cancelled"))
    await expect(ResearchDatabase.abortable(new Promise(() => {}), a.signal)).rejects.toThrow("cancelled")
  })

  test("without a signal the shared promise is returned as is", async () => {
    const shared = Promise.resolve(1)
    expect(ResearchDatabase.abortable(shared)).toBe(shared)
  })
})
