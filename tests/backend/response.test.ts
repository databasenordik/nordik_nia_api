import { expect, test } from "bun:test"
import { Schema } from "effect"
import { SessionMessage } from "@opencode/schema/session-message"
import { ResearchResponse } from "@nia/backend/response"

const message = (id: string, text: string, finish = "stop", retry = false) =>
  Schema.decodeUnknownSync(SessionMessage.Info)({
    id: `msg_${id}`,
    type: "assistant",
    agent: "research",
    model: { providerID: "fixture", id: "research-model" },
    time: { created: 0 },
    content: [{ type: "text", text }],
    finish,
    ...(finish === "error" ? { error: { type: "provider.invalid-output", message: "Interrupted stream" } } : {}),
    ...(retry
      ? { retry: { attempt: 2, at: 0, error: { type: "provider.invalid-output", message: "Interrupted stream" } } }
      : {}),
  })

test("an ordinary closing reply excludes completed intermediate text", () => {
  expect(
    ResearchResponse.text([message("tool", "Looking this up", "tool-calls"), message("reply", "84 records.")]),
  ).toBe("84 records.")
})

test("a recovered interrupted response preserves the count and its continuation", () => {
  expect(
    ResearchResponse.text([
      message("tool", "Looking this up", "tool-calls"),
      message("partial", "84 records", "error", true),
      message("continued", "in this list. Source: ev_example.", "error", true),
      message("final", "These are records, not certified distinct people."),
    ]),
  ).toBe("84 records\n\nin this list. Source: ev_example.\n\nThese are records, not certified distinct people.")
})

test("recovery does not include older completed or unretried failed answers", () => {
  expect(
    ResearchResponse.text([
      message("old", "Superseded answer."),
      message("failed", "Discard this failure.", "error"),
      message("partial", "84 records.", "error", true),
      message("final", "Source: ev_current."),
    ]),
  ).toBe("84 records.\n\nSource: ev_current.")
})

test("a continued output-limit response retains its prefix", () => {
  expect(
    ResearchResponse.text([message("limited", "84 records.", "length"), message("final", "Source: ev_current.")]),
  ).toBe("84 records.\n\nSource: ev_current.")
})

test("an incomplete final stream is not delivered as a complete answer", () => {
  expect(() => ResearchResponse.text([message("failed", "84 rec", "error", true)])).toThrow("did not complete")
  expect(() => ResearchResponse.text([message("limited", "84 rec", "length")])).toThrow("did not complete")
  expect(ResearchResponse.text([])).toBe("")
})
