export * as ResearchResponse from "./research-response.js"

import type { SessionMessage } from "@opencode/schema/session-message"

/** Messages are chronological and restricted to the current admitted prompt. */
export function text(messages: readonly SessionMessage.Info[]) {
  const last = messages.at(-1)
  if (last?.type !== "assistant") return ""
  if (last.error || last.finish !== "stop") throw new Error("The provider did not complete a reply; ask again")
  // The runner tells interrupted streams to continue, so their text is a prefix of the final reply.
  // Stop at completed steps/tool calls: they may contain superseded reasoning or earlier answers.
  const boundary = messages.findLastIndex(
    (message, index) =>
      index < messages.length - 1 &&
      (message.type !== "assistant" ||
        (message.finish !== "length" && (message.error?.type !== "provider.invalid-output" || !message.retry))),
  )
  return messages
    .slice(boundary + 1)
    .flatMap((message) =>
      message.type === "assistant"
        ? message.content.flatMap((part) => (part.type === "text" && part.text.trim() ? [part.text.trim()] : []))
        : [],
    )
    .join("\n\n")
}
