// Delivery instructions from nordikdriveapi's former Gemini reader. No research context is added.
export function voicePrompt(style: string, text: string) {
  return `Style: ${style}.\nRead the following TEXT exactly as written. Do not add or remove words.\n\nTEXT:\n${text}`
}
