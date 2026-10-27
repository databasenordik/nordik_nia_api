import { parseThemeDocument, type ThemeDocument } from "./index.js"
import opencode2 from "./nia.json" with { type: "json" }

let opencode2Theme: ThemeDocument | undefined

export function getOpenCode2Theme() {
  opencode2Theme ??= parseThemeDocument(opencode2, "opencode2")
  return opencode2Theme
}
