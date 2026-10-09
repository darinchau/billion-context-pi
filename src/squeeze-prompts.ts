export interface SummarizeRequest {
  toolName: string;
  toolArgs: string;
  output: string;
  userGoal: string;
}

export type Summarizer = (req: SummarizeRequest) => Promise<string>;


export function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  const half = Math.floor(max / 2);
  return `${text.slice(0, half)}\n\n[... ${text.length - max} chars omitted ...]\n\n${text.slice(-half)}`;
}

export function globMatch(pattern: string, value: string): boolean {
  const re = new RegExp(`^${pattern.split("*").map(escapeRe).join(".*")}$`, "i");
  return re.test(value);
}
function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}


export type PromptStyle = "squeeze" | "pi";
export const PROMPT_STYLES: PromptStyle[] = ["squeeze", "pi"];

/** "squeeze" style: terse, tool-output-specific prompt (pi-squeeze's own). */
export const SQUEEZE_SYSTEM = `You compress tool outputs for a coding agent. The agent's context is expensive, so you replace a tool output with a dense summary it will read instead of the original.
Rules:
- Keep everything the agent is likely to need later: file paths, line numbers, function/class names, exact error messages, failing test names, versions, key values, counts, and conclusions.
- Prefer exact snippets (short) over paraphrase for code and errors.
- Drop boilerplate, repetition, progress bars, and irrelevant noise.
- Say what was NOT found if the output shows absence (e.g. "no matches for X").
- Do not add advice or commentary. Output only the summary, plain text or terse markdown bullets.`;

/**
 * "pi" style: pi's built-in compaction prompts, copied verbatim from pi-coding-agent
 * dist/core/compaction/utils.js (SUMMARIZATION_SYSTEM_PROMPT) and compaction.js (SUMMARIZATION_PROMPT).
 * The input is serialized like pi's serializeConversation().
 */
export const PI_SYSTEM = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

export const PI_SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

function piArgs(argsJson: string): string {
  try {
    const args = JSON.parse(argsJson) as Record<string, unknown>;
    return Object.entries(args)
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
      .join(", ");
  } catch {
    return argsJson;
  }
}

export function compressorSystem(style: PromptStyle): string {
  return style === "pi" ? PI_SYSTEM : SQUEEZE_SYSTEM;
}

export function compressorPrompt(req: SummarizeRequest, words: number, style: PromptStyle = "squeeze"): string {
  if (style === "pi") {
    const conversation = [
      `[User]: ${req.userGoal || "(unknown)"}`,
      `[Assistant tool calls]: ${req.toolName}(${piArgs(req.toolArgs)})`,
      `[Tool result]: ${req.output}`,
    ].join("\n\n");
    return `<conversation>\n${conversation}\n</conversation>\n\n${PI_SUMMARIZATION_PROMPT}`;
  }
  return [
    `Agent's latest user request (for relevance):\n<goal>\n${req.userGoal || "(unknown)"}\n</goal>`,
    `Tool: ${req.toolName}\nArguments: ${req.toolArgs}`,
    `<tool_output>\n${req.output}\n</tool_output>`,
    `Summarize the tool output in at most ~${words} words (fewer if the output is simple).`,
  ].join("\n\n");
}
