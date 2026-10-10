/**
 * /acp-force: automatic blocking compression when the token meter exceeds a threshold.
 *
 * Design:
 *  - Fires inside the context-transform lock (same turn, blocking) so the provider
 *    request goes out already compressed.
 *  - Selects the oldest compressible ranges, skipping the most recent one.
 *  - Target is 80 % of the threshold (FORCE_TARGET_RATIO).
 *  - Summaries: cheap model (option A) with mechanical fallback (option B).
 *  - Persists an "acp-auto-force" custom entry so state-rebuild.ts can replay it
 *    when the .acp.json sidecar is missing (imported session).
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { CompressionState, CompressibleRange, CoreMessage } from "acp-kernel";
import { COMPRESS_PHILOSOPHY, HOW_TO_COMPRESS_RULES } from "acp-kernel";
import { logInfo, logWarn } from "./log.js";
import { sanitizeSummary } from "./summary-sanitize.js";
import type { SummarizeFn } from "./squeeze.js";

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------

export const ACP_AUTO_FORCE_CUSTOM_TYPE = "acp-auto-force";

/** Compress down to this fraction of the threshold (80 %). */
export const FORCE_TARGET_RATIO = 0.8;

/** Don't re-arm until T grows by this fraction from the exhaustion point. */
export const FORCE_BACKOFF_RATIO = 0.1;

// ------------------------------------------------------------------
// Types
// ------------------------------------------------------------------

export interface AutoForceRangeSpec {
  startRef: string;
  endRef: string;
  summary: string;
  topic?: string;
}

/** Written to the session via appendEntry for state-rebuild replay. */
export interface AutoForceRecord {
  ranges: AutoForceRangeSpec[];
  ts: number;
  summarizer: AutoForceSummarizer;
}

/** Which summarizer produced the batch: all model, all mechanical, or a mix
 *  (some ranges fell back after a model error/timeout). */
export type AutoForceSummarizer = "model" | "mechanical" | "mixed";

// ------------------------------------------------------------------
// Range selection (pure — easy to test)
// ------------------------------------------------------------------

/**
 * Pick the oldest ranges greedily until `forceTokens - reclaimed <= target`.
 * The last range in `compressibleRanges` (most recent by startIndex) is never
 * picked; at least 2 ranges must be present.
 *
 * @param compressibleRanges  Already viableRanges-filtered, sorted or unsorted.
 * @param forceTokens         Current token meter (T, excluding last tool call).
 * @param threshold           /acp-force N value.
 * @returns  { selected, exhausted }
 */
export function selectForceRanges(
  compressibleRanges: CompressibleRange[],
  forceTokens: number,
  threshold: number,
): { selected: CompressibleRange[]; exhausted: boolean } {
  if (compressibleRanges.length < 2) return { selected: [], exhausted: false };

  const target = Math.floor(threshold * FORCE_TARGET_RATIO);
  if (forceTokens <= threshold) return { selected: [], exhausted: false };

  // Sort oldest first by startIndex; fall back to startRef lexical order.
  const sorted = [...compressibleRanges].sort(
    (a, b) => (a.startIndex ?? 0) - (b.startIndex ?? 0) || a.startRef.localeCompare(b.startRef),
  );

  // Always keep the most recent range intact.
  const candidates = sorted.slice(0, sorted.length - 1);

  let reclaimed = 0;
  const selected: CompressibleRange[] = [];
  for (const range of candidates) {
    selected.push(range);
    reclaimed += range.tokens;
    if (forceTokens - reclaimed <= target) break;
  }

  const exhausted =
    selected.length === candidates.length && forceTokens - reclaimed > target;

  return { selected, exhausted };
}

// ------------------------------------------------------------------
// Mechanical summary (option B fallback)
// ------------------------------------------------------------------

const MECHANICAL_USER_CHARS = 400;
const MECHANICAL_ASSISTANT_CHARS = 200;
const MECHANICAL_TOOL_CHARS = 200;

/**
 * Build a compact mechanical summary from a list of CoreMessages.
 * Used when there is no cheap-model summarizer or when it times out/errors.
 */
export function mechanicalSummary(
  msgs: CoreMessage[],
  t0Summaries: Map<string, string>,
): string {
  const clip = (txt: string, max: number): string => (txt.length > max ? txt.slice(0, max) + "…" : txt);
  const flat = (txt: string | undefined): string => (txt ?? "").replace(/\s+/g, " ").trim();
  const lines: string[] = [];
  for (const m of msgs) {
    switch (m.contentType) {
      case "tool-call": {
        // Keep the tool name and clipped arguments so the summary still says
        // what was attempted (T0 summaries describe results, not calls).
        const args = flat(m.text);
        lines.push(`[tool-call ${m.toolName ?? "unknown"}] ${args ? clip(args, MECHANICAL_TOOL_CHARS) : "(no args)"}`);
        break;
      }
      case "tool-result": {
        const t0 = m.toolCallId ? t0Summaries.get(m.toolCallId) : undefined;
        if (t0) {
          lines.push(`[tool-result ${m.toolName ?? "unknown"}] ${clip(flat(t0), MECHANICAL_TOOL_CHARS)}`);
        } else {
          const txt = flat(m.text);
          const head = `[tool-result ${m.toolName ?? "unknown"}] (${(m.text ?? "").length} chars)`;
          lines.push(txt ? `${head} ${clip(txt, MECHANICAL_TOOL_CHARS)}` : head);
        }
        break;
      }
      default: {
        const txt = m.text ?? "";
        if (m.role === "user") {
          if (txt) lines.push(`[user ${m.id}] ${clip(txt, MECHANICAL_USER_CHARS)}`);
        } else if (m.role === "assistant") {
          if (txt) lines.push(`[assistant] ${clip(txt, MECHANICAL_ASSISTANT_CHARS)}`);
        }
        break;
      }
    }
  }
  return lines.join("\n") || "(empty range)";
}

// ------------------------------------------------------------------
// Collect the messages that fall inside a CompressibleRange
// ------------------------------------------------------------------

/**
 * Return the CoreMessages whose ids map to refs between range.startRef and
 * range.endRef (inclusive).  Uses state.messageRefs.byRef to translate refs
 * back to raw ids, then walks the messages array linearly.
 */
export function messagesForRange(
  range: CompressibleRange,
  messages: readonly CoreMessage[],
  state: CompressionState,
): CoreMessage[] {
  const byRef = state.messageRefs.byRef as Record<string, string>;
  const startRaw = byRef[range.startRef];
  const endRaw = byRef[range.endRef];
  if (!startRaw || !endRaw) return [];

  let inRange = false;
  const out: CoreMessage[] = [];
  for (const m of messages) {
    const baseId = m.id.includes("#") ? m.id.split("#")[0]! : m.id;
    if (baseId === startRaw) inRange = true;
    if (inRange) out.push(m);
    if (baseId === endRaw) break;
  }
  return out;
}

// ------------------------------------------------------------------
// Summarize a single range
// ------------------------------------------------------------------

async function summarizeRange(opts: {
  range: CompressibleRange;
  messages: readonly CoreMessage[];
  state: CompressionState;
  t0Summaries: Map<string, string>;
  summarizeFn: SummarizeFn | null;
  signal: AbortSignal;
  sid: string;
}): Promise<{ summary: string; usedModel: boolean }> {
  const { range, messages, state, t0Summaries, summarizeFn, signal, sid } = opts;
  const msgs = messagesForRange(range, messages, state);

  if (summarizeFn) {
    const content = msgs
      .map((m) => {
        if (m.contentType === "tool-result" && m.toolCallId) {
          const t0 = t0Summaries.get(m.toolCallId);
          return t0
            ? `[tool-result ${m.toolName} → T0]: ${t0}`
            : `[tool-result ${m.toolName}]: ${(m.text ?? "").slice(0, 2000)}`;
        }
        const label = m.toolName ? `${m.role} ${m.toolName}` : m.role;
        return `[${label} ${m.id}]: ${(m.text ?? "").slice(0, 2000)}`;
      })
      .join("\n\n");

    const system = `${COMPRESS_PHILOSOPHY}\n\n${HOW_TO_COMPRESS_RULES}`;
    const prompt =
      `Compress the following conversation range (${range.startRef}..${range.endRef}) ` +
      `into a dense summary following the compression rules above.\n\n${content}`;

    try {
      if (signal.aborted) throw new Error("aborted");
      const summary = await summarizeFn({} as never, system, prompt, signal);
      const trimmed = summary.trim();
      if (trimmed.length >= 50) return { summary: trimmed, usedModel: true };
    } catch (e) {
      logWarn("auto-force", {
        sid,
        event: "model-summarizer-failed",
        range: `${range.startRef}..${range.endRef}`,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  // Option B: mechanical fallback
  return { summary: mechanicalSummary(msgs, t0Summaries), usedModel: false };
}

// ------------------------------------------------------------------
// Main entry point — called from index.ts
// ------------------------------------------------------------------

export interface GenerateSummariesResult {
  rangeSpecs: AutoForceRangeSpec[];
  summarizer: AutoForceSummarizer;
}

/**
 * Generate summaries for all selected ranges.
 *
 * Runs model summarizations in parallel; falls back to mechanical on any
 * failure.  Respects `timeoutMs` via a shared AbortSignal.
 */
export async function generateSummaries(opts: {
  selected: CompressibleRange[];
  coreMessages: readonly CoreMessage[];
  state: CompressionState;
  t0Summaries: Map<string, string>;
  summarizeFn: SummarizeFn | null;
  timeoutMs: number;
  sid: string;
}): Promise<GenerateSummariesResult> {
  const { selected, coreMessages, state, t0Summaries, summarizeFn, timeoutMs, sid } = opts;

  const abort = new AbortController();
  const timeoutId = setTimeout(() => abort.abort(), timeoutMs);

  let anyModel = false;
  let anyMechanical = false;

  try {
    const results = await Promise.all(
      selected.map((range) =>
        summarizeRange({
          range,
          messages: coreMessages,
          state,
          t0Summaries,
          summarizeFn,
          signal: abort.signal,
          sid,
        }),
      ),
    );

    const rangeSpecs: AutoForceRangeSpec[] = results.map((r, i) => {
      if (r.usedModel) anyModel = true;
      else anyMechanical = true;
      const sanitized = sanitizeSummary(r.summary).text;
      const range = selected[i]!;
      return { startRef: range.startRef, endRef: range.endRef, summary: sanitized };
    });

    return {
      rangeSpecs,
      summarizer: anyModel && anyMechanical ? "mixed" : anyModel ? "model" : "mechanical",
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

// ------------------------------------------------------------------
// Persist the custom entry for state-rebuild replay
// ------------------------------------------------------------------

export function persistAutoForceEntry(
  pi: Pick<ExtensionAPI, "appendEntry">,
  record: AutoForceRecord,
  sid: string,
): void {
  if (typeof pi.appendEntry !== "function") return;
  try {
    pi.appendEntry(ACP_AUTO_FORCE_CUSTOM_TYPE, record);
  } catch (e) {
    logWarn("auto-force", {
      sid,
      event: "persist-entry-failed",
      error: e instanceof Error ? e.message : String(e),
    });
  }
}

// ------------------------------------------------------------------
// Log helper
// ------------------------------------------------------------------

export function logAutoForce(opts: {
  sid: string;
  ranges: number;
  before: number;
  reclaimed: number;
  target: number;
  after: number;
  summarizer: AutoForceSummarizer;
  warnings?: string[];
}): void {
  logInfo("auto-force", {
    sid: opts.sid,
    event: "auto-compressed",
    ranges: opts.ranges,
    before: opts.before,
    reclaimed: opts.reclaimed,
    target: opts.target,
    after: opts.after,
    summarizer: opts.summarizer,
    ...(opts.warnings && opts.warnings.length > 0 ? { warnings: opts.warnings.slice(0, 5) } : {}),
  });
}
