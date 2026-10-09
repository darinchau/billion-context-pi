import { createHash, randomBytes } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { SessionMessageEntry } from "@earendil-works/pi-coding-agent";
import { defaultCountTokens } from "acp-kernel";

type AgentMessage = SessionMessageEntry["message"];
import { extractText, splitLeadingRefTag } from "./messages.js";
import { rewriteTagTokens } from "./tag-tokens.js";
import {
  clip,
  compressorPrompt,
  compressorSystem,
  globMatch,
  PROMPT_STYLES,
  type PromptStyle,
  type SummarizeRequest,
} from "./squeeze-prompts.js";

export interface SqueezeConfig {
  enabled: boolean;
  compressorModel: string;
  targetModels: string[];
  keepRecentToolTurns: number;
  keepFirstToolTurns: number;
  minChars: number;
  summaryWords: number;
  maxRatio: number;
  maxCompressorInputChars: number;
  concurrency: number;
  maxSummaryTokens: number;
  promptStyle: PromptStyle;
}

export const DEFAULT_SQUEEZE: SqueezeConfig = {
  enabled: false,
  compressorModel: "",
  targetModels: [],
  keepRecentToolTurns: 2,
  keepFirstToolTurns: 1,
  minChars: 1500,
  summaryWords: 200,
  maxRatio: 0.6,
  maxCompressorInputChars: 400_000,
  concurrency: 4,
  maxSummaryTokens: 2048,
  promptStyle: "squeeze",
};

export const SQUEEZE_NUMERIC_KEYS = [
  "keepRecentToolTurns",
  "keepFirstToolTurns",
  "minChars",
  "summaryWords",
  "maxRatio",
  "maxCompressorInputChars",
  "concurrency",
  "maxSummaryTokens",
] as const;
type NumericKey = (typeof SQUEEZE_NUMERIC_KEYS)[number];

export const SQUEEZE_SETTABLE_KEYS = ["enabled", "compressorModel", "targetModels", "promptStyle", ...SQUEEZE_NUMERIC_KEYS] as const;

const SQUEEZE_TOOL_EXCLUSIONS = new Set(["compress", "decompress", "search_context", "acp_status", "acp_cache", "acp_rule"]);

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isNumericKey(k: string): k is NumericKey {
  return (SQUEEZE_NUMERIC_KEYS as readonly string[]).includes(k);
}

function validNumber(key: NumericKey, n: number): boolean {
  if (!Number.isFinite(n) || n < 0) return false;
  if (key === "maxRatio") return n > 0 && n <= 1;
  if (key === "concurrency") return Number.isInteger(n) && n >= 1;
  return Number.isInteger(n);
}

export function resolveSqueeze(raw: unknown): SqueezeConfig {
  const out: SqueezeConfig = { ...DEFAULT_SQUEEZE, targetModels: [] };
  if (!isRecord(raw)) return out;
  if (typeof raw.enabled === "boolean") out.enabled = raw.enabled;
  if (typeof raw.compressorModel === "string") out.compressorModel = raw.compressorModel.trim();
  if (Array.isArray(raw.targetModels)) out.targetModels = raw.targetModels.filter((t): t is string => typeof t === "string" && t.trim() !== "").map((t) => t.trim());
  if (typeof raw.promptStyle === "string" && (PROMPT_STYLES as readonly string[]).includes(raw.promptStyle)) out.promptStyle = raw.promptStyle as PromptStyle;
  for (const k of SQUEEZE_NUMERIC_KEYS) {
    const v = raw[k];
    if (typeof v === "number" && validNumber(k, v)) out[k] = v;
  }
  return out;
}

export function squeezeActiveFor(cfg: SqueezeConfig, activeModel: string | undefined): boolean {
  if (!cfg.enabled || !cfg.compressorModel || !activeModel) return false;
  if (activeModel.toLowerCase() === cfg.compressorModel.toLowerCase()) return false;
  if (cfg.targetModels.length === 0) return true;
  return cfg.targetModels.some((p) => globMatch(p, activeModel));
}

export interface T0Entry {
  summary: string;
  rawFile: string;
  toolName: string;
  originalChars: number;
  hash: string;
}

export interface T0File {
  version: 1;
  t0: Record<string, T0Entry>;
  failed: Record<string, string>;
}

export function squeezeRootDir(): string {
  return join(homedir() || tmpdir(), ".cache", "pi", "acp-squeeze");
}

function safeSegment(s: string): string {
  return s.replace(/[^A-Za-z0-9._-]/g, "_").slice(0, 80) || "session";
}

export function contentHash(text: string): string {
  return createHash("sha256").update(text).digest("hex").slice(0, 16);
}

export class T0Store {
  readonly dir: string;
  private data: T0File = { version: 1, t0: {}, failed: {} };
  private loaded = false;

  constructor(sessionId: string, root: string = squeezeRootDir()) {
    this.dir = join(root, safeSegment(sessionId));
  }

  get file(): string {
    return join(this.dir, "t0.json");
  }

  async load(): Promise<void> {
    if (this.loaded) return;
    this.loaded = true;
    try {
      const parsed: unknown = JSON.parse(await readFile(this.file, "utf8"));
      if (isRecord(parsed) && isRecord(parsed.t0)) {
        const t0: Record<string, T0Entry> = {};
        for (const [id, e] of Object.entries(parsed.t0)) {
          if (isRecord(e) && typeof e.summary === "string" && typeof e.rawFile === "string") {
            t0[id] = {
              summary: e.summary,
              rawFile: e.rawFile,
              toolName: typeof e.toolName === "string" ? e.toolName : "tool",
              originalChars: typeof e.originalChars === "number" ? e.originalChars : 0,
              hash: typeof e.hash === "string" ? e.hash : "",
            };
          }
        }
        this.data = { version: 1, t0, failed: {} };
      }
    } catch {
      this.data = { version: 1, t0: {}, failed: {} };
    }
  }

  get(id: string, hash: string): T0Entry | undefined {
    const e = this.data.t0[id];
    if (!e) return undefined;
    if (e.hash && e.hash !== hash) return undefined;
    if (!existsSync(e.rawFile)) return undefined;
    return e;
  }

  entries(): Record<string, T0Entry> {
    return this.data.t0;
  }

  hasFailed(id: string, hash: string): boolean {
    return this.data.failed[id] === hash;
  }

  markFailed(id: string, hash: string): void {
    this.data.failed[id] = hash;
  }

  async put(id: string, toolName: string, raw: string, summary: string): Promise<T0Entry> {
    await mkdir(this.dir, { recursive: true });
    const hash = contentHash(raw);
    const rawFile = join(this.dir, `${safeSegment(id)}-${hash}.txt`);
    if (!existsSync(rawFile)) await writeAtomic(rawFile, raw);
    const entry: T0Entry = { summary, rawFile, toolName, originalChars: raw.length, hash };
    this.data.t0[id] = entry;
    await this.flush();
    return entry;
  }

  private writeChain: Promise<void> = Promise.resolve();

  // Serialize index writes: concurrent rename onto the same target fails with
  // EPERM on Windows, and later writes must not be overtaken by earlier ones.
  private flush(): Promise<void> {
    const next = this.writeChain.then(() => writeAtomic(this.file, JSON.stringify({ version: 1, t0: this.data.t0 }, null, 2) + "\n"));
    this.writeChain = next.catch(() => undefined);
    return next;
  }

  stats(): { count: number; originalChars: number; summaryChars: number } {
    let originalChars = 0;
    let summaryChars = 0;
    const list = Object.values(this.data.t0);
    for (const e of list) {
      originalChars += e.originalChars;
      summaryChars += e.summary.length;
    }
    return { count: list.length, originalChars, summaryChars };
  }
}

async function writeAtomic(file: string, content: string): Promise<void> {
  const tmp = `${file}.${randomBytes(6).toString("hex")}.tmp`;
  try {
    await writeFile(tmp, content, "utf8");
    await rename(tmp, file);
  } catch (err) {
    await rm(tmp, { force: true }).catch(() => undefined);
    throw err;
  }
}

export function placeholderText(entry: T0Entry): string {
  return [
    `[acp-squeeze: this ${entry.toolName} output (${entry.originalChars} chars) was summarized by a smaller model to save context.`,
    `Full original output: ${entry.rawFile}`,
    `Use read on that file if you need exact details.]`,
    "",
    entry.summary,
  ].join("\n");
}

export interface SqueezeCandidate {
  id: string;
  toolName: string;
  toolCallId: string;
  text: string;
  hash: string;
  index: number;
}

interface ToolCallInfo {
  name: string;
  args: string;
  turn: number;
}

function toolCallsOf(msg: AgentMessage): { id: string; name: string; arguments: unknown }[] {
  if (msg.role !== "assistant" || !Array.isArray(msg.content)) return [];
  const out: { id: string; name: string; arguments: unknown }[] = [];
  for (const b of msg.content) {
    if (b.type === "toolCall") out.push({ id: b.id, name: b.name, arguments: b.arguments });
  }
  return out;
}

function stringifyArgs(a: unknown): string {
  if (typeof a === "string") return a;
  try {
    return JSON.stringify(a) ?? "";
  } catch {
    return "";
  }
}

export interface EligibilityInput {
  messages: readonly AgentMessage[];
  ids: readonly (string | undefined)[];
  covered: ReadonlySet<string>;
  protectedTools: readonly string[];
  cfg: SqueezeConfig;
}

export function findCandidates(input: EligibilityInput): { candidates: SqueezeCandidate[]; calls: Map<string, ToolCallInfo>; userGoal: string } {
  const { messages, ids, covered, cfg } = input;
  const calls = new Map<string, ToolCallInfo>();
  let turn = 0;
  for (const m of messages) {
    const tc = toolCallsOf(m);
    if (tc.length === 0) continue;
    for (const c of tc) calls.set(c.id, { name: c.name, args: stringifyArgs(c.arguments), turn });
    turn += 1;
  }
  const totalTurns = turn;
  const protectedSet = new Set([...input.protectedTools, ...SQUEEZE_TOOL_EXCLUSIONS]);
  let userGoal = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === "user") {
      userGoal = clip(extractText(m.content), 3000);
      break;
    }
  }
  const candidates: SqueezeCandidate[] = [];
  messages.forEach((m, index) => {
    if (m.role !== "toolResult") return;
    const id = ids[index];
    if (!id || covered.has(id)) return;
    if (m.isError) return;
    const owner = calls.get(m.toolCallId);
    if (!owner) return;
    if (owner.turn < cfg.keepFirstToolTurns) return;
    if (owner.turn >= totalTurns - cfg.keepRecentToolTurns) return;
    if (protectedSet.has(m.toolName)) return;
    const text = extractText(m.content);
    if (text.length < cfg.minChars) return;
    if (text.startsWith("[acp-squeeze:") || text.startsWith("[pi-squeeze:")) return;
    candidates.push({ id, toolName: m.toolName, toolCallId: m.toolCallId, text, hash: contentHash(text), index });
  });
  return { candidates, calls, userGoal };
}

export function applyT0(message: AgentMessage, entry: T0Entry): AgentMessage {
  if (message.role !== "toolResult") return message;
  const blocks = typeof message.content === "string" ? [{ type: "text" as const, text: message.content }] : message.content;
  let tag = "";
  for (const b of blocks) {
    if (b.type !== "text") continue;
    const split = splitLeadingRefTag(b.text);
    if (split) {
      tag = split.tag;
      break;
    }
    const trailing = /\n*(\x3cacp\s[^>]*\x3em\d{5}\x3c\/acp\x3e)\s*$/.exec(b.text);
    if (trailing && trailing[1]) {
      tag = trailing[1];
      break;
    }
  }
  const body = placeholderText(entry);
  const text = tag ? `${body}\n\n${rewriteTagTokens(tag.trim(), body)}` : body;
  const nonText = blocks.filter((b) => b.type !== "text");
  return { ...message, content: [...nonText, { type: "text", text }] };
}

export interface SqueezeApplyResult {
  messages: AgentMessage[];
  applied: number;
  tokensSaved: number;
}

export function applySqueeze(
  messages: readonly AgentMessage[],
  ids: readonly (string | undefined)[],
  covered: ReadonlySet<string>,
  store: T0Store,
): SqueezeApplyResult {
  let applied = 0;
  let tokensSaved = 0;
  const out = messages.map((m, index) => {
    const id = ids[index];
    if (!id || covered.has(id) || m.role !== "toolResult") return m;
    const text = extractText(m.content);
    const entry = store.get(id, contentHash(text));
    if (!entry) return m;
    const next = applyT0(m, entry);
    applied += 1;
    tokensSaved += Math.max(0, defaultCountTokens(text) - defaultCountTokens(placeholderText(entry)));
    return next;
  });
  return { messages: out, applied, tokensSaved };
}

export type SummarizeFn = (req: SummarizeRequest, system: string, prompt: string, signal?: AbortSignal) => Promise<string>;

export const SQUEEZE_RETRIES = 3;

export async function withRetry<T>(fn: () => Promise<T>, retries: number, delayMs: number, signal?: AbortSignal): Promise<T> {
  let last: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (signal?.aborted) throw new Error("aborted");
    try {
      return await fn();
    } catch (err) {
      last = err;
      if (attempt < retries && delayMs > 0) await new Promise((r) => setTimeout(r, delayMs * (attempt + 1)));
    }
  }
  throw last instanceof Error ? last : new Error(String(last));
}

export interface SummarizeJobResult {
  done: number;
  errors: string[];
}

export async function summarizeCandidates(
  candidates: readonly SqueezeCandidate[],
  calls: ReadonlyMap<string, ToolCallInfo>,
  userGoal: string,
  cfg: SqueezeConfig,
  store: T0Store,
  summarize: SummarizeFn,
  opts: { retryDelayMs?: number; signal?: AbortSignal } = {},
): Promise<SummarizeJobResult> {
  const errors: string[] = [];
  let done = 0;
  const queue = candidates.filter((c) => !store.get(c.id, c.hash) && !store.hasFailed(c.id, c.hash));
  const worker = async (): Promise<void> => {
    for (;;) {
      const c = queue.shift();
      if (!c) return;
      const call = calls.get(c.toolCallId);
      const req: SummarizeRequest = {
        toolName: c.toolName,
        toolArgs: clip(call?.args ?? "", 4000),
        output: clip(c.text, cfg.maxCompressorInputChars),
        userGoal,
      };
      try {
        const summary = (
          await withRetry(
            () => summarize(req, compressorSystem(cfg.promptStyle), compressorPrompt(req, cfg.summaryWords, cfg.promptStyle), opts.signal),
            SQUEEZE_RETRIES,
            opts.retryDelayMs ?? 1000,
            opts.signal,
          )
        ).trim();
        if (!summary) throw new Error("empty summary");
        if (summary.length > c.text.length * cfg.maxRatio) {
          store.markFailed(c.id, c.hash);
          continue;
        }
        await store.put(c.id, c.toolName, c.text, summary);
        done += 1;
      } catch (err) {
        store.markFailed(c.id, c.hash);
        errors.push(`${c.toolName} ${c.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(cfg.concurrency, queue.length)) }, worker));
  return { done, errors };
}

export function formatSqueezeStatus(modelName: string, prunedTokens: number, percent: number | null, fmt: (n: number) => string): string {
  const short = modelName.split("/").pop() || modelName;
  const pct = percent === null || !Number.isFinite(percent) ? "?" : `${Math.round(percent)}%`;
  return `squeeze:${short} \u25bc${fmt(Math.max(0, Math.round(prunedTokens)))} (${pct})`;
}
