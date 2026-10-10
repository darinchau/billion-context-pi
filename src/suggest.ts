export const SUGGEST_DIRECTIVE = "You SHOULD use this tool now to compress earlier context";
export const SUGGEST_RELEASE_RATIO = 0.8;
export const SUGGEST_USAGE = "Usage: /acp-suggest [N|off] [--save]   (N like 120000, 120k, 1.2m; 0 or off disables)";

export type SuggestCommandOp =
  | { kind: "show" }
  | { kind: "set"; threshold: number | null; save: boolean }
  | { kind: "error"; message: string };

export function parseThreshold(raw: string): number | null | undefined {
  const s = raw.trim().toLowerCase().replace(/[,_]/g, "");
  if (s === "off" || s === "0") return null;
  const m = /^(\d+(?:\.\d+)?)([km]?)$/.exec(s);
  if (!m || !m[1]) return undefined;
  const mult = m[2] === "k" ? 1_000 : m[2] === "m" ? 1_000_000 : 1;
  const n = Math.round(Number(m[1]) * mult);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return n === 0 ? null : n;
}

export function parseSuggestCommand(args: string): SuggestCommandOp {
  const tokens = args.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return { kind: "show" };
  const save = tokens.includes("--save");
  const rest = tokens.filter((t) => t !== "--save");
  if (rest.length !== 1 || !rest[0]) return { kind: "error", message: SUGGEST_USAGE };
  const threshold = parseThreshold(rest[0]);
  if (threshold === undefined) return { kind: "error", message: `invalid threshold "${rest[0]}". ${SUGGEST_USAGE}` };
  return { kind: "set", threshold, save };
}

export type SuggestTransition = "activated" | "released-compressed" | "released-below" | "disabled" | null;

export interface SuggestState {
  threshold: number | null;
  active: boolean;
  lastTokens: number;
}

export function newSuggestState(threshold: number | null): SuggestState {
  return { threshold, active: false, lastTokens: 0 };
}

export function evaluateSuggest(st: SuggestState, tokens: number, compressedSinceActive: boolean): SuggestTransition {
  st.lastTokens = tokens;
  if (st.threshold === null || st.threshold <= 0) {
    if (st.active) {
      st.active = false;
      return "disabled";
    }
    return null;
  }
  if (st.active) {
    if (compressedSinceActive) {
      st.active = false;
      return "released-compressed";
    }
    if (tokens < st.threshold * SUGGEST_RELEASE_RATIO) {
      st.active = false;
      return "released-below";
    }
    return null;
  }
  if (tokens >= st.threshold) {
    st.active = true;
    return "activated";
  }
  return null;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function augment(desc: unknown): string {
  const base = typeof desc === "string" ? desc : "";
  if (base.includes(SUGGEST_DIRECTIVE)) return base;
  return base ? `${SUGGEST_DIRECTIVE}. ${base}` : `${SUGGEST_DIRECTIVE}.`;
}

function rewriteTool(tool: unknown, name: string): { tool: unknown; hit: boolean } {
  if (!isRecord(tool)) return { tool, hit: false };
  if (tool.name === name && ("input_schema" in tool || "parameters" in tool || "description" in tool)) {
    return { tool: { ...tool, description: augment(tool.description) }, hit: true };
  }
  if (isRecord(tool.function) && tool.function.name === name) {
    return { tool: { ...tool, function: { ...tool.function, description: augment(tool.function.description) } }, hit: true };
  }
  if (Array.isArray(tool.functionDeclarations)) {
    let hit = false;
    const decls = tool.functionDeclarations.map((d: unknown) => {
      if (isRecord(d) && d.name === name) {
        hit = true;
        return { ...d, description: augment(d.description) };
      }
      return d;
    });
    return hit ? { tool: { ...tool, functionDeclarations: decls }, hit } : { tool, hit: false };
  }
  if (isRecord(tool.toolSpec) && tool.toolSpec.name === name) {
    return { tool: { ...tool, toolSpec: { ...tool.toolSpec, description: augment(tool.toolSpec.description) } }, hit: true };
  }
  return { tool, hit: false };
}

function rewriteToolList(list: unknown, name: string): { list: unknown; hit: boolean } {
  if (!Array.isArray(list)) return { list, hit: false };
  let hit = false;
  const next = list.map((t: unknown) => {
    const r = rewriteTool(t, name);
    if (r.hit) hit = true;
    return r.tool;
  });
  return hit ? { list: next, hit } : { list, hit: false };
}

export function applySuggestToPayload(payload: unknown, toolName = "compress"): { payload: unknown; hit: boolean } {
  if (!isRecord(payload)) return { payload, hit: false };
  const top = rewriteToolList(payload.tools, toolName);
  if (top.hit) return { payload: { ...payload, tools: top.list }, hit: true };
  if (isRecord(payload.config)) {
    const inner = rewriteToolList(payload.config.tools, toolName);
    if (inner.hit) return { payload: { ...payload, config: { ...payload.config, tools: inner.list } }, hit: true };
  }
  if (isRecord(payload.toolConfig)) {
    const inner = rewriteToolList(payload.toolConfig.tools, toolName);
    if (inner.hit) return { payload: { ...payload, toolConfig: { ...payload.toolConfig, tools: inner.list } }, hit: true };
  }
  return { payload, hit: false };
}

export function formatThreshold(n: number | null): string {
  if (n === null) return "off";
  if (n >= 1_000_000) return `${+(n / 1_000_000).toFixed(2)}m`;
  if (n >= 1_000) return `${+(n / 1_000).toFixed(1)}k`;
  return String(n);
}
