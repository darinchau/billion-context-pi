import type { ExtensionContext, SessionMessageEntry } from "@earendil-works/pi-coding-agent";
import type { CoreMessage } from "acp-kernel";
import { logInfo, logWarn } from "./log.js";
import { formatCompactTokens } from "./footer-status.js";
import { evaluateForce, newForceState, type ForceState, type ForceTransition } from "./force.js";
import {
  applySqueeze,
  findCandidates,
  formatSqueezeStatus,
  resolveSqueeze,
  squeezeActiveFor,
  summarizeCandidates,
  T0Store,
  type SqueezeConfig,
  type SummarizeFn,
} from "./squeeze.js";

type AgentMessage = SessionMessageEntry["message"];

export const SQUEEZE_STATUS_KEY = "acp-squeeze";

export interface SqueezeTurnInput {
  sid: string;
  ctx: ExtensionContext;
  modelKey: string | undefined;
  messages: AgentMessage[];
  coreMessages: readonly CoreMessage[];
  covered: ReadonlySet<string>;
  protectedTools: readonly string[];
  acpPrunedTokens: number;
}

export interface SqueezeTurnOutput {
  messages: AgentMessage[];
  applied: number;
  tokensSaved: number;
}

interface SessionSqueeze {
  store: T0Store;
  inflight: Promise<void> | null;
  abort: AbortController;
  lastPruned: number;
  lastSaved: number;
  lastApplied: number;
  errors: string[];
}

interface SessionForce {
  state: ForceState;
  sessionOverride: boolean;
  compressedSinceActive: boolean;
}

export function toolResultIds(coreMessages: readonly CoreMessage[]): Map<string, string> {
  const out = new Map<string, string>();
  for (const c of coreMessages) {
    if (c.contentType === "tool-result" && c.toolCallId) out.set(c.toolCallId, c.id);
  }
  return out;
}

export function modelKeyOf(model: unknown): string | undefined {
  if (typeof model !== "object" || model === null) return undefined;
  const m = model as { provider?: unknown; id?: unknown };
  if (typeof m.id !== "string") return undefined;
  return typeof m.provider === "string" ? `${m.provider}/${m.id}` : m.id;
}

export function contextSummarizer(ctx: ExtensionContext, cfg: SqueezeConfig): SummarizeFn | null {
  const [provider, ...rest] = cfg.compressorModel.split("/");
  if (!provider || rest.length === 0) return null;
  const registry = ctx.modelRegistry;
  if (!registry || typeof registry.find !== "function" || typeof registry.complete !== "function") return null;
  const model = registry.find(provider, rest.join("/"));
  if (!model) return null;
  return async (_req, system, prompt, signal) => {
    const res = await registry.complete(
      model,
      { systemPrompt: system, messages: [{ role: "user", content: [{ type: "text", text: prompt }], timestamp: Date.now() }] },
      { maxTokens: cfg.maxSummaryTokens, signal, cacheRetention: "none" },
    );
    if (res.stopReason === "error" || res.stopReason === "aborted") throw new Error(res.errorMessage ?? `compressor ${res.stopReason}`);
    return res.content.map((c) => (c.type === "text" ? c.text : "")).join("");
  };
}

export class SqueezeController {
  private sessions = new Map<string, SessionSqueeze>();
  private forces = new Map<string, SessionForce>();
  summarizerFactory: (ctx: ExtensionContext, cfg: SqueezeConfig) => SummarizeFn | null = contextSummarizer;
  storeRoot: string | undefined;
  retryDelayMs = 1000;

  constructor(private readonly adapter: () => { squeeze?: unknown; forceThreshold?: number | null }) {}

  config(): SqueezeConfig {
    return resolveSqueeze(this.adapter().squeeze);
  }

  configuredForceThreshold(): number | null {
    const t = this.adapter().forceThreshold;
    return typeof t === "number" && Number.isFinite(t) && t > 0 ? Math.round(t) : null;
  }

  session(sid: string): SessionSqueeze {
    let s = this.sessions.get(sid);
    if (!s) {
      s = { store: new T0Store(sid, this.storeRoot), inflight: null, abort: new AbortController(), lastPruned: 0, lastSaved: 0, lastApplied: 0, errors: [] };
      this.sessions.set(sid, s);
    }
    return s;
  }

  peek(sid: string): SessionSqueeze | undefined {
    return this.sessions.get(sid);
  }

  async process(input: SqueezeTurnInput): Promise<SqueezeTurnOutput> {
    const cfg = this.config();
    if (!squeezeActiveFor(cfg, input.modelKey)) return { messages: input.messages, applied: 0, tokensSaved: 0 };
    const s = this.session(input.sid);
    await s.store.load();
    const byCall = toolResultIds(input.coreMessages);
    const ids = input.messages.map((m) => (m.role === "toolResult" ? byCall.get(m.toolCallId) : undefined));
    const applied = applySqueeze(input.messages, ids, input.covered, s.store);
    s.lastApplied = applied.applied;
    s.lastSaved = applied.tokensSaved;
    s.lastPruned = input.acpPrunedTokens + applied.tokensSaved;
    if (applied.applied > 0) logInfo("squeeze", { sid: input.sid, event: "applied", count: applied.applied, saved: applied.tokensSaved });
    if (!s.inflight) {
      const found = findCandidates({ messages: input.messages, ids, covered: input.covered, protectedTools: input.protectedTools, cfg });
      const pending = found.candidates.filter((c) => !s.store.get(c.id, c.hash) && !s.store.hasFailed(c.id, c.hash));
      if (pending.length > 0) {
        const summarize = this.summarizerFactory(input.ctx, cfg);
        if (!summarize) {
          if (!s.errors.includes("compressor-unavailable")) {
            s.errors.push("compressor-unavailable");
            logWarn("squeeze", { sid: input.sid, event: "compressor-unavailable", model: cfg.compressorModel });
          }
        } else {
          logInfo("squeeze", { sid: input.sid, event: "summarize-start", count: pending.length });
          s.inflight = summarizeCandidates(pending, found.calls, found.userGoal, cfg, s.store, summarize, { retryDelayMs: this.retryDelayMs, signal: s.abort.signal })
            .then((r) => {
              if (r.errors.length > 0) s.errors.push(...r.errors.slice(0, 5));
              logInfo("squeeze", { sid: input.sid, event: "summarize-done", done: r.done, errors: r.errors.length });
            })
            .catch((err: unknown) => {
              logWarn("squeeze", { sid: input.sid, event: "summarize-failed", error: err instanceof Error ? err.message : String(err) });
            })
            .finally(() => {
              s.inflight = null;
            });
        }
      }
    }
    return { messages: applied.messages, applied: applied.applied, tokensSaved: applied.tokensSaved };
  }

  async idle(sid: string): Promise<void> {
    const s = this.sessions.get(sid);
    if (s?.inflight) await s.inflight;
  }

  updateStatus(ctx: ExtensionContext, sid: string, percent: number | null): string | undefined {
    const cfg = this.config();
    const show = cfg.enabled && cfg.compressorModel !== "" && squeezeActiveFor(cfg, modelKeyOf(ctx.model));
    const text = show ? formatSqueezeStatus(cfg.compressorModel, this.sessions.get(sid)?.lastPruned ?? 0, percent, formatCompactTokens) : undefined;
    if (ctx.hasUI) ctx.ui.setStatus(SQUEEZE_STATUS_KEY, text);
    return text;
  }

  clearStatus(ctx: ExtensionContext): void {
    if (ctx.hasUI) ctx.ui.setStatus(SQUEEZE_STATUS_KEY, undefined);
  }

  force(sid: string): SessionForce {
    let f = this.forces.get(sid);
    if (!f) {
      f = { state: newForceState(this.configuredForceThreshold()), sessionOverride: false, compressedSinceActive: false };
      this.forces.set(sid, f);
    }
    if (!f.sessionOverride) f.state.threshold = this.configuredForceThreshold();
    return f;
  }

  setForceThreshold(sid: string, threshold: number | null): void {
    const f = this.force(sid);
    f.state.threshold = threshold;
    f.sessionOverride = true;
  }

  evaluateForce(sid: string, tokens: number): ForceTransition {
    const f = this.force(sid);
    const t = evaluateForce(f.state, tokens, f.compressedSinceActive);
    if (t !== null) f.compressedSinceActive = false;
    if (t) logInfo("force-compress", { sid, event: t, tokens, threshold: f.state.threshold });
    return t;
  }

  forceActive(sid: string): boolean {
    const f = this.forces.get(sid);
    return !!f && f.state.active && f.state.threshold !== null;
  }

  noteCompressSuccess(sid: string): void {
    const f = this.forces.get(sid);
    if (f?.state.active) f.compressedSinceActive = true;
  }

  dropSession(sid: string): void {
    const s = this.sessions.get(sid);
    if (s) s.abort.abort();
    this.sessions.delete(sid);
    this.forces.delete(sid);
  }
}

const controllers = new WeakMap<object, SqueezeController>();

export function squeezeControllerFor(runtime: { readonly adapter: { squeeze?: unknown; forceThreshold?: number | null } }): SqueezeController {
  let c = controllers.get(runtime);
  if (!c) {
    c = new SqueezeController(() => runtime.adapter);
    controllers.set(runtime, c);
  }
  return c;
}
