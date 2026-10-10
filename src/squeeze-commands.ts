import type { ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { patchAcpJson } from "./config-write.js";
import { formatCompactTokens } from "./footer-status.js";
import { SUGGEST_USAGE, formatThreshold, parseSuggestCommand, parseThreshold } from "./suggest.js";
import { FORCE_TARGET_RATIO } from "./auto-force.js";
import { PROMPT_STYLES } from "./squeeze-prompts.js";
import { resolveSqueeze, SQUEEZE_NUMERIC_KEYS, SQUEEZE_SETTABLE_KEYS, squeezeActiveFor, type SqueezeConfig } from "./squeeze.js";
import { modelKeyOf, squeezeControllerFor } from "./squeeze-runtime.js";
import type { AcpRuntime } from "./runtime.js";

export const SQUEEZE_USAGE =
  "Usage: /acp-squeeze [status|on|off|model [provider/id]|targets [glob,...|none]|style [squeeze|pi]|set <key> <value>] [--project]";

export type SqueezeCommandOp =
  | { kind: "status" }
  | { kind: "toggle"; on: boolean }
  | { kind: "model"; value: string | null }
  | { kind: "targets"; value: string[] | null }
  | { kind: "style"; value: string | null }
  | { kind: "set"; key: string; value: unknown }
  | { kind: "error"; message: string };

export interface ParsedSqueeze {
  op: SqueezeCommandOp;
  scope: "global" | "project";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function parseSetValue(key: string, raw: string): { ok: true; value: unknown } | { ok: false; message: string } {
  if (!(SQUEEZE_SETTABLE_KEYS as readonly string[]).includes(key)) {
    return { ok: false, message: `unknown squeeze key "${key}". Keys: ${SQUEEZE_SETTABLE_KEYS.join(", ")}` };
  }
  if (key === "enabled") {
    const v = raw.toLowerCase();
    if (["on", "true", "1"].includes(v)) return { ok: true, value: true };
    if (["off", "false", "0"].includes(v)) return { ok: true, value: false };
    return { ok: false, message: `enabled expects on|off, got "${raw}"` };
  }
  if (key === "compressorModel") return raw.includes("/") ? { ok: true, value: raw } : { ok: false, message: `compressorModel must be provider/id, got "${raw}"` };
  if (key === "targetModels") return { ok: true, value: parseTargets(raw) };
  if (key === "promptStyle") {
    return (PROMPT_STYLES as readonly string[]).includes(raw) ? { ok: true, value: raw } : { ok: false, message: `promptStyle must be one of ${PROMPT_STYLES.join("|")}` };
  }
  const n = Number(raw);
  const probe = resolveSqueeze({ [key]: n });
  const numericKey = SQUEEZE_NUMERIC_KEYS.find((k) => k === key);
  if (!numericKey || !Number.isFinite(n) || probe[numericKey] !== n) return { ok: false, message: `invalid value for ${key}: "${raw}"` };
  return { ok: true, value: n };
}

function parseTargets(raw: string): string[] {
  if (raw.trim().toLowerCase() === "none" || raw.trim() === "") return [];
  return raw.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
}

export function parseSqueezeCommand(args: string): ParsedSqueeze {
  const tokens = args.trim().split(/\s+/).filter(Boolean);
  const scope: "global" | "project" = tokens.includes("--project") ? "project" : "global";
  const rest = tokens.filter((t) => t !== "--project" && t !== "--global");
  const sub = (rest[0] ?? "status").toLowerCase();
  const tail = rest.slice(1);
  const err = (message: string): ParsedSqueeze => ({ op: { kind: "error", message: `${message}. ${SQUEEZE_USAGE}` }, scope });
  switch (sub) {
    case "status":
      return tail.length === 0 ? { op: { kind: "status" }, scope } : err("status takes no arguments");
    case "on":
    case "off":
      return tail.length === 0 ? { op: { kind: "toggle", on: sub === "on" }, scope } : err(`${sub} takes no arguments`);
    case "model": {
      if (tail.length === 0) return { op: { kind: "model", value: null }, scope };
      if (tail.length !== 1 || !tail[0]?.includes("/")) return err("model expects provider/id");
      return { op: { kind: "model", value: tail[0] }, scope };
    }
    case "targets":
      return { op: { kind: "targets", value: tail.length === 0 ? null : parseTargets(tail.join(" ")) }, scope };
    case "style": {
      if (tail.length === 0) return { op: { kind: "style", value: null }, scope };
      const v = tail[0] ?? "";
      if (tail.length !== 1 || !(PROMPT_STYLES as readonly string[]).includes(v)) return err(`style must be one of ${PROMPT_STYLES.join("|")}`);
      return { op: { kind: "style", value: v }, scope };
    }
    case "set": {
      const key = tail[0];
      if (!key || tail.length < 2) return err("set expects <key> <value>");
      const parsed = parseSetValue(key, tail.slice(1).join(" "));
      if (!parsed.ok) return err(parsed.message);
      return { op: { kind: "set", key, value: parsed.value }, scope };
    }
    default:
      return err(`unknown subcommand "${sub}"`);
  }
}

export async function writeSqueezeKey(scope: "global" | "project", cwd: string, key: string, value: unknown) {
  return patchAcpJson(scope, cwd, (obj) => {
    const prev = isRecord(obj.squeeze) ? obj.squeeze : {};
    obj.squeeze = { ...prev, [key]: value };
  });
}

export function squeezeStatusText(runtime: AcpRuntime, ctx: ExtensionCommandContext): string {
  const cfg: SqueezeConfig = resolveSqueeze(runtime.adapter.squeeze);
  const ctl = squeezeControllerFor(runtime);
  const sid = ctx.sessionManager.getSessionId();
  const active = modelKeyOf(ctx.model);
  const sess = ctl.peek(sid);
  const stats = sess?.store.stats();
  const lines = [
    `acp-squeeze: ${cfg.enabled ? "on" : "off"}${runtime.refused ? " (ACP refused on this host, squeeze bypassed)" : ""}`,
    `compressor: ${cfg.compressorModel || "(unset)"}  style: ${cfg.promptStyle}`,
    `targets: ${cfg.targetModels.length ? cfg.targetModels.join(", ") : "(all non-compressor models)"}`,
    `active model: ${active ?? "?"} -> ${!runtime.refused && squeezeActiveFor(cfg, active) ? "squeezing" : "not squeezing"}`,
    `settings: keepRecentToolTurns=${cfg.keepRecentToolTurns} keepFirstToolTurns=${cfg.keepFirstToolTurns} minChars=${cfg.minChars} summaryWords=${cfg.summaryWords} maxRatio=${cfg.maxRatio} maxCompressorInputChars=${cfg.maxCompressorInputChars} concurrency=${cfg.concurrency} maxSummaryTokens=${cfg.maxSummaryTokens}`,
  ];
  if (sess && stats) {
    lines.push(
      `T0 cache: ${stats.count} summaries (${formatCompactTokens(stats.originalChars)} -> ${formatCompactTokens(stats.summaryChars)} chars) at ${sess.store.file}`,
      `last request: ${sess.lastApplied} T0 applied, ~${formatCompactTokens(sess.lastSaved)} tokens saved by squeeze, ~${formatCompactTokens(sess.lastPruned)} pruned total`,
    );
    const ids = Object.keys(sess.store.entries());
    if (ids.length > 0) lines.push(`T0 ids: ${ids.slice(0, 20).join(", ")}${ids.length > 20 ? ` (+${ids.length - 20} more)` : ""}`);
    if (sess.inflight) lines.push("summaries: in progress (applied on a later request)");
    if (sess.errors.length > 0) lines.push(`recent errors: ${sess.errors.slice(-3).join("; ")}`);
  }
  return lines.join("\n");
}

export async function handleSqueezeCommand(runtime: AcpRuntime, args: string, ctx: ExtensionCommandContext): Promise<void> {
  const { op, scope } = parseSqueezeCommand(args);
  if (op.kind === "error") {
    ctx.ui.notify(op.message, "error");
    return;
  }
  if (op.kind === "status") {
    ctx.ui.notify(squeezeStatusText(runtime, ctx));
    return;
  }
  let key: string;
  let value: unknown;
  if (op.kind === "toggle") {
    key = "enabled";
    value = op.on;
  } else if (op.kind === "model") {
    key = "compressorModel";
    value = op.value ?? (await pickModel(ctx));
  } else if (op.kind === "targets") {
    key = "targetModels";
    value = op.value ?? (await pickTargets(ctx));
  } else if (op.kind === "style") {
    key = "promptStyle";
    value = op.value ?? (ctx.hasUI ? await ctx.ui.select("acp-squeeze: prompt style", [...PROMPT_STYLES]) : undefined);
  } else {
    key = op.key;
    value = op.value;
  }
  if (value === undefined) {
    ctx.ui.notify(squeezeStatusText(runtime, ctx));
    return;
  }
  const res = await writeSqueezeKey(scope, ctx.cwd, key, value);
  if (!res.ok) {
    ctx.ui.notify(res.message, "error");
    return;
  }
  await runtime.reloadConfig(ctx.cwd);
  const lines = [`squeeze.${key} = ${JSON.stringify(value)} written to ${res.file}`];
  const effective: Record<string, unknown> = { ...resolveSqueeze(runtime.adapter.squeeze) };
  if (JSON.stringify(effective[key]) !== JSON.stringify(value)) lines.push(`effective squeeze.${key} is ${JSON.stringify(effective[key])} (overridden by project config)`);
  const ctl = squeezeControllerFor(runtime);
  const sid = ctx.sessionManager.getSessionId();
  if (key === "enabled" && value === false) ctl.clearStatus(ctx);
  else ctl.updateStatus(ctx, sid, ctx.getContextUsage?.()?.percent ?? null);
  ctx.ui.notify(lines.join("\n"));
}

async function pickModel(ctx: ExtensionCommandContext): Promise<string | undefined> {
  if (!ctx.hasUI) return undefined;
  const models = ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`);
  if (models.length === 0) return undefined;
  return ctx.ui.select("acp-squeeze: compressor (cheap) model", models);
}

async function pickTargets(ctx: ExtensionCommandContext): Promise<string[] | undefined> {
  if (!ctx.hasUI) return undefined;
  const raw = await ctx.ui.input("acp-squeeze: target models (comma-separated globs, 'none' for all)", "");
  return raw === undefined ? undefined : parseTargets(raw);
}

export const FORCE_USAGE = "Usage: /acp-force [N|off] [--save]   (N like 120000, 120k, 1.2m; 0 or off disables)";

type ForceCommandOp =
  | { kind: "show" }
  | { kind: "set"; threshold: number | null; save: boolean }
  | { kind: "error"; message: string };

export function parseForceCommand(args: string): ForceCommandOp {
  const tokens = args.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return { kind: "show" };
  const save = tokens.includes("--save");
  const rest = tokens.filter((t) => t !== "--save");
  if (rest.length !== 1 || !rest[0]) return { kind: "error", message: FORCE_USAGE };
  const threshold = parseThreshold(rest[0]);
  if (threshold === undefined) return { kind: "error", message: `invalid threshold "${rest[0]}". ${FORCE_USAGE}` };
  return { kind: "set", threshold, save };
}

export async function handleForceCommand(runtime: AcpRuntime, args: string, ctx: ExtensionCommandContext): Promise<void> {
  const op = parseForceCommand(args);
  if (op.kind === "error") {
    ctx.ui.notify(op.message, "error");
    return;
  }
  const ctl = squeezeControllerFor(runtime);
  const sid = ctx.sessionManager.getSessionId();
  if (op.kind === "show") {
    const f = ctl.force(sid);
    const thresh = f.threshold;
    const target = thresh !== null ? Math.floor(thresh * FORCE_TARGET_RATIO) : null;
    const lastTokens = ctl.forceLastTokens(sid);
    const backoff = lastTokens === null ? (f.exhaustedAt > 0 ? "unknown (no turn yet)" : "no") : ctl.inBackoff(sid, lastTokens) ? "yes" : "no";
    const lines = [
      `acp-force threshold: ${formatThreshold(thresh)}${f.sessionOverride ? " (session)" : runtime.adapter.forceThreshold ? " (config)" : ""}`,
      `target after compress: ${target !== null ? formatThreshold(target) : "n/a"} (80% of threshold)`,
      `timeout: ${f.timeoutMs}ms`,
      `in backoff: ${backoff}`,
      FORCE_USAGE,
    ];
    ctx.ui.notify(lines.join("\n"));
    return;
  }
  ctl.setForceThreshold(sid, op.threshold);
  const lines = [`acp-force threshold set to ${formatThreshold(op.threshold)} for this session`];
  if (op.save) {
    const res = await patchAcpJson("global", ctx.cwd, (obj) => {
      if (op.threshold === null) delete obj.forceThreshold;
      else obj.forceThreshold = op.threshold;
    });
    if (!res.ok) {
      ctx.ui.notify(res.message, "error");
      return;
    }
    await runtime.reloadConfig(ctx.cwd);
    lines.push(`saved to ${res.file}`);
  }
  ctx.ui.notify(lines.join("\n"));
}

export async function handleSuggestCommand(runtime: AcpRuntime, args: string, ctx: ExtensionCommandContext): Promise<void> {
  const op = parseSuggestCommand(args);
  if (op.kind === "error") {
    ctx.ui.notify(op.message, "error");
    return;
  }
  const ctl = squeezeControllerFor(runtime);
  const sid = ctx.sessionManager.getSessionId();
  if (op.kind === "show") {
    const f = ctl.suggest(sid);
    ctx.ui.notify(
      [
        `acp-suggest threshold: ${formatThreshold(f.state.threshold)}${f.sessionOverride ? " (session)" : runtime.adapter.suggestThreshold ? " (config)" : ""}`,
        `estimated input tokens: ${f.state.lastTokens > 0 ? formatCompactTokens(f.state.lastTokens) : "unknown (no request yet)"}`,
        `active: ${f.state.active && f.state.threshold !== null ? "yes" : "no"}`,
        SUGGEST_USAGE,
      ].join("\n"),
    );
    return;
  }
  ctl.setSuggestThreshold(sid, op.threshold);
  const lines = [`acp-suggest threshold set to ${formatThreshold(op.threshold)} for this session`];
  if (op.save) {
    const res = await patchAcpJson("global", ctx.cwd, (obj) => {
      if (op.threshold === null) delete obj.suggestThreshold;
      else obj.suggestThreshold = op.threshold;
    });
    if (!res.ok) {
      ctx.ui.notify(res.message, "error");
      return;
    }
    await runtime.reloadConfig(ctx.cwd);
    lines.push(`saved to ${res.file}`);
  }
  if (op.threshold === null) ctl.evaluateSuggest(sid, ctl.suggest(sid).state.lastTokens);
  ctx.ui.notify(lines.join("\n"));
}
