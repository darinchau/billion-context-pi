import { promises as fs, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { randomBytes } from "node:crypto";
import { parseAcpJson } from "./user-config.js";
import { userConfigPath } from "./config-dir.js";
import { resolveDelegate, type AdapterConfig } from "./config.js";

export type SetTarget = "rules" | "delegate";

export type SetCommandOp =
  | { kind: "show" }
  | { kind: "set"; target: SetTarget; on: boolean; scope: "global" | "project" }
  | { kind: "error"; message: string };

const SET_USAGE = "Usage: /acp-set [rules|delegate] [on|off] [--project]";

export function parseSetCommand(args: string): SetCommandOp {
  const tokens = args.trim().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return { kind: "show" };
  let scope: "global" | "project" = "global";
  const rest: string[] = [];
  for (const t of tokens) {
    if (t === "--project") scope = "project";
    else if (t === "--global") scope = "global";
    else rest.push(t.toLowerCase());
  }
  if (rest.length !== 2) return { kind: "error", message: SET_USAGE };
  const [target, value] = rest;
  if (target !== "rules" && target !== "delegate") {
    return { kind: "error", message: `unknown feature "${target}". ${SET_USAGE}` };
  }
  if (value !== "on" && value !== "off") {
    return { kind: "error", message: `expected on|off, got "${value}". ${SET_USAGE}` };
  }
  return { kind: "set", target, on: value === "on", scope };
}

export interface SetResult {
  ok: boolean;
  file: string;
  message: string;
}

function acpJsonPath(scope: "global" | "project", cwd: string): string {
  const root = scope === "global" ? homedir() : cwd;
  return userConfigPath(root, "acp.json");
}

async function readExisting(file: string): Promise<{ raw: string; obj: Record<string, unknown> } | { error: string }> {
  let raw: string;
  try {
    raw = await fs.readFile(file, "utf8");
  } catch (e: unknown) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return { raw: "", obj: {} };
    return { error: `cannot read ${file}: ${(e as Error).message}` };
  }
  const parsed = parseAcpJson(file, raw);
  if (parsed.status === "failed") {
    return { error: `${file} has invalid JSON and cannot be patched — fix it by hand first (${parsed.reason})` };
  }
  if (parsed.status === "repaired") {
    return { error: `${file} has non-strict JSON (BOM / trailing commas / unquoted keys) — fix it by hand first; /acp-set refuses to rewrite a repaired file` };
  }
  return { raw, obj: parsed.value ?? {} };
}

function patchValue(obj: Record<string, unknown>, target: SetTarget, on: boolean): void {
  if (target === "rules") {
    obj.rules = on;
    return;
  }
  const existing = obj.delegate;
  if (existing && typeof existing === "object" && !Array.isArray(existing)) {
    (existing as Record<string, unknown>).enabled = on;
  } else {
    obj.delegate = on;
  }
}

async function atomicWrite(file: string, content: string): Promise<void> {
  const dir = dirname(file);
  await fs.mkdir(dir, { recursive: true });
  const tmp = join(dir, `.acp-set-${randomBytes(4).toString("hex")}.tmp`);
  try {
    await fs.writeFile(tmp, content, "utf8");
    await fs.rename(tmp, file);
  } catch (e) {
    await fs.unlink(tmp).catch(() => undefined);
    throw e;
  }
}

export async function setConfigValue(
  target: SetTarget,
  on: boolean,
  scope: "global" | "project",
  cwd: string,
): Promise<SetResult> {
  const file = acpJsonPath(scope, cwd);
  const existing = await readExisting(file);
  if ("error" in existing) return { ok: false, file, message: existing.error };
  const obj = { ...existing.obj };
  patchValue(obj, target, on);
  await atomicWrite(file, JSON.stringify(obj, null, 2) + "\n");
  return { ok: true, file, message: `${target} = ${on} written to ${file}` };
}

export function checkOverride(
  target: SetTarget,
  scope: "global" | "project",
  cwd: string,
): string | undefined {
  if (scope !== "global") return undefined;
  const projectFile = userConfigPath(cwd, "acp.json");
  const globalFile = acpJsonPath("global", cwd);
  if (projectFile === globalFile) return undefined;
  let raw: string;
  try {
    raw = readFileSync(projectFile, "utf8");
  } catch {
    return undefined;
  }
  const parsed = parseAcpJson(projectFile, raw);
  if (parsed.status === "failed" || !parsed.value) return undefined;
  if (target in parsed.value) {
    return `note: project ${projectFile} also sets "${target}" and overrides the global value`;
  }
  return undefined;
}

export function currentValue(target: SetTarget, adapter: AdapterConfig): boolean {
  if (target === "rules") return adapter.rules === true;
  return resolveDelegate(adapter).enabled;
}
