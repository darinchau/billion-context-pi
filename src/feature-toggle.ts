import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { RULE_TOOL_NAME } from "acp-kernel";
import type { AcpRuntime } from "./runtime.js";
import { resolveDelegate } from "./config.js";
import { makeRuleTool } from "./rule-tool.js";
import { makeDelegateTool, makeDelegateWaitTool, makeDelegateCancelTool } from "./delegate-tool.js";

export const DELEGATE_TOOL_NAMES = ["acp_delegate", "acp_delegate_wait", "acp_delegate_cancel"] as const;

const registered = new WeakMap<ExtensionAPI, Set<string>>();

function registeredFor(pi: ExtensionAPI): Set<string> {
  let s = registered.get(pi);
  if (!s) {
    s = new Set();
    registered.set(pi, s);
  }
  return s;
}

export function rulesWanted(runtime: AcpRuntime): boolean {
  return runtime.adapter.rules === true;
}

export function delegateWanted(runtime: AcpRuntime): boolean {
  return resolveDelegate(runtime.adapter).enabled && !runtime.delegateStoodDown;
}

export function registerRuleTool(pi: ExtensionAPI, runtime: AcpRuntime): void {
  const s = registeredFor(pi);
  if (s.has(RULE_TOOL_NAME)) return;
  pi.registerTool(makeRuleTool(runtime));
  s.add(RULE_TOOL_NAME);
}

export function registerDelegateTools(pi: ExtensionAPI): void {
  const s = registeredFor(pi);
  if (s.has(DELEGATE_TOOL_NAMES[0])) return;
  pi.registerTool(makeDelegateTool(pi));
  pi.registerTool(makeDelegateWaitTool(pi));
  pi.registerTool(makeDelegateCancelTool(pi));
  for (const n of DELEGATE_TOOL_NAMES) s.add(n);
}

function setActive(pi: ExtensionAPI, names: readonly string[], on: boolean): void {
  const active = new Set(pi.getActiveTools());
  let changed = false;
  for (const n of names) {
    if (on && !active.has(n)) {
      active.add(n);
      changed = true;
    } else if (!on && active.has(n)) {
      active.delete(n);
      changed = true;
    }
  }
  if (changed) pi.setActiveTools([...active]);
}

export function applyFeatureToggles(pi: ExtensionAPI, runtime: AcpRuntime): void {
  const rules = rulesWanted(runtime);
  if (rules) registerRuleTool(pi, runtime);
  if (registeredFor(pi).has(RULE_TOOL_NAME)) setActive(pi, [RULE_TOOL_NAME], rules);
  const delegate = delegateWanted(runtime);
  if (delegate) registerDelegateTools(pi);
  if (registeredFor(pi).has(DELEGATE_TOOL_NAMES[0])) setActive(pi, DELEGATE_TOOL_NAMES, delegate);
}
