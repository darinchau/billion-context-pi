/**
 * Tests for the delegate role allowlist (`delegate.agents` in acp.json).
 *
 * Coverage:
 *  1. applyAgentConfig: `enabled: false` removes a built-in role
 *  2. runDelegate: a disabled role fails with "disabled by config", not "unknown"
 *  3. applyAgentConfig: custom role is added (description + tools)
 *  4. applyAgentConfig: custom role without a prompt is skipped
 *  5. applyAgentConfig: invalid role names are skipped
 *  6. makeDelegateTool: tool description + param description reflect the roster
 *  7. buildChildArgs: custom role gets its --tools allowlist
 *  8. resetAgents: restores the built-in roster
 *  9. built-in overrides: description and tools
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyAgentConfig, resetAgents, makeDelegateTool, buildChildArgs } from "../src/delegate-tool.js";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";

/** Minimal ctx mock — buildChildArgs reads ctx.model and sessionManager. */
function mockCtx(host: "pi" | "omp" = "pi"): ExtensionContext {
  const sessionManager =
    host === "pi"
      ? { buildContextEntries: () => [] }
      : { getBranch: () => [] };
  return { model: { provider: "test", id: "test-model" }, sessionManager } as unknown as ExtensionContext;
}

/** Parse the --tools value from cliArgs, or null if absent. */
function getToolsValue(cliArgs: string[]): string | null {
  const i = cliArgs.indexOf("--tools");
  return i >= 0 ? (cliArgs[i + 1] ?? null) : null;
}

test("applyAgentConfig: enabled:false removes a built-in role from the roster", () => {
  try {
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const tool = makeDelegateTool(pi);
    const desc = tool.description ?? "";
    assert.ok(desc.includes("reviewer"), "default roster advertises reviewer");

    applyAgentConfig({ reviewer: { enabled: false } });
    const after = makeDelegateTool(pi).description ?? "";
    assert.ok(!after.includes("reviewer -"), "disabled role no longer advertised");
    assert.ok(after.includes("worker -"), "other roles untouched");
  } finally {
    resetAgents();
  }
});

test("runDelegate: disabled role fails with a config error, unknown role with the generic error", async () => {
  try {
    applyAgentConfig({ oracle: { enabled: false } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const tool = makeDelegateTool(pi);
    const ctx = { ...mockCtx("pi"), cwd: process.cwd() } as unknown as ExtensionContext;

    const disabled = await tool.execute("t-disabled", { agent: "oracle", task: "x", async: false }, undefined, undefined, ctx);
    const disabledText = (disabled.content[0] as { text?: string }).text ?? "";
    assert.ok(disabledText.includes("disabled by the delegate.agents config"), `disabled message: ${disabledText}`);
    assert.ok(disabledText.includes("worker"), "error lists the available roles");

    const unknown = await tool.execute("t-unknown", { agent: "nope", task: "x", async: false }, undefined, undefined, ctx);
    const unknownText = (unknown.content[0] as { text?: string }).text ?? "";
    assert.ok(unknownText.startsWith("Unknown agent"), `unknown message: ${unknownText}`);
  } finally {
    resetAgents();
  }
});

test("applyAgentConfig: custom role is added with its description and toolset", () => {
  try {
    applyAgentConfig({
      auditor: {
        prompt: "You are a security auditor with read-only access. Report findings with file:line references. Do NOT modify any files.",
        description: "read-only security audit",
        tools: "read,bash",
      },
    });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const desc = makeDelegateTool(pi).description ?? "";
    assert.ok(desc.includes("auditor - read-only security audit"), "custom role advertised with its description");
    assert.ok(desc.includes("[tools: read,bash"), "custom toolset listed");
  } finally {
    resetAgents();
  }
});

test("applyAgentConfig: custom role without a prompt is skipped", () => {
  try {
    applyAgentConfig({ promptless: { description: "no prompt" } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const desc = makeDelegateTool(pi).description ?? "";
    assert.ok(!desc.includes("promptless"), "role without a prompt is not added");
  } finally {
    resetAgents();
  }
});

test("applyAgentConfig: invalid role names are skipped", () => {
  try {
    applyAgentConfig({ "Bad Name": { prompt: "x" }, "1starts-with-digit": { prompt: "x" } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const desc = makeDelegateTool(pi).description ?? "";
    assert.ok(!desc.includes("Bad Name") && !desc.includes("1starts"), "invalid names rejected");
  } finally {
    resetAgents();
  }
});

test("makeDelegateTool: agent param description lists the configured roster", () => {
  try {
    applyAgentConfig({ worker: { enabled: false } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const tool = makeDelegateTool(pi);
    const agentParam = (tool.parameters as { properties: { agent: { description?: string } } }).properties.agent;
    const d = agentParam.description ?? "";
    assert.ok(d.includes("reviewer"), "param description enumerates reviewer");
    assert.ok(!d.includes("worker,"), "param description omits the disabled role");
  } finally {
    resetAgents();
  }
});

test("buildChildArgs: custom role gets its --tools allowlist", async () => {
  try {
    applyAgentConfig({ auditor: { prompt: "audit things", tools: "read,bash" } });
    const { cliArgs } = await buildChildArgs({ agent: "auditor", task: "x" }, "audit things", mockCtx("pi"), "del_allow_1");
    assert.ok(String(getToolsValue(cliArgs)).startsWith("read,bash"), "custom role runs with its configured toolset");
  } finally {
    resetAgents();
  }
});

test("resetAgents restores the built-in roster", () => {
  try {
    applyAgentConfig({ reviewer: { enabled: false }, auditor: { prompt: "x" } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    let desc = makeDelegateTool(pi).description ?? "";
    assert.ok(!desc.includes("reviewer -") && desc.includes("auditor"));
    resetAgents();
    desc = makeDelegateTool(pi).description ?? "";
    assert.ok(desc.includes("reviewer -"), "built-in restored");
    assert.ok(!desc.includes("auditor"), "custom role removed");
  } finally {
    resetAgents();
  }
});

test("applyAgentConfig: built-in overrides apply description and tools", () => {
  try {
    applyAgentConfig({ oracle: { description: "wise counsel", tools: "read" } });
    const pi = {} as Parameters<typeof makeDelegateTool>[0];
    const desc = makeDelegateTool(pi).description ?? "";
    assert.ok(desc.includes("oracle - wise counsel"), "description override wins over built-in blurb");
    assert.ok(desc.includes("[tools: read +"), "tools override listed");
  } finally {
    resetAgents();
  }
});
