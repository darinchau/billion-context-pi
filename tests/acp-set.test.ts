import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { RULE_TOOL_NAME } from "acp-kernel";
import { createAcpExtension } from "../src/index.js";
import { CONFIG_DIR_NAME } from "../src/config-dir.js";
import { parseSetCommand, setConfigValue } from "../src/config-write.js";
import { tmpPath } from "./tmp-path.js";

type HomeEnv = { HOME: string | undefined; USERPROFILE: string | undefined };

async function withHome<T>(fn: (home: string, cwd: string) => Promise<T>): Promise<T> {
  const saved: HomeEnv = { HOME: process.env.HOME, USERPROFILE: process.env.USERPROFILE };
  const home = await mkdtemp(path.join(tmpdir(), "acp-set-home-"));
  const cwd = await mkdtemp(path.join(tmpdir(), "acp-set-cwd-"));
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  try {
    return await fn(home, cwd);
  } finally {
    process.env.HOME = saved.HOME;
    process.env.USERPROFILE = saved.USERPROFILE;
    await rm(home, { recursive: true, force: true });
    await rm(cwd, { recursive: true, force: true });
  }
}

const globalFile = (home: string) => path.join(home, CONFIG_DIR_NAME, "acp.json");
const projectFile = (cwd: string) => path.join(cwd, CONFIG_DIR_NAME, "acp.json");

test("parseSetCommand: show, set, scope flag, and errors", () => {
  assert.deepEqual(parseSetCommand(""), { kind: "show" });
  assert.deepEqual(parseSetCommand("rules on"), { kind: "set", target: "rules", on: true, scope: "global" });
  assert.deepEqual(parseSetCommand("DELEGATE Off --project"), { kind: "set", target: "delegate", on: false, scope: "project" });
  assert.equal(parseSetCommand("rules").kind, "error");
  assert.equal(parseSetCommand("squeeze on").kind, "error");
  assert.equal(parseSetCommand("rules maybe").kind, "error");
});

test("setConfigValue creates acp.json and preserves unrelated keys", async () => {
  await withHome(async (home, cwd) => {
    const r1 = await setConfigValue("rules", true, "global", cwd);
    assert.ok(r1.ok, r1.message);
    assert.deepEqual(JSON.parse(await readFile(globalFile(home), "utf8")), { rules: true });

    await writeFile(globalFile(home), JSON.stringify({ autoUpdate: false, rules: true }));
    const r2 = await setConfigValue("rules", false, "global", cwd);
    assert.ok(r2.ok);
    assert.deepEqual(JSON.parse(await readFile(globalFile(home), "utf8")), { autoUpdate: false, rules: false });
  });
});

test("setConfigValue sets delegate.enabled when delegate is an object, keeping its other fields", async () => {
  await withHome(async (home, cwd) => {
    await mkdir(path.dirname(globalFile(home)), { recursive: true });
    await writeFile(globalFile(home), JSON.stringify({ delegate: { maxDepth: 2, enabled: true } }));
    const r = await setConfigValue("delegate", false, "global", cwd);
    assert.ok(r.ok);
    assert.deepEqual(JSON.parse(await readFile(globalFile(home), "utf8")), { delegate: { maxDepth: 2, enabled: false } });
  });
});

test("setConfigValue refuses to rewrite repaired or broken JSON and leaves it untouched", async () => {
  await withHome(async (home, cwd) => {
    await mkdir(path.dirname(globalFile(home)), { recursive: true });
    const repaired = '{ rules: true, }';
    await writeFile(globalFile(home), repaired);
    const r1 = await setConfigValue("rules", false, "global", cwd);
    assert.equal(r1.ok, false);
    assert.equal(await readFile(globalFile(home), "utf8"), repaired);

    const broken = '{ "rules": ';
    await writeFile(globalFile(home), broken);
    const r2 = await setConfigValue("rules", false, "global", cwd);
    assert.equal(r2.ok, false);
    assert.equal(await readFile(globalFile(home), "utf8"), broken);
  });
});

test("--project writes the project acp.json, not the global one", async () => {
  await withHome(async (home, cwd) => {
    const r = await setConfigValue("delegate", false, "project", cwd);
    assert.ok(r.ok);
    assert.deepEqual(JSON.parse(await readFile(projectFile(cwd), "utf8")), { delegate: false });
    await assert.rejects(readFile(globalFile(home), "utf8"));
  });
});

function captureApi() {
  const handlers = new Map<string, ((event: unknown, ctx: unknown) => unknown)[]>();
  const tools: { name: string }[] = [];
  let active: string[] = [];
  const api = {
    on(event: string, handler: (e: unknown, ctx: unknown) => unknown) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
    commands: new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>(),
    registerTool(tool: { name: string }) {
      tools.push(tool);
      active.push(tool.name);
    },
    registerCommand(name: string, options: { handler: (args: string, ctx: unknown) => Promise<void> }) {
      this.commands.set(name, options);
    },
    registerShortcut() {},
    getActiveTools: () => [...active],
    setActiveTools: (names: string[]) => {
      active = [...names];
    },
  };
  return { api, handlers, tools, active: () => active };
}

test("/acp-set rules on|off registers and toggles the rule tool live", async () => {
  await withHome(async (_home, cwd) => {
    const { api, tools, active } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false, delegate: false })(api as never);
    const notes: string[] = [];
    const stateFile = tmpPath("acp-set-live.session.json");
    const ctx = {
      cwd,
      mode: "rpc",
      hasUI: false,
      ui: { notify: (m: string) => notes.push(m), setStatus: () => {} },
      model: { contextWindow: 200_000 },
      sessionManager: { getBranch: () => [], getSessionId: () => "acp-set-live", getSessionFile: () => stateFile },
    };
    const cmd = api.commands.get("acp-set");
    assert.ok(cmd, "acp-set registered");
    assert.equal(tools.some((t) => t.name === RULE_TOOL_NAME), false);

    await cmd.handler("rules on", ctx);
    assert.ok(tools.some((t) => t.name === RULE_TOOL_NAME), "rule tool registered live");
    assert.ok(active().includes(RULE_TOOL_NAME));

    await cmd.handler("rules off", ctx);
    assert.equal(active().includes(RULE_TOOL_NAME), false, "rule tool deactivated");
    assert.equal(tools.filter((t) => t.name === RULE_TOOL_NAME).length, 1, "never registered twice");

    await cmd.handler("rules on", ctx);
    assert.ok(active().includes(RULE_TOOL_NAME), "reactivated");
    assert.equal(tools.filter((t) => t.name === RULE_TOOL_NAME).length, 1);

    notes.length = 0;
    await cmd.handler("", ctx);
    assert.match(notes[0] ?? "", /rules: on/);
    assert.match(notes[0] ?? "", /delegate: off/);
    await rm(`${stateFile}.acp.json`, { force: true });
  });
});
