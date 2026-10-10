import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createAcpExtension } from "../src/index.js";
import { userConfigPath } from "../src/config-dir.js";
import { applySuggestToPayload, evaluateSuggest, SUGGEST_DIRECTIVE, newSuggestState, parseSuggestCommand, parseThreshold } from "../src/suggest.js";

test("parseThreshold: plain, k/m suffix, separators, off", () => {
  assert.equal(parseThreshold("120000"), 120_000);
  assert.equal(parseThreshold("120k"), 120_000);
  assert.equal(parseThreshold("1.5M"), 1_500_000);
  assert.equal(parseThreshold("120_000"), 120_000);
  assert.equal(parseThreshold("0"), null);
  assert.equal(parseThreshold("off"), null);
  assert.equal(parseThreshold("-5"), undefined);
  assert.equal(parseThreshold("lots"), undefined);
});

test("parseSuggestCommand", () => {
  assert.deepEqual(parseSuggestCommand(""), { kind: "show" });
  assert.deepEqual(parseSuggestCommand("100k --save"), { kind: "set", threshold: 100_000, save: true });
  assert.deepEqual(parseSuggestCommand("off"), { kind: "set", threshold: null, save: false });
  assert.equal(parseSuggestCommand("1 2").kind, "error");
  assert.equal(parseSuggestCommand("abc").kind, "error");
});

test("evaluateSuggest: activation, hysteresis, compress release, disable", () => {
  const st = newSuggestState(100);
  assert.equal(evaluateSuggest(st, 99, false), null);
  assert.equal(evaluateSuggest(st, 100, false), "activated");
  assert.equal(evaluateSuggest(st, 85, false), null, "stays active within the 80% band");
  assert.equal(evaluateSuggest(st, 79, false), "released-below");
  assert.equal(evaluateSuggest(st, 120, false), "activated");
  assert.equal(evaluateSuggest(st, 120, true), "released-compressed");
  assert.equal(st.active, false);
  assert.equal(evaluateSuggest(st, 120, false), "activated", "re-arms next request if still above");
  st.threshold = null;
  assert.equal(evaluateSuggest(st, 500, false), "disabled");
  assert.equal(evaluateSuggest(st, 500, false), null);
});

test("applySuggestToPayload: anthropic, openai chat, responses, gemini, bedrock shapes; does not mutate", () => {
  const anth = { tools: [{ name: "read", description: "r", input_schema: {} }, { name: "compress", description: "Compress ranges", input_schema: {} }] };
  const a = applySuggestToPayload(anth);
  assert.ok(a.hit);
  const t = (a.payload as typeof anth).tools[1]!;
  assert.ok(t.description.startsWith(SUGGEST_DIRECTIVE));
  assert.ok(t.description.endsWith("Compress ranges"));
  assert.equal(anth.tools[1]!.description, "Compress ranges");
  assert.equal(applySuggestToPayload(a.payload).payload !== a.payload, true);
  const twice = (applySuggestToPayload(a.payload).payload as typeof anth).tools[1]!.description;
  assert.equal(twice.split(SUGGEST_DIRECTIVE).length, 2, "idempotent");

  const oai = { tools: [{ type: "function", function: { name: "compress", description: "d", parameters: {} } }] };
  assert.match(JSON.stringify(applySuggestToPayload(oai).payload), new RegExp(SUGGEST_DIRECTIVE));
  const resp = { tools: [{ type: "function", name: "compress", description: "d", parameters: {} }] };
  assert.ok(applySuggestToPayload(resp).hit);
  const gem = { config: { tools: [{ functionDeclarations: [{ name: "compress", description: "d" }] }] } };
  assert.ok(applySuggestToPayload(gem).hit);
  const bed = { toolConfig: { tools: [{ toolSpec: { name: "compress", description: "d" } }] } };
  assert.ok(applySuggestToPayload(bed).hit);
  assert.equal(applySuggestToPayload({ tools: [{ name: "read", description: "r", input_schema: {} }] }).hit, false);
  assert.equal(applySuggestToPayload("x").hit, false);
});

function captureApi() {
  const handlers = new Map<string, ((event: any, ctx: any) => any)[]>();
  const api = {
    on(event: string, handler: (e: any, ctx: any) => any) {
      const list = handlers.get(event) ?? [];
      list.push(handler);
      handlers.set(event, list);
    },
    tools: [] as any[],
    commands: new Map<string, any>(),
    registerTool(tool: any) { this.tools.push(tool); },
    registerCommand(name: string, options: any) { this.commands.set(name, options); },
    getActiveTools: () => [],
    setActiveTools: () => {},
    getAllTools: () => [],
  };
  return { api, handlers };
}

async function withHome<T>(fn: (home: string, cwd: string) => Promise<T>): Promise<T> {
  const keys = ["HOME", "USERPROFILE", "BILLION_CONTEXT_NATIVE", "BILLION_CONTEXT_PROXY"] as const;
  const saved = Object.fromEntries(keys.map((k) => [k, process.env[k]]));
  const home = await mkdtemp(path.join(tmpdir(), "acp-suggest-home-"));
  const cwd = await mkdtemp(path.join(tmpdir(), "acp-suggest-cwd-"));
  process.env.HOME = home;
  process.env.USERPROFILE = home;
  delete process.env.BILLION_CONTEXT_NATIVE;
  delete process.env.BILLION_CONTEXT_PROXY;
  try {
    return await fn(home, cwd);
  } finally {
    for (const [k, v] of Object.entries(saved)) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    await rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
    await rm(cwd, { recursive: true, force: true, maxRetries: 10, retryDelay: 50 });
  }
}

const text = (n: number) => "word ".repeat(n);

function ctxFor(entries: any[], cwd: string, notes: string[]) {
  return {
    mode: "rpc",
    hasUI: false,
    cwd,
    ui: { notify: (m: string) => notes.push(m), confirm: async () => true, select: async () => undefined, input: async () => "", setStatus: () => {} },
    model: { contextWindow: 200_000, id: "test-model", provider: "p", api: "anthropic-messages" },
    getContextUsage: () => null,
    sessionManager: { buildContextEntries: () => entries, getSessionId: () => "force-e2e", getSessionFile: () => path.join(cwd, "s.jsonl") },
  };
}

const payload = () => ({ tools: [{ name: "compress", description: "Compress.", input_schema: {} }] });

async function request(handlers: Map<string, any[]>, ctx: any) {
  for (const h of handlers.get("context") ?? []) await h({ type: "context", messages: [] }, ctx);
  let out: unknown;
  for (const h of handlers.get("before_provider_request") ?? []) {
    const r = await h({ type: "before_provider_request", payload: out ?? payload() }, ctx);
    if (r !== undefined) out = r;
  }
  return out;
}

const forced = (p: unknown) => JSON.stringify(p ?? {}).includes(SUGGEST_DIRECTIVE);

test("e2e /acp-suggest: injects above threshold, releases after compress, off disables, --save persists", async () => {
  await withHome(async (home, cwd) => {
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ autoUpdate: false }) + "\n");
    const entries: any[] = [{ type: "message", id: "u0", parentId: null, timestamp: "", message: { role: "user", content: text(3000), timestamp: 0 } }];
    const { api, handlers } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(api as any);
    const notes: string[] = [];
    const ctx = ctxFor(entries, cwd, notes);
    const cmd = api.commands.get("acp-suggest");

    assert.equal(forced(await request(handlers, ctx)), false, "no threshold: nothing injected");
    await cmd.handler("1k", ctx);
    assert.equal(forced(await request(handlers, ctx)), true, "above threshold: injected");
    await cmd.handler("", ctx);
    assert.match(notes.at(-1) ?? "", /threshold: 1k \(session\)[\s\S]*active: yes/);

    for (const h of handlers.get("tool_result") ?? []) {
      await h({ type: "tool_result", toolName: "compress", toolCallId: "x", input: {}, isError: false, content: [{ type: "text", text: "▣ ACP | 3.0K → 0.2K tokens (~2.8K reclaimed, 1 block)" }] }, ctx);
    }
    const afterCompress = await request(handlers, ctx);
    assert.equal(forced(afterCompress), false, "released for the request after a successful compress");
    assert.equal(forced(await request(handlers, ctx)), true, "re-arms on the next request if still above");

    await cmd.handler("off", ctx);
    assert.equal(forced(await request(handlers, ctx)), false);

    await cmd.handler("50k --save", ctx);
    assert.deepEqual(JSON.parse(await readFile(cfgFile, "utf8")), { autoUpdate: false, suggestThreshold: 50_000 });
    assert.equal(forced(await request(handlers, ctx)), false, "below 50k");
    await cmd.handler("bogus", ctx);
    assert.match(notes.at(-1) ?? "", /invalid threshold/);
  });
});

test("e2e /acp-suggest: configured threshold applies, proxied host bypasses", async () => {
  await withHome(async (home, cwd) => {
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ suggestThreshold: 500 }));
    const entries: any[] = [{ type: "message", id: "u0", parentId: null, timestamp: "", message: { role: "user", content: text(3000), timestamp: 0 } }];
    const { api, handlers } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(api as any);
    const ctx = ctxFor(entries, cwd, []);
    assert.equal(forced(await request(handlers, ctx)), true);

    const p = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(p.api as any);
    const pctx: any = ctxFor(entries, cwd, []);
    pctx.model.baseUrl = "http://127.0.0.1:1/bili/https://api.example.com/v1";
    assert.equal(forced(await request(p.handlers, pctx)), false);
  });
});
