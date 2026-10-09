import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { createAcpExtension } from "../src/index.js";
import { userConfigPath } from "../src/config-dir.js";
import {
  applySqueeze,
  findCandidates,
  formatSqueezeStatus,
  resolveSqueeze,
  squeezeActiveFor,
  summarizeCandidates,
  T0Store,
  contentHash,
} from "../src/squeeze.js";
import { parseSqueezeCommand } from "../src/squeeze-commands.js";

type HomeEnv = Record<"HOME" | "USERPROFILE" | "BILLION_CONTEXT_NATIVE" | "BILLION_CONTEXT_PROXY", string | undefined>;

async function withHome<T>(fn: (home: string, cwd: string) => Promise<T>): Promise<T> {
  const saved: HomeEnv = {
    HOME: process.env.HOME,
    USERPROFILE: process.env.USERPROFILE,
    BILLION_CONTEXT_NATIVE: process.env.BILLION_CONTEXT_NATIVE,
    BILLION_CONTEXT_PROXY: process.env.BILLION_CONTEXT_PROXY,
  };
  delete process.env.BILLION_CONTEXT_NATIVE;
  delete process.env.BILLION_CONTEXT_PROXY;
  const home = await mkdtemp(path.join(tmpdir(), "acp-sq-home-"));
  const cwd = await mkdtemp(path.join(tmpdir(), "acp-sq-cwd-"));
  process.env.HOME = home;
  process.env.USERPROFILE = home;
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

const BIG = (label: string) => `${label} ` + "lorem ipsum dolor sit amet ".repeat(120);

function asst(callId: string, name = "bash") {
  return { role: "assistant", content: [{ type: "toolCall", id: callId, name, arguments: { command: `echo ${callId}` } }], timestamp: 0 } as any;
}
function result(callId: string, text: string, name = "bash", isError = false) {
  return { role: "toolResult", toolCallId: callId, toolName: name, content: [{ type: "text", text }], isError, timestamp: 0 } as any;
}
function user(text: string) {
  return { role: "user", content: text, timestamp: 0 } as any;
}

function conversation(turns: number) {
  const messages: any[] = [user("goal: inspect repo")];
  const ids: (string | undefined)[] = [undefined];
  for (let i = 0; i < turns; i++) {
    messages.push(asst(`c${i}`));
    ids.push(undefined);
    messages.push(result(`c${i}`, BIG(`out${i}`)));
    ids.push(`m0000${i}`);
  }
  return { messages, ids };
}

const ON = resolveSqueeze({ enabled: true, compressorModel: "cheap/mini" });

test("resolveSqueeze: defaults off, validates fields", () => {
  const d = resolveSqueeze(undefined);
  assert.equal(d.enabled, false);
  assert.equal(d.compressorModel, "");
  const c = resolveSqueeze({ enabled: true, compressorModel: " p/m ", minChars: -1, maxRatio: 2, concurrency: 0, promptStyle: "pi", targetModels: ["a*", 3] });
  assert.equal(c.compressorModel, "p/m");
  assert.equal(c.minChars, 1500);
  assert.equal(c.maxRatio, 0.6);
  assert.equal(c.concurrency, 4);
  assert.equal(c.promptStyle, "pi");
  assert.deepEqual(c.targetModels, ["a*"]);
});

test("squeezeActiveFor: disabled, unset, same model, targets", () => {
  assert.equal(squeezeActiveFor(resolveSqueeze({ enabled: false, compressorModel: "a/b" }), "x/y"), false);
  assert.equal(squeezeActiveFor(resolveSqueeze({ enabled: true }), "x/y"), false);
  assert.equal(squeezeActiveFor(ON, "cheap/mini"), false);
  assert.equal(squeezeActiveFor(ON, "x/y"), true);
  const t = resolveSqueeze({ enabled: true, compressorModel: "cheap/mini", targetModels: ["anthropic/*"] });
  assert.equal(squeezeActiveFor(t, "anthropic/opus"), true);
  assert.equal(squeezeActiveFor(t, "openai/gpt"), false);
});

test("findCandidates excludes first, recent, small, error, protected, covered and ACP tools", () => {
  const { messages, ids } = conversation(6);
  messages.push(asst("cx", "compress"), result("cx", BIG("compress"), "compress"));
  ids.push(undefined, "m00090");
  messages[4] = result("c1", "tiny");
  messages[6] = result("c2", BIG("err"), "bash", true);
  const { candidates } = findCandidates({ messages, ids, covered: new Set(["m00003"]), protectedTools: [], cfg: ON });
  assert.deepEqual(candidates.map((c) => c.id), ["m00004"], "c1 small, c2 error, c3 covered; trailing compress turn keeps c5 recent");
  const r2 = findCandidates({ messages, ids, covered: new Set(), protectedTools: [], cfg: ON });
  assert.deepEqual(r2.candidates.map((c) => c.id), ["m00003", "m00004"]);
  const r3 = findCandidates({ messages, ids, covered: new Set(), protectedTools: ["bash"], cfg: ON });
  assert.deepEqual(r3.candidates, []);
  assert.equal(r2.userGoal, "goal: inspect repo");
});

test("T0 store: summaries keyed by stable message id, raw output preserved, reload and recovery", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "acp-t0-"));
  try {
    const { messages, ids } = conversation(6);
    const store = new T0Store("sess/1", root);
    await store.load();
    const found = findCandidates({ messages, ids, covered: new Set(), protectedTools: [], cfg: ON });
    let calls = 0;
    const r = await summarizeCandidates(found.candidates, found.calls, found.userGoal, ON, store, async (req) => {
      calls++;
      assert.match(req.output, /^out\d/);
      return `summary of ${req.output.slice(0, 4)}`;
    }, { retryDelayMs: 0 });
    assert.equal(r.done, found.candidates.length);
    assert.equal(calls, found.candidates.length);
    const entry = store.entries()["m00001"];
    assert.ok(entry);
    assert.equal(entry.summary, "summary of out1");
    assert.equal(await readFile(entry.rawFile, "utf8"), BIG("out1"));

    const reloaded = new T0Store("sess/1", root);
    await reloaded.load();
    assert.ok(reloaded.get("m00001", contentHash(BIG("out1"))));
    assert.equal(reloaded.get("m00001", contentHash("changed")), undefined, "content change invalidates");
    await rm(entry.rawFile);
    assert.equal(reloaded.get("m00001", contentHash(BIG("out1"))), undefined, "missing raw file invalidates");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("summarizeCandidates retries, rejects oversize summaries, records errors", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "acp-t0-"));
  try {
    const { messages, ids } = conversation(5);
    const store = new T0Store("s", root);
    const found = findCandidates({ messages, ids, covered: new Set(), protectedTools: [], cfg: ON });
    assert.equal(found.candidates.length, 2);
    let n = 0;
    const r = await summarizeCandidates(found.candidates, found.calls, "", { ...ON, concurrency: 1 }, store, async (req) => {
      n++;
      if (req.output.startsWith("out1") && n < 3) throw new Error("flaky");
      if (req.output.startsWith("out2")) return "x".repeat(req.output.length);
      return "ok";
    }, { retryDelayMs: 0 });
    assert.equal(r.done, 1);
    assert.ok(store.entries()["m00001"]);
    assert.equal(store.entries()["m00002"], undefined);
    const again = await summarizeCandidates(found.candidates, found.calls, "", ON, store, async () => assert.fail("no retry of cached/failed"), { retryDelayMs: 0 });
    assert.equal(again.done, 0);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("applySqueeze keeps the ACP ref tag stable and skips covered messages", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "acp-t0-"));
  try {
    const tag = "\x3cacp tokens=\"700\" type=\"bash\"\x3em00001\x3c/acp\x3e";
    const raw = BIG("out1");
    const msgs: any[] = [asst("c1"), { ...result("c1", raw), content: [{ type: "text", text: `${raw}\n\n${tag}` }] }];
    const ids = [undefined, "m00001"];
    const store = new T0Store("s", root);
    await store.put("m00001", "bash", raw, "short summary");
    const out = applySqueeze(msgs, ids, new Set(), store);
    assert.equal(out.applied, 1);
    const text = out.messages[1].content[0].text as string;
    assert.match(text, /^\[acp-squeeze: this bash output/);
    assert.match(text, /short summary/);
    assert.match(text, /\x3cacp tokens="[^"]+" type="bash"\x3em00001\x3c\/acp\x3e$/);
    assert.ok(out.tokensSaved > 0);
    assert.equal(msgs[1].content[0].text, `${raw}\n\n${tag}`, "input not mutated");
    const covered = applySqueeze(msgs, ids, new Set(["m00001"]), store);
    assert.equal(covered.applied, 0, "ACP block coverage takes precedence over T0");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("formatSqueezeStatus", () => {
  assert.equal(formatSqueezeStatus("openai/gpt-mini", 12_345, 42.4, (n) => `${Math.round(n / 1000)}k`), "squeeze:gpt-mini \u25bc12k (42%)");
  assert.equal(formatSqueezeStatus("m", 0, null, String), "squeeze:m \u25bc0 (?)");
});

test("parseSqueezeCommand", () => {
  assert.deepEqual(parseSqueezeCommand("").op, { kind: "status" });
  assert.deepEqual(parseSqueezeCommand("on").op, { kind: "toggle", on: true });
  assert.equal(parseSqueezeCommand("off --project").scope, "project");
  assert.deepEqual(parseSqueezeCommand("model a/b").op, { kind: "model", value: "a/b" });
  assert.equal(parseSqueezeCommand("model nope").op.kind, "error");
  assert.deepEqual(parseSqueezeCommand("targets a/*, b/c").op, { kind: "targets", value: ["a/*", "b/c"] });
  assert.deepEqual(parseSqueezeCommand("targets none").op, { kind: "targets", value: [] });
  assert.deepEqual(parseSqueezeCommand("style pi").op, { kind: "style", value: "pi" });
  assert.equal(parseSqueezeCommand("style fancy").op.kind, "error");
  assert.deepEqual(parseSqueezeCommand("set minChars 2000").op, { kind: "set", key: "minChars", value: 2000 });
  assert.equal(parseSqueezeCommand("set minChars -5").op.kind, "error");
  assert.equal(parseSqueezeCommand("set bogus 1").op.kind, "error");
  assert.equal(parseSqueezeCommand("compact").op.kind, "error", "compaction is not ported");
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

function entriesFor(turns: number) {
  const entries: any[] = [{ type: "message", id: "u0", parentId: null, timestamp: "", message: user("goal: inspect repo") }];
  for (let i = 0; i < turns; i++) {
    entries.push({ type: "message", id: `a${i}`, parentId: null, timestamp: "", message: asst(`c${i}`) });
    entries.push({ type: "message", id: `r${i}`, parentId: null, timestamp: "", message: result(`c${i}`, BIG(`out${i}`)) });
  }
  return entries;
}

function ctxFor(entries: any[], cwd: string, stateFile: string, statuses: Map<string, string | undefined>, completes: { n: number }) {
  return {
    mode: "rpc",
    hasUI: true,
    cwd,
    ui: { notify: () => {}, confirm: async () => true, select: async () => undefined, input: async () => "", setStatus: (k: string, v: string | undefined) => statuses.set(k, v), setWidget: () => {}, theme: { fg: (_c: string, s: string) => s } },
    model: { contextWindow: 200_000, id: "big", provider: "main" },
    getContextUsage: () => null,
    modelRegistry: {
      find: (p: string, id: string) => (p === "cheap" && id === "mini" ? { provider: p, id } : undefined),
      complete: async (_m: unknown, c: any) => {
        completes.n++;
        const prompt: string = c.messages[0].content[0].text;
        const m = /out\d+/.exec(prompt);
        return { stopReason: "stop", content: [{ type: "text", text: `S(${m?.[0]})` }] };
      },
      getAvailable: () => [],
    },
    sessionManager: { buildContextEntries: () => entries, getSessionId: () => "sq-e2e", getSessionFile: () => stateFile },
  };
}

async function runContext(handlers: Map<string, any[]>, ctx: any) {
  let out: any;
  for (const h of handlers.get("context") ?? []) {
    const r = await h({ type: "context", messages: [] }, ctx);
    if (r) out = r;
  }
  return out;
}

const textOfResult = (out: any, callId: string): string => {
  const m = out.messages.find((x: any) => x.role === "toolResult" && x.toolCallId === callId);
  return m.content.map((b: any) => b.text ?? "").join("");
};

test("e2e: squeeze disabled leaves the ACP output identical and creates no T0 cache", async () => {
  await withHome(async (home, cwd) => {
    const entries = entriesFor(6);
    const stateFile = path.join(cwd, "s.jsonl");
    const statuses = new Map<string, string | undefined>();
    const completes = { n: 0 };
    const baseline = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(baseline.api as any);
    const a = await runContext(baseline.handlers, ctxFor(entries, cwd, stateFile, statuses, completes));
    await rm(`${stateFile}.acp.json`, { force: true });
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ squeeze: { enabled: false, compressorModel: "cheap/mini" } }));
    const off = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(off.api as any);
    const b = await runContext(off.handlers, ctxFor(entries, cwd, stateFile, statuses, completes));
    assert.deepEqual(b.messages, a.messages);
    assert.equal(completes.n, 0);
    assert.equal(statuses.get("acp-squeeze"), undefined);
    assert.equal(existsSync(path.join(home, ".cache", "pi", "acp-squeeze")), false);
  });
});

test("e2e: squeeze summarizes asynchronously, applies on a later request, keeps tags, shows status", async () => {
  await withHome(async (home, cwd) => {
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ squeeze: { enabled: true, compressorModel: "cheap/mini" } }));
    const entries = entriesFor(6);
    const stateFile = path.join(cwd, "s.jsonl");
    const statuses = new Map<string, string | undefined>();
    const completes = { n: 0 };
    const { api, handlers } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(api as any);
    const ctx = ctxFor(entries, cwd, stateFile, statuses, completes);
    const first = await runContext(handlers, ctx);
    assert.match(statuses.get("acp-squeeze") ?? "", /^squeeze:mini \u25bc0 \(/, "nothing pruned before any summary or ACP block");
    assert.doesNotMatch(textOfResult(first, "c2"), /acp-squeeze/, "first request is not rewritten");
    const tagBefore = /\x3cacp [^>]*\x3e(m\d{5})\x3c\/acp\x3e/.exec(textOfResult(first, "c2"))?.[1];
    assert.ok(tagBefore);
    for (let i = 0; i < 50 && completes.n < 3; i++) await new Promise((r) => setTimeout(r, 10));
    await new Promise((r) => setTimeout(r, 50));
    assert.equal(completes.n, 3, "c1..c3 eligible (first turn and last 2 kept)");
    const second = await runContext(handlers, ctx);
    const t2 = textOfResult(second, "c2");
    assert.match(t2, /^\[acp-squeeze: this bash output/);
    assert.match(t2, /S\(out2\)/);
    assert.equal(/\x3cacp [^>]*\x3e(m\d{5})\x3c\/acp\x3e/.exec(t2)?.[1], tagBefore, "ref id stable");
    assert.doesNotMatch(textOfResult(second, "c0"), /acp-squeeze/);
    assert.doesNotMatch(textOfResult(second, "c5"), /acp-squeeze/);
    assert.match(statuses.get("acp-squeeze") ?? "", /^squeeze:mini \u25bc\S+ \(\d+%\)$/);
    const pruned = /\u25bc(\S+)/.exec(statuses.get("acp-squeeze") ?? "")?.[1] ?? "0";
    const prunedN = pruned.endsWith("k") ? Number(pruned.slice(0, -1)) * 1000 : Number(pruned);
    const saved3 = [BIG("out1"), BIG("out2"), BIG("out3")].reduce((s, t) => s + t.length, 0) / 4;
    assert.ok(prunedN > saved3 * 0.5 && prunedN < saved3 * 1.2, `pruned ${prunedN} should approximate the three replaced outputs (~${saved3})`);
    const t0 = JSON.parse(await readFile(path.join(home, ".cache", "pi", "acp-squeeze", "sq-e2e", "t0.json"), "utf8"));
    assert.equal(Object.keys(t0.t0).length, 3, JSON.stringify(Object.keys(t0.t0)) + " tag=" + tagBefore);
    assert.deepEqual(Object.keys(t0.t0).sort(), ["r1", "r2", "r3"], "keyed by stable session message id");
    const rawFile = t0.t0["r2"].rawFile;
    assert.equal(await readFile(rawFile, "utf8"), BIG("out2"));
    for (const h of handlers.get("session_shutdown") ?? []) await h({ type: "session_shutdown" }, ctx);
  });
});

test("e2e: refused (proxied) host bypasses squeeze", async () => {
  await withHome(async (home, cwd) => {
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ squeeze: { enabled: true, compressorModel: "cheap/mini" } }));
    const statuses = new Map<string, string | undefined>();
    const completes = { n: 0 };
    const { api, handlers } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(api as any);
    const ctx = ctxFor(entriesFor(6), cwd, path.join(cwd, "s.jsonl"), statuses, completes);
    (ctx.model as any).baseUrl = "http://127.0.0.1:1/bili/https://api.example.com/v1";
    const out = await runContext(handlers, ctx);
    assert.equal(out, undefined);
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(completes.n, 0);
    assert.equal(statuses.get("acp-squeeze"), undefined);
  });
});

test("/acp-squeeze writes under acp.json squeeze, preserves other keys, live reloads", async () => {
  await withHome(async (home, cwd) => {
    const cfgFile = userConfigPath(home, "acp.json");
    await mkdir(path.dirname(cfgFile), { recursive: true });
    await writeFile(cfgFile, JSON.stringify({ autoUpdate: false, squeeze: { minChars: 999 } }, null, 2) + "\n");
    const { api, handlers } = captureApi();
    createAcpExtension({ modelContextLimit: 200_000, autoUpdate: false })(api as any);
    const notes: string[] = [];
    const statuses = new Map<string, string | undefined>();
    const ctx: any = { ...ctxFor(entriesFor(1), cwd, path.join(cwd, "s.jsonl"), statuses, { n: 0 }), ui: { notify: (m: string) => notes.push(m), setStatus: (k: string, v: string | undefined) => statuses.set(k, v), select: async () => undefined, input: async () => undefined } };
    await runContext(handlers, ctx);
    const cmd = api.commands.get("acp-squeeze");
    await cmd.handler("model cheap/mini", ctx);
    await cmd.handler("on", ctx);
    const saved = JSON.parse(await readFile(cfgFile, "utf8"));
    assert.deepEqual(saved, { autoUpdate: false, squeeze: { minChars: 999, compressorModel: "cheap/mini", enabled: true } });
    assert.match(statuses.get("acp-squeeze") ?? "", /^squeeze:mini/);
    await cmd.handler("status", ctx);
    assert.match(notes.at(-1) ?? "", /acp-squeeze: on[\s\S]*minChars=999/);
    await cmd.handler("off", ctx);
    assert.equal(statuses.get("acp-squeeze"), undefined);
    await writeFile(cfgFile, "{ bad json,, }");
    await cmd.handler("on", ctx);
    assert.match(notes.at(-1) ?? "", /invalid JSON|repaired|refuses/);
    assert.equal(await readFile(cfgFile, "utf8"), "{ bad json,, }");
  });
});
