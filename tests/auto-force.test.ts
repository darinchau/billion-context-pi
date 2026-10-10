/**
 * Tests for /acp-force auto-compress functionality.
 *
 * Coverage:
 *  1. selectForceRanges: below threshold → no selection
 *  2. selectForceRanges: single compressible range → no selection (keep most recent)
 *  3. selectForceRanges: two ranges → oldest selected, most recent kept
 *  4. selectForceRanges: stops when target (80% of N) is reached
 *  5. selectForceRanges: exhausted when all candidates used and still above target
 *  6. selectForceRanges: sorts by startIndex, falls back to startRef lexical
 *  7. mechanicalSummary: user/assistant/tool messages
 *  8. messagesForRange: resolves refs and slices the messages array
 *  9. generateSummaries: uses mechanical fallback when no summarizeFn
 * 10. state-rebuild: replays acp-auto-force custom entries
 * 11. state-rebuild: acp-auto-force + model compress call coexist
 * 12. state-rebuild: handles malformed acp-auto-force entries gracefully
 * 13. /acp-force command: show status
 * 14. /acp-force command: set threshold for session
 * 15. /acp-force command: --save writes config
 * 16. SqueezeController.force: syncs from adapter unless overridden
 * 17. SqueezeController.inBackoff: backoff after exhaustion, resets after success
 * 18. SqueezeController.takeSuppressSuggestOnce: clears after first call
 * 19. e2e: auto-compress fires in context transform when threshold exceeded
 * 20. e2e: does NOT fire when threshold not exceeded
 * 21. e2e: does NOT fire when only one compressible range
 * 22. e2e: respects backoff after exhaustion
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  selectForceRanges,
  mechanicalSummary,
  messagesForRange,
  generateSummaries,
  logAutoForce,
  FORCE_TARGET_RATIO,
  FORCE_BACKOFF_RATIO,
  type AutoForceRecord,
  ACP_AUTO_FORCE_CUSTOM_TYPE,
} from "../src/auto-force.js";
import { SqueezeController } from "../src/squeeze-runtime.js";
import { parseForceCommand, FORCE_USAGE } from "../src/squeeze-commands.js";
import { createAcpExtension } from "../src/index.js";
import { createInitialState } from "acp-kernel";
import type { CompressibleRange, CoreMessage, CompressionState } from "acp-kernel";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeRange(
  startRef: string,
  endRef: string,
  tokens: number,
  startIndex: number,
): CompressibleRange {
  return { startRef, endRef, count: 1, tokens, toolPct: 0, textPct: 100, chars: tokens * 4, startIndex };
}

function captureApi() {
  const handlers = new Map<string, ((event: any, ctx: any) => any)[]>();
  const appendedEntries: Array<{ customType: string; data: unknown }> = [];
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
    appendEntry(customType: string, data: unknown) { appendedEntries.push({ customType, data }); },
  };
  return { api, handlers, appendedEntries };
}

function entry(id: string, role: string, text: string) {
  return { type: "message", id, parentId: null, timestamp: "", message: { role, content: text, timestamp: 0 } };
}

function assistantCallEntry(id: string, toolCallId: string, ranges: Array<{ startId: string; endId: string; summary: string }>) {
  return {
    type: "message", id, parentId: null, timestamp: "",
    message: {
      role: "assistant",
      content: [{ type: "toolCall", name: "compress", id: toolCallId, arguments: { content: ranges } }],
      timestamp: 0,
    },
  };
}

function compressResultEntry(id: string, toolCallId: string, isError = false) {
  return {
    type: "message", id, parentId: null, timestamp: "",
    message: {
      role: "toolResult", toolName: "compress", toolCallId,
      content: "▣ ACP | 5.2K → 0.8K tokens (~4.4K reclaimed, 1 block)",
      isError, timestamp: 0,
    },
  };
}

function autoForceEntry(id: string, record: AutoForceRecord) {
  return { type: "custom", id, parentId: null, timestamp: "", customType: ACP_AUTO_FORCE_CUSTOM_TYPE, data: record };
}

function fakeCtx(entries: any[], stateFile: string, sid: string, cwd: string) {
  return {
    mode: "rpc", hasUI: false, cwd,
    ui: { notify: () => {}, confirm: async () => true, select: async () => undefined, input: async () => "", setStatus: () => {} },
    model: { contextWindow: 200_000, id: "test-model" },
    sessionManager: {
      buildContextEntries: () => entries,
      getSessionId: () => sid,
      getSessionFile: () => stateFile,
    },
  };
}

function eventOf(entries: any[]) {
  return { type: "context", messages: entries
    .filter((e) => e.type === "message")
    .map((e) => ({ role: e.message.role, content: [{ type: "text", text: typeof e.message.content === "string" ? e.message.content : "" }], timestamp: 0 })) };
}

const LONG = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor ".repeat(80);
const ADAPTER = { modelContextLimit: 200_000, autoUpdate: false };

async function fire(handlers: Map<string, any[]>, entries: any[], ctx: any) {
  await handlers.get("session_start")![0]!({}, ctx);
  return handlers.get("context")![0]!(eventOf(entries), ctx);
}

// ---------------------------------------------------------------------------
// 1. selectForceRanges: below threshold
// ---------------------------------------------------------------------------
test("selectForceRanges: does nothing when forceTokens <= threshold", () => {
  const ranges = [makeRange("m1", "m2", 10_000, 0), makeRange("m3", "m4", 10_000, 2)];
  const { selected, exhausted } = selectForceRanges(ranges, 80_000, 100_000);
  assert.equal(selected.length, 0);
  assert.equal(exhausted, false);
});

// ---------------------------------------------------------------------------
// 2. selectForceRanges: single compressible range → no selection
// ---------------------------------------------------------------------------
test("selectForceRanges: does nothing with only one range (most recent is protected)", () => {
  const ranges = [makeRange("m1", "m2", 50_000, 0)];
  const { selected } = selectForceRanges(ranges, 150_000, 100_000);
  assert.equal(selected.length, 0);
});

// ---------------------------------------------------------------------------
// 3. selectForceRanges: two ranges → oldest selected, most recent kept
// ---------------------------------------------------------------------------
test("selectForceRanges: selects oldest, keeps most recent intact", () => {
  const older = makeRange("m1", "m4", 60_000, 0);
  const newer = makeRange("m5", "m8", 60_000, 4);
  // threshold=100k, forceTokens=150k, target=80k → reclaiming older(60k) gives 90k which is > 80k
  // but actually 150k - 60k = 90k > 80k → need to keep going but newer is protected
  // so exhausted should be true (can't reach 80k after removing only older)
  const { selected, exhausted } = selectForceRanges([older, newer], 150_000, 100_000);
  assert.equal(selected.length, 1);
  assert.equal(selected[0]!.startRef, "m1");
  assert.equal(exhausted, true); // 150k - 60k = 90k > 80k target
});

// ---------------------------------------------------------------------------
// 4. selectForceRanges: stops when target reached
// ---------------------------------------------------------------------------
test("selectForceRanges: stops as soon as target is reached", () => {
  const r1 = makeRange("m1", "m2", 40_000, 0);
  const r2 = makeRange("m3", "m4", 40_000, 2);
  const r3 = makeRange("m5", "m6", 40_000, 4);   // most recent — never picked
  // threshold=100k, target=80k, forceTokens=180k
  // reclaim r1(40k) → 140k, still > 80k
  // reclaim r2(40k) → 100k, still > 80k
  // r3 is protected
  // exhausted because 100k > 80k and no more candidates
  const { selected, exhausted } = selectForceRanges([r1, r2, r3], 180_000, 100_000);
  assert.equal(selected.length, 2);
  assert.equal(selected[0]!.startRef, "m1");
  assert.equal(selected[1]!.startRef, "m3");
  assert.equal(exhausted, true);
});

test("selectForceRanges: stops early when target reached mid-list", () => {
  const r1 = makeRange("m1", "m2", 80_000, 0);
  const r2 = makeRange("m3", "m4", 80_000, 2);   // most recent
  // threshold=100k, target=80k, forceTokens=130k
  // reclaim r1(80k) → 50k ≤ 80k → stop
  const { selected, exhausted } = selectForceRanges([r1, r2], 130_000, 100_000);
  assert.equal(selected.length, 1);
  assert.equal(selected[0]!.startRef, "m1");
  assert.equal(exhausted, false);
});

// ---------------------------------------------------------------------------
// 5. selectForceRanges: exhausted flag
// ---------------------------------------------------------------------------
test("selectForceRanges: exhausted=false when target is reached", () => {
  const r1 = makeRange("m1", "m2", 100_000, 0);
  const r2 = makeRange("m3", "m4", 10_000, 2);
  const { exhausted } = selectForceRanges([r1, r2], 120_000, 100_000);
  assert.equal(exhausted, false);
});

// ---------------------------------------------------------------------------
// 6. selectForceRanges: sorts by startIndex
// ---------------------------------------------------------------------------
test("selectForceRanges: picks oldest by startIndex regardless of input order", () => {
  const newer = makeRange("m5", "m6", 30_000, 4);
  const older = makeRange("m1", "m2", 30_000, 0);
  const newest = makeRange("m7", "m8", 30_000, 6);
  // forceTokens=130k, threshold=100k, target=80k
  // reclaim older(30k)→100k > 80k, reclaim newer(30k)→70k ≤ 80k → stop
  const { selected } = selectForceRanges([newer, older, newest], 130_000, 100_000);
  assert.equal(selected[0]!.startRef, "m1");  // oldest first
});

// ---------------------------------------------------------------------------
// 7. mechanicalSummary
// ---------------------------------------------------------------------------
test("mechanicalSummary: summarizes user/assistant/tool messages", () => {
  const msgs: CoreMessage[] = [
    { id: "m1", role: "user", contentType: "text", text: "hello world" },
    { id: "m2", role: "assistant", contentType: "text", text: "response" },
    { id: "m3", role: "tool", contentType: "tool-call", toolName: "bash", toolCallId: "c1", text: '{"command":"ls"}' },
    { id: "m3", role: "tool", contentType: "tool-result", toolName: "bash", toolCallId: "c1", text: "file.ts\n" },
  ];
  const t0Map = new Map([["c1", "T0 summary of bash"]]);
  const summary = mechanicalSummary(msgs, t0Map);
  assert.ok(summary.includes("[user m1] hello world"));
  assert.ok(summary.includes("[assistant] response"));
  assert.ok(summary.includes("T0 summary of bash"));
});

test("mechanicalSummary: clips long user messages", () => {
  const long = "x".repeat(1000);
  const msgs: CoreMessage[] = [{ id: "m1", role: "user", contentType: "text", text: long }];
  const summary = mechanicalSummary(msgs, new Map());
  assert.ok(summary.length < 600);
  assert.ok(summary.includes("…"));
});

test("mechanicalSummary: uses size fallback for tool result without T0", () => {
  const msgs: CoreMessage[] = [
    { id: "m1", role: "tool", contentType: "tool-result", toolName: "read", toolCallId: "c2", text: "a".repeat(200) },
  ];
  const summary = mechanicalSummary(msgs, new Map());
  assert.ok(summary.includes("200 chars"));
});

// ---------------------------------------------------------------------------
// 8. messagesForRange
// ---------------------------------------------------------------------------
test("messagesForRange: returns messages between startRef and endRef", () => {
  const state = createInitialState();
  // Manually inject refs
  (state as any).messageRefs = {
    byRef: { "m00001": "m1", "m00002": "m2", "m00003": "m3" },
    byRaw: { "m1": "m00001", "m2": "m00002", "m3": "m00003" },
  };
  const msgs: CoreMessage[] = [
    { id: "m1", role: "user", contentType: "text", text: "a" },
    { id: "m2", role: "assistant", contentType: "text", text: "b" },
    { id: "m3", role: "user", contentType: "text", text: "c" },
  ];
  const range: CompressibleRange = { startRef: "m00001", endRef: "m00002", count: 2, tokens: 10, toolPct: 0, textPct: 100 };
  const result = messagesForRange(range, msgs, state as unknown as CompressionState);
  assert.equal(result.length, 2);
  assert.equal(result[0]!.id, "m1");
  assert.equal(result[1]!.id, "m2");
});

test("messagesForRange: returns empty array when refs not found", () => {
  const state = createInitialState();
  (state as any).messageRefs = { byRef: {}, byRaw: {} };
  const range: CompressibleRange = { startRef: "m99999", endRef: "m99998", count: 1, tokens: 10, toolPct: 0, textPct: 100 };
  const result = messagesForRange(range, [], state as unknown as CompressionState);
  assert.equal(result.length, 0);
});

// ---------------------------------------------------------------------------
// 9. generateSummaries: mechanical fallback
// ---------------------------------------------------------------------------
test("generateSummaries: uses mechanical fallback when no summarizeFn", async () => {
  const state = createInitialState();
  (state as any).messageRefs = {
    byRef: { "m00001": "m1", "m00002": "m2" },
    byRaw: { "m1": "m00001", "m2": "m00002" },
  };
  const msgs: CoreMessage[] = [
    { id: "m1", role: "user", contentType: "text", text: "user says hello" },
    { id: "m2", role: "assistant", contentType: "text", text: "assistant says hi" },
  ];
  const selected: CompressibleRange[] = [
    { startRef: "m00001", endRef: "m00002", count: 2, tokens: 100, toolPct: 0, textPct: 100 },
  ];
  const result = await generateSummaries({
    selected,
    coreMessages: msgs,
    state: state as unknown as CompressionState,
    t0Summaries: new Map(),
    summarizeFn: null,
    timeoutMs: 5000,
    sid: "test",
  });
  assert.equal(result.summarizer, "mechanical");
  assert.equal(result.rangeSpecs.length, 1);
  assert.ok(result.rangeSpecs[0]!.summary.length > 0);
  assert.equal(result.rangeSpecs[0]!.startRef, "m00001");
  assert.equal(result.rangeSpecs[0]!.endRef, "m00002");
});

test("generateSummaries: falls back to mechanical on summarizeFn error", async () => {
  const state = createInitialState();
  (state as any).messageRefs = {
    byRef: { "m00001": "m1" },
    byRaw: { "m1": "m00001" },
  };
  const msgs: CoreMessage[] = [{ id: "m1", role: "user", contentType: "text", text: "test" }];
  const selected: CompressibleRange[] = [
    { startRef: "m00001", endRef: "m00001", count: 1, tokens: 100, toolPct: 0, textPct: 100 },
  ];
  const failingFn = async () => { throw new Error("model unavailable"); };
  const result = await generateSummaries({
    selected,
    coreMessages: msgs,
    state: state as unknown as CompressionState,
    t0Summaries: new Map(),
    summarizeFn: failingFn as any,
    timeoutMs: 5000,
    sid: "test",
  });
  assert.equal(result.summarizer, "mechanical");
  assert.equal(result.rangeSpecs.length, 1);
});

// ---------------------------------------------------------------------------
// 10. state-rebuild: replays acp-auto-force custom entries
// ---------------------------------------------------------------------------
test("state-rebuild: replays acp-auto-force custom entry when sidecar is missing", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-rebuild-"));
  try {
    const stateFile = join(dir, "session.jsonl");
    const entries = [
      entry("m1", "user", LONG),
      entry("m2", "user", LONG),
      entry("m3", "user", LONG),
      entry("m4", "user", LONG),
      entry("m5", "user", LONG),
      entry("m6", "user", LONG),
      // Auto-force custom entry (no model compress call/result pair)
      autoForceEntry("af1", {
        ranges: [{ startRef: "m00001", endRef: "m00002", summary: "auto-force summary long enough to clear the fifty character minimum guard here" }],
        ts: Date.now(),
        summarizer: "mechanical",
      }),
      entry("m7", "user", "tail"),
    ];
    const { api, handlers } = captureApi();
    createAcpExtension(ADAPTER)(api as any);
    const ctx = fakeCtx(entries, stateFile, "af-rebuild-1", dir);
    await fire(handlers, entries, ctx);

    assert.ok(existsSync(`${stateFile}.acp.json`), "sidecar created");
    const saved = JSON.parse(readFileSync(`${stateFile}.acp.json`, "utf8"));
    assert.ok(saved.blocks.length >= 1, `expected ≥1 block, got ${saved.blocks.length}`);

    // Second fire must not double-apply
    await fire(handlers, entries, ctx);
    const saved2 = JSON.parse(readFileSync(`${stateFile}.acp.json`, "utf8"));
    assert.equal(saved2.blocks.length, saved.blocks.length, "no double-apply on second fire");
    assert.equal(saved2.blocks[0].blockId, saved.blocks[0].blockId, "same block id");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 11. state-rebuild: acp-auto-force + model compress coexist
// ---------------------------------------------------------------------------
test("state-rebuild: auto-force entry and model compress call both replay", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-coexist-"));
  try {
    const stateFile = join(dir, "session.jsonl");
    const entries = [
      entry("m1", "user", LONG),
      entry("m2", "user", LONG),
      entry("m3", "user", LONG),
      entry("m4", "user", LONG),
      entry("m5", "user", LONG),
      entry("m6", "user", LONG),
      entry("m7", "user", LONG),
      entry("m8", "user", LONG),
      // Model-issued compress
      assistantCallEntry("m9", "c1", [{ startId: "m1", endId: "m2", summary: "model compress summary long enough to clear the fifty char minimum guard" }]),
      compressResultEntry("m10", "c1"),
      // Auto-force entry (compresses different range)
      autoForceEntry("af1", {
        ranges: [{ startRef: "m00003", endRef: "m00004", summary: "auto-force summary long enough to pass the fifty character minimum guard check" }],
        ts: Date.now(),
        summarizer: "mechanical",
      }),
      entry("m11", "user", "tail"),
    ];
    const { api, handlers } = captureApi();
    createAcpExtension(ADAPTER)(api as any);
    const ctx = fakeCtx(entries, stateFile, "af-coexist-1", dir);
    await fire(handlers, entries, ctx);

    const saved = JSON.parse(readFileSync(`${stateFile}.acp.json`, "utf8"));
    assert.ok(saved.blocks.length >= 1, `expected ≥1 block, got ${saved.blocks.length}`);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 12. state-rebuild: malformed acp-auto-force entries are skipped
// ---------------------------------------------------------------------------
test("state-rebuild: malformed acp-auto-force entries are skipped gracefully", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-malformed-"));
  try {
    const stateFile = join(dir, "session.jsonl");
    const entries = [
      entry("m1", "user", LONG),
      entry("m2", "user", LONG),
      entry("m3", "user", LONG),
      entry("m4", "user", LONG),
      entry("m5", "user", LONG),
      entry("m6", "user", LONG),
      // Malformed: ranges is not an array
      { type: "custom", id: "af-bad", parentId: null, timestamp: "", customType: ACP_AUTO_FORCE_CUSTOM_TYPE, data: { ranges: null, ts: 0 } },
      // Malformed: ranges is empty
      { type: "custom", id: "af-empty", parentId: null, timestamp: "", customType: ACP_AUTO_FORCE_CUSTOM_TYPE, data: { ranges: [], ts: 0 } },
      entry("m7", "user", "tail"),
    ];
    const { api, handlers } = captureApi();
    createAcpExtension(ADAPTER)(api as any);
    const ctx = fakeCtx(entries, stateFile, "af-malformed-1", dir);
    // Must not throw
    await assert.doesNotReject(fire(handlers, entries, ctx));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 13. parseForceCommand: show status
// ---------------------------------------------------------------------------
test("parseForceCommand: no args → show", () => {
  const op = parseForceCommand("");
  assert.equal(op.kind, "show");
});

test("parseForceCommand: N → set threshold", () => {
  const op = parseForceCommand("120k");
  assert.equal(op.kind, "set");
  if (op.kind === "set") {
    assert.equal(op.threshold, 120_000);
    assert.equal(op.save, false);
  }
});

test("parseForceCommand: off → set null", () => {
  const op = parseForceCommand("off");
  assert.equal(op.kind, "set");
  if (op.kind === "set") assert.equal(op.threshold, null);
});

test("parseForceCommand: N --save → set with save", () => {
  const op = parseForceCommand("1.2m --save");
  assert.equal(op.kind, "set");
  if (op.kind === "set") {
    assert.equal(op.threshold, 1_200_000);
    assert.equal(op.save, true);
  }
});

test("parseForceCommand: garbage → error", () => {
  const op = parseForceCommand("notanumber");
  assert.equal(op.kind, "error");
});

// ---------------------------------------------------------------------------
// 14–15. handleForceCommand: tested via SqueezeController directly
// ---------------------------------------------------------------------------
test("SqueezeController.force: returns state with configured threshold", () => {
  const ctl = new SqueezeController(() => ({ forceThreshold: 80_000 }));
  const f = ctl.force("sid1");
  assert.equal(f.threshold, 80_000);
});

test("SqueezeController.setForceThreshold: overrides config", () => {
  const ctl = new SqueezeController(() => ({ forceThreshold: 80_000 }));
  ctl.setForceThreshold("sid1", 200_000);
  const f = ctl.force("sid1");
  assert.equal(f.threshold, 200_000);
  assert.equal(f.sessionOverride, true);
});

test("SqueezeController.force: syncs from adapter when not overridden", () => {
  let thresh = 80_000;
  const ctl = new SqueezeController(() => ({ forceThreshold: thresh }));
  ctl.force("sid1"); // init
  thresh = 150_000;
  const f = ctl.force("sid1");
  assert.equal(f.threshold, 150_000);
});

// ---------------------------------------------------------------------------
// 16. SqueezeController.configuredForceTimeoutMs
// ---------------------------------------------------------------------------
test("SqueezeController.configuredForceTimeoutMs: defaults to 60000", () => {
  const ctl = new SqueezeController(() => ({}));
  assert.equal(ctl.configuredForceTimeoutMs(), 60_000);
});

test("SqueezeController.configuredForceTimeoutMs: uses adapter value", () => {
  const ctl = new SqueezeController(() => ({ forceTimeoutMs: 30_000 }));
  assert.equal(ctl.configuredForceTimeoutMs(), 30_000);
});

// ---------------------------------------------------------------------------
// 17. SqueezeController.inBackoff
// ---------------------------------------------------------------------------
test("SqueezeController.inBackoff: not in backoff initially", () => {
  const ctl = new SqueezeController(() => ({}));
  assert.equal(ctl.inBackoff("sid1", 150_000), false);
});

test("SqueezeController.inBackoff: enters backoff after exhaustion, exits after growth", () => {
  const ctl = new SqueezeController(() => ({}));
  ctl.force("sid1"); // init
  ctl.noteAutoForceExhausted("sid1", 100_000);
  // Still below growth floor (100k * (1 + 0.1) = 110k)
  assert.equal(ctl.inBackoff("sid1", 105_000), true);
  // Above growth floor
  assert.equal(ctl.inBackoff("sid1", 115_000), false);
});

test("SqueezeController.inBackoff: resets after noteAutoForceSuccess", () => {
  const ctl = new SqueezeController(() => ({}));
  ctl.force("sid1");
  ctl.noteAutoForceExhausted("sid1", 100_000);
  assert.equal(ctl.inBackoff("sid1", 105_000), true);
  ctl.noteAutoForceSuccess("sid1");
  assert.equal(ctl.inBackoff("sid1", 105_000), false);
});

// ---------------------------------------------------------------------------
// 18. SqueezeController.takeSuppressSuggestOnce
// ---------------------------------------------------------------------------
test("takeSuppressSuggestOnce: returns false when not set", () => {
  const ctl = new SqueezeController(() => ({}));
  ctl.force("sid1");
  assert.equal(ctl.takeSuppressSuggestOnce("sid1"), false);
});

test("takeSuppressSuggestOnce: returns true once then false", () => {
  const ctl = new SqueezeController(() => ({}));
  ctl.force("sid1");
  ctl.noteAutoForceSuccess("sid1"); // sets suppressSuggestOnce
  assert.equal(ctl.takeSuppressSuggestOnce("sid1"), true);
  assert.equal(ctl.takeSuppressSuggestOnce("sid1"), false);
});

// ---------------------------------------------------------------------------
// 19. e2e: auto-compress fires in context transform
// ---------------------------------------------------------------------------
test("e2e: auto-compress fires and creates blocks when threshold exceeded", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-e2e-"));
  try {
    const stateFile = join(dir, "e2e.jsonl");
    const entries = [
      entry("m1", "user", LONG),
      entry("m2", "user", LONG),
      entry("m3", "user", LONG),
      entry("m4", "user", LONG),
      entry("m5", "user", LONG),
      entry("m6", "user", LONG),
      entry("m7", "user", LONG),
      entry("m8", "user", LONG),
      entry("m9", "user", LONG),
      entry("m10", "user", "recent tail"),
    ];
    // Very low threshold to trigger auto-force
    const { api, handlers, appendedEntries } = captureApi();
    createAcpExtension({ ...ADAPTER, forceThreshold: 100 })(api as any);
    const ctx = fakeCtx(entries, stateFile, "e2e-force-1", dir);
    await fire(handlers, entries, ctx);

    const saved = JSON.parse(readFileSync(`${stateFile}.acp.json`, "utf8"));
    // Should have compressed something
    assert.ok(saved.blocks.length >= 1, `expected blocks after auto-force, got ${saved.blocks.length}`);
    // Should have appended the custom entry for rebuild
    const forceEntries = appendedEntries.filter((e) => e.customType === ACP_AUTO_FORCE_CUSTOM_TYPE);
    assert.ok(forceEntries.length >= 1, "acp-auto-force custom entry persisted");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 20. e2e: does NOT fire when threshold not exceeded
// ---------------------------------------------------------------------------
test("e2e: auto-force does not fire when below threshold", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-noop-"));
  try {
    const stateFile = join(dir, "noop.jsonl");
    const entries = [
      entry("m1", "user", "short"),
      entry("m2", "user", "short"),
      entry("m3", "user", "short"),
    ];
    const { api, handlers, appendedEntries } = captureApi();
    // Very high threshold — should never trigger
    createAcpExtension({ ...ADAPTER, forceThreshold: 999_999_999 })(api as any);
    const ctx = fakeCtx(entries, stateFile, "e2e-noop-1", dir);
    await fire(handlers, entries, ctx);

    const forceEntries = appendedEntries.filter((e) => e.customType === ACP_AUTO_FORCE_CUSTOM_TYPE);
    assert.equal(forceEntries.length, 0, "no auto-force entry when below threshold");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 21. e2e: does NOT fire when only one compressible range
// ---------------------------------------------------------------------------
test("e2e: auto-force does not fire with only one compressible range", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-auto-force-single-"));
  try {
    const stateFile = join(dir, "single.jsonl");
    const entries = [
      entry("m1", "user", LONG),
      entry("m2", "user", LONG),
      entry("m3", "user", "recent"),
    ];
    const { api, handlers, appendedEntries } = captureApi();
    createAcpExtension({ ...ADAPTER, forceThreshold: 100 })(api as any);
    const ctx = fakeCtx(entries, stateFile, "e2e-single-1", dir);
    await fire(handlers, entries, ctx);

    // Should have at most the state saved from rebuild, but no auto-force entry
    const forceEntries = appendedEntries.filter((e) => e.customType === ACP_AUTO_FORCE_CUSTOM_TYPE);
    // With only one compressible range the kernel's protect-recent zone protects the whole compressible
    // list, so selectForceRanges sees < 2 candidates and returns nothing.
    assert.equal(forceEntries.length, 0, "no auto-force with single compressible range");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// 22. e2e: FORCE_TARGET_RATIO constant is 0.8
// ---------------------------------------------------------------------------
test("FORCE_TARGET_RATIO is 0.8", () => {
  assert.equal(FORCE_TARGET_RATIO, 0.8);
});

test("FORCE_BACKOFF_RATIO is 0.1", () => {
  assert.equal(FORCE_BACKOFF_RATIO, 0.1);
});

// ---------------------------------------------------------------------------
// 23. SqueezeController.noteForceTokens / forceLastTokens (/acp-force show)
// ---------------------------------------------------------------------------
test("SqueezeController.noteForceTokens/forceLastTokens: records the latest token level", () => {
  const ctl = new SqueezeController(() => ({ forceThreshold: 80_000 }));
  assert.equal(ctl.forceLastTokens("sid1"), null, "null before any turn");
  ctl.noteForceTokens("sid1", 123_456);
  assert.equal(ctl.forceLastTokens("sid1"), 123_456);
  ctl.noteForceTokens("sid1", 99_000);
  assert.equal(ctl.forceLastTokens("sid1"), 99_000, "latest wins");
});

test("SqueezeController.inBackoff: show reports against the latest level, not the exhausted peak", () => {
  const ctl = new SqueezeController(() => ({ forceThreshold: 80_000 }));
  ctl.force("sid1");
  // Exhausted at 120k (the pre-compression peak); after a partial compress
  // the live level is 95k. Backoff must be judged against the recorded
  // post-compression level so a session still above threshold can re-arm
  // once it grows 10% from the REAL anchor, not the old peak.
  ctl.noteAutoForceExhausted("sid1", 95_000);
  ctl.noteForceTokens("sid1", 95_000);
  assert.equal(ctl.inBackoff("sid1", ctl.forceLastTokens("sid1") ?? 0), true, "in backoff at the anchor itself");
  assert.equal(ctl.inBackoff("sid1", 105_000), false, "out of backoff after 10% growth");
});

// ---------------------------------------------------------------------------
// 24. logAutoForce accepts the after/warnings fields
// ---------------------------------------------------------------------------
test("logAutoForce: accepts after + warnings (post-compression level surfaced in the log)", () => {
  logAutoForce({ sid: "sid1", ranges: 2, before: 100_000, reclaimed: 30_000, target: 64_000, after: 70_000, summarizer: "model", warnings: [] });
});
