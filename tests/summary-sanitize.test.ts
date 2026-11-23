import { test } from "node:test";
import assert from "node:assert/strict";
import {
  countUnicodeEscapes,
  decodeUnicodeEscapes,
  findUnverifiableUserQuote,
  looksLikeStubSummary,
  sanitizeSummary,
  STUB_SUMMARY_CHARS,
  UNESCAPE_THRESHOLD,
} from "../src/summary-sanitize.js";

// issue #309: small models emit summaries whose CJK text arrives as literal
// \uXXXX runs (double-escaped on the wire). The kernel stores/renders summaries
// verbatim, so the adapter normalizes at ingest. Also detects unverifiable
// user-quote claims (hallucinated phrases stored as fact — the loop amplifier).

test("decodeUnicodeEscapes decodes basic CJK escapes", () => {
  assert.equal(decodeUnicodeEscapes("\\u5408\\u5e76"), "合并");
});

test("decodeUnicodeEscapes decodes surrogate pairs to a single code point", () => {
  assert.equal(decodeUnicodeEscapes("\\ud83d\\ude00"), "\u{1F600}");
});

test("decodeUnicodeEscapes keeps a lone high surrogate as one char (no crash)", () => {
  const out = decodeUnicodeEscapes("\\ud800");
  assert.equal(out.length, 1);
  assert.equal(out.charCodeAt(0), 0xd800);
});

test("decodeUnicodeEscapes leaves non-unicode escapes and invalid hex untouched", () => {
  assert.equal(decodeUnicodeEscapes("a\\nb"), "a\\nb");
  assert.equal(decodeUnicodeEscapes("\\\\x"), "\\\\x");
  assert.equal(decodeUnicodeEscapes("\\uZZZZ"), "\\uZZZZ");
  assert.equal(decodeUnicodeEscapes("\\u12"), "\\u12");
});

test("decodeUnicodeEscapes mixes decoded runs with surrounding text", () => {
  assert.equal(decodeUnicodeEscapes("line1 \\u4f60\\u597d tail"), "line1 你好 tail");
});

test("countUnicodeEscapes counts all literal escape runs", () => {
  assert.equal(countUnicodeEscapes("\\u5408\\u5e76\\u7ee7"), 3);
  assert.equal(countUnicodeEscapes("plain text \\uZZZZ"), 0);
});

test("sanitizeSummary leaves clean summaries untouched", () => {
  const s = sanitizeSummary("合并了 295 继续下一个 rebase (m00012)");
  assert.equal(s.unescaped, false);
  assert.equal(s.text, "合并了 295 继续下一个 rebase (m00012)");
});

test("sanitizeSummary does NOT decode at or below the threshold (legit examples survive)", () => {
  const s = "\\u5408".repeat(UNESCAPE_THRESHOLD);
  const out = sanitizeSummary(s);
  assert.equal(out.unescaped, false);
  assert.equal(out.text, s);
});

test("sanitizeSummary decodes above the threshold (double-escape failure mode)", () => {
  const s = "摘要: " + "\\u5408".repeat(UNESCAPE_THRESHOLD + 1) + " 结束";
  const out = sanitizeSummary(s);
  assert.equal(out.unescaped, true);
  assert.equal(out.text, "摘要: " + "合".repeat(UNESCAPE_THRESHOLD + 1) + " 结束");
  assert.ok(!out.text.includes("\\u5408"));
});

test("findUnverifiableUserQuote flags the incident shape: user quote without ref", () => {
  const flagged = findUnverifiableUserQuote("CURRENT TASK: user '合并了 下一个' = proceed to NEXT open issue");
  assert.ok(flagged !== null);
});

test("findUnverifiableUserQuote flags 'user verbatim' claims without ref", () => {
  const flagged = findUnverifiableUserQuote("user verbatim 'go ahead' captured above");
  assert.ok(flagged !== null);
});

test("findUnverifiableUserQuote passes when a message ref is present", () => {
  assert.equal(findUnverifiableUserQuote("user verbatim 'go ahead' (m00012)"), null);
  assert.equal(findUnverifiableUserQuote("CURRENT TASK: user '合并了' per m00045"), null);
});

test("findUnverifiableUserQuote ignores summaries with no user-quote claim", () => {
  assert.equal(findUnverifiableUserQuote('error "ENOENT: no such file" while reading /tmp/x'), null);
  assert.equal(findUnverifiableUserQuote("merged PR #299, next: #299 open issues remain"), null);
});

test("findUnverifiableUserQuote flags Chinese 原话 claims without ref", () => {
  assert.ok(findUnverifiableUserQuote("用户原话\"合并了\"表示继续") !== null);
});

// ─── display-stub guard (silent data loss) ────────────────────────────────
// The host renders a consumed compress call as a short ellipsis-clamped
// preview; models occasionally copy that preview back as a NEW summary, which
// then becomes the range's only stored record (observed: 199-char summaries cut
// mid-word, e.g. "...(move-fre…", "...## Done so far…").

test("looksLikeStubSummary flags the observed real-world stubs", () => {
  const observed = [
    "TASK AS OF THIS BLOCK: Phase 4 census wiring — native image Read WORKS (move-fre\u2026",
    "TASK AS OF THIS BLOCK: Phase 4 census wiring in progress. Native image Read works.  ## Done so far\u2026",
  ];
  for (const s of observed) {
    assert.equal(looksLikeStubSummary(s), true, `missed stub: ${s.slice(-40)}`);
    assert.ok(s.length <= STUB_SUMMARY_CHARS);
  }
});

test("looksLikeStubSummary flags the kernel hidden-summary placeholder", () => {
  assert.equal(looksLikeStubSummary(`${"S".repeat(200)}\u2026`), true);
});

test("looksLikeStubSummary flags short previews ending in three dots", () => {
  assert.equal(looksLikeStubSummary("Merged the runner and the census pass..."), true);
});

test("looksLikeStubSummary passes real summaries", () => {
  assert.equal(looksLikeStubSummary("Fixed validateSummaryLength in src/config.ts:592 — cap is 20000 chars."), false);
  // Longer than a stub: an intentional trailing ellipsis is the model's call.
  assert.equal(looksLikeStubSummary(`${"x".repeat(STUB_SUMMARY_CHARS + 1)}\u2026`), false);
  assert.equal(looksLikeStubSummary(""), false);
});
