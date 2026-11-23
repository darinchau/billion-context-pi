// Issue #309: small models sometimes emit summaries whose CJK text arrived as
// literal \uXXXX sequences (double-escaped on the wire, so one JSON.parse leaves
// 6-char "\u5408" runs in the parsed string). The kernel stores and renders
// summaries verbatim, so the corruption would persist into every future prompt
// and re-contaminate tier-2/tier-3 distillations. Normalize at ingest instead.

const UNICODE_ESCAPE_RE = /\\u[0-9a-fA-F]{4}/g;

export function countUnicodeEscapes(s: string): number {
  const m = s.match(UNICODE_ESCAPE_RE);
  return m ? m.length : 0;
}

// Decodes ONLY \uXXXX runs (incl. surrogate pairs) — \n, \\ etc. stay literal,
// so legitimate escape examples in prose are never mangled below the threshold.
export function decodeUnicodeEscapes(s: string): string {
  let out = "";
  let i = 0;
  while (i < s.length) {
    if (s.charAt(i) === "\\" && s.charAt(i + 1) === "u") {
      const hex = s.slice(i + 2, i + 6);
      if (/^[0-9a-fA-F]{4}$/.test(hex)) {
        const high = parseInt(hex, 16);
        const next = s.slice(i + 6, i + 12);
        if (high >= 0xd800 && high <= 0xdbff && /^\\u[0-9a-fA-F]{4}$/.test(next)) {
          const low = parseInt(next.slice(2, 6), 16);
          if (low >= 0xdc00 && low <= 0xdfff) {
            out += String.fromCodePoint(((high - 0xd800) << 10) + (low - 0xdc00) + 0x10000);
            i += 12;
            continue;
          }
        }
        out += String.fromCharCode(high);
        i += 6;
        continue;
      }
    }
    out += s.charAt(i);
    i++;
  }
  return out;
}

// Strictly-more-than semantics: a summary legitimately quoting a few \uXXXX
// examples must survive; dense runs (>20) are the double-escape failure mode.
export const UNESCAPE_THRESHOLD = 20;

export function sanitizeSummary(s: string): { text: string; unescaped: boolean } {
  if (countUnicodeEscapes(s) <= UNESCAPE_THRESHOLD) return { text: s, unescaped: false };
  return { text: decodeUnicodeEscapes(s), unescaped: true };
}

// Issue #309 phenomenon B: a hallucinated user phrase was stored as
// `user verbatim '...'` / `CURRENT TASK: user '...'`. Without an mNNNNN ref the
// quote is unverifiable and later readers treat it as fact — the loop amplifier.
// Detection only: callers log evidence ([warn] summary-unverifiable-quote),
// they NEVER rewrite the model's text.
const USER_QUOTE_CLAIMS: RegExp[] = [
  /\buser\s+verbatim\b/i,
  /\bverbatim\s+user\b/i,
  /\buser\s*[:：]\s*["'"“”‘’]/i,
  /\buser\s+["'"“”‘’]/i,
  /\buser\s+said\s+["'"“”‘’]/i,
  /["'"“”‘’][^"'\n]{1,200}["'"“”‘’]\s*\(\s*(?:from\s+)?user\s*\)/i,
  /用户(?:的)?原话/,
];

/**
 * Display-stub guard (data-loss fix).
 *
 * The host renders an already-consumed compress call as a short ellipsis-
 * clamped preview (`<~200 chars>…`) instead of the full arguments. Models
 * occasionally copy that preview back as the summary of a NEW range — observed
 * in the wild as 199-char summaries cut mid-word (e.g. "…Native image Read
 * WORKS (move-fre…", "…## Done so far…"), which then get stored as the
 * range's only remaining record. That is silent, unrecoverable data loss, so
 * such a summary is rejected instead of persisted.
 *
 * Shape rules (keep narrow — a rejection costs the model one retry):
 *   1. the kernel's hidden-summary placeholder: 190+ "S" + "…" (0.0.101+), or
 *   2. short (<= STUB_SUMMARY_CHARS) text that ends in an ellipsis, i.e. a
 *      preview cut mid-content. Real summaries are told to end with a period.
 */
/**
 * Fixed per-range summary cap.
 *
 * Measured on 3 776 successful compress ranges (1 807 calls, 286 sessions):
 * p50 1 413 / p75 2 962 / p90 5 836 / p95 8 397 / p99 13 895 / max 40 624
 * chars. 12 000 sits just under p99, so it clips ~1.6 % of ranges while
 * forcing the split into separate blocks those ranges actually need — which
 * is cheaper than one oversized summary (every active block's summary is
 * re-sent each turn).
 *
 * Deliberately NOT overridable per call: the model's self-set override was
 * the escape hatch it reached for on every hard range (9 sessions hit
 * `Summary too long` with sizes up to 40 624, and three of those had already
 * lowered the cap on themselves). A fixed number makes the failure mode
 * "split the range", not "retry larger".
 */
export const MAX_SUMMARY_CHARS = 12_000;

export const STUB_SUMMARY_CHARS = 200;

const KERNEL_PLACEHOLDER_RE = /^S{190,}\u2026$/;
const TRAILING_ELLIPSIS_RE = /(?:\u2026|\.\.\.)$/;

export function looksLikeStubSummary(summary: string): boolean {
  const text = summary.trim();
  if (text.length === 0) return false;
  if (KERNEL_PLACEHOLDER_RE.test(text)) return true;
  if (text.length > STUB_SUMMARY_CHARS) return false;
  return TRAILING_ELLIPSIS_RE.test(text);
}

const MESSAGE_REF_RE = /\bm\d{4,}\b/i;

// null when no claim matches, or when the summary carries at least one mNNNNN
// ref (quotes then point at verifiable messages).
export function findUnverifiableUserQuote(summary: string): string | null {
  if (MESSAGE_REF_RE.test(summary)) return null;
  for (const re of USER_QUOTE_CLAIMS) {
    const m = re.exec(summary);
    if (m) return m[0].slice(0, 80);
  }
  return null;
}
