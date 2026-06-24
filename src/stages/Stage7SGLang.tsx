/**
 * Stage 7 — SGLang vs vLLM
 *
 * The finale: a side-by-side comparison of two engines that share Stages 1–6's
 * fundamentals but diverge on two axes that matter for large MoE serving.
 *
 * Config-driven diagram component (like Stages 5/6). No engine changes.
 *
 * 1. Prefix caching — INTERACTIVE. The user toggles a set of demo requests that
 *    share a system-prompt prefix. We tokenize them (deterministic demo
 *    tokenizer) and build a radix tree. Left panel = vLLM (flat blocks + hash
 *    match, each request stores its own copy). Right panel = SGLang (radix tree,
 *    shared prefix stored once). A token-count readout shows the memory saved.
 *
 * 2. MoE serving — diagram. Left = tensor-parallel KV replicated across every
 *    rank (wasteful). Right = SGLang's DP-attention (each rank holds distinct
 *    requests' KV) + expert parallelism for the FFN.
 */

import { useState } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { tokenize } from "../engine/tokenizer";
import {
  buildRadixTree,
  flatTokenCount,
  radixTokenCount,
  isShared,
  type RadixInput,
  type RadixNode,
} from "../content/radixTree";
import { color, space, radius, font, sectionLabel, notePanel, statusTint } from "../theme";

// ─── Demo requests for the prefix-cache comparison ────────────────────────────
// A shared system prompt + per-request continuation. Toggling which are active
// changes how much prefix overlap there is.

const SHARED_SYSTEM_PROMPT = "You are a helpful assistant";

interface DemoRequest {
  id: number;
  label: string;
  /** Full prompt = shared system prompt + this continuation. */
  continuation: string;
}

const DEMO_REQUESTS: DemoRequest[] = [
  { id: 0, label: "Chat A", continuation: "summarize this article" },
  { id: 1, label: "Chat B", continuation: "translate this to French" },
  { id: 2, label: "Chat C", continuation: "summarize this report" },
];

function fullPrompt(r: DemoRequest): string {
  return `${SHARED_SYSTEM_PROMPT} ${r.continuation}`;
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const outerStyle: React.CSSProperties = {
  padding: `${space.xxl}px ${space.xl}px`,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
  display: "flex",
  flexDirection: "column",
  gap: 28,
};

const subHeaderStyle: React.CSSProperties = {
  ...sectionLabel,
  fontSize: font.size.md,
  letterSpacing: "0.08em",
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: space.sm,
  marginBottom: space.xs,
};

const compareRowStyle: React.CSSProperties = {
  display: "flex",
  gap: space.xl,
  alignItems: "stretch",
  flexWrap: "wrap",
};

function panelStyle(accent: string): React.CSSProperties {
  return {
    flex: "1 1 320px",
    minWidth: 280,
    background: color.panelBgInset,
    border: `1px solid ${accent}`,
    borderRadius: radius.lg + 2,
    padding: `14px ${space.xl}px`,
    display: "flex",
    flexDirection: "column",
    gap: space.lg,
  };
}

function panelTitleStyle(accent: string): React.CSSProperties {
  return {
    fontSize: font.size.lg,
    fontWeight: font.weight.bold,
    color: accent,
    display: "flex",
    alignItems: "center",
    gap: space.md,
  };
}

const requestToggleRowStyle: React.CSSProperties = {
  display: "flex",
  gap: space.sm,
  flexWrap: "wrap",
  marginBottom: space.xs,
};

function requestToggleStyle(active: boolean): React.CSSProperties {
  return {
    padding: `${space.xs}px 10px`,
    borderRadius: radius.sm,
    border: `1px solid ${active ? color.prefill : color.borderStrong}`,
    background: active ? `${color.prefill}22` : color.panelBgInset,
    color: active ? color.prefill : color.textFaint,
    cursor: "pointer",
    fontSize: font.size.md,
    fontFamily: font.mono,
    fontWeight: active ? font.weight.bold : font.weight.normal,
  };
}

const tokenChipStyle = (shared: boolean): React.CSSProperties => ({
  display: "inline-block",
  padding: `1px ${space.sm}px`,
  borderRadius: radius.sm,
  margin: 1,
  fontSize: font.size.sm,
  fontFamily: font.mono,
  background: shared ? `${color.accent}22` : color.panelBg,
  border: `1px solid ${shared ? color.accent : color.border}`,
  color: shared ? color.accent : color.textMuted,
});

const requestRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  gap: space.xs,
  marginBottom: space.xs,
};

const reqLabelStyle: React.CSSProperties = {
  fontSize: font.size.sm,
  fontFamily: font.mono,
  color: color.prefill,
  minWidth: 52,
  fontWeight: font.weight.bold,
};

const noteStyle: React.CSSProperties = {
  fontSize: font.size.md,
  color: color.textFaint,
  lineHeight: 1.5,
};

const savingsStyle = (good: boolean): React.CSSProperties => ({
  ...notePanel,
  marginTop: space.xs,
  padding: `${space.md}px ${space.lg}px`,
  fontSize: font.size.md,
  color: good ? color.decode : color.textMuted,
  border: good ? `1px solid ${color.decode}66` : `1px solid ${color.border}`,
  background: good ? `${color.decode}11` : color.panelBg,
  fontFamily: font.mono,
});

// ─── Radix tree rendering ─────────────────────────────────────────────────────

function RadixTreeView({ root }: { root: RadixNode }) {
  // Render children of root (root itself is an empty sentinel).
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: space.xs }}>
      {root.children.map((child) => (
        <RadixBranch key={child.id} node={child} depth={0} />
      ))}
    </div>
  );
}

function RadixBranch({ node, depth }: { node: RadixNode; depth: number }) {
  const shared = isShared(node);
  return (
    <div style={{ marginLeft: depth * 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: space.sm, flexWrap: "wrap" }}>
        <span style={{ color: color.border, fontFamily: font.mono, fontSize: font.size.sm }}>
          {depth > 0 ? "└" : "▸"}
        </span>
        {node.segment.map((tok, i) => (
          <span key={i} style={tokenChipStyle(shared)}>
            {tok}
          </span>
        ))}
        <span style={{ fontSize: font.size.xs, color: shared ? color.accent : color.textFaint }}>
          {shared ? `shared by ${node.requestIds.length}` : `req ${node.requestIds[0]}`}
        </span>
      </div>
      {node.children.map((c) => (
        <RadixBranch key={c.id} node={c} depth={depth + 1} />
      ))}
    </div>
  );
}

// ─── MoE serving comparison (static diagram) ──────────────────────────────────

const NUM_RANKS = 4;

function MoeRankCard({
  rank,
  kvLabel,
  kvAccent,
  expertLabel,
}: {
  rank: number;
  kvLabel: string;
  kvAccent: string;
  expertLabel: string;
}) {
  return (
    <div
      style={{
        flex: "1 1 100px",
        minWidth: 90,
        background: color.panelBgInset,
        border: `1px solid ${color.border}`,
        borderRadius: radius.lg,
        padding: `${space.md}px 10px`,
        display: "flex",
        flexDirection: "column",
        gap: space.sm,
      }}
    >
      <div style={{ fontSize: font.size.sm, fontWeight: font.weight.bold, color: color.textMuted }}>
        GPU {rank}
      </div>
      <div
        style={{
          fontSize: font.size.xs,
          padding: `3px ${space.sm}px`,
          borderRadius: radius.sm,
          background: `${kvAccent}22`,
          border: `1px solid ${kvAccent}`,
          color: kvAccent,
          fontFamily: font.mono,
        }}
      >
        {kvLabel}
      </div>
      <div style={{ ...statusTint(color.accent), fontSize: font.size.xs, fontFamily: font.mono, borderRadius: radius.sm }}>
        {expertLabel}
      </div>
    </div>
  );
}

// ─── Component ─────────────────────────────────────────────────────────────────

export function Stage7SGLang(_props: StageProps) {
  // Which demo requests are active in the prefix-cache comparison.
  const [activeIds, setActiveIds] = useState<number[]>([0, 1, 2]);

  function toggleRequest(id: number) {
    setActiveIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id].sort()
    );
  }

  const activeRequests = DEMO_REQUESTS.filter((r) => activeIds.includes(r.id));

  // Tokenize each active request's full prompt (deterministic demo tokenizer).
  const inputs: RadixInput[] = activeRequests.map((r) => ({
    requestId: r.id,
    tokens: tokenize(fullPrompt(r)).map((t) => t.text),
  }));

  const radixRoot = buildRadixTree(inputs);
  const flatTokens = flatTokenCount(inputs);
  const radixTokens = inputs.length > 0 ? radixTokenCount(radixRoot) : 0;
  const saved = flatTokens - radixTokens;
  const savedPct = flatTokens > 0 ? Math.round((saved / flatTokens) * 100) : 0;

  // Longest token prefix shared by ALL active requests — what both engines cache
  // once. (vLLM finds it via hashed blocks; SGLang via the radix-tree root path.)
  const sharedPrefix: string[] = [];
  if (inputs.length > 0) {
    const first = inputs[0].tokens;
    for (let i = 0; i < first.length; i++) {
      if (inputs.every((inp) => inp.tokens[i] === first[i])) {
        sharedPrefix.push(first[i]);
      } else break;
    }
  }

  return (
    <div style={outerStyle} aria-label="SGLang vs vLLM visualization">
      {/* ── Comparison 1: Prefix caching ── */}
      <div>
        <h3 style={subHeaderStyle}>
          1. <Term tokenKey="prefixCache">Prefix Caching</Term> — Hash-Matched Pool vs Radix Tree
        </h3>
        <p style={{ ...noteStyle, marginBottom: space.lg }}>
          These demo requests all begin with the same system prompt{" "}
          <span style={{ color: color.accent, fontFamily: font.mono }}>
            "{SHARED_SYSTEM_PROMPT}"
          </span>
          . Both engines reuse that shared prefix's KV (automatic prefix caching, on by default) —
          so both store the same {radixTokens} token slots. The difference is the data structure
          they use to find and share it. Toggle requests to change the overlap.
        </p>

        <div style={requestToggleRowStyle} role="group" aria-label="Toggle demo requests">
          {DEMO_REQUESTS.map((r) => (
            <button
              key={r.id}
              style={requestToggleStyle(activeIds.includes(r.id))}
              onClick={() => toggleRequest(r.id)}
              aria-pressed={activeIds.includes(r.id)}
              title={fullPrompt(r)}
            >
              {r.label}: …{r.continuation}
            </button>
          ))}
        </div>

        {inputs.length === 0 ? (
          <p style={noteStyle}>Enable at least one request to compare.</p>
        ) : (
          <div style={compareRowStyle}>
            {/* vLLM panel */}
            <div style={panelStyle(color.prefill)}>
              <div style={panelTitleStyle(color.prefill)}>vLLM — hash-matched blocks</div>
              <p style={noteStyle}>
                vLLM hashes each block of tokens. Identical leading blocks across requests get the
                same hash, so the shared prefix is stored once and reused; each request then has its
                own unique-tail blocks. Same sharing, different bookkeeping than a tree.
              </p>
              {sharedPrefix.length > 0 && (
                <div style={requestRowStyle}>
                  <span style={reqLabelStyle}>shared</span>
                  {sharedPrefix.map((tok, i) => (
                    <span key={i} style={tokenChipStyle(true)}>
                      {tok}
                    </span>
                  ))}
                  <span style={{ fontSize: font.size.xs, color: color.accent }}>
                    1 cached copy ({sharedPrefix.length} tok)
                  </span>
                </div>
              )}
              {inputs.map((inp) => {
                const tail = inp.tokens.slice(sharedPrefix.length);
                return (
                  <div key={inp.requestId} style={requestRowStyle}>
                    <span style={reqLabelStyle}>req {inp.requestId}</span>
                    {tail.length === 0 ? (
                      <span style={{ fontSize: font.size.xs, color: color.textFaint }}>(prefix only)</span>
                    ) : (
                      tail.map((tok, i) => (
                        <span key={i} style={tokenChipStyle(false)}>
                          {tok}
                        </span>
                      ))
                    )}
                  </div>
                );
              })}
              <div style={savingsStyle(saved > 0)}>
                stores {radixTokens} token slots
                {saved > 0 && ` — shared prefix cached once`}
              </div>
            </div>

            {/* SGLang panel */}
            <div style={panelStyle(color.accent)}>
              <div style={panelTitleStyle(color.accent)}>
                <Term tokenKey="radixAttention">SGLang — radix tree</Term>
              </div>
              <p style={noteStyle}>
                Same sharing, expressed as a tree: the shared prefix is one path near the root
                (highlighted), and sequences branch only at the first differing token. Prefix reuse
                is the native structure rather than a hash lookup, which also makes overlapping
                sub-branches (multi-turn chat, tree-of-thought) cheap.{" "}
                <Term tokenKey="lruEviction">LRU eviction</Term> reclaims cold leaves first.
              </p>
              <RadixTreeView root={radixRoot} />
              <div style={savingsStyle(saved > 0)}>
                stores {radixTokens} token slots
                {saved > 0 && ` — ${saved} fewer than ${flatTokens} naive copies (${savedPct}% saved)`}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Comparison 2: KV placement for large MoE ── */}
      <div>
        <h3 style={subHeaderStyle}>
          2. Serving a Large <Term tokenKey="moe">MoE</Term> — Two Independent Axes
        </h3>
        <p style={{ ...noteStyle, marginBottom: space.lg }}>
          A common confusion: <em>where KV lives</em> and <em>where experts live</em> are separate
          decisions. KV placement is set by the <strong>attention</strong> parallelism;{" "}
          <Term tokenKey="expertParallel">expert parallelism</Term> only shards the{" "}
          <strong>FFN experts</strong> and doesn't touch the KV cache. Both engines support all of
          the options below — the table contrasts the <em>choices</em>, not the engines.
        </p>

        {/* Axis A: KV placement (attention parallelism) */}
        <p style={{ ...noteStyle, color: color.textMuted, fontWeight: font.weight.bold, marginBottom: space.sm }}>
          Axis A — KV placement (attention)
        </p>
        <div style={{ ...compareRowStyle, marginBottom: space.xl }}>
          {/* TP-attention (sharded; MLA replicates) */}
          <div style={panelStyle(color.prefill)}>
            <div style={panelTitleStyle(color.prefill)}>
              <Term tokenKey="tensorParallel">TP-attention</Term>
            </div>
            <p style={noteStyle}>
              Attention heads are sharded across ranks, so normally each rank holds only{" "}
              <em>its heads'</em> slice of every request's KV — not a full copy.{" "}
              <strong>The catch is MLA</strong> (DeepSeek): its KV is one tiny per-token latent that
              can't be split across heads, so it ends up <em>replicated</em> on every rank — the
              memory waste DP-attention fixes.
            </p>
            <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" }}>
              {Array.from({ length: NUM_RANKS }, (_, i) => (
                <MoeRankCard
                  key={i}
                  rank={i}
                  kvLabel="KV: heads shard (MLA: full copy)"
                  kvAccent={color.warn}
                  expertLabel="attention layer"
                />
              ))}
            </div>
            <div style={savingsStyle(false)}>
              GQA: sharded · MLA: replicated {NUM_RANKS}× (wasteful)
            </div>
          </div>

          {/* DP-attention (distinct per rank) */}
          <div style={panelStyle(color.decode)}>
            <div style={panelTitleStyle(color.decode)}>
              <Term tokenKey="dpAttention">DP-attention</Term>
            </div>
            <p style={noteStyle}>
              Each rank runs attention on a distinct subset of requests, holding only{" "}
              <em>their</em> KV — no duplication, even for MLA. Pioneered in SGLang for DeepSeek;
              now also in vLLM (each data-parallel engine keeps an independent KV cache).
            </p>
            <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" }}>
              {Array.from({ length: NUM_RANKS }, (_, i) => (
                <MoeRankCard
                  key={i}
                  rank={i}
                  kvLabel={`KV: req ${i} only`}
                  kvAccent={color.decode}
                  expertLabel="attention layer"
                />
              ))}
            </div>
            <div style={savingsStyle(true)}>KV stored 1× (distinct per rank)</div>
          </div>
        </div>

        {/* Axis B: expert placement (EP) */}
        <p style={{ ...noteStyle, color: color.textMuted, fontWeight: font.weight.bold, marginBottom: space.sm }}>
          Axis B — expert placement (<Term tokenKey="expertParallel">EP</Term>, FFN only)
        </p>
        <div style={panelStyle(color.accent)}>
          <p style={noteStyle}>
            Independent of KV: the MoE FFN's experts are sharded across GPUs, and each token's
            router sends it (all-to-all) to whichever rank holds its top-k experts. This is about{" "}
            <strong>compute/weights, not KV</strong> — a token's KV stays wherever attention put it.{" "}
            <strong>Both engines support EP</strong> (with DeepEP backends and expert load
            balancing); it is not SGLang-specific.
          </p>
          <div style={{ display: "flex", gap: space.md, flexWrap: "wrap", marginTop: space.xs }}>
            {Array.from({ length: NUM_RANKS }, (_, i) => (
              <MoeRankCard
                key={i}
                rank={i}
                kvLabel="(KV set by Axis A)"
                kvAccent={color.borderStrong}
                expertLabel={`experts ${i * 2}-${i * 2 + 1}`}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Punchline */}
      <div
        style={{
          ...notePanel,
          border: `1px solid ${color.accent}66`,
          padding: `${space.lg}px ${space.xl}px`,
          fontSize: font.size.base,
          color: color.textPrimary,
          lineHeight: 1.6,
        }}
      >
        <strong style={{ color: color.accent }}>Punchline:</strong> vLLM and SGLang now share the
        building blocks — automatic prefix caching, TP/EP, and{" "}
        <Term tokenKey="dpAttention">DP-attention</Term> are available in both. SGLang{" "}
        <em>pioneered</em> <Term tokenKey="radixAttention">RadixAttention</Term> and DP-attention and
        has historically led on DeepSeek-class EP serving (DeepEP, computation/communication{" "}
        <Term tokenKey="overlappedScheduling">overlap</Term>, a{" "}
        <Term tokenKey="programmableFrontend">programmable frontend</Term>). The honest difference
        today is defaults, maturity, and specific optimizations — not raw capability.
      </div>
    </div>
  );
}
