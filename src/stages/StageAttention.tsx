/**
 * Attention — KV-cache-shaping attention techniques.
 *
 * Two sub-views, both config/UI-driven diagrams:
 *   1. Sliding-window: a token strip where only the last W tokens are "live" in
 *      the KV cache; older ones are evicted as the window slides. A W slider
 *      shows KV memory going from O(length) to O(W) — the long-context payoff.
 *   2. Attention variants (MHA / GQA / MLA): how K/V heads are shared or
 *      compressed to shrink the cache, with a relative cache-size bar.
 */

import { useState } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel } from "../theme";

const SEQ_LEN = 24; // illustrative sequence length for the sliding-window strip

// ─── Styles ──────────────────────────────────────────────────────────────────

const outerStyle: React.CSSProperties = {
  padding: `${space.xl}px ${space.md}px`,
  display: "flex",
  flexDirection: "column",
  gap: space.xxl,
  fontFamily: font.sans,
  fontSize: font.size.lg,
  color: color.textPrimary,
};

const sectionHeadingStyle: React.CSSProperties = {
  ...sectionLabel,
  margin: `0 0 ${space.md}px`,
};

const sliderRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  marginBottom: space.lg,
};

const tokenStripStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 3,
  marginBottom: space.md,
};

function tokenCellStyle(live: boolean): React.CSSProperties {
  return {
    width: 22,
    height: 22,
    borderRadius: radius.sm,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: font.size.xs,
    fontFamily: font.mono,
    background: live ? `${color.accent}33` : color.panelBgInset,
    border: `1px solid ${live ? color.accent : color.border}`,
    color: live ? color.accent : color.textFaint,
  };
}

const variantRowStyle: React.CSSProperties = {
  display: "flex",
  gap: space.md,
  flexWrap: "wrap",
};

function variantCardStyle(active: boolean): React.CSSProperties {
  return {
    flex: "1 1 200px",
    minWidth: 180,
    background: color.panelBgInset,
    border: `1px solid ${active ? color.accent : color.border}`,
    borderRadius: radius.lg,
    padding: `${space.lg}px`,
    display: "flex",
    flexDirection: "column",
    gap: space.sm,
    cursor: "pointer",
  };
}

const cacheBarTrackStyle: React.CSSProperties = {
  height: 10,
  background: color.panelBg,
  borderRadius: radius.sm,
  overflow: "hidden",
};

// ─── Attention variants ───────────────────────────────────────────────────────

interface Variant {
  key: string;
  name: string;
  termKey: string;
  kvHeads: string;
  relCache: number; // 0..1 relative KV-cache size for the bar
  note: string;
}

const QUERY_HEADS = 32;
const VARIANTS: Variant[] = [
  {
    key: "mha",
    name: "MHA",
    termKey: "mha",
    kvHeads: `${QUERY_HEADS} KV heads`,
    relCache: 1,
    note: "One KV head per query head — biggest cache.",
  },
  {
    key: "gqa",
    name: "GQA",
    termKey: "gqa",
    kvHeads: "8 KV heads",
    relCache: 0.25,
    note: "Query heads grouped 4:1 onto shared KV heads — common default.",
  },
  {
    key: "mla",
    name: "MLA",
    termKey: "mla",
    kvHeads: "1 latent vector",
    relCache: 0.07,
    note: "K/V compressed to one latent per token — smallest, needs DP-attention.",
  },
];

// ─── Component ─────────────────────────────────────────────────────────────────

export function StageAttention(_props: StageProps) {
  const [windowSize, setWindowSize] = useState(8);
  const [variant, setVariant] = useState("gqa");

  const liveStart = Math.max(0, SEQ_LEN - windowSize);
  const active = VARIANTS.find((v) => v.key === variant) ?? VARIANTS[1];

  return (
    <div style={outerStyle} aria-label="Attention techniques visualization">
      {/* ── Sliding window ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          1. <Term tokenKey="slidingWindow">Sliding-Window</Term> Attention
        </h3>
        <div style={sliderRowStyle}>
          <span style={{ minWidth: 110 }}>Window (W) = {windowSize}</span>
          <input
            type="range"
            min={2}
            max={SEQ_LEN}
            step={1}
            value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value))}
            aria-label="Sliding window size"
            style={{ flex: 1 }}
          />
        </div>
        <div style={tokenStripStyle} aria-label="Token sequence with sliding window">
          {Array.from({ length: SEQ_LEN }, (_, i) => {
            const live = i >= liveStart;
            return (
              <span key={i} style={tokenCellStyle(live)} title={live ? "in KV cache" : "evicted"}>
                {i}
              </span>
            );
          })}
        </div>
        <p style={{ margin: 0, fontSize: font.size.base }}>
          A {SEQ_LEN}-token sequence keeps only{" "}
          <strong style={{ color: color.accent }}>{windowSize}</strong> tokens live in cache (the
          rest are evicted). KV memory is{" "}
          <strong>O(W)</strong> instead of O(length) — this is what makes 100k–1M-token contexts
          affordable. Newest token attends back W positions.
        </p>
      </div>

      {/* ── Attention variants ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          2. KV-Head <Term tokenKey="attentionVariant">Variants</Term>
        </h3>
        <div style={variantRowStyle} role="group" aria-label="Attention variants">
          {VARIANTS.map((v) => (
            <div
              key={v.key}
              style={variantCardStyle(variant === v.key)}
              onClick={() => setVariant(v.key)}
              aria-pressed={variant === v.key}
            >
              <div style={{ fontSize: font.size.lg, fontWeight: font.weight.bold }}>
                <Term tokenKey={v.termKey}>{v.name}</Term>
              </div>
              <div style={{ fontSize: font.size.md, fontFamily: font.mono, color: color.accent }}>
                {v.kvHeads}
              </div>
              <div style={cacheBarTrackStyle}>
                <div
                  style={{
                    height: "100%",
                    width: `${Math.round(v.relCache * 100)}%`,
                    background: color.accent,
                    borderRadius: radius.sm,
                  }}
                />
              </div>
              <div style={{ fontSize: font.size.md }}>{v.note}</div>
            </div>
          ))}
        </div>
        <p style={{ margin: `${space.md}px 0 0`, fontSize: font.size.base }}>
          Relative KV-cache size for {QUERY_HEADS} query heads:{" "}
          <Term tokenKey="mha">MHA</Term> stores all {QUERY_HEADS} →{" "}
          <Term tokenKey="gqa">GQA</Term> shares them down to 8 (~4× smaller) →{" "}
          <Term tokenKey="mla">MLA</Term> compresses to one latent (~14× smaller). Smaller cache =
          longer contexts and bigger batches on the same GPU. Selected:{" "}
          <strong style={{ color: color.accent }}>{active.name}</strong>.
        </p>
      </div>
    </div>
  );
}
