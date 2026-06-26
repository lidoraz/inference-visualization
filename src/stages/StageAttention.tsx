/**
 * Attention — fundamentals first, then memory-shaping techniques.
 *
 * Panel order:
 *   1. Q/K/V Computation step-through (5 steps)
 *   2. Flash Attention tiling (Standard vs Flash toggle)
 *   3. Sliding-window attention
 *   4. KV-head variants (MHA / GQA / MLA)
 */

import { useState, useEffect } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, notePanel } from "../theme";

const SEQ_LEN = 24;

// ─── Fixed matrix data (hardcoded for reproducibility) ──────────────────────
// 5 tokens × 8 dims, values in [0, 1]

const Q_DATA = [
  [0.9, 0.2, 0.7, 0.1, 0.8, 0.3, 0.6, 0.4],
  [0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6],
  [0.5, 0.5, 0.9, 0.3, 0.6, 0.1, 0.8, 0.2],
  [0.1, 0.6, 0.4, 0.7, 0.3, 0.9, 0.2, 0.8],
  [0.7, 0.3, 0.2, 0.5, 0.9, 0.4, 0.1, 0.6],
];

const K_DATA = [
  [0.4, 0.7, 0.3, 0.8, 0.1, 0.6, 0.9, 0.2],
  [0.8, 0.1, 0.6, 0.2, 0.9, 0.4, 0.3, 0.7],
  [0.2, 0.9, 0.5, 0.6, 0.4, 0.8, 0.1, 0.3],
  [0.6, 0.4, 0.8, 0.1, 0.7, 0.2, 0.5, 0.9],
  [0.3, 0.6, 0.1, 0.9, 0.5, 0.3, 0.7, 0.4],
];

const V_DATA = [
  [0.6, 0.1, 0.8, 0.4, 0.2, 0.9, 0.3, 0.7],
  [0.2, 0.8, 0.4, 0.7, 0.6, 0.1, 0.9, 0.3],
  [0.9, 0.3, 0.1, 0.6, 0.8, 0.5, 0.4, 0.2],
  [0.4, 0.6, 0.7, 0.2, 0.3, 0.8, 0.1, 0.9],
  [0.7, 0.4, 0.3, 0.8, 0.1, 0.2, 0.6, 0.5],
];

function dot(a: number[], b: number[]) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function softmaxRow(row: number[]): number[] {
  const max = Math.max(...row);
  const exps = row.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// Precompute score matrix: softmax(Q @ K^T / sqrt(d_k))
const D_K = 8;
const SCORE_DATA: number[][] = Q_DATA.map((qRow) => {
  const rawScores = K_DATA.map((kRow) => dot(qRow, kRow) / Math.sqrt(D_K));
  return softmaxRow(rawScores);
});

// Precompute output: softmax_scores @ V
const OUTPUT_DATA: number[][] = SCORE_DATA.map((scoreRow) => {
  return Array.from({ length: 8 }, (_, d) =>
    scoreRow.reduce((sum, w, t) => sum + w * V_DATA[t][d], 0)
  );
});

// Normalize output to [0,1] for display
const outputMax = Math.max(...OUTPUT_DATA.flat());
const OUTPUT_NORM = OUTPUT_DATA.map((row) => row.map((v) => v / outputMax));

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
  relCache: number;
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

// ─── QKV matrix sub-component ────────────────────────────────────────────────

const MATRIX_LABELS: Record<string, { termKey: string; baseColor: string }> = {
  Q: { termKey: "queryMatrix", baseColor: "#89b4fa" }, // blue
  K: { termKey: "keyMatrix", baseColor: "#a6e3a1" },   // green
  V: { termKey: "valueMatrix", baseColor: color.accent },
  scores: { termKey: "attentionScore", baseColor: "#f38ba8" }, // red/pink
  output: { termKey: "scaledDotProduct", baseColor: "#cba6f7" }, // mauve
};

function MatrixGrid({
  data,
  label,
  active,
  square = false,
}: {
  data: number[][];
  label: string;
  active: boolean;
  square?: boolean;
}) {
  const { termKey, baseColor } = MATRIX_LABELS[label];
  const cols = data[0].length;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: space.xs, alignItems: "center" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gap: 2,
          opacity: active ? 1 : 0.35,
          border: `2px solid ${active ? baseColor : color.border}`,
          borderRadius: radius.sm,
          padding: 3,
          transition: "opacity 0.3s, border-color 0.3s",
        }}
      >
        {data.map((row, r) =>
          row.map((val, c) => (
            <div
              key={`${r}-${c}`}
              style={{
                width: square ? 18 : 12,
                height: 18,
                borderRadius: 2,
                background: `color-mix(in srgb, ${baseColor} ${Math.round(val * 100)}%, ${color.panelBgInset})`,
              }}
            />
          ))
        )}
      </div>
      <div
        style={{
          fontSize: font.size.sm,
          color: active ? baseColor : color.textFaint,
          fontWeight: active ? font.weight.bold : font.weight.normal,
          transition: "color 0.3s",
        }}
      >
        <Term tokenKey={termKey}>{label}</Term>
      </div>
    </div>
  );
}

function ArrowRight({ active }: { active: boolean }) {
  return (
    <div
      style={{
        alignSelf: "center",
        color: active ? color.accent : color.textFaint,
        fontSize: font.size.xl,
        opacity: active ? 1 : 0.3,
        transition: "opacity 0.3s",
        userSelect: "none",
        marginBottom: 22,
      }}
    >
      →
    </div>
  );
}

const QKV_STEPS = [
  {
    label: "Project inputs to Q, K, V",
    caption: "Each token is projected into three spaces — what it asks (Q), what it offers (K), what it contains (V).",
    active: ["Q", "K", "V"],
  },
  {
    label: "Compute attention scores (Q × Kᵀ / √d_k)",
    caption: "Q × Kᵀ asks: how much should each token attend to every other?",
    active: ["Q", "K", "scores"],
  },
  {
    label: "Apply softmax row-wise",
    caption: "Scores become weights — each row sums to 1.",
    active: ["scores"],
  },
  {
    label: "Weight V by softmax scores",
    caption: "Each output is a blend of V rows, weighted by where it attended.",
    active: ["scores", "V", "output"],
  },
  {
    label: "Output added to residual stream",
    caption: "The result is added back to the residual stream and passed to the next layer.",
    active: ["output"],
  },
];

function QKVPanel() {
  const [step, setStep] = useState(0);
  const current = QKV_STEPS[step];

  return (
    <div>
      {/* Step indicator + nav */}
      <div style={{ display: "flex", alignItems: "center", gap: space.md, marginBottom: space.lg }}>
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          style={navBtnStyle(step === 0)}
          aria-label="Previous step"
        >
          ← Prev
        </button>
        <span style={{ fontSize: font.size.md, color: color.textMuted, minWidth: 50, textAlign: "center" }}>
          {step + 1} / {QKV_STEPS.length}
        </span>
        <button
          onClick={() => setStep((s) => Math.min(QKV_STEPS.length - 1, s + 1))}
          disabled={step === QKV_STEPS.length - 1}
          style={navBtnStyle(step === QKV_STEPS.length - 1)}
          aria-label="Next step"
        >
          Next →
        </button>
        <div style={{ marginLeft: space.sm }}>
          <span style={{ fontSize: font.size.base, color: color.textPrimary }}>
            {step === QKV_STEPS.length - 1 ? (
              <Term tokenKey="residualStream">{current.label}</Term>
            ) : step === 2 ? (
              <><Term tokenKey="softmax">Apply softmax</Term> row-wise</>
            ) : step === 1 ? (
              <>Compute <Term tokenKey="attentionScore">attention scores</Term> (Q × Kᵀ / √d_k)</>
            ) : step === 3 ? (
              <>Weight V by softmax scores → <Term tokenKey="scaledDotProduct">output</Term></>
            ) : (
              <>Project inputs to <Term tokenKey="queryMatrix">Q</Term>, <Term tokenKey="keyMatrix">K</Term>, <Term tokenKey="valueMatrix">V</Term></>
            )}
          </span>
          <div style={{ ...notePanel, marginTop: space.xs, padding: `${space.xs}px ${space.sm}px` }}>
            {current.caption}
          </div>
        </div>
      </div>

      {/* Matrix diagram */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: space.md,
          flexWrap: "wrap",
          padding: `${space.md}px 0`,
        }}
      >
        <MatrixGrid data={Q_DATA} label="Q" active={current.active.includes("Q")} />
        <ArrowRight active={current.active.includes("Q") && current.active.includes("K")} />
        <MatrixGrid data={K_DATA} label="K" active={current.active.includes("K")} />
        <ArrowRight active={current.active.includes("scores")} />
        <MatrixGrid data={SCORE_DATA} label="scores" active={current.active.includes("scores")} square />
        <ArrowRight active={current.active.includes("scores") && current.active.includes("V")} />
        <MatrixGrid data={V_DATA} label="V" active={current.active.includes("V")} />
        <ArrowRight active={current.active.includes("output")} />
        <MatrixGrid data={OUTPUT_NORM} label="output" active={current.active.includes("output")} />
      </div>
    </div>
  );
}

function navBtnStyle(disabled: boolean): React.CSSProperties {
  return {
    padding: `${space.xs}px ${space.sm}px`,
    background: disabled ? color.panelBgInset : color.panelBg,
    border: `1px solid ${color.border}`,
    borderRadius: radius.sm,
    color: disabled ? color.textFaint : color.textPrimary,
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: font.size.sm,
    fontFamily: font.sans,
  };
}

// ─── Flash Attention panel ────────────────────────────────────────────────────

const TILE_ROWS = 4;
const TILE_COLS = 4;
const TILE_COUNT = TILE_ROWS * TILE_COLS;

function FlashPanel() {
  const [mode, setMode] = useState<"standard" | "flash">("standard");
  const [activeTile, setActiveTile] = useState(0);

  useEffect(() => {
    if (mode !== "flash") return;
    const id = setInterval(() => setActiveTile((t) => (t + 1) % TILE_COUNT), 700);
    return () => clearInterval(id);
  }, [mode]);

  return (
    <div>
      {/* Toggle cards */}
      <div style={variantRowStyle} role="group" aria-label="Flash attention mode">
        {(["standard", "flash"] as const).map((m) => (
          <div
            key={m}
            style={variantCardStyle(mode === m)}
            onClick={() => setMode(m)}
            aria-pressed={mode === m}
          >
            <div style={{ fontSize: font.size.lg, fontWeight: font.weight.bold }}>
              {m === "standard" ? (
                <Term tokenKey="scaledDotProduct">Standard Attention</Term>
              ) : (
                <Term tokenKey="flashAttention">Flash Attention</Term>
              )}
            </div>
            <div style={{ fontSize: font.size.md, color: color.textMuted }}>
              {m === "standard"
                ? "Full N×N matrix written to HBM"
                : "Tiled — one block at a time in SRAM"}
            </div>
          </div>
        ))}
      </div>

      {/* Diagram */}
      <div style={{ marginTop: space.lg, display: "flex", gap: space.xl, flexWrap: "wrap", alignItems: "flex-start" }}>
        {mode === "standard" ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: space.sm }}>
            <div
              style={{
                width: 160,
                height: 160,
                background: `${color.danger}22`,
                border: `2px solid ${color.danger}`,
                borderRadius: radius.md,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: font.size.sm,
                color: color.danger,
                fontWeight: font.weight.bold,
              }}
            >
              N × N
            </div>
            <div style={{ fontSize: font.size.sm, color: color.textMuted }}>
              <Term tokenKey="hbm">HBM</Term> — full matrix materialized
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: space.sm }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: `repeat(${TILE_COLS}, 38px)`,
                gridTemplateRows: `repeat(${TILE_ROWS}, 38px)`,
                gap: 3,
              }}
            >
              {Array.from({ length: TILE_COUNT }, (_, i) => {
                const isActive = i === activeTile;
                return (
                  <div
                    key={i}
                    style={{
                      background: isActive ? `${color.accent}44` : color.panelBgInset,
                      border: `2px solid ${isActive ? color.accent : color.border}`,
                      borderRadius: radius.sm,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: font.size.xs,
                      color: isActive ? color.accent : color.textFaint,
                      fontWeight: isActive ? font.weight.bold : font.weight.normal,
                      transition: "background 0.15s, border-color 0.15s",
                    }}
                  >
                    {isActive ? "▶" : ""}
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: font.size.sm, color: color.textMuted, textAlign: "center" }}>
              Active tile → <Term tokenKey="sram">SRAM</Term>
              <br />
              rest stays in <Term tokenKey="hbm">HBM</Term>
            </div>
          </div>
        )}

        {/* Legend */}
        <div style={{ display: "flex", flexDirection: "column", gap: space.sm, justifyContent: "center" }}>
          <div style={{ fontSize: font.size.sm, color: color.textMuted, fontWeight: font.weight.bold, marginBottom: 2 }}>
            Memory
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
            <div style={{ width: 80, height: 12, background: color.danger, borderRadius: radius.sm, opacity: 0.7 }} />
            <span style={{ fontSize: font.size.xs, color: color.textMuted }}><Term tokenKey="hbm">HBM</Term> — large, slow</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: space.sm }}>
            <div style={{ width: 28, height: 12, background: color.accent, borderRadius: radius.sm, opacity: 0.8 }} />
            <span style={{ fontSize: font.size.xs, color: color.textMuted }}><Term tokenKey="sram">SRAM</Term> — small, fast</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Component ─────────────────────────────────────────────────────────────────

export function StageAttention(_props: StageProps) {
  const [windowSize, setWindowSize] = useState(8);
  const [variant, setVariant] = useState("gqa");

  const liveStart = Math.max(0, SEQ_LEN - windowSize);
  const active = VARIANTS.find((v) => v.key === variant) ?? VARIANTS[1];

  return (
    <div style={outerStyle} aria-label="Attention techniques visualization">

      {/* ── 1. Q/K/V Computation ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          1. <Term tokenKey="scaledDotProduct">Scaled Dot-Product</Term> Attention
        </h3>
        <QKVPanel />
      </div>

      {/* ── 2. Flash Attention ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          2. <Term tokenKey="flashAttention">Flash Attention</Term>
        </h3>
        <FlashPanel />
      </div>

      {/* ── 3. Sliding window ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          3. <Term tokenKey="slidingWindow">Sliding-Window</Term> Attention
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
        <p style={{ ...notePanel, margin: `${space.md}px 0 0` }}>
          Only the last <strong style={{ color: color.accent }}>{windowSize}</strong> tokens stay live — KV memory is O(W) instead of O(length).
        </p>
      </div>

      {/* ── 4. Attention variants ── */}
      <div>
        <h3 style={sectionHeadingStyle}>
          4. KV-Head <Term tokenKey="attentionVariant">Variants</Term>
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
        <p style={{ ...notePanel, margin: `${space.md}px 0 0` }}>
          {active.key === "mha" && <>Baseline — every query head stores its own K/V.</>}
          {active.key === "gqa" && <>~{Math.round(1 / active.relCache)}× smaller KV cache than MHA — the modern default.</>}
          {active.key === "mla" && <>~{Math.round(1 / active.relCache)}× smaller than MHA, but K/V can't be split across heads — requires <Term tokenKey="dpAttention">DP-attention</Term> under tensor parallelism.</>}
        </p>
      </div>
    </div>
  );
}
