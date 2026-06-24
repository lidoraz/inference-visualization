/**
 * Stage 6 — Model Features: FP8 Quantization + Mixture of Experts (MoE)
 *
 * Config-driven diagram component. No engine changes.
 *
 * Two sub-views:
 *   1. FP8 memory bars — shows weights + KV cache memory halving when
 *      switching from FP16 (2 bytes/elem) to FP8 (1 byte/elem).
 *   2. MoE expert routing — a router selects top-k of E experts per token;
 *      unselected experts are grayed out. Router uses a deterministic hash of
 *      (tokenId, layerIndex) so the layout is stable across re-renders.
 *
 * Determinism note: expert selection uses a simple deterministic hash
 *   hash(tokenId, layerIdx) = (tokenId * 2654435761 ^ layerIdx * 40503) & 0xffffffff
 * so the same token always routes to the same experts, making the diagram
 * stable and reproducible without Math.random().
 */

import { useState } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, notePanel, requestColors } from "../theme";

type QuantMode = "fp16" | "fp8";

// ─── Deterministic hash for expert routing ────────────────────────────────────
// Uses a simple integer hash to assign tokens to experts without Math.random().

function deterministicHash(a: number, b: number): number {
  // Fibonacci hashing + XOR to spread bits
  return ((a * 2654435761) ^ (b * 40503)) >>> 0;
}

/** Pick top-k expert indices for a given token + layer using a deterministic hash. */
function pickExperts(tokenId: number, layerIdx: number, numExperts: number, topK: number): number[] {
  // Score each expert deterministically
  const scores = Array.from({ length: numExperts }, (_, e) =>
    deterministicHash(tokenId + e, layerIdx + e * 7)
  );
  // Sort by score descending, return top-k indices
  return scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((x) => x.idx)
    .sort((a, b) => a - b); // sort ascending for stable display order
}

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_PARAMS_B = 8; // 8B parameter model for illustration
const FP16_BYTES_PER_PARAM = 2;
const FP8_BYTES_PER_PARAM = 1;
const KV_TOKENS = 4096; // illustrative KV cache token count
const KV_DIMS = 4096; // hidden dim for KV estimation
const NUM_EXPERTS = 8;
const TOP_K = 2;
const MAX_MOE_TOKENS = 4; // few enough that some experts stay unselected (skipped)
const DEMO_TOKENS = [10, 42, 77, 128, 255, 512]; // fallback ids when no request exists

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

const subSectionStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 14,
};

const subHeaderStyle: React.CSSProperties = {
  ...sectionLabel,
  fontSize: font.size.md,
  letterSpacing: "0.08em",
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: space.sm,
};

const toggleRowStyle: React.CSSProperties = {
  display: "flex",
  gap: space.sm,
  alignItems: "center",
};

function toggleButtonStyle(active: boolean, accent: string): React.CSSProperties {
  return {
    padding: `5px 14px`,
    borderRadius: radius.md,
    border: `1px solid ${active ? accent : color.borderStrong}`,
    background: active ? `${accent}22` : color.border,
    color: active ? accent : color.textPrimary,
    cursor: "pointer",
    fontSize: font.size.base,
    fontWeight: active ? font.weight.bold : font.weight.normal,
    transition: "all 0.15s",
  };
}

const memBarRowStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

function MemoryBar({
  label,
  valueGB,
  maxGB,
  accent,
}: {
  label: string;
  valueGB: number;
  maxGB: number;
  accent: string;
}) {
  const pct = Math.min(valueGB / maxGB, 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: space.xs }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: font.size.md }}>
        <span>{label}</span>
        <span style={{ fontFamily: font.mono, color: accent }}>
          {valueGB.toFixed(1)} GB
        </span>
      </div>
      <div
        style={{
          height: 18,
          background: color.border,
          borderRadius: radius.sm,
          overflow: "hidden",
        }}
        aria-label={`${label}: ${valueGB.toFixed(1)} GB`}
      >
        <div
          style={{
            height: "100%",
            width: `${Math.round(pct * 100)}%`,
            background: accent,
            borderRadius: radius.sm,
            transition: "width 0.4s ease",
            display: "flex",
            alignItems: "center",
            paddingLeft: space.sm,
          }}
        >
          {pct > 0.25 && (
            <span style={{ fontSize: font.size.xs, color: color.panelBg, fontWeight: font.weight.bold }}>
              {Math.round(pct * 100)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

const moeLayerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.lg,
};

const routerBoxStyle: React.CSSProperties = {
  background: color.panelBgInset,
  border: `1px solid ${color.prefill}`,
  borderRadius: radius.lg,
  padding: `10px 14px`,
  textAlign: "center",
  fontSize: font.size.md,
  color: color.prefill,
  fontWeight: font.weight.bold,
};

function expertBoxStyle(selected: boolean): React.CSSProperties {
  return {
    flex: "1 1 80px",
    minWidth: 64,
    padding: `${space.md}px ${space.sm}px`,
    borderRadius: radius.lg,
    border: `1px solid ${selected ? color.accent : color.border}`,
    background: selected ? `${color.accent}22` : color.panelBgInset,
    color: selected ? color.accent : color.border,
    textAlign: "center",
    fontSize: font.size.md,
    fontWeight: selected ? font.weight.bold : font.weight.normal,
    transition: "all 0.2s ease",
  };
}


const TOKEN_COLORS = requestColors;

// ─── MoE routing diagram (tokens → experts, with literal arrows) ──────────────

interface RoutedToken {
  tokenId: number;
  text: string;
  colorIdx: number;
  experts: number[];
}

/**
 * Renders a token row (top) and an expert row (bottom) with SVG lines drawn
 * from each token to its top-k experts. Column centers are evenly spaced and
 * expressed as percentages, so no DOM measurement is needed. The two rows live
 * at fixed vertical bands inside a relative box; the SVG overlays the gap.
 */
function MoeRoutingDiagram({ tokens }: { tokens: RoutedToken[] }) {
  const DIAGRAM_H = 150; // px — vertical span the arrows cross
  const tokN = tokens.length;
  const tokX = (i: number) => ((i + 0.5) / tokN) * 100; // % center of token i
  const expX = (e: number) => ((e + 0.5) / NUM_EXPERTS) * 100; // % center of expert e

  return (
    <div style={{ position: "relative" }}>
      {/* Token row */}
      <div style={{ display: "flex", gap: space.xs }}>
        {tokens.map((t, i) => (
          <div
            key={`${t.tokenId}-${i}`}
            style={{
              flex: 1,
              textAlign: "center",
              padding: `3px ${space.xs}px`,
              borderRadius: radius.sm,
              border: `1px solid ${TOKEN_COLORS[t.colorIdx % TOKEN_COLORS.length]}`,
              background: `${TOKEN_COLORS[t.colorIdx % TOKEN_COLORS.length]}22`,
              color: TOKEN_COLORS[t.colorIdx % TOKEN_COLORS.length],
              fontSize: font.size.sm,
              fontFamily: font.mono,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
            title={`token "${t.text}" → experts ${t.experts.map((e) => `E${e}`).join(", ")}`}
          >
            {t.text}
          </div>
        ))}
      </div>

      {/* Arrow layer */}
      <svg
        width="100%"
        height={DIAGRAM_H}
        viewBox={`0 0 100 ${DIAGRAM_H}`}
        preserveAspectRatio="none"
        style={{ display: "block" }}
        aria-hidden="true"
      >
        {tokens.flatMap((t, i) =>
          t.experts.map((e) => (
            <line
              key={`${i}-${e}`}
              x1={tokX(i)}
              y1={0}
              x2={expX(e)}
              y2={DIAGRAM_H}
              stroke={TOKEN_COLORS[t.colorIdx % TOKEN_COLORS.length]}
              strokeWidth={1}
              strokeOpacity={0.7}
              vectorEffect="non-scaling-stroke"
            />
          ))
        )}
      </svg>

      {/* Expert row */}
      <div style={{ display: "flex", gap: space.xs }}>
        {Array.from({ length: NUM_EXPERTS }, (_, e) => {
          const selected = tokens.some((t) => t.experts.includes(e));
          return (
            <div key={e} style={{ ...expertBoxStyle(selected), flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: font.size.xs, marginBottom: 2 }}>E{e}</div>
              {selected ? "active" : "skip"}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const insightBoxStyle: React.CSSProperties = {
  ...notePanel,
  padding: `10px 14px`,
  lineHeight: 1.65,
};

// ─── Component ────────────────────────────────────────────────────────────────

export function Stage6ModelFeatures({ engine, config }: StageProps) {
  const [quant, setQuant] = useState<QuantMode>(config.quant ?? "fp16");
  const [moeOn, setMoeOn] = useState<boolean>(config.moe ?? false);
  // Bumping this re-rolls routing by shifting the layer index fed to the hash.
  const [routingSeed, setRoutingSeed] = useState(0);

  // ── FP8 memory calculations ──────────────────────────────────────────────
  const bytesPerParam = quant === "fp8" ? FP8_BYTES_PER_PARAM : FP16_BYTES_PER_PARAM;
  const weightsGB = (MODEL_PARAMS_B * 1e9 * bytesPerParam) / 1e9;
  const kvCacheGB = (KV_TOKENS * KV_DIMS * 2 * bytesPerParam) / 1e9; // 2 for K+V
  const maxGB = (MODEL_PARAMS_B * 1e9 * FP16_BYTES_PER_PARAM) / 1e9 * 1.2; // headroom

  // ── MoE token source: the seeded request's real prompt tokens (first few),
  // falling back to fixed demo ids if no request exists. ─────────────────────
  const liveReq = engine.requests.find((r) => r.promptTokens.length > 0);
  const moeTokens: { id: number; text: string }[] = liveReq
    ? liveReq.promptTokens.slice(0, MAX_MOE_TOKENS).map((t) => ({ id: t.id, text: t.text }))
    : DEMO_TOKENS.map((id) => ({ id, text: `t${id}` }));

  const layerIdx = routingSeed; // re-roll shifts which "layer" the router emulates
  const tokenExpertMap = moeTokens.map((tok, i) => ({
    tokenId: tok.id,
    text: tok.text,
    colorIdx: i,
    experts: pickExperts(tok.id, layerIdx, NUM_EXPERTS, TOP_K),
  }));

  return (
    <div style={outerStyle} aria-label="Model Features visualization">
      {/* ── Sub-view 1: FP8 Memory Bars ── */}
      <div style={subSectionStyle} aria-label="FP8 quantization sub-view">
        <h3 style={subHeaderStyle}>
          1. <Term tokenKey="quantization">Quantization</Term> — FP16 vs FP8 Memory
        </h3>

        <div style={toggleRowStyle}>
          <span style={{ fontSize: font.size.md, color: color.textFaint }}>Precision:</span>
          {(["fp16", "fp8"] as QuantMode[]).map((q) => (
            <button
              key={q}
              style={toggleButtonStyle(quant === q, q === "fp8" ? color.decode : color.prefill)}
              onClick={() => setQuant(q)}
              aria-pressed={quant === q}
            >
              {q.toUpperCase()}
            </button>
          ))}
        </div>

        <div style={memBarRowStyle}>
          <MemoryBar
            label={`Model Weights (${MODEL_PARAMS_B}B params)`}
            valueGB={weightsGB}
            maxGB={maxGB}
            accent={quant === "fp8" ? color.decode : color.prefill}
          />
          <MemoryBar
            label={`KV Cache (illustrative, ${KV_TOKENS.toLocaleString()} tokens)`}
            valueGB={kvCacheGB}
            maxGB={maxGB * 0.4}
            accent={quant === "fp8" ? color.decode : color.prefill}
          />
        </div>

        <div style={insightBoxStyle}>
          <strong style={{ color: color.textPrimary }}>
            {quant === "fp8" ? "FP8 active" : "FP16 baseline"}:{" "}
          </strong>
          weights {weightsGB.toFixed(1)} GB, KV {kvCacheGB.toFixed(2)} GB.{" "}
          {quant === "fp8" ? (
            <>
              <Term tokenKey="fp8">FP8</Term> uses 1 byte/element vs FP16's 2 — memory is halved.
              Modern H100/B200 GPUs have native FP8 matrix-multiply units for additional throughput
              gains. (The KV bar is illustrative — the exact per-token formula is on the Paged KV
              Cache stage.)
            </>
          ) : (
            <>
              Standard FP16 baseline. Toggle to <strong>FP8</strong> to see the bars halve — the
              same model fits in half the GPU memory.
            </>
          )}
        </div>
      </div>

      {/* ── Sub-view 2: MoE Expert Routing ── */}
      <div style={subSectionStyle} aria-label="MoE expert routing sub-view">
        <h3 style={subHeaderStyle}>
          2. <Term tokenKey="moe">Mixture of Experts</Term> — Top-{TOP_K} of {NUM_EXPERTS} Expert Routing
        </h3>

        <div style={toggleRowStyle}>
          <span style={{ fontSize: font.size.md, color: color.textFaint }}>MoE:</span>
          <button
            style={toggleButtonStyle(!moeOn, color.textFaint)}
            onClick={() => setMoeOn(false)}
            aria-pressed={!moeOn}
          >
            OFF (dense)
          </button>
          <button
            style={toggleButtonStyle(moeOn, color.accent)}
            onClick={() => setMoeOn(true)}
            aria-pressed={moeOn}
          >
            ON (MoE)
          </button>
        </div>

        {!moeOn ? (
          <div style={insightBoxStyle}>
            <strong style={{ color: color.textPrimary }}>Dense FFN (MoE off): </strong>
            every token passes through the same single FFN block. Total and active parameters are
            the same. Toggle MoE <strong>ON</strong> to see expert routing.
          </div>
        ) : (
          <div style={moeLayerStyle}>
            {/* Router */}
            <div style={routerBoxStyle}>
              <Term tokenKey="expertRouting">Router</Term>
              {" "}— selects top-{TOP_K} of {NUM_EXPERTS} experts per token
            </div>

            {/* Token → expert arrow diagram */}
            <div>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: space.sm,
                }}
              >
                <p style={{ fontSize: font.size.sm, color: color.textFaint, margin: 0 }}>
                  {liveReq
                    ? "Routing this request's prompt tokens — each line goes to a chosen expert; unselected experts are skipped:"
                    : "Each token routes to its top-k experts; unselected experts are skipped:"}
                </p>
                <button
                  style={toggleButtonStyle(false, color.accent)}
                  onClick={() => setRoutingSeed((s) => s + 1)}
                  aria-label="Re-roll expert routing"
                  title="Re-roll the router's expert choices"
                >
                  ↻ Re-roll
                </button>
              </div>
              <MoeRoutingDiagram tokens={tokenExpertMap} />
            </div>

            {/* Active params insight */}
            <div style={insightBoxStyle}>
              <p style={{ margin: 0 }}>
                <strong style={{ color: color.textPrimary }}>
                  <Term tokenKey="activeParams">Active vs Total Parameters:</Term>
                </strong>{" "}
                This model has {NUM_EXPERTS} experts. Each token activates only top-
                {TOP_K} ({Math.round((TOP_K / NUM_EXPERTS) * 100)}% of experts). Total
                parameters (memory) scale with all {NUM_EXPERTS} experts, but per-token
                compute (FLOPs) only scales with {TOP_K}.
              </p>
              <p style={{ margin: `${space.sm}px 0 0` }}>
                Example: 8B total params with 8 experts → each token only runs ≈2B active
                params. <Term tokenKey="fp8">FP8</Term> + <Term tokenKey="moe">MoE</Term> together
                (e.g. DeepSeek-V3) pack hundreds of billions of total parameters into a cluster
                while keeping per-token cost low.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
