/**
 * Speculative Decoding — illustrative stage.
 *
 * A small draft model proposes k tokens; the target verifies all k in one pass
 * and accepts the longest correct prefix. Two sliders (draft length k, draft
 * acceptance rate) drive a visual of the draft tokens (accepted / first reject /
 * discarded) and the resulting speedup (tokens per target pass = 1 + accepted).
 *
 * Config/UI-driven diagram — no engine change. Deterministic: how many tokens
 * are accepted is derived from k and the acceptance rate, not random.
 */

import { useState } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel } from "../theme";

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
  marginBottom: space.md,
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.sm,
  flexWrap: "wrap",
  marginBottom: space.md,
};

const laneLabelStyle: React.CSSProperties = {
  minWidth: 96,
  fontSize: font.size.md,
  color: color.textMuted,
};

type CellKind = "target" | "accepted" | "reject" | "discarded";

function cellStyle(kind: CellKind): React.CSSProperties {
  const map: Record<CellKind, { bg: string; border: string; fg: string }> = {
    target: { bg: `${color.prefill}33`, border: color.prefill, fg: color.prefill },
    accepted: { bg: `${color.decode}33`, border: color.decode, fg: color.decode },
    reject: { bg: `${color.danger}33`, border: color.danger, fg: color.danger },
    discarded: { bg: color.panelBgInset, border: color.border, fg: color.textFaint },
  };
  const c = map[kind];
  return {
    minWidth: 30,
    height: 30,
    padding: `0 ${space.sm}px`,
    borderRadius: radius.sm,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: font.size.md,
    fontFamily: font.mono,
    background: c.bg,
    border: `1px solid ${c.border}`,
    color: c.fg,
  };
}

const statChipStyle: React.CSSProperties = {
  flex: "1 1 140px",
  minWidth: 130,
  background: color.panelBgInset,
  border: `1px solid ${color.border}`,
  borderRadius: radius.lg,
  padding: `${space.lg}px`,
  display: "flex",
  flexDirection: "column",
  gap: 2,
};

const statValueStyle: React.CSSProperties = {
  fontSize: font.size.xxl,
  fontWeight: font.weight.bold,
  fontFamily: font.mono,
  color: color.accent,
};

const statLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: color.textFaint,
};

// ─── Component ─────────────────────────────────────────────────────────────────

export function StageSpeculative(_props: StageProps) {
  const [k, setK] = useState(4); // draft length
  const [acceptPct, setAcceptPct] = useState(70); // per-token acceptance rate %

  // Deterministic: accepted count = floor(k * rate). The target then corrects
  // the first non-accepted token, so a verify pass yields accepted + 1 tokens
  // (capped at k + 1 when the whole draft is accepted).
  const accepted = Math.floor((k * acceptPct) / 100);
  const allAccepted = accepted >= k;
  const tokensPerPass = Math.min(accepted + 1, k + 1);
  const speedup = tokensPerPass; // vanilla decode = 1 token per target pass

  return (
    <div style={outerStyle} aria-label="Speculative decoding visualization">
      {/* Controls */}
      <div>
        <div style={sliderRowStyle}>
          <span style={{ minWidth: 150 }}>
            <Term tokenKey="draftModel">Draft</Term> length k = {k}
          </span>
          <input
            type="range"
            min={1}
            max={8}
            step={1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            aria-label="Draft length"
            style={{ flex: 1 }}
          />
        </div>
        <div style={sliderRowStyle}>
          <span style={{ minWidth: 150 }}>
            <Term tokenKey="acceptanceRate">Acceptance</Term> = {acceptPct}%
          </span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={acceptPct}
            onChange={(e) => setAcceptPct(Number(e.target.value))}
            aria-label="Acceptance rate"
            style={{ flex: 1 }}
          />
        </div>
      </div>

      {/* Draft → verify → accept flow */}
      <div>
        <h3 style={sectionHeadingStyle}>One Target Pass</h3>

        <div style={rowStyle} aria-label="Draft proposal">
          <span style={laneLabelStyle}>
            <Term tokenKey="draftModel">Draft</Term> proposes
          </span>
          {Array.from({ length: k }, (_, i) => (
            <span key={i} style={cellStyle("discarded")} title={`drafted token ${i + 1}`}>
              d{i + 1}
            </span>
          ))}
        </div>

        <div style={rowStyle} aria-label="Verification result">
          <span style={laneLabelStyle}>
            <Term tokenKey="verification">Target</Term> verifies
          </span>
          {Array.from({ length: k }, (_, i) => {
            const kind: CellKind = i < accepted ? "accepted" : i === accepted ? "reject" : "discarded";
            const label = i < accepted ? "✓" : i === accepted ? "✗" : "·";
            const title =
              i < accepted
                ? "accepted (matched target)"
                : i === accepted
                ? "first mismatch — target corrects it"
                : "discarded (after a mismatch)";
            return (
              <span key={i} style={cellStyle(kind)} title={title}>
                {label}
              </span>
            );
          })}
          {/* The target always emits one correct token (the corrected/next one). */}
          <span style={{ fontSize: font.size.md, color: color.textFaint }}>→</span>
          <span style={cellStyle("target")} title="target's own correct token (always emitted)">
            {allAccepted ? "+1" : "fix"}
          </span>
        </div>

        <p style={{ margin: 0, fontSize: font.size.base }}>
          {accepted} of {k} draft tokens accepted{allAccepted ? " (all)" : ""}, plus 1 the target
          emits itself ={" "}
          <strong style={{ color: color.accent }}>{tokensPerPass} tokens</strong> from this one
          pass.
        </p>
      </div>

      {/* Speedup readout */}
      <div>
        <h3 style={sectionHeadingStyle}>Speedup vs Vanilla Decode</h3>
        <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" }}>
          <div style={statChipStyle}>
            <span style={statValueStyle}>1</span>
            <span style={statLabelStyle}>Vanilla: tokens / pass</span>
          </div>
          <div style={statChipStyle}>
            <span style={statValueStyle}>{tokensPerPass}</span>
            <span style={statLabelStyle}>Speculative: tokens / pass</span>
          </div>
          <div style={statChipStyle}>
            <span style={{ ...statValueStyle, color: color.decode }}>{speedup.toFixed(1)}×</span>
            <span style={statLabelStyle}>Speedup (same target cost)</span>
          </div>
        </div>
        <p style={{ margin: `${space.md}px 0 0`, fontSize: font.size.base }}>
          A verify pass costs the same as decoding one token, so yielding {tokensPerPass} tokens is a{" "}
          {speedup.toFixed(1)}× decode speedup — and the output is provably identical to plain
          sampling. Push acceptance up (a better-aligned draft) or k up to see the gain grow; a low
          acceptance rate wastes the draft's guesses.
        </p>
      </div>
    </div>
  );
}
