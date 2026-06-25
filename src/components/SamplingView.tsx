/**
 * SamplingView — illustrative next-token sampling controls (Stage 1 sub-view).
 *
 * Purely educational: a fixed set of candidate next-tokens with base logits is
 * reshaped by temperature, then truncated by top-k / top-p, then renormalized
 * and drawn as a probability bar chart. No engine involvement — this visualizes
 * the sampling step conceptually so the determinism↔creativity tradeoff is felt.
 */

import { useState } from "react";
import { Term } from "./Term";
import { CANDIDATES, computeDistribution } from "../content/sampling";
import { color, space, radius, font, statusTint } from "../theme";

// ─── styles ──────────────────────────────────────────────────────────────────

const sliderRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  fontSize: font.size.md,
  color: color.textMuted,
};

const sliderLabelStyle: React.CSSProperties = {
  minWidth: 96,
  display: "flex",
  alignItems: "center",
  gap: space.xs,
};

const valueStyle: React.CSSProperties = {
  minWidth: 32,
  fontFamily: font.mono,
  color: color.textPrimary,
  textAlign: "right",
};

const barRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  marginBottom: 3,
};

const tokenLabelStyle: React.CSSProperties = {
  minWidth: 64,
  fontFamily: font.mono,
  fontSize: font.size.md,
  textAlign: "right",
};

const barTrackStyle: React.CSSProperties = {
  flex: 1,
  height: 16,
  background: color.panelBgInset,
  borderRadius: radius.sm,
  overflow: "hidden",
};

// ─── component ─────────────────────────────────────────────────────────────────

export function SamplingView() {
  const [temperature, setTemperature] = useState(1.0);
  const [topK, setTopK] = useState(CANDIDATES.length);
  const [topP, setTopP] = useState(1.0);

  const dist = computeDistribution(temperature, topK, topP);
  const isGreedy = temperature <= 0.01;
  const keptCount = dist.filter((d) => d.kept).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: space.lg }}>
      {/* Sliders */}
      <div style={{ display: "flex", flexDirection: "column", gap: space.md }}>
        <label style={sliderRowStyle}>
          <span style={sliderLabelStyle}>
            <Term tokenKey="temperature">Temperature</Term>
          </span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            aria-label="Temperature"
            style={{ flex: 1 }}
          />
          <span style={valueStyle}>{temperature.toFixed(1)}</span>
        </label>

        <label style={sliderRowStyle}>
          <span style={sliderLabelStyle}>
            <Term tokenKey="topK">Top-k</Term>
          </span>
          <input
            type="range"
            min={1}
            max={CANDIDATES.length}
            step={1}
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            aria-label="Top-k"
            style={{ flex: 1 }}
          />
          <span style={valueStyle}>{topK}</span>
        </label>

        <label style={sliderRowStyle}>
          <span style={sliderLabelStyle}>
            <Term tokenKey="topP">Top-p</Term>
          </span>
          <input
            type="range"
            min={0.1}
            max={1}
            step={0.05}
            value={topP}
            onChange={(e) => setTopP(Number(e.target.value))}
            aria-label="Top-p"
            style={{ flex: 1 }}
          />
          <span style={valueStyle}>{topP.toFixed(2)}</span>
        </label>
      </div>

      {/* Top-K vs Top-P comparison grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: space.md }}>
        {/* Top-K card */}
        <div
          style={{
            background: color.panelBgInset,
            border: `1px solid ${color.prefill}44`,
            borderRadius: radius.md,
            padding: space.md,
            display: "flex",
            flexDirection: "column",
            gap: space.xs,
          }}
        >
          <span
            style={{
              fontSize: font.size.sm,
              fontWeight: font.weight.bold,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              color: color.prefill,
              fontFamily: font.sans,
            }}
          >
            Top-K
          </span>
          <span
            style={{
              display: "inline-block",
              padding: `2px 8px`,
              borderRadius: radius.pill,
              fontSize: font.size.xs,
              fontWeight: font.weight.semibold,
              fontFamily: font.sans,
              background: `${color.prefill}22`,
              color: color.prefill,
              border: `1px solid ${color.prefill}44`,
              alignSelf: "flex-start",
            }}
          >
            Fixed pool
          </span>
          <span style={{ fontFamily: font.mono, fontSize: font.size.sm, color: color.textPrimary }}>
            &le; {topK} tokens, fixed count
          </span>
          <span style={{ fontSize: font.size.xs, color: color.textFaint }}>
            Ignores model confidence
          </span>
          <span
            style={{
              display: "inline-block",
              padding: `2px 8px`,
              borderRadius: radius.pill,
              fontSize: font.size.xs,
              fontWeight: font.weight.semibold,
              fontFamily: font.sans,
              background: `${color.prefill}22`,
              color: color.prefill,
              border: `1px solid ${color.prefill}44`,
              alignSelf: "flex-start",
            }}
          >
            coding · factual
          </span>
        </div>

        {/* Top-P card */}
        <div
          style={{
            background: color.panelBgInset,
            border: `1px solid ${color.decode}44`,
            borderRadius: radius.md,
            padding: space.md,
            display: "flex",
            flexDirection: "column",
            gap: space.xs,
          }}
        >
          <span
            style={{
              fontSize: font.size.sm,
              fontWeight: font.weight.bold,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              color: color.decode,
              fontFamily: font.sans,
            }}
          >
            Top-P
          </span>
          <span
            style={{
              display: "inline-block",
              padding: `2px 8px`,
              borderRadius: radius.pill,
              fontSize: font.size.xs,
              fontWeight: font.weight.semibold,
              fontFamily: font.sans,
              background: `${color.decode}22`,
              color: color.decode,
              border: `1px solid ${color.decode}44`,
              alignSelf: "flex-start",
            }}
          >
            Nucleus sampling
          </span>
          <span style={{ fontFamily: font.mono, fontSize: font.size.sm, color: color.textPrimary }}>
            &le; {(topP * 100).toFixed(0)}% cumulative mass, dynamic
          </span>
          <span style={{ fontSize: font.size.xs, color: color.textFaint }}>
            Contracts when model is certain
          </span>
          <span
            style={{
              display: "inline-block",
              padding: `2px 8px`,
              borderRadius: radius.pill,
              fontSize: font.size.xs,
              fontWeight: font.weight.semibold,
              fontFamily: font.sans,
              background: `${color.decode}22`,
              color: color.decode,
              border: `1px solid ${color.decode}44`,
              alignSelf: "flex-start",
            }}
          >
            creative · chat
          </span>
        </div>
      </div>

      {/* Distribution bars */}
      <div>
        {dist.map((d) => (
          <div key={d.text} style={barRowStyle}>
            <span
              style={{
                ...tokenLabelStyle,
                color: d.kept ? color.textPrimary : color.textFaint,
                textDecoration: d.kept ? "none" : "line-through",
              }}
            >
              {d.text}
            </span>
            <div style={barTrackStyle}>
              <div
                style={{
                  height: "100%",
                  width: `${Math.round(d.prob * 100)}%`,
                  background: d.kept ? color.accent : color.border,
                  borderRadius: radius.sm,
                  transition: "width 0.2s ease",
                }}
              />
            </div>
            <span style={{ ...valueStyle, color: d.kept ? color.textMuted : color.textFaint }}>
              {(d.prob * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      {/* Status chips */}
      <div style={{ display: "flex", gap: space.sm, flexWrap: "wrap", alignItems: "center" }}>
        {isGreedy ? (
          <span style={statusTint(color.accent)}>greedy — deterministic</span>
        ) : (
          <>
            <span style={statusTint(color.prefill)}>Top-K: {topK} tokens max</span>
            <span style={statusTint(color.decode)}>Top-P: {(topP * 100).toFixed(0)}% mass</span>
            <span style={statusTint(color.accent)}>{keptCount} / {CANDIDATES.length} in pool</span>
          </>
        )}
      </div>
    </div>
  );
}
