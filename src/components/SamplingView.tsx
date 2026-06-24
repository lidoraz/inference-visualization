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
import { color, space, radius, font } from "../theme";

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
      <p style={{ margin: 0, fontSize: font.size.base, color: color.textMuted, lineHeight: 1.6 }}>
        Each <Term tokenKey="decode">decode</Term> step produces a probability over the whole
        vocabulary; a sampling strategy picks the next token. Drag the knobs to reshape this
        illustrative distribution.
      </p>

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

      <p style={{ margin: 0, fontSize: font.size.sm, color: color.textFaint, lineHeight: 1.5 }}>
        {isGreedy ? (
          <>
            Temperature 0 = <Term tokenKey="greedy">greedy decoding</Term>: all mass on the single
            most-likely token, fully deterministic.
          </>
        ) : (
          <>
            {keptCount} of {CANDIDATES.length} tokens kept (struck-through ones are excluded by
            top-k / top-p). Lower temperature sharpens toward the top token; higher flattens for
            more variety.
          </>
        )}
      </p>
    </div>
  );
}
