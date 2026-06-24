/**
 * KvMemoryBreakdown — bridges model architecture to KV-cache memory.
 *
 * Makes the formula concrete and interactive: adjustable num_layers,
 * num_kv_heads, head_dim, and dtype feed
 *
 *   bytes_per_token = num_layers × 2 (K,V) × num_kv_heads × head_dim × dtype_bytes
 *   bytes_per_block = bytes_per_token × blockSize
 *   total_cache     = bytes_per_block × kvCacheBlocks
 *
 * so a user can see how the model's shape sets the cost of one token, one block,
 * and the whole cache shown in Stage 2. Arch params are local UI state — purely
 * educational, they do not affect the simulation engine.
 */

import { useState } from "react";
import { Term } from "./Term";
import { color, space, radius, font, palette } from "../theme";

interface KvMemoryBreakdownProps {
  /** Tokens per block — comes from the live sim config so the bridge is honest. */
  blockSize: number;
  /** Number of physical blocks in the cache — also from live config. */
  kvCacheBlocks: number;
}

type Dtype = "fp16" | "fp8";
const DTYPE_BYTES: Record<Dtype, number> = { fp16: 2, fp8: 1 };

// Illustrative defaults roughly matching an ~8B model with grouped-query attention.
const DEFAULTS = { numLayers: 32, kvHeads: 8, headDim: 128 };

// Real-model presets so the abstract numbers map to recognizable models. Only
// the params that drive KV-cache size are used (layers, KV heads, head_dim);
// the note line gives human context. Values from each model's HuggingFace
// config.json (see commit message for source URLs).
interface ModelPreset {
  name: string;
  numLayers: number;
  kvHeads: number;
  headDim: number;
  note: string;
}

const MODEL_PRESETS: ModelPreset[] = [
  {
    name: "GPT-2 (124M)",
    numLayers: 12,
    kvHeads: 12, // MHA: KV heads == query heads
    headDim: 64,
    note: "124M dense, multi-head attention — tiny KV footprint.",
  },
  {
    name: "GPT-OSS-120B",
    numLayers: 36,
    kvHeads: 8, // GQA
    headDim: 64,
    note: "~117B MoE (128 experts, top-4), GQA shrinks KV to 8 heads.",
  },
  {
    name: "GLM-5.2 (753B)",
    numLayers: 78,
    kvHeads: 64, // full MHA — no GQA reduction
    headDim: 192, // explicit in config (not hidden_size/heads)
    note: "753B MoE (256 experts, top-8). Full MHA (64 KV heads) → huge KV per token.",
  },
];

// ─── formatting ────────────────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

// ─── styles ──────────────────────────────────────────────────────────────────

const presetRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.sm,
  marginBottom: space.md,
};

const presetButtonStyle = (active: boolean): React.CSSProperties => ({
  padding: `${space.xs}px ${space.md + 2}px`,
  borderRadius: radius.sm + 1,
  border: `1px solid ${active ? color.accent : color.borderStrong}`,
  background: active ? `${color.accent}22` : color.panelBgInset,
  color: active ? color.accent : color.textMuted,
  cursor: "pointer",
  fontSize: font.size.md,
  fontWeight: active ? font.weight.bold : font.weight.normal,
  transition: "all 0.15s",
});

const presetNoteStyle: React.CSSProperties = {
  fontSize: font.size.sm,
  color: color.accent,
  margin: `0 0 ${space.lg}px`,
  lineHeight: 1.5,
};

const paramsRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.lg,
  marginBottom: 14,
};

const paramLabelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 3,
  fontSize: font.size.sm,
  color: color.textMuted,
};

const paramInputStyle: React.CSSProperties = {
  padding: `${space.xs}px ${space.md}px`,
  borderRadius: radius.sm,
  border: `1px solid ${color.borderStrong}`,
  background: color.panelBgInset,
  color: color.textPrimary,
  fontSize: font.size.base,
  width: 72,
  fontFamily: font.mono,
};

const dtypeToggleStyle = (active: boolean): React.CSSProperties => ({
  padding: `${space.xs}px ${space.md + 2}px`,
  borderRadius: radius.sm,
  border: `1px solid ${active ? color.prefill : color.borderStrong}`,
  background: active ? `${color.prefill}22` : color.panelBgInset,
  color: active ? color.prefill : color.textFaint,
  cursor: "pointer",
  fontSize: font.size.md,
  fontFamily: font.mono,
  fontWeight: active ? font.weight.bold : font.weight.normal,
});

const formulaStyle: React.CSSProperties = {
  background: palette.crust,
  border: `1px solid ${color.border}`,
  borderRadius: radius.md,
  padding: `${space.md + 2}px ${space.lg}px`,
  fontFamily: font.mono,
  fontSize: font.size.md,
  color: color.textPrimary,
  lineHeight: 1.9,
  overflowX: "auto",
};

const dimStyle: React.CSSProperties = { color: color.textFaint };
const kvalStyle: React.CSSProperties = { color: color.prefill, fontWeight: font.weight.bold };

const ladderRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.md + 2,
  marginTop: space.lg,
};

const ladderChipStyle = (accent: string): React.CSSProperties => ({
  flex: "1 1 150px",
  background: color.panelBgInset,
  border: `1px solid ${accent}`,
  borderRadius: radius.lg,
  padding: `${space.md + 2}px ${space.lg}px`,
  display: "flex",
  flexDirection: "column",
  gap: space.xs,
});

const ladderLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: color.textFaint,
};

const ladderValueStyle = (accent: string): React.CSSProperties => ({
  fontSize: font.size.xxl,
  fontWeight: font.weight.bold,
  fontFamily: font.mono,
  color: accent,
});

const ladderSubStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  color: color.textFaint,
  fontFamily: font.mono,
};

// ─── component ─────────────────────────────────────────────────────────────────

function NumberParam({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: React.ReactNode;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <label style={paramLabelStyle}>
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.max(min, Math.min(max, Number(e.target.value))))}
        style={paramInputStyle}
      />
    </label>
  );
}

export function KvMemoryBreakdown({ blockSize, kvCacheBlocks }: KvMemoryBreakdownProps) {
  const [numLayers, setNumLayers] = useState(DEFAULTS.numLayers);
  const [kvHeads, setKvHeads] = useState(DEFAULTS.kvHeads);
  const [headDim, setHeadDim] = useState(DEFAULTS.headDim);
  const [dtype, setDtype] = useState<Dtype>("fp16");

  const dtypeBytes = DTYPE_BYTES[dtype];
  const bytesPerToken = numLayers * 2 * kvHeads * headDim * dtypeBytes;
  const bytesPerBlock = bytesPerToken * blockSize;
  const totalCacheBytes = bytesPerBlock * kvCacheBlocks;

  // A preset is "active" when all three KV-driving params match it exactly.
  const activePreset = MODEL_PRESETS.find(
    (p) => p.numLayers === numLayers && p.kvHeads === kvHeads && p.headDim === headDim
  );

  function applyPreset(p: ModelPreset) {
    setNumLayers(p.numLayers);
    setKvHeads(p.kvHeads);
    setHeadDim(p.headDim);
  }

  return (
    <div>
      {/* Real-model presets */}
      <div style={presetRowStyle} aria-label="Model presets">
        <span style={{ fontSize: font.size.sm, color: color.textFaint, alignSelf: "center" }}>Preset:</span>
        {MODEL_PRESETS.map((p) => (
          <button
            key={p.name}
            style={presetButtonStyle(activePreset?.name === p.name)}
            onClick={() => applyPreset(p)}
            aria-pressed={activePreset?.name === p.name}
            title={p.note}
          >
            {p.name}
          </button>
        ))}
      </div>
      {activePreset && (
        <p style={presetNoteStyle}>{activePreset.note}</p>
      )}

      {/* Adjustable architecture params */}
      <div style={paramsRowStyle} aria-label="Model architecture parameters">
        <NumberParam
          label={<Term tokenKey="modelLayers">num_layers</Term>}
          value={numLayers}
          min={1}
          max={160}
          onChange={setNumLayers}
        />
        <NumberParam
          label={<Term tokenKey="kvHeads">num_kv_heads</Term>}
          value={kvHeads}
          min={1}
          max={128}
          onChange={setKvHeads}
        />
        <NumberParam
          label={<Term tokenKey="headDim">head_dim</Term>}
          value={headDim}
          min={1}
          max={512}
          onChange={setHeadDim}
        />
        <div style={paramLabelStyle}>
          <span>dtype</span>
          <div style={{ display: "flex", gap: space.xs }}>
            {(["fp16", "fp8"] as Dtype[]).map((d) => (
              <button
                key={d}
                style={dtypeToggleStyle(dtype === d)}
                onClick={() => setDtype(d)}
                aria-pressed={dtype === d}
              >
                {d.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* The formula, with live numbers substituted. Note: no <Term> inside this
          box — its overflowX:auto clips the term's popover. The bytes/token term
          is explained in the surrounding copy instead. */}
      <div style={formulaStyle} aria-label="Bytes per token formula">
        <div>
          <span style={{ color: color.textPrimary, fontWeight: font.weight.bold }}>bytes/token</span> ={" "}
          <span style={dimStyle}>layers</span> × <span style={dimStyle}>2 (K,V)</span> ×{" "}
          <span style={dimStyle}>kv_heads</span> × <span style={dimStyle}>head_dim</span> ×{" "}
          <span style={dimStyle}>dtype</span>
        </div>
        <div>
          {"= "}
          <span style={kvalStyle}>{numLayers}</span> × <span style={kvalStyle}>2</span> ×{" "}
          <span style={kvalStyle}>{kvHeads}</span> × <span style={kvalStyle}>{headDim}</span> ×{" "}
          <span style={kvalStyle}>{dtypeBytes}</span> ={" "}
          <span style={{ color: color.decode, fontWeight: font.weight.bold }}>
            {bytesPerToken.toLocaleString()} bytes
          </span>{" "}
          <span style={dimStyle}>({formatBytes(bytesPerToken)})</span>
        </div>
      </div>

      {/* token → block → cache ladder */}
      <div style={ladderRowStyle} aria-label="Memory scaling from token to full cache">
        <div style={ladderChipStyle(color.decode)}>
          <span style={ladderLabelStyle}>Per Token</span>
          <span style={ladderValueStyle(color.decode)}>{formatBytes(bytesPerToken)}</span>
          <span style={ladderSubStyle}>architecture × dtype</span>
        </div>
        <div style={ladderChipStyle(color.prefill)}>
          <span style={ladderLabelStyle}>Per Block</span>
          <span style={ladderValueStyle(color.prefill)}>{formatBytes(bytesPerBlock)}</span>
          <span style={ladderSubStyle}>
            × blockSize {blockSize}
          </span>
        </div>
        <div style={ladderChipStyle(color.warn)}>
          <span style={ladderLabelStyle}>Whole Cache</span>
          <span style={ladderValueStyle(color.warn)}>{formatBytes(totalCacheBytes)}</span>
          <span style={ladderSubStyle}>
            × {kvCacheBlocks} blocks
          </span>
        </div>
      </div>

      <p style={{ fontSize: font.size.md, color: color.textFaint, lineHeight: 1.6, marginTop: space.lg, marginBottom: 0 }}>
        Each of the {kvCacheBlocks} blocks shown above holds {blockSize} tokens, and every token
        costs <span style={{ color: color.decode }}>{formatBytes(bytesPerToken)}</span> set entirely by
        the model's shape. Increase the layers or KV heads and watch the whole cache grow — this is
        why bigger/deeper models need more GPU memory per token, and why GQA (fewer{" "}
        <Term tokenKey="kvHeads">KV heads</Term>) and <Term tokenKey="fp8">FP8</Term> are such
        effective levers.
      </p>
    </div>
  );
}
