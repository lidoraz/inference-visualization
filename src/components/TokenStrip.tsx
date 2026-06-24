/**
 * TokenStrip — renders the token sequence for a single request.
 *
 * Props:
 *   promptTokens   — tokens from the original prompt
 *   decodedTokens  — tokens generated so far
 *   phase          — current phase ("prefill" | "decode")
 *
 * Layout: horizontal strip of chips.
 *   - Prompt tokens have a distinct background (blue-toned).
 *   - Decoded tokens are green-toned.
 *   - The most recent decoded token is highlighted with a brighter border.
 *   - A small phase badge is shown to the right.
 */

import type { Token, Phase, RequestStatus } from "../engine/types";
import { Term } from "./Term";
import { color, space, radius, font, statusTint } from "../theme";

interface TokenStripProps {
  promptTokens: Token[];
  decodedTokens: Token[];
  phase: Phase;
  /** Request status — drives strip styling: finished is grayed, others colored. */
  status?: RequestStatus;
}

// ─── styles ──────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.sm,
  fontFamily: font.mono,
  fontSize: font.size.md,
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 3,
  alignItems: "center",
};

// textTransform and letterSpacing are omitted here so that the Term dotted-underline
// button inside these labels renders in normal title case rather than shouting ALL-CAPS.
const groupLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  fontWeight: font.weight.semibold,
  color: color.textMuted,
  marginRight: space.xs,
  flexShrink: 0,
};

const chipBase: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  padding: `2px ${space.sm}px`,
  borderRadius: radius.sm,
  border: "1px solid transparent",
  whiteSpace: "nowrap",
  lineHeight: 1.4,
};

// Accent-tinted chip: translucent fill + matching text + semi-opaque border,
// the same language as theme's statusTint (kept inline here to extend chipBase).
function tintChip(accent: string): React.CSSProperties {
  return {
    ...chipBase,
    background: `${accent}22`,
    color: accent,
    borderColor: `${accent}66`,
  };
}

const promptChipStyle = tintChip(color.prefill);
const decodeChipStyle = tintChip(color.decode);

const latestChipStyle: React.CSSProperties = {
  ...decodeChipStyle,
  borderColor: color.decode,
  boxShadow: `0 0 0 1px ${color.decode}`,
  fontWeight: font.weight.bold,
};

const phaseBadgeStyle = (phase: Phase): React.CSSProperties => ({
  ...statusTint(phase === "prefill" ? color.textMuted : color.decode),
  fontSize: font.size.sm,
  marginLeft: space.md,
});

const dividerStyle: React.CSSProperties = {
  width: 1,
  height: 20,
  background: color.borderStrong,
  marginInline: space.xs,
  flexShrink: 0,
};

// ─── Component ────────────────────────────────────────────────────────────────

// Status → strip treatment. Only finished/cancelled gray out; live states stay
// colored (via their accent) so running vs. queued vs. swapped read distinctly.
const STATUS_META: Record<RequestStatus, { label: string; accent: string; dim: boolean }> = {
  waiting: { label: "Queued", accent: color.swapped, dim: false },
  running: { label: "Running", accent: color.decode, dim: false },
  swapped: { label: "Swapped", accent: color.warn, dim: false },
  finished: { label: "Finished", accent: color.textFaint, dim: true },
  cancelled: { label: "Cancelled", accent: color.danger, dim: true },
};

function statusBadgeStyle(accent: string): React.CSSProperties {
  return { ...statusTint(accent), fontSize: font.size.sm, marginLeft: space.md };
}

export function TokenStrip({ promptTokens, decodedTokens, phase, status }: TokenStripProps) {
  const lastDecodeIdx = decodedTokens.length - 1;
  const meta = status ? STATUS_META[status] : null;
  // Gray the whole strip only when the request is finished/cancelled.
  const dimStyle: React.CSSProperties = meta?.dim
    ? { opacity: 0.5, filter: "grayscale(0.8)" }
    : {};

  return (
    <div style={{ ...containerStyle, ...dimStyle }} aria-label="Token sequence">
      <div style={rowStyle}>
        {/* Prompt group */}
        {promptTokens.length > 0 && (
          <>
            <span style={groupLabelStyle}><Term tokenKey="prefill">Prompt</Term></span>
            {promptTokens.map((tok) => (
              <span key={tok.id} style={promptChipStyle} title={`id:${tok.id}`}>
                {tok.text}
              </span>
            ))}
          </>
        )}

        {/* Divider between groups when both present */}
        {promptTokens.length > 0 && decodedTokens.length > 0 && (
          <span style={dividerStyle} aria-hidden="true" />
        )}

        {/* Decoded group */}
        {decodedTokens.length > 0 && (
          <>
            <span style={groupLabelStyle}><Term tokenKey="decode">Generated</Term></span>
            {decodedTokens.map((tok, i) => (
              <span
                key={tok.id}
                style={i === lastDecodeIdx ? latestChipStyle : decodeChipStyle}
                title={i === lastDecodeIdx ? `Latest (id:${tok.id})` : `id:${tok.id}`}
                aria-current={i === lastDecodeIdx ? "true" : undefined}
              >
                {tok.text}
              </span>
            ))}
          </>
        )}

        {/* Phase badge */}
        <span style={phaseBadgeStyle(phase)} aria-label={`Phase: ${phase}`}>
          {phase}
        </span>

        {/* Status badge — distinguishes running / queued / swapped / finished */}
        {meta && (
          <span style={statusBadgeStyle(meta.accent)} aria-label={`Status: ${meta.label}`}>
            {meta.label}
          </span>
        )}
      </div>

      {/* Summary line */}
      <div style={{ fontSize: font.size.sm, color: color.textFaint, fontFamily: font.sans }}>
        {promptTokens.length} prompt token{promptTokens.length !== 1 ? "s" : ""}
        {decodedTokens.length > 0 &&
          ` · ${decodedTokens.length} generated`}
      </div>
    </div>
  );
}
