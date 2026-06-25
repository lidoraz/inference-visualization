/**
 * Multi-Token Prediction (MTP) — interactive visualizer.
 *
 * Demonstrates how DeepSeek-style MTP works: a single MTP head (or chain of
 * heads) branches off the SAME final hidden state `h` that feeds the main LM
 * head, speculatively predicting k extra tokens per forward pass. Because the
 * demo target phrase is fixed, all predictions are always correct, letting the
 * user focus on the structural insight rather than stochastic acceptance.
 *
 * Modes:
 *   standard — 1 token/pass (no MTP)
 *   mtp1     — main head + 1 MTP head = up to 2 tokens/pass
 *   mtp3     — main head + 3 chained MTP heads = up to 4 tokens/pass
 */

import { useState, useRef } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, panel } from "../theme";
import { type Mode, nextPass, initialState, isDone as mtpIsDone } from "./mtp";

// ─── Constants ────────────────────────────────────────────────────────────────

const TARGET_PHRASE = [
  "The", "capital", "of", "France", "is", "Paris", ",", "which", "is", "beautiful", ".",
];

// ─── Static styles ────────────────────────────────────────────────────────────

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

const statLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: color.textFaint,
};

const modeBtnBase: React.CSSProperties = {
  padding: `${space.sm}px ${space.lg}px`,
  borderRadius: radius.md,
  border: `1px solid ${color.border}`,
  background: color.panelBgInset,
  color: color.textMuted,
  fontFamily: font.sans,
  fontSize: font.size.base,
  cursor: "pointer",
  transition: "all 0.2s ease",
};

const modeBtnActive: React.CSSProperties = {
  ...modeBtnBase,
  background: `${color.accent}22`,
  border: `1px solid ${color.accent}`,
  color: color.accent,
  fontWeight: font.weight.semibold,
};

const actionBtnStyle: React.CSSProperties = {
  padding: `${space.md}px ${space.xl}px`,
  borderRadius: radius.md,
  border: `1px solid ${color.decode}`,
  background: `${color.decode}22`,
  color: color.decode,
  fontFamily: font.mono,
  fontSize: font.size.base,
  fontWeight: font.weight.semibold,
  cursor: "pointer",
  transition: "all 0.2s ease",
};

const actionBtnDisabledStyle: React.CSSProperties = {
  ...actionBtnStyle,
  opacity: 0.4,
  cursor: "not-allowed",
  border: `1px solid ${color.border}`,
  background: color.panelBgInset,
  color: color.textFaint,
};

const resetBtnStyle: React.CSSProperties = {
  padding: `${space.md}px ${space.xl}px`,
  borderRadius: radius.md,
  border: `1px solid ${color.border}`,
  background: color.panelBgInset,
  color: color.textMuted,
  fontFamily: font.sans,
  fontSize: font.size.base,
  cursor: "pointer",
  transition: "all 0.2s ease",
};

const archPanelStyle: React.CSSProperties = {
  ...panel,
  background: color.panelBgInset,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: space.md,
  padding: `${space.xl}px`,
  transition: "all 0.3s ease",
};

const arrowStyle: React.CSSProperties = {
  fontFamily: font.mono,
  fontSize: font.size.xl,
  color: color.textFaint,
  lineHeight: 1,
};

const trunkBoxBase: React.CSSProperties = {
  width: 200,
  padding: `${space.lg}px ${space.xl}px`,
  borderRadius: radius.lg,
  border: `1px solid ${color.border}`,
  background: color.panelBg,
  textAlign: "center",
  fontFamily: font.mono,
  fontSize: font.size.base,
  color: color.textPrimary,
  transition: "all 0.3s ease",
};

function trunkBoxAnimating(): React.CSSProperties {
  return {
    ...trunkBoxBase,
    background: `${color.prefill}22`,
    border: `1px solid ${color.prefill}`,
    boxShadow: `0 0 12px ${color.prefill}55`,
    color: color.prefill,
  };
}

function headBoxStyle(accent: string, animating: boolean): React.CSSProperties {
  return {
    padding: `${space.md}px ${space.lg}px`,
    borderRadius: radius.md,
    border: `1px solid ${animating ? accent : color.border}`,
    background: animating ? `${accent}22` : color.panelBg,
    textAlign: "center",
    fontFamily: font.mono,
    fontSize: font.size.sm,
    color: animating ? accent : color.textMuted,
    transition: "all 0.3s ease",
    minWidth: 100,
    boxShadow: animating ? `0 0 8px ${accent}44` : "none",
  };
}

function confirmedTokenChip(isNew: boolean): React.CSSProperties {
  return {
    padding: `${space.xs}px ${space.md}px`,
    borderRadius: radius.sm,
    border: `1px solid ${color.decode}`,
    background: isNew ? `${color.decode}44` : `${color.decode}22`,
    color: color.decode,
    fontFamily: font.mono,
    fontSize: font.size.md,
    fontWeight: isNew ? font.weight.semibold : font.weight.normal,
    transition: "all 0.3s ease",
    whiteSpace: "nowrap" as const,
  };
}

const stagingTokenChip: React.CSSProperties = {
  padding: `${space.xs}px ${space.md}px`,
  borderRadius: radius.sm,
  border: `1px dashed ${color.waiting}`,
  background: "transparent",
  color: color.waiting,
  fontFamily: font.mono,
  fontSize: font.size.md,
  whiteSpace: "nowrap" as const,
  transition: "all 0.3s ease",
};

const separatorDotStyle: React.CSSProperties = {
  color: color.textFaint,
  fontFamily: font.mono,
  fontSize: font.size.xl,
  lineHeight: 1,
  padding: `0 ${space.xs}px`,
  alignSelf: "center",
};

const legendChipBase: React.CSSProperties = {
  display: "inline-block",
  padding: `2px ${space.md}px`,
  borderRadius: radius.sm,
  fontFamily: font.mono,
  fontSize: font.size.xs,
};

// ─── Architecture diagram sub-components ─────────────────────────────────────

function HiddenStateLabel() {
  return (
    <span
      style={{
        fontFamily: font.mono,
        fontSize: font.size.xs,
        color: color.textFaint,
        letterSpacing: "0.04em",
      }}
    >
      h<sub>t</sub> (hidden state)
    </span>
  );
}

function TokenChipArch({
  label,
  accent,
  dashed,
}: {
  label: string;
  accent: string;
  dashed?: boolean;
}) {
  return (
    <span
      style={{
        padding: `${space.xs}px ${space.md}px`,
        borderRadius: radius.sm,
        border: dashed ? `1px dashed ${accent}` : `1px solid ${accent}`,
        background: dashed ? "transparent" : `${accent}22`,
        color: accent,
        fontFamily: font.mono,
        fontSize: font.size.xs,
        whiteSpace: "nowrap" as const,
      }}
    >
      {label}
    </span>
  );
}

function BranchColumn({
  headLabel,
  tokenLabel,
  accent,
  dashed,
  isAnimating,
}: {
  headLabel: string;
  tokenLabel: string;
  accent: string;
  dashed?: boolean;
  isAnimating: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: space.sm,
      }}
    >
      <div style={headBoxStyle(accent, isAnimating)}>{headLabel}</div>
      <span style={arrowStyle}>↓</span>
      <TokenChipArch label={tokenLabel} accent={accent} dashed={dashed} />
    </div>
  );
}

function MtpChainColumn({ isAnimating }: { isAnimating: boolean }) {
  const heads = [
    { label: "MTP Head 1", token: "T_n+1 ○" },
    { label: "MTP Head 2", token: "T_n+2 ○" },
    { label: "MTP Head 3", token: "T_n+3 ○" },
  ];
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: space.xs,
      }}
    >
      {heads.map((h, i) => (
        <div
          key={h.label}
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: space.xs,
          }}
        >
          <div style={headBoxStyle(color.waiting, isAnimating)}>{h.label}</div>
          <span style={arrowStyle}>↓</span>
          <TokenChipArch label={h.token} accent={color.waiting} dashed />
          {i < 2 && (
            <span
              style={{
                fontSize: font.size.xs,
                color: color.textFaint,
                fontFamily: font.sans,
                fontStyle: "italic",
                marginTop: space.xs,
              }}
            >
              ↓ feeds into
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

function ArchDiagram({
  mode,
  isAnimating,
}: {
  mode: Mode;
  isAnimating: boolean;
}) {
  const captionMap: Record<Mode, string> = {
    standard: "Main model emits 1 token per forward pass.",
    mtp1: "MTP head reuses the same hidden state h to speculatively predict 1 extra token.",
    mtp3: "3 MTP heads chain off h, each feeding into the next — 3 extra speculative tokens per pass.",
  };

  return (
    <div
      style={{
        ...archPanelStyle,
        boxShadow: isAnimating ? `0 0 16px ${color.prefill}33` : "none",
      }}
    >
      <HiddenStateLabel />
      <span style={arrowStyle}>↓</span>
      <div style={isAnimating ? trunkBoxAnimating() : trunkBoxBase}>
        Transformer Trunk
      </div>
      <span style={arrowStyle}>↓</span>

      <div
        style={{
          display: "flex",
          flexDirection: "row",
          alignItems: "flex-start",
          justifyContent: "center",
          gap: space.xxl,
        }}
      >
        <BranchColumn
          headLabel="Main Head"
          tokenLabel="T_n ✓ confirmed"
          accent={color.decode}
          isAnimating={isAnimating}
        />

        {mode === "mtp1" && (
          <BranchColumn
            headLabel="MTP Head"
            tokenLabel="T_n+1 ○ staging"
            accent={color.waiting}
            dashed
            isAnimating={isAnimating}
          />
        )}

        {mode === "mtp3" && <MtpChainColumn isAnimating={isAnimating} />}
      </div>

      <span
        style={{
          fontSize: font.size.xs,
          color: color.textFaint,
          fontFamily: font.sans,
          textAlign: "center",
          maxWidth: 380,
          lineHeight: 1.5,
        }}
      >
        {captionMap[mode]}
      </span>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export function StageSpeculative(_props: StageProps) {
  const [mode, setMode] = useState<Mode>("standard");
  const [simState, setSimState] = useState(initialState());
  const [isAnimating, setIsAnimating] = useState(false);
  const [newFromIdx, setNewFromIdx] = useState<number | null>(null);
  const animTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { confirmed, staging, passes } = simState;
  const done = mtpIsDone(simState, TARGET_PHRASE);

  function triggerAnimation(fromIdx: number) {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setIsAnimating(true);
    setNewFromIdx(fromIdx);
    animTimerRef.current = setTimeout(() => {
      setIsAnimating(false);
      setNewFromIdx(null);
    }, 650);
  }

  function handleNextPass() {
    if (done) return;
    const prevConfirmedLen = simState.confirmed.length + simState.staging.length;
    const next = nextPass(simState, mode, TARGET_PHRASE);
    setSimState(next);
    triggerAnimation(prevConfirmedLen);
  }

  function handleModeChange(next: Mode) {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setMode(next);
    setSimState(initialState());
    setIsAnimating(false);
    setNewFromIdx(null);
  }

  function handleReset() {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setSimState(initialState());
    setIsAnimating(false);
    setNewFromIdx(null);
  }

  // Metrics
  const tokensPerPass = passes > 0 ? (confirmed.length / passes).toFixed(2) : "—";

  // Status line
  let statusLine: string;
  if (passes === 0) {
    statusLine = "Press Next Forward Pass to begin.";
  } else if (done) {
    statusLine = `Complete! ${TARGET_PHRASE.length} tokens in ${passes} passes.`;
  } else {
    const newCount = newFromIdx !== null ? confirmed.length - newFromIdx : 0;
    if (mode === "standard") {
      statusLine = `Pass ${passes}: main head confirmed "${confirmed[confirmed.length - 1]}".`;
    } else {
      const stagingCount = staging.length;
      statusLine =
        `Pass ${passes}: ${newCount} token${newCount !== 1 ? "s" : ""} confirmed` +
        (stagingCount > 0 ? `, ${stagingCount} staged speculatively.` : ".");
    }
  }

  return (
    <div style={outerStyle} aria-label="Multi-Token Prediction visualization">

      {/* Zone A — Controls & Metrics */}
      <div style={{ display: "flex", flexDirection: "column", gap: space.lg }}>

        {/* Mode selector */}
        <div
          style={{ display: "flex", gap: space.md, flexWrap: "wrap" as const }}
          role="group"
          aria-label="Select MTP mode"
        >
          {(["standard", "mtp1", "mtp3"] as Mode[]).map((m) => {
            const labels: Record<Mode, string> = {
              standard: "Standard",
              mtp1: "MTP-1",
              mtp3: "MTP-3",
            };
            return (
              <button
                key={m}
                style={mode === m ? modeBtnActive : modeBtnBase}
                onClick={() => handleModeChange(m)}
                aria-pressed={mode === m}
              >
                {labels[m]}
              </button>
            );
          })}
        </div>

        {/* Action row */}
        <div style={{ display: "flex", gap: space.md, alignItems: "center" }}>
          <button
            style={done ? actionBtnDisabledStyle : actionBtnStyle}
            onClick={handleNextPass}
            disabled={done}
            aria-label="Advance one forward pass"
          >
            ▶ Next Forward Pass
          </button>
          <button style={resetBtnStyle} onClick={handleReset} aria-label="Reset simulation">
            Reset
          </button>
        </div>

        {/* Metrics row */}
        <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" as const }}>
          <div style={statChipStyle}>
            <span
              style={{
                fontSize: font.size.xxl,
                fontWeight: font.weight.bold,
                fontFamily: font.mono,
                color: color.accent,
              }}
            >
              {passes}
            </span>
            <span style={statLabelStyle}>Forward Passes</span>
          </div>
          <div style={statChipStyle}>
            <span
              style={{
                fontSize: font.size.xxl,
                fontWeight: font.weight.bold,
                fontFamily: font.mono,
                color: color.accent,
              }}
            >
              {confirmed.length}
            </span>
            <span style={statLabelStyle}>Tokens Confirmed</span>
          </div>
          <div style={statChipStyle}>
            <span
              style={{
                fontSize: font.size.xxl,
                fontWeight: font.weight.bold,
                fontFamily: font.mono,
                color: color.decode,
              }}
            >
              {tokensPerPass}
              {passes > 0 ? "×" : ""}
            </span>
            <span style={statLabelStyle}>Tokens / Pass</span>
          </div>
          {mode !== "standard" && (
            <div style={statChipStyle}>
              <span
                style={{
                  fontSize: font.size.xxl,
                  fontWeight: font.weight.bold,
                  fontFamily: font.mono,
                  color: color.waiting,
                }}
              >
                {staging.length}
              </span>
              <span style={statLabelStyle}>Staging</span>
            </div>
          )}
        </div>
      </div>

      {/* Zone B — Architecture Diagram */}
      <div>
        <h3 style={sectionHeadingStyle}>
          <Term tokenKey="mtp">Model Architecture</Term>
        </h3>
        <ArchDiagram mode={mode} isAnimating={isAnimating} />
      </div>

      {/* Zone C — Output Stream */}
      <div>
        <h3 style={sectionHeadingStyle}>Output Stream</h3>

        {/* Token tape */}
        <div
          style={{
            background: color.panelBgInset,
            border: `1px solid ${color.border}`,
            borderRadius: radius.lg,
            padding: `${space.lg}px`,
            overflowX: "auto",
          }}
          aria-label="Output token stream"
          aria-live="polite"
        >
          <div
            style={{
              display: "flex",
              flexWrap: "wrap" as const,
              gap: space.sm,
              alignItems: "center",
              minHeight: 36,
            }}
          >
            {confirmed.length === 0 && staging.length === 0 ? (
              <span
                style={{
                  color: color.textFaint,
                  fontFamily: font.mono,
                  fontSize: font.size.base,
                }}
              >
                No tokens yet — press Next Forward Pass to start.
              </span>
            ) : (
              <>
                {confirmed.map((text, idx) => {
                  const isNew = newFromIdx !== null && isAnimating && idx >= newFromIdx;
                  return (
                    <span key={idx} style={confirmedTokenChip(isNew)} title={`Token ${idx + 1}`}>
                      {text}
                    </span>
                  );
                })}
                {staging.length > 0 && (
                  <>
                    <span style={separatorDotStyle} aria-hidden="true">
                      ·
                    </span>
                    {staging.map((st) => (
                      <span
                        key={st.id}
                        style={stagingTokenChip}
                        title="Speculatively staged — pending verification"
                      >
                        {st.text}
                      </span>
                    ))}
                  </>
                )}
              </>
            )}
          </div>
        </div>

        {/* Legend */}
        <div
          style={{
            display: "flex",
            gap: space.md,
            marginTop: space.md,
            alignItems: "center",
            flexWrap: "wrap" as const,
          }}
          aria-label="Token legend"
        >
          <span
            style={{
              ...legendChipBase,
              border: `1px solid ${color.decode}`,
              background: `${color.decode}22`,
              color: color.decode,
            }}
          >
            Confirmed
          </span>
          {mode !== "standard" && (
            <span
              style={{
                ...legendChipBase,
                border: `1px dashed ${color.waiting}`,
                background: "transparent",
                color: color.waiting,
              }}
            >
              Staging
            </span>
          )}
        </div>

        {/* Status line */}
        <p
          style={{
            margin: `${space.md}px 0 0`,
            fontSize: font.size.sm,
            color: color.textFaint,
            fontFamily: font.sans,
          }}
        >
          {statusLine}
        </p>
      </div>
    </div>
  );
}
