/**
 * Multi-Token Prediction (MTP) — interactive visualizer.
 *
 * Modes:
 *   standard  — 1 token/pass (no MTP)
 *   mtp1      — main head + MTP module ×1 = up to 2 tokens/pass
 *   mtp3      — main head + MTP module ×3 = up to 4 tokens/pass
 *
 * Acceptance rate slider controls how many staged tokens are correct.
 * Wrong predictions are shown as red strikethrough with the correction.
 */

import { useState, useRef, useEffect } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, panel } from "../theme";
import {
  type Mode,
  type StagingToken,
  type VerifiedToken,
  nextPass,
  initialState,
  isDone as mtpIsDone,
  draftSizeForMode,
} from "./mtp";
import { sampleSentences } from "../content/sampleSentences";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function pickPhrase(): string[] {
  return sampleSentences[Math.floor(Math.random() * sampleSentences.length)].split(" ");
}

const ACCEPTANCE_DECAY = 0.55;

/** Per-slot acceptance probability: base * decay^i */
function slotAcceptProb(baseRate: number, slot: number): number {
  return baseRate * Math.pow(ACCEPTANCE_DECAY, slot);
}

/** Builds staging tokens with randomized correct/wrong decisions per slot. */
function buildStagingTokens(
  confirmedLen: number,
  draftSize: number,
  targetPhrase: string[],
  baseAcceptPct: number,
  _passIndex: number,
  idStart: number
): StagingToken[] {
  const tokens: StagingToken[] = [];
  for (let i = 0; i < draftSize; i++) {
    const correctWord = targetPhrase[confirmedLen + i];
    if (correctWord === undefined) break;
    const prob = slotAcceptProb(baseAcceptPct, i);
    if (Math.random() * 100 < prob) {
      tokens.push({ id: idStart + i, text: correctWord });
    } else {
      tokens.push({ id: idStart + i, text: "<token>" });
    }
  }
  return tokens;
}

// ─── Styles ───────────────────────────────────────────────────────────────────

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
  flex: "1 1 120px",
  minWidth: 110,
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

const sliderRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  fontSize: font.size.md,
  color: color.textMuted,
};

const legendChipBase: React.CSSProperties = {
  display: "inline-block",
  padding: `2px ${space.md}px`,
  borderRadius: radius.sm,
  fontFamily: font.mono,
  fontSize: font.size.xs,
};

// ─── Token chip styles ────────────────────────────────────────────────────────

function confirmedChipStyle(isNew: boolean): React.CSSProperties {
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

const stagingChipStyle: React.CSSProperties = {
  padding: `${space.xs}px ${space.md}px`,
  borderRadius: radius.sm,
  border: `1px dashed ${color.waiting}`,
  background: "transparent",
  color: color.waiting,
  fontFamily: font.mono,
  fontSize: font.size.md,
  whiteSpace: "nowrap" as const,
};


// ─── Verification result row ──────────────────────────────────────────────────

function VerificationRow({ tokens }: { tokens: VerifiedToken[] }) {
  if (tokens.length === 0) return null;
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap" as const,
        gap: space.sm,
        alignItems: "center",
        padding: `${space.md}px ${space.lg}px`,
        background: color.panelBgDeep,
        border: `1px solid ${color.border}`,
        borderRadius: radius.md,
        marginTop: space.md,
      }}
    >
      <span style={{ ...sectionLabel, marginRight: space.sm }}>Last verification</span>
      {tokens.map((v) => {
        if (v.status === "accepted") {
          return (
            <span
              key={v.id}
              style={{
                padding: `1px ${space.md}px`,
                borderRadius: radius.sm,
                border: `1px solid ${color.decode}`,
                background: `${color.decode}22`,
                color: color.decode,
                fontFamily: font.mono,
                fontSize: font.size.sm,
                whiteSpace: "nowrap" as const,
              }}
              title="Accepted — prediction matched"
            >
              ✓ {v.predicted}
            </span>
          );
        }
        if (v.status === "rejected") {
          return (
            <span
              key={v.id}
              style={{ display: "inline-flex", alignItems: "center", gap: space.xs, whiteSpace: "nowrap" as const }}
              title={`Rejected — predicted "${v.predicted}", correct is "${v.truth}"`}
            >
              <span
                style={{
                  padding: `1px ${space.md}px`,
                  borderRadius: radius.sm,
                  border: `1px solid ${color.danger}`,
                  background: `${color.danger}22`,
                  color: color.danger,
                  fontFamily: font.mono,
                  fontSize: font.size.sm,
                  textDecoration: "line-through",
                }}
              >
                {v.predicted}
              </span>
              <span style={{ color: color.textFaint, fontSize: font.size.xs }}>→</span>
              <span
                style={{
                  padding: `1px ${space.md}px`,
                  borderRadius: radius.sm,
                  border: `1px solid ${color.decode}`,
                  background: `${color.decode}22`,
                  color: color.decode,
                  fontFamily: font.mono,
                  fontSize: font.size.sm,
                }}
              >
                {v.truth}
              </span>
            </span>
          );
        }
        // discarded
        return (
          <span
            key={v.id}
            style={{
              padding: `1px ${space.md}px`,
              borderRadius: radius.sm,
              border: `1px solid ${color.border}`,
              background: color.panelBgInset,
              color: color.textFaint,
              fontFamily: font.mono,
              fontSize: font.size.sm,
              textDecoration: "line-through",
              whiteSpace: "nowrap" as const,
            }}
            title="Discarded — came after a rejection"
          >
            {v.predicted}
          </span>
        );
      })}
    </div>
  );
}

// ─── Architecture diagram ─────────────────────────────────────────────────────

function TokenChipArch({ label, accent, dashed }: { label: string; accent: string; dashed?: boolean }) {
  return (
    <span
      style={{
        padding: `1px ${space.md}px`,
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

function headBox(accent: string, animating: boolean, label: string, sub?: string): React.CSSProperties {
  return {
    padding: `${space.sm}px ${space.md}px`,
    borderRadius: radius.md,
    border: `1px solid ${animating ? accent : color.border}`,
    background: animating ? `${accent}22` : color.panelBg,
    textAlign: "center" as const,
    fontFamily: font.mono,
    fontSize: font.size.xs,
    color: animating ? accent : color.textMuted,
    transition: "all 0.3s ease",
    minWidth: 90,
    boxShadow: animating ? `0 0 8px ${accent}44` : "none",
  };
  void label; void sub;
}

function ArchDiagram({
  mode,
  isAnimating,
  passes,
  confirmed,
  staging,
  targetPhrase,
}: {
  mode: Mode;
  isAnimating: boolean;
  passes: number;
  confirmed: string[];
  staging: StagingToken[];
  targetPhrase: string[];
}) {
  // Only show the last confirmed token after at least one pass has run
  const lastConfirmedToken = passes > 0 ? (confirmed[confirmed.length - 1] ?? "…") : "…";
  // The token that fed INTO the LLM this pass is the one confirmed the step before
  const llmInputToken = confirmed.length >= 2 ? confirmed[confirmed.length - 2] : null;
  const nextTokens = Array.from({ length: 3 }, (_, i) => staging[i]?.text ?? "…");
  const steps = draftSizeForMode(mode);
  const ac = color.waiting;


  const containerStyle: React.CSSProperties = {
    ...panel,
    background: color.panelBgInset,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: space.sm,
    padding: `${space.md}px ${space.lg}px`,
    transition: "all 0.3s ease",
    boxShadow: isAnimating ? `0 0 14px ${color.prefill}33` : "none",
  };

  // Compact LLM + h row
  const llmBoxStyle: React.CSSProperties = {
    padding: `${space.sm}px ${space.xl}px`,
    borderRadius: radius.md,
    border: `1px solid ${isAnimating ? color.prefill : color.border}`,
    background: isAnimating ? `${color.prefill}22` : color.panelBg,
    textAlign: "center" as const,
    fontFamily: font.mono,
    fontSize: font.size.xs,
    color: isAnimating ? color.prefill : color.textPrimary,
    transition: "all 0.3s ease",
    boxShadow: isAnimating ? `0 0 10px ${color.prefill}44` : "none",
  };

  const hChipStyle: React.CSSProperties = {
    padding: `1px ${space.md}px`,
    borderRadius: radius.sm,
    border: `1px solid ${color.info}55`,
    background: `${color.info}11`,
    fontFamily: font.mono,
    fontSize: font.size.xs,
    color: color.info,
  };

  const arrowSm: React.CSSProperties = {
    fontFamily: font.mono,
    fontSize: font.size.base,
    color: color.textFaint,
    lineHeight: 1,
  };

  return (
    <div style={containerStyle}>
      {/* Compact LLM stack */}
      <span style={{ fontFamily: font.mono, fontSize: font.size.xs, color: color.info }}>
        <Term tokenKey="embeddingLookup">
          emb({llmInputToken != null ? `"${llmInputToken}"` : "t"})
        </Term>
      </span>
      <span style={arrowSm}>↓</span>
      <div style={llmBoxStyle}>LLM (N layers)</div>
      <span style={arrowSm}>↓</span>
      <span style={hChipStyle}>
        <Term tokenKey="hiddenState">h</Term>
      </span>

      {/* Two-column branch from h */}
      <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", justifyContent: "center", gap: space.xxl, width: "100%" }}>

        {/* Left: LM Head */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: space.sm }}>
          <span style={arrowSm}>↓</span>
          <div style={headBox(color.decode, isAnimating, "LM Head")}>
            <div><Term tokenKey="lmHeadTranspose">LM Head</Term></div>
            <div style={{ fontSize: font.size.xs, color: isAnimating ? color.decode : color.textFaint }}>
              <Term tokenKey="lmHeadTranspose">embᵀ</Term>
            </div>
          </div>
          <span style={arrowSm}>↓</span>
          <TokenChipArch label={lastConfirmedToken} accent={color.decode} />
          <span style={{ fontSize: font.size.xs, color: color.textFaint, fontStyle: "italic" as const }}>confirmed</span>
        </div>

        {/* Right: MTP Module — only in mtp modes */}
        {mode !== "standard" && (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: space.sm }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <span style={arrowSm}>↓</span>
              <span style={{ fontSize: font.size.xs, color: color.info, fontFamily: font.mono }}>
                <Term tokenKey="embeddingLookup">
                  emb({lastConfirmedToken === "…" ? "?" : `"${lastConfirmedToken}"`})
                </Term>
              </span>
            </div>
            <div style={{ position: "relative" as const }}>
              <div style={{ ...headBox(ac, isAnimating, "MTP"), minWidth: 120 }}>
                <div><Term tokenKey="mtp">MTP Module</Term></div>
                <div style={{ fontSize: font.size.xs, color: isAnimating ? ac : color.textFaint }}>own transformer block</div>
              </div>
              {steps > 1 && (
                <span style={{
                  position: "absolute" as const, top: -7, right: -7,
                  background: isAnimating ? ac : color.panelBgInset,
                  border: `1px solid ${isAnimating ? ac : color.border}`,
                  color: isAnimating ? color.panelBg : color.textFaint,
                  borderRadius: radius.pill,
                  fontFamily: font.mono, fontSize: font.size.xs, fontWeight: font.weight.bold,
                  padding: "1px 5px", lineHeight: 1.4, transition: "all 0.3s ease",
                }}>
                  ×{steps}
                </span>
              )}
            </div>
            <span style={arrowSm}>↓</span>
            <div style={{ display: "flex", gap: space.sm, flexWrap: "wrap" as const, justifyContent: "center" }}>
              {Array.from({ length: steps }, (_, i) => (
                <TokenChipArch key={i} label={nextTokens[i]} accent={ac} dashed />
              ))}
            </div>
            <span style={{ fontSize: font.size.xs, color: color.textFaint, fontStyle: "italic" as const, textAlign: "center" as const }}>
              {steps > 1 ? `${steps} recursive steps` : "speculative"}
            </span>
          </div>
        )}
      </div>

    </div>
  );
}

// ─── Legend (collapsible) ─────────────────────────────────────────────────────

function LegendRow({ mode }: { mode: Mode }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: space.md }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: color.textFaint, fontFamily: font.sans, fontSize: font.size.xs,
          padding: 0, display: "flex", alignItems: "center", gap: space.xs,
        }}
        aria-expanded={open}
      >
        <span style={{ fontFamily: font.mono, fontSize: "0.65em" }}>{open ? "▾" : "▸"}</span>
        Color legend
      </button>
      {open && (
        <div style={{ display: "flex", gap: space.md, marginTop: space.sm, flexWrap: "wrap" as const }}>
          <span style={{ ...legendChipBase, border: `1px solid ${color.decode}`, background: `${color.decode}22`, color: color.decode }}>Confirmed (prev)</span>
          <span style={{ ...legendChipBase, border: `1px solid ${color.decode}`, background: `${color.decode}44`, color: color.decode, fontWeight: font.weight.semibold }}>New this pass</span>
          {mode !== "standard" && <>
            <span style={{ ...legendChipBase, border: `1px dashed ${color.waiting}`, background: "transparent", color: color.waiting }}>Staging (speculative)</span>
            <span style={{ ...legendChipBase, border: `1px solid ${color.danger}`, background: `${color.danger}22`, color: color.danger, textDecoration: "line-through" }}>Rejected</span>
            <span style={{ ...legendChipBase, border: `1px solid ${color.border}`, background: color.panelBgInset, color: color.textFaint, textDecoration: "line-through" }}>Discarded</span>
          </>}
        </div>
      )}
    </div>
  );
}

// ─── Step log (collapsible) ───────────────────────────────────────────────────

function LogPanel({ logLines, logEndRef }: { logLines: string[]; logEndRef: React.RefObject<HTMLDivElement | null> }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: color.textFaint, fontFamily: font.sans, fontSize: font.size.xs,
          padding: 0, display: "flex", alignItems: "center", gap: space.xs,
        }}
        aria-expanded={open}
      >
        <span style={{ fontFamily: font.mono, fontSize: "0.65em" }}>{open ? "▾" : "▸"}</span>
        Step log ({logLines.filter(l => l.startsWith("Pass")).length} passes)
      </button>
      {open && (
        <pre style={{
          marginTop: space.sm,
          background: color.panelBgInset,
          border: `1px solid ${color.border}`,
          borderRadius: radius.lg,
          padding: `${space.lg}px`,
          fontFamily: font.mono,
          fontSize: font.size.xs,
          color: color.textSecondary,
          lineHeight: 1.7,
          maxHeight: 320,
          overflowY: "auto",
          whiteSpace: "pre-wrap" as const,
          wordBreak: "break-word" as const,
          margin: `${space.sm}px 0 0`,
        }}>
          {logLines.join("\n")}
          <div ref={logEndRef} />
        </pre>
      )}
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export function StageSpeculative(_props: StageProps) {
  const [mode, setMode] = useState<Mode>("standard");
  const [acceptPct, setAcceptPct] = useState(75);
  const [targetPhrase, setTargetPhrase] = useState<string[]>(pickPhrase);
  const [simState, setSimState] = useState(initialState());
  const [isAnimating, setIsAnimating] = useState(false);
  // splitIdx: index in confirmed[] where "new this pass" begins — persists between passes
  const [splitIdx, setSplitIdx] = useState<number | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const animTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const { confirmed, staging, passes, lastVerification } = simState;
  const done = mtpIsDone(simState, targetPhrase);

  function triggerAnimation(newSplitIdx: number) {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setIsAnimating(true);
    setSplitIdx(newSplitIdx);
    animTimerRef.current = setTimeout(() => setIsAnimating(false), 650);
    // splitIdx intentionally NOT cleared — it stays to keep the separator visible
  }

  function handleNextPass() {
    if (done) return;
    const draftSize = draftSizeForMode(mode);

    // How many staging tokens will be accepted on this pass
    let acceptedCount = 0;
    for (let i = 0; i < staging.length; i++) {
      if (staging[i].text === targetPhrase[confirmed.length + i]) acceptedCount++;
      else break;
    }

    // After this pass confirmed grows by acceptedCount + 1 (main head)
    const newConfirmedLen = confirmed.length + acceptedCount + 1;

    const newStaging = buildStagingTokens(
      newConfirmedLen,
      draftSize,
      targetPhrase,
      acceptPct,
      passes + 1,
      simState.idCounter + draftSize
    );

    // splitIdx = current confirmed.length so the newly confirmed tokens light up
    const next = nextPass(simState, mode, targetPhrase, newStaging);
    setSimState(next);
    triggerAnimation(confirmed.length);

    // Build log entry
    const p = passes + 1;
    const lines: string[] = [];
    if (mode === "standard") {
      const emitted = targetPhrase[confirmed.length] ?? "…";
      lines.push(`Pass ${p} [standard]  emit="${emitted}"  confirmed=[${[...confirmed, emitted].join(", ")}]`);
    } else {
      const stagingDesc = staging.length > 0
        ? (() => {
            let rejected = false;
            return staging.map((st, i) => {
              if (rejected) return `"${st.text}"~`;
              const truth = targetPhrase[confirmed.length + i];
              if (st.text === truth) return `"${st.text}"✓`;
              rejected = true;
              return `"${st.text}"✗→"${truth}"`;
            }).join("  ");
          })()
        : "(none)";
      const verif = next.lastVerification.map((v) =>
        v.status === "accepted" ? `"${v.predicted}"✓`
        : v.status === "rejected" ? `"${v.predicted}"✗→"${v.truth}"`
        : `"${v.predicted}"~`
      ).join("  ");
      const newConfirmedTokens = next.confirmed.slice(confirmed.length);
      lines.push(
        `Pass ${p} [${mode}]`,
        `  staging_in:    ${stagingDesc || "(none)"}`,
        `  verification:  ${verif || "(first pass — no staging yet)"}`,
        `  new_confirmed: [${newConfirmedTokens.join(", ")}]`,
        `  new_staging:   [${newStaging.map(s => `"${s.text}"`).join(", ")}]`,
        `  confirmed:     [${next.confirmed.join(", ")}]  (${next.confirmed.length}/${targetPhrase.length})`,
      );
    }
    setLogLines((prev) => [...prev, ...lines, ""]);
  }

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines]);

  function handleModeChange(next: Mode) {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setMode(next);
    setSimState(initialState());
    setIsAnimating(false);
    setSplitIdx(null);
    setLogLines([]);
  }

  function handleReset() {
    if (animTimerRef.current) clearTimeout(animTimerRef.current);
    setTargetPhrase(pickPhrase());
    setSimState(initialState());
    setIsAnimating(false);
    setSplitIdx(null);
    setLogLines([]);
  }

  const tokensPerPass = passes > 0 ? (confirmed.length / passes).toFixed(2) : "—";

  let statusLine: string;
  if (passes === 0) {
    statusLine = "Press Next Forward Pass to begin.";
  } else if (done) {
    statusLine = `Complete! ${targetPhrase.length} tokens in ${passes} passes.`;
  } else if (mode === "standard") {
    statusLine = `Pass ${passes}: confirmed "${confirmed[confirmed.length - 1]}".`;
  } else {
    const accepted = lastVerification.filter((v) => v.status === "accepted").length;
    const rejected = lastVerification.filter((v) => v.status === "rejected").length;
    const discarded = lastVerification.filter((v) => v.status === "discarded").length;
    const parts = [];
    if (accepted > 0) parts.push(`${accepted} accepted`);
    if (rejected > 0) parts.push(`${rejected} rejected`);
    if (discarded > 0) parts.push(`${discarded} discarded`);
    statusLine = `Pass ${passes}: ${parts.join(", ")} — main head corrects & confirms.`;
  }

  return (
    <div style={outerStyle} aria-label="Multi-Token Prediction visualization">

      {/* Zone A — Controls & Metrics */}
      <div style={{ display: "flex", flexDirection: "column", gap: space.lg }}>

        {/* Mode selector */}
        <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" as const }} role="group" aria-label="MTP mode">
          {(["standard", "mtp1", "mtp3"] as Mode[]).map((m) => {
            const labels: Record<Mode, string> = { standard: "Standard", mtp1: "MTP (k=1)", mtp3: "MTP (k=3)" };
            return (
              <button key={m} style={mode === m ? modeBtnActive : modeBtnBase}
                onClick={() => handleModeChange(m)} aria-pressed={mode === m}>
                {labels[m]}
              </button>
            );
          })}
        </div>

        {/* Acceptance rate slider — only for MTP modes */}
        {mode !== "standard" && (
          <div style={{ display: "flex", flexDirection: "column", gap: space.sm }}>
            <label style={sliderRowStyle}>
              <span style={{ minWidth: 140 }}>
                <Term tokenKey="acceptanceRate">Slot 0</Term> accept = {acceptPct}%
              </span>
              <input type="range" min={10} max={100} step={5} value={acceptPct}
                onChange={(e) => setAcceptPct(Number(e.target.value))}
                aria-label="Acceptance rate" style={{ flex: 1 }} />
            </label>
            {/* Decay readout + expected tokens/pass */}
            <div style={{ display: "flex", gap: space.md, paddingLeft: 140, alignItems: "center", flexWrap: "wrap" as const }}>
              {Array.from({ length: draftSizeForMode(mode) }, (_, i) => {
                const pct = Math.round(slotAcceptProb(acceptPct, i));
                const accent = pct >= 60 ? color.decode : pct >= 30 ? color.waiting : color.danger;
                return (
                  <span key={i} style={{
                    fontFamily: font.mono, fontSize: font.size.xs,
                    color: accent,
                    border: `1px solid ${accent}55`,
                    background: `${accent}11`,
                    borderRadius: radius.sm,
                    padding: `1px ${space.sm}px`,
                  }}>
                    k={i + 1}: {pct}%
                  </span>
                );
              })}
              {(() => {
                // E[tokens/pass] = 1 + Σ_d P(all slots 0..d accepted)
                const k = draftSizeForMode(mode);
                let expected = 1;
                let cumProb = 1;
                for (let i = 0; i < k; i++) {
                  cumProb *= slotAcceptProb(acceptPct, i) / 100;
                  expected += cumProb;
                }
                return (
                  <span style={{
                    fontFamily: font.mono, fontSize: font.size.xs,
                    color: color.textSecondary,
                    marginLeft: space.sm,
                  }}>
                    expected ≈ {expected.toFixed(2)}/pass
                  </span>
                );
              })()}
            </div>
          </div>
        )}

        {/* Action row */}
        <div style={{ display: "flex", gap: space.md, alignItems: "center" }}>
          <button style={done ? actionBtnDisabledStyle : actionBtnStyle}
            onClick={handleNextPass} disabled={done} aria-label="Advance one forward pass">
            ▶ Next Forward Pass
          </button>
          <button style={resetBtnStyle} onClick={handleReset} aria-label="Reset">Reset</button>
        </div>

        {/* Metrics */}
        <div style={{ display: "flex", gap: space.md, flexWrap: "wrap" as const }}>
          <div style={statChipStyle}>
            <span style={{ fontSize: font.size.xxl, fontWeight: font.weight.bold, fontFamily: font.mono, color: color.accent }}>{passes}</span>
            <span style={statLabelStyle}>Forward Passes</span>
          </div>
          <div style={statChipStyle}>
            <span style={{ fontSize: font.size.xxl, fontWeight: font.weight.bold, fontFamily: font.mono, color: color.accent }}>{confirmed.length}</span>
            <span style={statLabelStyle}>Confirmed</span>
          </div>
          <div style={statChipStyle}>
            <span style={{ fontSize: font.size.xxl, fontWeight: font.weight.bold, fontFamily: font.mono, color: color.decode }}>
              {tokensPerPass}{passes > 0 ? "×" : ""}
            </span>
            <span style={statLabelStyle}>Actual tokens/pass</span>
          </div>
          {mode !== "standard" && (
            <div style={statChipStyle}>
              <span style={{ fontSize: font.size.xxl, fontWeight: font.weight.bold, fontFamily: font.mono, color: color.waiting }}>{staging.length}</span>
              <span style={statLabelStyle}>Staging</span>
            </div>
          )}
        </div>
      </div>

      {/* Zone B — Architecture Diagram (compact) */}
      <div>
        <h3 style={sectionHeadingStyle}><Term tokenKey="mtp">Decode Architecture</Term></h3>
        <ArchDiagram mode={mode} isAnimating={isAnimating} passes={passes}
          confirmed={confirmed} staging={staging} targetPhrase={targetPhrase} />
      </div>

      {/* Zone C — Output Stream */}
      <div>
        <h3 style={sectionHeadingStyle}>Output Stream</h3>

        {/* Token tape: [prev] | [new this pass] -- [staging] */}
        <div style={{
          background: color.panelBgInset, border: `1px solid ${color.border}`,
          borderRadius: radius.lg, padding: `${space.lg}px`, overflowX: "auto",
        }} aria-label="Output token stream" aria-live="polite">
          <div style={{ display: "flex", flexWrap: "wrap" as const, gap: space.sm, alignItems: "center", minHeight: 36 }}>
            {confirmed.length === 0 && staging.length === 0 ? (
              <span style={{ color: color.textFaint, fontFamily: font.mono, fontSize: font.size.base }}>
                No tokens yet — press Next Forward Pass to start.
              </span>
            ) : (
              <>
                {confirmed.map((text, idx) => {
                  const isNew = splitIdx !== null && idx >= splitIdx;
                  return (
                    <span key={idx} style={confirmedChipStyle(isNew)} title={`Token ${idx + 1}`}>{text}</span>
                  );
                })}
                {/* Vertical divider before staging */}
                {staging.length > 0 && (
                  <>
                    <span
                      style={{
                        alignSelf: "stretch",
                        width: 1,
                        background: color.border,
                        margin: `0 ${space.xs}px`,
                        flexShrink: 0,
                      }}
                      aria-hidden="true"
                    />
                    {staging.map((st) => (
                      <span key={st.id} style={stagingChipStyle} title="Speculatively staged — pending verification">{st.text}</span>
                    ))}
                  </>
                )}
              </>
            )}
          </div>
        </div>

        {/* Verification result for last pass */}
        {mode !== "standard" && <VerificationRow tokens={lastVerification} />}

        {/* Legend toggle */}
        <LegendRow mode={mode} />

        {/* Status line */}
        <p style={{ margin: `${space.md}px 0 0`, fontSize: font.size.sm, color: color.textFaint, fontFamily: font.sans }}>
          {statusLine}
        </p>
      </div>

      {/* Zone D — Step Log (collapsible) */}
      {logLines.length > 0 && (
        <LogPanel logLines={logLines} logEndRef={logEndRef} />
      )}
    </div>
  );
}
