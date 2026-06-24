/**
 * Stage 1 — Single-Request Lifecycle
 * Visualizes the prefill → decode lifecycle for a single request.
 */

import type { StageProps } from "./types";
import type { Request } from "../engine/types";
import { TokenStrip } from "../components/TokenStrip";
import { Term } from "../components/Term";
import { BlockGrid } from "../components/BlockGrid";
import { color, space, radius, font, sectionLabel, notePanel, statusTint } from "../theme";

// ─── helpers ─────────────────────────────────────────────────────────────────

/** Pick the most interesting request to focus on for Stage 1. */
function pickFocusRequest(requests: Request[]): Request | null {
  if (requests.length === 0) return null;

  // Prefer running (prefill or decode in progress)
  const running = requests.find((r) => r.status === "running");
  if (running) return running;

  // Then waiting
  const waiting = requests.find((r) => r.status === "waiting");
  if (waiting) return waiting;

  // Then the most recently finished (highest id, non-cancelled)
  const finished = [...requests]
    .filter((r) => r.status === "finished")
    .sort((a, b) => b.id - a.id);
  if (finished.length > 0) return finished[0];

  // Fall back to any non-cancelled request with the highest id
  const any = [...requests]
    .filter((r) => r.status !== "cancelled")
    .sort((a, b) => b.id - a.id);
  return any.length > 0 ? any[0] : null;
}

// ─── sub-components ──────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: Request["status"] }) {
  // Status is the one place color is meaningful. Each badge tints a translucent
  // version of its accent (derived from the theme palette) — no off-palette hex.
  const map: Record<Request["status"], { label: string; accent: string }> = {
    waiting: { label: "Waiting", accent: color.accent },
    running: { label: "Running", accent: color.decode },
    swapped: { label: "Swapped", accent: color.warn },
    finished: { label: "Finished", accent: color.prefill },
    cancelled: { label: "Cancelled", accent: color.danger },
  };
  const s = map[status];
  return (
    <span style={statusTint(s.accent)} aria-label={`Request status: ${s.label}`}>
      {s.label}
    </span>
  );
}

function PhaseReadout({
  request,
  tick,
}: {
  request: Request;
  tick: number;
}) {
  const { phase, status, decodedTokens, maxDecode } = request;
  const decoded = decodedTokens.length;

  const dividerStyle: React.CSSProperties = { color: color.border };
  const subLabelStyle: React.CSSProperties = {
    fontSize: font.size.xs,
    color: color.textFaint,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
  };

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: space.xl,
        alignItems: "center",
        padding: `10px 14px`,
        borderRadius: radius.lg,
        background: color.panelBg,
        border: `1px solid ${color.border}`,
        fontFamily: font.sans,
        fontSize: font.size.base,
        color: color.textPrimary,
      }}
      aria-label="Phase and progress readout"
    >
      {/* Tick */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <span style={subLabelStyle}>Tick</span>
        <span style={{ fontWeight: font.weight.bold, fontSize: font.size.xl, fontFamily: font.mono, color: color.prefill }}>
          {tick}
        </span>
      </div>

      <span style={dividerStyle}>|</span>

      {/* Status */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <span style={subLabelStyle}>Status</span>
        <StatusBadge status={status} />
      </div>

      <span style={dividerStyle}>|</span>

      {/* Phase */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <span style={subLabelStyle}>Phase</span>
        <span style={{ fontWeight: font.weight.semibold }}>
          {status === "waiting" ? (
            <span style={{ color: color.textMuted }}>—</span>
          ) : (
            <Term tokenKey={phase === "prefill" ? "prefill" : "decode"}>{phase}</Term>
          )}
        </span>
      </div>

      {/* Decode progress (only meaningful in decode phase or finished) */}
      {(phase === "decode" || status === "finished") && (
        <>
          <span style={dividerStyle}>|</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
            <span style={subLabelStyle}>Decoded</span>
            <span style={{ fontFamily: font.mono, fontWeight: font.weight.semibold, color: color.decode }}>
              {decoded} / {maxDecode}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

/**
 * One quiet, neutral status line whose text changes with the request's phase.
 * Replaces the previous set of differently-colored callout panels — color is
 * reserved for the status badge; the depth lives in the Term tooltips + guide.
 */
function ContextLine({ children }: { children: React.ReactNode }) {
  return (
    <div role="note" style={notePanel}>
      {children}
    </div>
  );
}

function DecodeProgressBar({ decoded, maxDecode }: { decoded: number; maxDecode: number }) {
  const pct = maxDecode > 0 ? Math.min(decoded / maxDecode, 1) : 0;
  return (
    <div aria-label={`Decode progress: ${decoded} of ${maxDecode} tokens`}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: font.size.sm,
          fontFamily: font.sans,
          color: color.textFaint,
          marginBottom: space.xs,
        }}
      >
        <span>Decode progress</span>
        <span style={{ fontFamily: font.mono }}>
          {decoded} / {maxDecode} tokens
        </span>
      </div>
      <div
        style={{
          height: 8,
          borderRadius: radius.sm,
          background: color.border,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${Math.round(pct * 100)}%`,
            background: pct >= 1 ? color.prefill : color.decode,
            borderRadius: radius.sm,
            transition: "width 0.15s ease",
          }}
        />
      </div>
    </div>
  );
}

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h3 style={{ margin: `0 0 ${space.sm}px 0`, ...sectionLabel }}>
      {children}
    </h3>
  );
}

function EmptyState() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: space.lg,
        padding: `48px ${space.xxl}px`,
        textAlign: "center",
        color: color.textFaint,
        fontFamily: font.sans,
        fontSize: font.size.lg,
      }}
      role="status"
      aria-live="polite"
    >
      <span style={{ fontSize: 36, opacity: 0.5 }} aria-hidden="true">◌</span>
      <p style={{ margin: 0, fontWeight: font.weight.semibold, color: color.textMuted }}>No requests yet</p>
      <p style={{ margin: 0, maxWidth: 340, lineHeight: 1.6 }}>
        Add a request using the controls to begin. Type a sentence and press{" "}
        <strong>Add</strong>.
      </p>
      <p style={{ margin: 0, fontSize: font.size.base, maxWidth: 360, lineHeight: 1.6 }}>
        This stage focuses on the lifecycle of a single request:{" "}
        <Term tokenKey="prefill">prefill</Term> processes all prompt tokens in one pass,
        then <Term tokenKey="decode">decode</Term> emits one new token per step until done.
      </p>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export function Stage1Lifecycle({ engine }: StageProps) {
  const { tick, requests, blocks } = engine;
  const focusReq = pickFocusRequest(requests);

  if (!focusReq) {
    return (
      <div
        style={{
          padding: `0 ${space.md}px`,
          color: color.textMuted,
          fontFamily: font.sans,
          fontSize: font.size.lg,
        }}
        aria-label="Single-Request Lifecycle visualization"
      >
        <EmptyState />
      </div>
    );
  }

  const { phase, status, promptTokens, decodedTokens, maxDecode } = focusReq;
  const isDecode = phase === "decode";
  const isPrefill = phase === "prefill" && status === "running";
  const isWaiting = status === "waiting";
  const isFinished = status === "finished";

  // Only blocks for this request
  const reqBlocks = blocks.filter(
    (b) => b.requestId === focusReq.id || focusReq.blockTable.includes(b.id)
  );

  return (
    <div
      style={{
        padding: `${space.xl}px ${space.md}px`,
        display: "flex",
        flexDirection: "column",
        gap: 20,
        fontFamily: font.sans,
        color: color.textPrimary,
      }}
      aria-label="Single-Request Lifecycle visualization"
    >
      {/* Header */}
      <div>
        <h2
          style={{
            margin: `0 0 ${space.xs}px 0`,
            fontSize: font.size.xl,
            fontWeight: font.weight.bold,
            color: color.textPrimary,
            display: "flex",
            alignItems: "center",
            gap: space.md,
          }}
        >
          Request #{focusReq.id}
          <span style={{ fontSize: font.size.sm, fontWeight: font.weight.normal, color: color.textFaint, fontFamily: font.mono }}>
            arrived tick {focusReq.arrivalTick}
          </span>
        </h2>
      </div>

      {/* Phase / status readout */}
      <div>
        <SectionHeading>Status</SectionHeading>
        <PhaseReadout request={focusReq} tick={tick} />
      </div>

      {/* Contextual line — one neutral panel, text changes with the phase */}
      {isWaiting && (
        <ContextLine>
          In the <Term tokenKey="waitingQueue">waiting queue</Term> — the scheduler admits it when
          the batch has capacity. Press Step to advance.
        </ContextLine>
      )}
      {isPrefill && (
        <ContextLine>
          <Term tokenKey="prefill">Prefill</Term> — all {promptTokens.length} prompt token
          {promptTokens.length !== 1 ? "s" : ""} processed in one pass, populating the{" "}
          <Term tokenKey="kvCache">KV cache</Term>. No output token yet.
        </ContextLine>
      )}
      {isDecode && (
        <ContextLine>
          <Term tokenKey="decode">Decode</Term> — one token per step,{" "}
          <Term tokenKey="vocabSampling">sampled</Term> from the vocabulary and appended.
        </ContextLine>
      )}
      {isFinished && (
        <ContextLine>
          Finished — {decodedTokens.length} token{decodedTokens.length !== 1 ? "s" : ""} generated,{" "}
          <Term tokenKey="kvCache">KV cache</Term> blocks released.
        </ContextLine>
      )}

      {/* Token strip */}
      <div>
        <SectionHeading>Token Sequence</SectionHeading>
        {focusReq.promptText && (
          <div
            style={{
              marginBottom: space.md,
              padding: `${space.md}px ${space.lg}px`,
              borderRadius: radius.lg,
              background: color.panelBg,
              border: `1px solid ${color.border}`,
              fontSize: font.size.base,
              lineHeight: 1.6,
              color: color.textMuted,
            }}
          >
            <span style={{ color: color.textFaint }}>Your prompt</span>{" "}
            <span style={{ color: color.textPrimary, fontStyle: "italic" }}>
              “{focusReq.promptText}”
            </span>
            <br />
            <Term tokenKey="tokenizer">tokenizes</Term> into{" "}
            <strong style={{ color: color.prefill }}>{promptTokens.length}</strong> token
            {promptTokens.length !== 1 ? "s" : ""} ↓
          </div>
        )}
        <TokenStrip
          promptTokens={promptTokens}
          decodedTokens={decodedTokens}
          phase={phase}
          status={status}
        />
      </div>

      {/* Decode progress bar (decode phase or finished) */}
      {(isDecode || isFinished) && (
        <div>
          <SectionHeading>Decode Progress</SectionHeading>
          <DecodeProgressBar decoded={decodedTokens.length} maxDecode={maxDecode} />
        </div>
      )}

      {/* KV cache view */}
      <div>
        <SectionHeading>
          <Term tokenKey="kvCache">KV Cache</Term> Allocation
        </SectionHeading>
        {reqBlocks.length > 0 ? (
          <BlockGrid blocks={reqBlocks} requests={[focusReq]} />
        ) : (
          <p
            style={{ margin: 0, fontSize: font.size.base, color: color.textFaint, fontStyle: "italic" }}
            aria-label="No KV cache blocks allocated yet"
          >
            {isWaiting
              ? "No blocks allocated yet — waiting for scheduler admission."
              : isFinished
              ? "Blocks released after completion."
              : "No blocks currently allocated for this request."}
          </p>
        )}
      </div>

    </div>
  );
}
