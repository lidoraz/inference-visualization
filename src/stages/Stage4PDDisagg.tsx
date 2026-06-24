/**
 * Stage 4 — PD Disaggregation
 *
 * Shows how splitting prefill and decode onto separate GPU workers enables
 * independent scaling and optimization of each phase.
 *
 * Worker assignment: derived purely from request.phase in the component
 * (no engine changes). A request in phase "prefill" belongs to the Prefill
 * Worker; a request in phase "decode" (and finished requests) belongs to the
 * Decode Worker. This is valid because in a real disaggregated system the
 * handoff happens exactly at the prefill→decode boundary — and modeling it
 * this way keeps Stages 1–3 completely unaffected.
 */

import { useState } from "react";
import type { StageProps } from "./types";
import type { Request, Block } from "../engine/types";
import { BlockGrid } from "../components/BlockGrid";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, panel, statusTint } from "../theme";

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** A request lives on the prefill worker if it is running and still in prefill phase. */
function isPrefillWorker(r: Request): boolean {
  return r.status === "running" && r.phase === "prefill";
}

/** A request lives on the decode worker if it is running in decode phase, or finished. */
function isDecodeWorker(r: Request): boolean {
  return (r.status === "running" && r.phase === "decode") || r.status === "finished";
}

/** Returns blocks that belong to the given set of requests. */
function blocksFor(allBlocks: Block[], requestIds: Set<number>): Block[] {
  return allBlocks.map((b) =>
    b.requestId !== null && requestIds.has(b.requestId)
      ? b
      : { ...b, requestId: null, usedSlots: 0 }
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const outerStyle: React.CSSProperties = {
  padding: `20px ${space.xl}px`,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
  display: "flex",
  flexDirection: "column",
  gap: space.xxl,
};

const panelsRowStyle: React.CSSProperties = {
  display: "flex",
  gap: space.xl,
  alignItems: "stretch",
  flexWrap: "wrap",
};

function workerPanelStyle(accent: string): React.CSSProperties {
  return {
    ...panel,
    flex: "1 1 280px",
    minWidth: 240,
    background: color.panelBgInset,
    border: `1px solid ${accent}`,
    padding: `14px ${space.xl}px`,
    display: "flex",
    flexDirection: "column",
    gap: space.lg,
  };
}

const workerTitleStyle: React.CSSProperties = {
  ...sectionLabel,
  marginBottom: space.xs,
};

const transferArrowWrapStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: `0 ${space.xs}px`,
  gap: space.sm,
  minWidth: 64,
};

function transferArrowStyle(active: boolean): React.CSSProperties {
  return {
    fontSize: 22,
    color: active ? color.waiting : color.borderStrong,
    transition: "color 0.25s ease",
    userSelect: "none",
  };
}

function transferLabelStyle(active: boolean): React.CSSProperties {
  return {
    fontSize: font.size.xs,
    fontWeight: font.weight.bold,
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    color: active ? color.waiting : color.borderStrong,
    transition: "color 0.25s ease",
    textAlign: "center",
    whiteSpace: "nowrap",
  };
}

function requestChipStyle(phase: "prefill" | "decode", status: string): React.CSSProperties {
  const isFinished = status === "finished";
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: space.sm,
    padding: `${space.xs}px 10px`,
    borderRadius: radius.md,
    border: `1px solid ${isFinished ? color.borderStrong : phase === "prefill" ? color.prefill : color.decode}`,
    background: isFinished ? color.panelBg : phase === "prefill" ? `${color.prefill}18` : `${color.decode}18`,
    color: isFinished ? color.textFaint : color.textPrimary,
    fontSize: font.size.md,
    fontFamily: font.mono,
    cursor: "default",
  };
}

const phaseBadgeStyle = (phase: "prefill" | "decode"): React.CSSProperties => ({
  fontSize: font.size.xs,
  fontWeight: font.weight.bold,
  padding: "1px 5px",
  borderRadius: radius.sm,
  background: phase === "prefill" ? color.prefill : color.decode,
  color: color.panelBg,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
});

const emptyPanelStyle: React.CSSProperties = {
  color: color.borderStrong,
  fontStyle: "italic",
  fontSize: font.size.md,
  padding: `${space.md}px 0`,
};

const chipsWrapStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.sm,
};

const subSectionLabelStyle: React.CSSProperties = {
  ...sectionLabel,
  marginBottom: space.xs,
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function RequestChip({ request }: { request: Request }) {
  const tokenCount = request.promptTokens.length + request.decodedTokens.length;
  return (
    <div style={requestChipStyle(request.phase, request.status)}>
      <span>Req {request.id}</span>
      {request.status !== "finished" && (
        <span style={phaseBadgeStyle(request.phase)}>{request.phase}</span>
      )}
      {request.status === "finished" && (
        <span style={statusTint(color.borderStrong)}>
          done
        </span>
      )}
      <span style={{ fontSize: font.size.xs, color: color.textFaint }}>{tokenCount} tok</span>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export function Stage4PDDisagg({ engine }: StageProps) {
  const { requests, blocks, tick } = engine;

  // Partition requests by worker
  const prefillRequests = requests.filter(isPrefillWorker);
  const decodeRequests = requests.filter(isDecodeWorker);
  const waitingRequests = requests.filter((r) => r.status === "waiting" || r.status === "swapped");

  // Detect transfer: any request that just crossed prefill→decode this tick.
  // We approximate "just transferred" by checking for decode-phase running requests
  // whose decoded token count is 0 (first decode step = just arrived from prefill).
  const [prevTick, setPrevTick] = useState<number>(-1);
  const [transferring, setTransferring] = useState<boolean>(false);

  if (tick !== prevTick) {
    const justTransferred = requests.some(
      (r) => r.status === "running" && r.phase === "decode" && r.decodedTokens.length === 0
    );
    setTransferring(justTransferred);
    setPrevTick(tick);
  }

  // Build per-worker block views
  const prefillIds = new Set(prefillRequests.map((r) => r.id));
  const decodeIds = new Set(decodeRequests.map((r) => r.id));
  const prefillBlocks = blocksFor(blocks, prefillIds);
  const decodeBlocks = blocksFor(blocks, decodeIds);

  const hasAnyRequests = requests.length > 0;

  return (
    <div style={outerStyle} aria-label="PD Disaggregation visualization">
      {/* Tick counter */}
      <div style={{ fontSize: font.size.md, color: color.textFaint }}>Tick {tick}</div>

      {/* Waiting queue (outside both workers) */}
      {waitingRequests.length > 0 && (
        <div>
          <p style={subSectionLabelStyle}>Waiting / Swapped Queue</p>
          <div style={chipsWrapStyle}>
            {waitingRequests.map((r) => (
              <div
                key={r.id}
                style={{
                  ...requestChipStyle(r.phase, r.status),
                  borderColor: color.waiting,
                  background: `${color.waiting}18`,
                }}
              >
                <span>Req {r.id}</span>
                <span style={statusTint(color.waiting)}>
                  {r.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main two-panel layout */}
      <div style={panelsRowStyle}>
        {/* Prefill Worker */}
        <div
          style={workerPanelStyle(color.prefill)}
          aria-label="Prefill Worker panel"
        >
          <div style={workerTitleStyle}>
            <Term tokenKey="prefillWorker">Prefill Worker</Term>
          </div>
          <div>
            <p style={subSectionLabelStyle}>Active Requests</p>
            {prefillRequests.length > 0 ? (
              <div style={chipsWrapStyle}>
                {prefillRequests.map((r) => (
                  <RequestChip key={r.id} request={r} />
                ))}
              </div>
            ) : (
              <p style={emptyPanelStyle}>No requests in prefill</p>
            )}
          </div>
          <div>
            <p style={subSectionLabelStyle}>KV Blocks (prefill side)</p>
            <BlockGrid blocks={prefillBlocks} requests={prefillRequests} />
          </div>
          <div style={{ fontSize: font.size.sm, color: color.textFaint, lineHeight: 1.5 }}>
            Compute-bound: processes the entire prompt in one shot. Optimized for
            high arithmetic intensity (large matrix multiplies).
          </div>
        </div>

        {/* KV Transfer arrow */}
        <div style={transferArrowWrapStyle} aria-label="KV Transfer indicator">
          <span style={transferArrowStyle(transferring)}>→</span>
          <span style={transferLabelStyle(transferring)}>
            <Term tokenKey="kvTransfer">KV Transfer</Term>
          </span>
          {transferring && (
            <span
              style={{
                fontSize: font.size.xs,
                color: color.waiting,
                textAlign: "center",
                animation: "none",
              }}
            >
              active
            </span>
          )}
        </div>

        {/* Decode Worker */}
        <div
          style={workerPanelStyle(color.decode)}
          aria-label="Decode Worker panel"
        >
          <div style={workerTitleStyle}>
            <Term tokenKey="decodeWorker">Decode Worker</Term>
          </div>
          <div>
            <p style={subSectionLabelStyle}>Active Requests</p>
            {decodeRequests.length > 0 ? (
              <div style={chipsWrapStyle}>
                {decodeRequests.map((r) => (
                  <RequestChip key={r.id} request={r} />
                ))}
              </div>
            ) : (
              <p style={emptyPanelStyle}>No requests in decode</p>
            )}
          </div>
          <div>
            <p style={subSectionLabelStyle}>KV Blocks (decode side)</p>
            <BlockGrid blocks={decodeBlocks} requests={decodeRequests} />
          </div>
          <div style={{ fontSize: font.size.sm, color: color.textFaint, lineHeight: 1.5 }}>
            Memory-bandwidth-bound: reads the full KV cache every step for one new
            token. Optimized for high HBM bandwidth efficiency.
          </div>
        </div>
      </div>

      {/* Empty state */}
      {!hasAnyRequests && (
        <p
          style={{
            textAlign: "center",
            color: color.textFaint,
            fontStyle: "italic",
            padding: `${space.xxl}px 0`,
          }}
        >
          Add requests to see them flow from the Prefill Worker to the Decode Worker.
        </p>
      )}
    </div>
  );
}
