/**
 * Stage 2 — Paged KV Cache
 * Teaches paged KV cache: fixed-size physical blocks + per-request block table
 * mapping logical → physical block ids. Shows fill levels, internal fragmentation,
 * and why paging beats contiguous allocation.
 */

import type { StageProps } from "./types";
import type { Request, Block } from "../engine/types";
import { BlockGrid } from "../components/BlockGrid";
import { TokenStrip } from "../components/TokenStrip";
import { Term } from "../components/Term";
import { KvMemoryBreakdown } from "../components/KvMemoryBreakdown";
import { color, space, radius, font, sectionLabel, panel, notePanel } from "../theme";

// ─── helpers ─────────────────────────────────────────────────────────────────

/** Returns the most recent non-cancelled request that has blocks or is running */
function findActiveRequest(requests: Request[]): Request | null {
  const active = requests
    .filter((r) => r.status !== "cancelled" && r.status !== "finished")
    .sort((a, b) => b.arrivalTick - a.arrivalTick);
  if (active.length > 0) return active[0];

  // Fall back to most recently arrived finished request (so user can see what happened)
  const finished = requests
    .filter((r) => r.status === "finished")
    .sort((a, b) => b.arrivalTick - a.arrivalTick);
  return finished.length > 0 ? finished[0] : null;
}

/** Computes wasted slots in the last allocated block for a request */
function computeFragmentation(
  activeReq: Request | null,
  blocks: Block[]
): { wastedSlots: number; lastBlock: Block | null } {
  if (!activeReq || activeReq.blockTable.length === 0) {
    return { wastedSlots: 0, lastBlock: null };
  }
  const lastPhysId = activeReq.blockTable[activeReq.blockTable.length - 1];
  const blockMap = new Map(blocks.map((b) => [b.id, b]));
  const lastBlock = blockMap.get(lastPhysId) ?? null;
  if (!lastBlock) return { wastedSlots: 0, lastBlock: null };
  const wasted = lastBlock.tokenSlots - lastBlock.usedSlots;
  return { wastedSlots: wasted, lastBlock };
}

// ─── styles ──────────────────────────────────────────────────────────────────

const outerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 20,
  padding: `${space.xxl}px 20px`,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
  maxWidth: 780,
};

const sectionHeaderStyle: React.CSSProperties = {
  ...sectionLabel,
  marginBottom: space.lg,
};

const explainerStyle: React.CSSProperties = {
  lineHeight: 1.65,
  color: color.textMuted,
  fontSize: font.size.base,
};

const statsRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.lg,
  marginTop: space.md,
};

const statChipStyle: React.CSSProperties = {
  background: color.panelBgInset,
  border: `1px solid ${color.border}`,
  borderRadius: radius.md,
  padding: `${space.sm}px ${space.lg}px`,
  fontSize: font.size.md,
  fontFamily: font.mono,
  color: color.textPrimary,
  display: "flex",
  flexDirection: "column",
  gap: 2,
};

const statLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  fontWeight: font.weight.semibold,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  color: color.textFaint,
  fontFamily: font.sans,
};

const statValueStyle: React.CSSProperties = {
  fontSize: font.size.lg,
  fontWeight: font.weight.bold,
  color: color.prefill,
};

const emptyStyle: React.CSSProperties = {
  color: color.textFaint,
  fontStyle: "italic",
  fontSize: font.size.base,
  padding: `${space.lg}px 0`,
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function BlockStats({
  blocks,
  blockSize,
  activeReq,
}: {
  blocks: Block[];
  blockSize: number;
  activeReq: Request | null;
}) {
  const usedCount = blocks.filter((b) => b.requestId !== null).length;
  const totalCount = blocks.length;
  const { wastedSlots, lastBlock } = computeFragmentation(activeReq, blocks);
  const hasPartialBlock = lastBlock !== null && wastedSlots > 0;

  return (
    <div>
      <p style={sectionHeaderStyle}>Block Statistics</p>
      <div style={statsRowStyle}>
        <div style={statChipStyle}>
          <span style={statLabelStyle}>Blocks Used</span>
          <span style={statValueStyle}>
            {usedCount} / {totalCount}
          </span>
        </div>
        <div style={statChipStyle}>
          <span style={statLabelStyle}>Block Size</span>
          <span style={statValueStyle}>{blockSize} tokens</span>
        </div>
        <div style={statChipStyle}>
          <span style={statLabelStyle}>Free Blocks</span>
          <span
            style={{
              ...statValueStyle,
              color: totalCount - usedCount <= 2 ? color.danger : color.decode,
            }}
          >
            {totalCount - usedCount}
          </span>
        </div>
      </div>

      {hasPartialBlock && (
        <p style={{ ...notePanel, marginTop: space.md, color: color.accent }} role="note">
          <strong>
            <Term tokenKey="internalFragmentation">Internal fragmentation</Term>:
          </strong>{" "}
          the last block (P{lastBlock!.id}) holds{" "}
          <strong>{lastBlock!.usedSlots}</strong> of{" "}
          <strong>{lastBlock!.tokenSlots}</strong> slots — {wastedSlots} slot
          {wastedSlots !== 1 ? "s" : ""} wasted. Paging bounds this waste to
          &lt;1 block per request.
        </p>
      )}

      {activeReq &&
        lastBlock !== null &&
        wastedSlots === 0 &&
        lastBlock.usedSlots === blockSize && (
          <p
            style={{ ...notePanel, marginTop: space.md, color: color.decode }}
            role="note"
          >
            Last block is fully packed — no{" "}
            <Term tokenKey="internalFragmentation">internal fragmentation</Term>{" "}
            right now.
          </p>
        )}
    </div>
  );
}


// ─── Main component ───────────────────────────────────────────────────────────

export function Stage2PagedKV({ engine, config, selectedRequestId, onSelectRequest }: StageProps) {
  const { blocks, requests } = engine;

  // Active requests for the block-table readout (running + swapped)
  const runningRequests = requests.filter(
    (r) => r.status === "running" || r.status === "swapped"
  );

  // Active request for token strip and fragmentation analysis
  const activeReq = findActiveRequest(requests);

  // If user has a selected request and it's still live, prefer it
  const displayReq =
    selectedRequestId !== null
      ? (requests.find((r) => r.id === selectedRequestId) ?? activeReq)
      : activeReq;

  const hasAnyRequest = requests.length > 0;

  return (
    <div style={outerStyle} aria-label="Paged KV Cache visualization">
      {/* Token strip for active request */}
      <div style={panel}>
        <p style={sectionHeaderStyle}>Active Request — Token Sequence</p>
        {displayReq && displayReq.promptTokens.length > 0 ? (
          <div>
            {/* Request selector when multiple non-cancelled requests exist */}
            {requests.filter((r) => r.status !== "cancelled").length > 1 && (
              <div
                style={{
                  display: "flex",
                  gap: space.sm,
                  flexWrap: "wrap",
                  marginBottom: 10,
                }}
                role="group"
                aria-label="Select request"
              >
                {requests
                  .filter((r) => r.status !== "cancelled")
                  .map((r) => (
                    <button
                      key={r.id}
                      onClick={() => onSelectRequest(r.id)}
                      style={{
                        padding: "3px 10px",
                        borderRadius: radius.sm,
                        border: `1px solid ${
                          displayReq?.id === r.id ? color.prefill : color.border
                        }`,
                        background:
                          displayReq?.id === r.id ? `${color.prefill}22` : color.panelBgInset,
                        color:
                          displayReq?.id === r.id ? color.prefill : color.textFaint,
                        cursor: "pointer",
                        fontSize: font.size.sm,
                        fontFamily: font.mono,
                      }}
                      aria-pressed={displayReq?.id === r.id}
                    >
                      Req {r.id}{" "}
                      <span style={{ opacity: 0.7 }}>({r.status})</span>
                    </button>
                  ))}
              </div>
            )}
            <TokenStrip
              promptTokens={displayReq.promptTokens}
              decodedTokens={displayReq.decodedTokens}
              phase={displayReq.phase}
              status={displayReq.status}
            />
            {displayReq.blockTable.length > 0 && (
              <p
                style={{
                  marginTop: space.md,
                  fontSize: font.size.md,
                  color: color.textFaint,
                  fontFamily: font.mono,
                }}
              >
                Tokens 0–{config.blockSize - 1} → block P
                {displayReq.blockTable[0]}; tokens{" "}
                {config.blockSize}–{config.blockSize * 2 - 1} → block P
                {displayReq.blockTable[1] ?? "?"}{" "}
                {displayReq.blockTable.length > 2 && (
                  <span>… ({displayReq.blockTable.length} blocks total)</span>
                )}
              </p>
            )}
          </div>
        ) : (
          <p style={emptyStyle}>
            {hasAnyRequest
              ? "Request is waiting to be scheduled — no KV cache blocks yet."
              : "Add a request to watch its KV cache fill block by block."}
          </p>
        )}
      </div>

      {/* Centerpiece: BlockGrid */}
      <div style={panel}>
        <p style={sectionHeaderStyle}>Physical KV Cache Blocks</p>
        <BlockGrid blocks={blocks} requests={runningRequests} />
      </div>

      {/* Block stats + fragmentation */}
      <div style={panel}>
        <BlockStats
          blocks={blocks}
          blockSize={config.blockSize}
          activeReq={displayReq}
        />
      </div>

      {/* Architecture → memory bridge: what one block actually costs */}
      <div style={panel}>
        <p style={sectionHeaderStyle}>
          From Model Shape to <Term tokenKey="kvCache">KV Cache</Term> Memory
        </p>
        <p style={{ ...explainerStyle, marginTop: -4, marginBottom: 14 }}>
          A <Term tokenKey="block">block</Term> is an abstraction over real GPU memory. What it
          actually costs is fixed by the model's architecture: layers, KV heads, head dimension, and
          dtype. The <Term tokenKey="bytesPerToken">bytes per token</Term> below set the cost of one
          token; adjust the parameters to see one block and the whole cache scale with it.
        </p>
        <KvMemoryBreakdown blockSize={config.blockSize} kvCacheBlocks={config.kvCacheBlocks} />
      </div>
    </div>
  );
}
