/**
 * BlockGrid — KV cache block visualizer.
 *
 * Props:
 *   blocks    — all physical Block objects
 *   requests  — (optional) running requests; when provided, shows a
 *               logical→physical block-table readout per request
 *
 * Each cell shows:
 *   - Color derived from requestId (via requestColor helper)
 *   - usedSlots / tokenSlots fill bar
 *   - Block id label
 *
 * A color legend maps hue → request id when blocks are allocated.
 */

import type { Block, Request } from "../engine/types";
import { requestColor, requestColorLight } from "./requestColor";
import { Term } from "./Term";
import { color, space, radius, font, sectionLabel } from "../theme";

interface BlockGridProps {
  blocks: Block[];
  requests?: Request[];
}

// ─── styles ──────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.lg,
  fontFamily: font.mono,
  fontSize: font.size.md,
};

// Heading that contains a glossary <Term>: title case (no uppercase transform,
// which would shout and clash with the dotted-underline term button).
const termHeadingStyle: React.CSSProperties = {
  fontSize: font.size.base,
  fontWeight: font.weight.bold,
  color: color.textPrimary,
  marginBottom: space.sm,
  fontFamily: font.sans,
};

const gridStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
};

function blockCellStyle(block: Block): React.CSSProperties {
  const allocated = block.requestId !== null;
  return {
    position: "relative",
    width: 52,
    height: 44,
    borderRadius: radius.sm + 1,
    border: `2px solid ${allocated ? requestColor(block.requestId!) : color.border}`,
    background: allocated
      ? requestColorLight(block.requestId!, 0.25)
      : color.panelBgInset,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    cursor: "default",
  };
}

const blockIdStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  color: color.textMuted,
  lineHeight: 1,
};

const fillBarContainerStyle: React.CSSProperties = {
  width: "80%",
  height: 5,
  background: color.border,
  borderRadius: radius.sm - 1,
  overflow: "hidden",
  marginTop: 3,
};

function fillBarStyle(fill: number, requestId: number | null): React.CSSProperties {
  return {
    height: "100%",
    width: `${Math.round(fill * 100)}%`,
    background: requestId !== null ? requestColor(requestId, 0.9) : color.borderStrong,
    borderRadius: radius.sm - 1,
  };
}

const usageTextStyle: React.CSSProperties = {
  fontSize: 9,
  color: color.textFaint,
  lineHeight: 1,
  marginTop: 2,
};

// ─── Legend ───────────────────────────────────────────────────────────────────

function Legend({ blocks }: { blocks: Block[] }) {
  const seen = new Map<number, number>(); // requestId → count
  for (const b of blocks) {
    if (b.requestId !== null) {
      seen.set(b.requestId, (seen.get(b.requestId) ?? 0) + 1);
    }
  }
  if (seen.size === 0) return null;

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: space.sm, marginTop: space.xs }}>
      <span style={{ ...sectionLabel, marginBottom: 0 }}>Legend:</span>
      {[...seen.entries()].map(([rid, count]) => (
        <div
          key={rid}
          style={{
            display: "flex",
            alignItems: "center",
            gap: space.xs,
            fontSize: font.size.sm,
            fontFamily: font.sans,
            color: color.textPrimary,
          }}
        >
          <span
            style={{
              display: "inline-block",
              width: 12,
              height: 12,
              borderRadius: radius.sm - 1,
              background: requestColor(rid),
            }}
            aria-hidden="true"
          />
          <span>Req {rid}</span>
          <span style={{ color: color.textFaint }}>({count} blk{count !== 1 ? "s" : ""})</span>
        </div>
      ))}
    </div>
  );
}

// ─── Block table readout per request ─────────────────────────────────────────

function BlockTableReadout({ requests, blocks }: { requests: Request[]; blocks: Block[] }) {
  const running = requests.filter((r) => r.status === "running" || r.status === "swapped");
  if (running.length === 0) return null;

  const blockMap = new Map(blocks.map((b) => [b.id, b]));

  return (
    <div style={{ borderTop: `1px solid ${color.border}`, paddingTop: space.md }}>
      <p style={termHeadingStyle}>
        <Term tokenKey="blockTable">Block Tables</Term>
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: space.sm }}>
        {running.map((req) => (
          <div
            key={req.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: space.xs,
              flexWrap: "wrap",
            }}
          >
            <span
              style={{
                fontSize: font.size.sm,
                fontFamily: font.sans,
                color: requestColor(req.id),
                minWidth: 48,
                fontWeight: font.weight.semibold,
              }}
            >
              Req {req.id}
            </span>
            <span style={{ color: color.textFaint, fontSize: font.size.xs }}>→</span>
            {req.blockTable.length === 0 ? (
              <span style={{ fontSize: font.size.xs, color: color.textFaint }}>no blocks</span>
            ) : (
              req.blockTable.map((physId, logIdx) => {
                const phys = blockMap.get(physId);
                return (
                  <span
                    key={logIdx}
                    title={
                      `Logical block ${logIdx} → physical block ${physId}` +
                      (phys ? ` (${phys.usedSlots}/${phys.tokenSlots} token slots used)` : "") +
                      `. The request's block #${logIdx} in sequence order lives in cache block ${physId}.`
                    }
                    style={{
                      fontSize: font.size.xs,
                      padding: `1px 5px`,
                      borderRadius: radius.sm - 1,
                      background: requestColorLight(req.id, 0.3),
                      border: `1px solid ${requestColor(req.id, 0.5)}`,
                      color: color.textPrimary,
                      fontFamily: font.mono,
                    }}
                  >
                    L{logIdx}:P{physId}
                    {phys && (
                      <span style={{ color: color.textFaint }}>
                        {" "}({phys.usedSlots}/{phys.tokenSlots})
                      </span>
                    )}
                  </span>
                );
              })
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Component ────────────────────────────────────────────────────────────────

export function BlockGrid({ blocks, requests }: BlockGridProps) {
  return (
    <div style={containerStyle} aria-label="KV cache block grid">
      <p style={termHeadingStyle}>
        <Term tokenKey="kvCache">KV Cache</Term> ({blocks.length}{" "}
        <Term tokenKey="block">blocks</Term>)
      </p>
      <div style={gridStyle} role="list">
        {blocks.map((block) => {
          const fill =
            block.tokenSlots > 0 ? block.usedSlots / block.tokenSlots : 0;
          return (
            <div
              key={block.id}
              style={blockCellStyle(block)}
              role="listitem"
              title={
                block.requestId !== null
                  ? `Block ${block.id} — Req ${block.requestId} (${block.usedSlots}/${block.tokenSlots} slots)`
                  : `Block ${block.id} — free`
              }
            >
              <span style={blockIdStyle}>#{block.id}</span>
              <div style={fillBarContainerStyle}>
                <div style={fillBarStyle(fill, block.requestId)} />
              </div>
              <span style={usageTextStyle}>
                {block.usedSlots}/{block.tokenSlots}
              </span>
            </div>
          );
        })}
        {blocks.length === 0 && (
          <span style={{ color: color.textFaint, fontFamily: font.sans, fontSize: font.size.base }}>
            No blocks allocated.
          </span>
        )}
      </div>
      <Legend blocks={blocks} />
      {requests && <BlockTableReadout requests={requests} blocks={blocks} />}
    </div>
  );
}
