/**
 * QueueLanes — request queue status visualizer.
 *
 * Props:
 *   requests         — all Request objects
 *   selectedRequestId — currently selected request (highlighted)
 *   onSelectRequest  — callback when a chip is clicked
 *
 * Displays requests grouped into lanes by status:
 *   waiting | running | swapped | finished | cancelled
 *
 * Each chip shows: request id, decoded/maxDecode progress.
 * Clicking a chip fires onSelectRequest(id).
 * Selected chip is visually highlighted.
 * Colors match BlockGrid via the shared requestColor helper.
 */

import type { Request, RequestStatus } from "../engine/types";
import { requestColor, requestColorLight } from "./requestColor";
import { Term } from "./Term";
import { palette, color, space, radius, font } from "../theme";

interface QueueLanesProps {
  requests: Request[];
  selectedRequestId: number | null;
  onSelectRequest(id: number): void;
}

// ─── Lane order and labels ────────────────────────────────────────────────────

const LANE_ORDER: RequestStatus[] = [
  "running",
  "waiting",
  "swapped",
  "finished",
  "cancelled",
];

const LANE_LABELS: Record<RequestStatus, string> = {
  running: "Running",
  waiting: "Waiting",
  swapped: "Swapped",
  finished: "Finished",
  cancelled: "Cancelled",
};

// Accent color per lane (used for the lane's label).
const LANE_COLOR: Record<RequestStatus, string> = {
  running: color.decode,
  waiting: color.waiting,
  swapped: color.swapped,
  finished: color.textFaint,
  cancelled: color.danger,
};

// ─── styles ──────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.md + 2,
  fontFamily: font.sans,
  fontSize: font.size.base,
};

const laneStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.xs,
};

const laneHeaderStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.sm,
};

// Statuses that have a Term glossary link — rendered without uppercase/letterSpacing
// so the dotted-underline button looks natural.
const TERM_STATUSES = new Set<RequestStatus>(["running", "waiting", "swapped"]);

const laneNameStyle = (status: RequestStatus): React.CSSProperties => ({
  fontSize: font.size.sm,
  fontWeight: font.weight.bold,
  // Remove textTransform/letterSpacing for Term-wrapped labels so the dotted
  // underline doesn't shout in ALL-CAPS.
  ...(TERM_STATUSES.has(status)
    ? {}
    : { textTransform: "uppercase", letterSpacing: "0.07em" }),
  color: LANE_COLOR[status],
});

const laneCountStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  color: color.textFaint,
};

const chipsRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 5,
  minHeight: 28,
};

function chipStyle(
  req: Request,
  isSelected: boolean
): React.CSSProperties {
  // Finished/cancelled requests are done — gray them out wherever they render
  // so attention stays on the live (waiting/running/swapped) ones.
  const dimmed = req.status === "finished" || req.status === "cancelled";

  if (dimmed) {
    return {
      display: "inline-flex",
      alignItems: "center",
      gap: 5,
      padding: `${space.xs}px 9px`,
      borderRadius: radius.md,
      border: `2px solid ${color.border}`,
      background: color.panelBgInset,
      color: color.textFaint,
      cursor: "pointer",
      fontSize: font.size.md,
      fontFamily: font.mono,
      fontWeight: font.weight.normal,
      opacity: 0.6,
      filter: "grayscale(0.7)",
      transition: "box-shadow 0.1s",
      userSelect: "none",
    };
  }

  const baseColor = requestColor(req.id, 0.85);
  const bgColor = requestColorLight(req.id, isSelected ? 0.35 : 0.18);
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    padding: `${space.xs}px 9px`,
    borderRadius: radius.md,
    border: `2px solid ${isSelected ? baseColor : requestColor(req.id, 0.4)}`,
    background: bgColor,
    color: color.textPrimary,
    cursor: "pointer",
    fontSize: font.size.md,
    fontFamily: font.mono,
    fontWeight: isSelected ? font.weight.bold : font.weight.normal,
    boxShadow: isSelected ? `0 0 0 2px ${requestColor(req.id, 0.5)}` : "none",
    transition: "box-shadow 0.1s",
    userSelect: "none",
  };
}

const progressBarContainerStyle: React.CSSProperties = {
  width: 36,
  height: 4,
  background: color.border,
  borderRadius: 2,
  overflow: "hidden",
};

function progressBarStyle(req: Request): React.CSSProperties {
  const pct = req.maxDecode > 0
    ? Math.min(1, req.decodedTokens.length / req.maxDecode)
    : 0;
  return {
    height: "100%",
    width: `${Math.round(pct * 100)}%`,
    background: requestColor(req.id, 0.8),
    borderRadius: 2,
  };
}

const emptyLaneStyle: React.CSSProperties = {
  fontSize: font.size.sm,
  color: palette.surface1,
  fontStyle: "italic",
  padding: "2px 0",
};

// ─── Component ────────────────────────────────────────────────────────────────

export function QueueLanes({ requests, selectedRequestId, onSelectRequest }: QueueLanesProps) {
  // Group requests by status, preserving lane order
  const grouped = new Map<RequestStatus, Request[]>();
  for (const status of LANE_ORDER) {
    grouped.set(status, []);
  }
  for (const req of requests) {
    const list = grouped.get(req.status);
    if (list) list.push(req);
    // status not in LANE_ORDER (shouldn't happen) — silently skip
  }

  // Only render lanes that have content or are "primary" statuses
  const primaryStatuses: RequestStatus[] = ["running", "waiting", "swapped"];

  return (
    <div style={containerStyle} aria-label="Request queue lanes">
      {LANE_ORDER.map((status) => {
        const list = grouped.get(status) ?? [];
        const isPrimary = primaryStatuses.includes(status);
        // Hide finished/cancelled lanes when empty to reduce noise
        if (!isPrimary && list.length === 0) return null;

        return (
          <div key={status} style={laneStyle}>
            <div style={laneHeaderStyle}>
              <span style={laneNameStyle(status)}>
                {status === "running" ? (
                  <Term tokenKey="runningQueue">{LANE_LABELS[status]}</Term>
                ) : status === "waiting" ? (
                  <Term tokenKey="waitingQueue">{LANE_LABELS[status]}</Term>
                ) : status === "swapped" ? (
                  <Term tokenKey="swapping">{LANE_LABELS[status]}</Term>
                ) : (
                  LANE_LABELS[status]
                )}
              </span>
              <span style={laneCountStyle}>({list.length})</span>
            </div>
            <div style={chipsRowStyle} role="list" aria-label={`${LANE_LABELS[status]} requests`}>
              {list.length === 0 ? (
                <span style={emptyLaneStyle}>—</span>
              ) : (
                list.map((req) => {
                  const isSelected = req.id === selectedRequestId;
                  // Show why a request is still queued (set by the scheduler).
                  const showReason =
                    (status === "waiting" || status === "swapped") && !!req.waitReason;
                  return (
                    <button
                      key={req.id}
                      style={chipStyle(req, isSelected)}
                      onClick={() => onSelectRequest(req.id)}
                      aria-pressed={isSelected}
                      title={showReason ? req.waitReason : undefined}
                      aria-label={
                        showReason
                          ? `Request ${req.id}, waiting: ${req.waitReason}`
                          : `Request ${req.id}, ${req.decodedTokens.length} of ${req.maxDecode} decoded`
                      }
                      role="listitem"
                    >
                      <span>#{req.id}</span>
                      <div style={progressBarContainerStyle} aria-hidden="true">
                        <div style={progressBarStyle(req)} />
                      </div>
                      <span style={{ fontSize: font.size.xs, color: color.textMuted }}>
                        {req.decodedTokens.length}/{req.maxDecode}
                      </span>
                      {showReason && (
                        <span style={{ fontSize: font.size.xs, color: color.waiting }} aria-hidden="true">
                          ⓘ
                        </span>
                      )}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
