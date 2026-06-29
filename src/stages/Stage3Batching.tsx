/**
 * Stage 3 — Continuous Batching + Preemption
 *
 * Teaches continuous batching under load:
 *   - Multiple requests flow through waiting → running queues.
 *   - Scheduler admits up to maxBatchSize and a tokenBudget each step.
 *   - When the KV cache fills, the scheduler PREEMPTS (swaps out) a running request.
 *   - Finishing or cancelling frees blocks so a waiting request gets admitted.
 */

import type { StageProps } from "./types";
import type { Request, Block } from "../engine/types";
import { QueueLanes } from "../components/QueueLanes";
import { BlockGrid } from "../components/BlockGrid";
import { HighwayPanel } from "../components/HighwayPanel";
import { Term } from "../components/Term";
import { latencyMetrics } from "../content/metrics";
import { color, space, radius, font, sectionLabel, notePanel } from "../theme";

// ─── Derived metrics helpers ─────────────────────────────────────────────────

function countByStatus(requests: Request[]) {
  let waiting = 0;
  let running = 0;
  let swapped = 0;
  let finished = 0;
  let cancelled = 0;
  for (const r of requests) {
    if (r.status === "waiting") waiting++;
    else if (r.status === "running") running++;
    else if (r.status === "swapped") swapped++;
    else if (r.status === "finished") finished++;
    else if (r.status === "cancelled") cancelled++;
  }
  return { waiting, running, swapped, finished, cancelled };
}

function kvUsage(blocks: Block[]) {
  const total = blocks.length;
  const used = blocks.filter((b) => b.requestId !== null).length;
  return { used, total };
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

const sectionStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const sectionTitleStyle: React.CSSProperties = {
  ...sectionLabel,
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: space.sm,
};

const metricsRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: 10,
};

function metricPillStyle(accent: string): React.CSSProperties {
  return {
    display: "inline-flex",
    flexDirection: "column",
    alignItems: "center",
    padding: `${space.sm}px 14px`,
    borderRadius: radius.lg,
    background: color.panelBg,
    border: `1px solid ${accent}`,
    minWidth: 72,
    gap: 2,
  };
}

const metricValueStyle: React.CSSProperties = {
  fontSize: font.size.xxl,
  fontWeight: font.weight.bold,
  fontFamily: font.mono,
  color: color.textPrimary,
  lineHeight: 1,
};

// textTransform and letterSpacing omitted: some labels contain <Term> elements
// whose dotted-underline button looks bad in ALL-CAPS.
const metricLabelStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  color: color.textMuted,
};

const kvBarOuterStyle: React.CSSProperties = {
  height: 8,
  background: color.border,
  borderRadius: radius.sm,
  overflow: "hidden",
  flexGrow: 1,
  minWidth: 60,
  alignSelf: "center",
};

function kvBarInnerStyle(pct: number): React.CSSProperties {
  // KV pressure: green (low) → warn (medium) → danger (high), from the palette.
  const fill = pct > 0.85 ? color.danger : pct > 0.6 ? color.warn : color.decode;
  return {
    height: "100%",
    width: `${Math.round(pct * 100)}%`,
    background: fill,
    borderRadius: radius.sm,
    transition: "width 0.2s ease",
  };
}


const emptyStateStyle: React.CSSProperties = {
  textAlign: "center",
  color: color.textFaint,
  fontStyle: "italic",
  padding: `32px ${space.xl}px`,
  fontSize: font.size.lg,
};

// ─── Metric Pill component ───────────────────────────────────────────────────

function MetricPill({
  value,
  label,
  accent,
  ariaLabel,
}: {
  value: string | number;
  label: React.ReactNode;
  accent: string;
  /** Plain-string aria label. When label is a ReactNode (Term), pass this explicitly. */
  ariaLabel?: string;
}) {
  const resolvedAriaLabel =
    ariaLabel ?? (typeof label === "string" ? `${label}: ${value}` : `${value}`);
  return (
    <div style={metricPillStyle(accent)} aria-label={resolvedAriaLabel}>
      <span style={metricValueStyle}>{value}</span>
      <span style={metricLabelStyle}>{label}</span>
    </div>
  );
}

// ─── Component ───────────────────────────────────────────────────────────────

export function Stage3Batching({
  engine,
  config,
  selectedRequestId,
  onSelectRequest,
}: StageProps) {
  const { requests, blocks, tick } = engine;
  const counts = countByStatus(requests);
  const kv = kvUsage(blocks);
  const kvPct = kv.total > 0 ? kv.used / kv.total : 0;

  // Active = every request still in the system minus the finished ones — the
  // live load the scheduler is juggling (waiting + running + swapped).
  const activeCount = requests.length - counts.finished;

  const metrics = latencyMetrics(requests, tick);

  const runningRequests = requests.filter((r) => r.status === "running");
  const hasAnyRequests = requests.length > 0;

  // Distinct reasons requests are currently held in the waiting/swapped queue,
  // recorded by the scheduler. Surfaced so the held-back behavior isn't a mystery.
  const waitReasons = Array.from(
    new Set(
      requests
        .filter((r) => (r.status === "waiting" || r.status === "swapped") && r.waitReason)
        .map((r) => r.waitReason as string)
    )
  );

  return (
    <div style={outerStyle} aria-label="Continuous Batching visualization">
      {/* ── Live metrics strip ── */}
      <section style={sectionStyle} aria-label="Live metrics">
        <h3 style={sectionTitleStyle}>Live Metrics — Tick {tick}</h3>
        <div style={metricsRowStyle}>
          <MetricPill value={tick} label="Tick" accent={color.borderStrong} />
          <MetricPill
            value={activeCount}
            label="Active"
            ariaLabel={`Active requests: ${activeCount}`}
            accent={color.prefill}
          />
          <MetricPill
            value={counts.waiting}
            label={<Term tokenKey="waitingQueue">Waiting</Term>}
            ariaLabel={`Waiting: ${counts.waiting}`}
            accent={color.waiting}
          />
          <MetricPill
            value={counts.running}
            label={<Term tokenKey="runningQueue">Running</Term>}
            ariaLabel={`Running: ${counts.running}`}
            accent={color.decode}
          />
          <MetricPill
            value={counts.swapped}
            label={<Term tokenKey="swapping">Swapped</Term>}
            ariaLabel={`Swapped: ${counts.swapped}`}
            accent={color.swapped}
          />
          <MetricPill value={counts.finished} label="Finished" accent={color.textFaint} />
          <MetricPill
            value={`${kv.used}/${kv.total}`}
            label={<Term tokenKey="kvCache">KV Blocks</Term>}
            ariaLabel={`KV Blocks: ${kv.used}/${kv.total}`}
            accent={kvPct > 0.85 ? color.danger : kvPct > 0.6 ? color.warn : color.info}
          />
          <MetricPill
            value={config.maxBatchSize}
            label={<Term tokenKey="maxBatchSize">Max Batch</Term>}
            ariaLabel={`Max Batch: ${config.maxBatchSize}`}
            accent={color.prefill}
          />
          <MetricPill
            value={config.tokenBudget}
            label={<Term tokenKey="tokenBudget">Tok Budget</Term>}
            ariaLabel={`Tok Budget: ${config.tokenBudget}`}
            accent={color.info}
          />
        </div>
        {/* KV usage bar */}
        <div
          style={{ display: "flex", gap: space.md, alignItems: "center" }}
          aria-label={`KV cache usage: ${kv.used} of ${kv.total} blocks (${Math.round(kvPct * 100)}%)`}
        >
          <span style={{ fontSize: font.size.sm, color: color.textMuted, whiteSpace: "nowrap" }}>
            KV pressure
          </span>
          <div style={kvBarOuterStyle}>
            <div style={kvBarInnerStyle(kvPct)} />
          </div>
          <span
            style={{
              fontSize: font.size.sm,
              fontFamily: font.mono,
              color: kvPct > 0.85 ? color.danger : color.textMuted,
              whiteSpace: "nowrap",
            }}
          >
            {Math.round(kvPct * 100)}%
          </span>
        </div>
      </section>

      {/* ── Latency & throughput metrics ── */}
      <section style={sectionStyle} aria-label="Latency and throughput metrics">
        <h3
          style={sectionTitleStyle}
          title="All values in ticks (the demo's time unit). TTFT = ticks from arrival to first token; ITL = ticks per token during decode; Throughput = tokens/tick across the batch."
        >
          Latency &amp; Throughput
        </h3>
        <div style={metricsRowStyle}>
          <MetricPill
            value={metrics.avgTtft !== null ? `${metrics.avgTtft.toFixed(1)}` : "—"}
            label={<Term tokenKey="ttft">TTFT</Term>}
            ariaLabel={`Average TTFT: ${metrics.avgTtft?.toFixed(1) ?? "n/a"} ticks`}
            accent={color.prefill}
          />
          <MetricPill
            value={metrics.avgItl !== null ? `${metrics.avgItl.toFixed(1)}` : "—"}
            label={<Term tokenKey="itl">ITL</Term>}
            ariaLabel={`Average ITL: ${metrics.avgItl?.toFixed(1) ?? "n/a"} ticks/token`}
            accent={color.decode}
          />
          <MetricPill
            value={metrics.throughput.toFixed(2)}
            label={<Term tokenKey="throughput">Throughput</Term>}
            ariaLabel={`Throughput: ${metrics.throughput.toFixed(2)} tokens per tick`}
            accent={color.accent}
          />
        </div>
      </section>

      {/* ── Sequence Progress (highway) ── */}
      {hasAnyRequests && (
        <section style={sectionStyle} aria-label="Sequence progress visualization">
          <h3 style={sectionTitleStyle}>Sequence Progress</h3>
          <HighwayPanel engine={engine} config={config} />
        </section>
      )}

      {/* ── Queue Lanes (centerpiece) ── */}
      <section style={sectionStyle} aria-label="Request queues">
        <h3 style={sectionTitleStyle}>Request Queues</h3>
        {hasAnyRequests ? (
          <>
            {waitReasons.length > 0 && (
              <div
                role="note"
                aria-label="Why requests are waiting"
                style={{
                  ...notePanel,
                  color: color.waiting,
                  marginBottom: 10,
                }}
              >
                <strong>Why some requests aren't running this step:</strong>
                <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                  {waitReasons.map((reason) => (
                    <li key={reason}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}
            <QueueLanes
              requests={requests}
              selectedRequestId={selectedRequestId}
              onSelectRequest={onSelectRequest}
            />
          </>
        ) : (
          <p style={emptyStateStyle}>
            Add several requests to watch the scheduler batch them. Set a small KV Cache Blocks
            (Engine Setup) first to trigger preemption under load.
          </p>
        )}
      </section>

      {/* ── Block Grid (KV cache) ── */}
      <section style={sectionStyle} aria-label="KV cache block grid">
        <h3 style={sectionTitleStyle}>Shared KV Cache</h3>
        <BlockGrid blocks={blocks} requests={runningRequests} />
      </section>
    </div>
  );
}
