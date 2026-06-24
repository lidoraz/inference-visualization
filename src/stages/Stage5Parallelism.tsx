/**
 * Stage 5 — Parallelism Strategies (TP / PP / EP / DP)
 *
 * Config-driven diagram component. No engine changes — reads engine.requests
 * and config.numGPUs / config.parallelism to render a visual layout showing
 * how work is distributed across N GPUs under each parallelism strategy.
 *
 * Local state holds the active parallelism mode (seeded from config.parallelism);
 * toggling it updates only the diagram, not the engine.
 */

import { useState } from "react";
import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { color, space, radius, font, sectionLabel, notePanel, requestColors } from "../theme";

type ParMode = "tp" | "pp" | "ep" | "dp";

// ─── Styles ──────────────────────────────────────────────────────────────────

const outerStyle: React.CSSProperties = {
  padding: `${space.xxl}px ${space.xl}px`,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
  display: "flex",
  flexDirection: "column",
  gap: space.xxl,
};

const modeSelectorStyle: React.CSSProperties = {
  display: "flex",
  gap: space.sm,
  flexWrap: "wrap",
};

function modeButtonStyle(active: boolean, accent: string): React.CSSProperties {
  return {
    padding: `${space.sm}px ${space.xl}px`,
    borderRadius: radius.md,
    border: `1px solid ${active ? accent : color.borderStrong}`,
    background: active ? `${accent}22` : color.border,
    color: active ? accent : color.textPrimary,
    cursor: "pointer",
    fontSize: font.size.base,
    fontWeight: active ? font.weight.bold : font.weight.normal,
    transition: "all 0.15s",
  };
}

const gpuGridStyle: React.CSSProperties = {
  display: "flex",
  gap: space.lg,
  flexWrap: "wrap",
};

function gpuCardStyle(accent: string): React.CSSProperties {
  return {
    flex: "1 1 160px",
    minWidth: 140,
    background: color.panelBgInset,
    border: `1px solid ${accent}`,
    borderRadius: radius.lg + 2,
    padding: `${space.lg}px 14px`,
    display: "flex",
    flexDirection: "column",
    gap: space.md,
  };
}

const gpuHeaderStyle: React.CSSProperties = {
  ...sectionLabel,
  letterSpacing: "0.08em",
  borderBottom: `1px solid ${color.border}`,
  paddingBottom: space.sm,
};

const gpuContentStyle: React.CSSProperties = {
  fontSize: font.size.md,
  color: color.textPrimary,
  lineHeight: 1.6,
  display: "flex",
  flexDirection: "column",
  gap: space.xs,
};

function layerChipStyle(highlight?: boolean): React.CSSProperties {
  return {
    padding: `2px ${space.md}px`,
    borderRadius: radius.sm,
    background: highlight ? `${color.accent}22` : color.panelBgDeep,
    border: `1px solid ${highlight ? color.accent : color.border}`,
    fontSize: font.size.sm,
    fontFamily: font.mono,
    color: highlight ? color.accent : color.prefill,
    whiteSpace: "nowrap",
  };
}

function requestChipStyle(chipColor: string): React.CSSProperties {
  return {
    padding: `2px ${space.md}px`,
    borderRadius: radius.sm,
    background: `${chipColor}22`,
    border: `1px solid ${chipColor}`,
    fontSize: font.size.sm,
    fontFamily: font.mono,
    color: chipColor,
    whiteSpace: "nowrap",
  };
}

const expertChipBase: React.CSSProperties = {
  padding: `2px ${space.md}px`,
  borderRadius: radius.sm,
  fontSize: font.size.sm,
  fontFamily: font.mono,
  whiteSpace: "nowrap",
};

const infoBoxStyle: React.CSSProperties = {
  ...notePanel,
  padding: `${space.lg}px ${space.xl}px`,
  lineHeight: 1.65,
};

const allReduceNoteStyle: React.CSSProperties = {
  background: color.panelBgInset,
  border: `1px solid ${color.borderStrong}`,
  borderRadius: radius.md,
  padding: `${space.sm}px 10px`,
  fontSize: font.size.sm,
  color: color.prefill,
  textAlign: "center",
  marginTop: space.xs,
};

const whenToUseStyle: React.CSSProperties = {
  ...notePanel,
  marginTop: space.sm,
  padding: `${space.sm}px 10px`,
  fontSize: font.size.md,
  color: color.decode,
  border: `1px solid ${color.decode}66`,
  background: `${color.decode}11`,
  lineHeight: 1.5,
};

// ─── Per-mode GPU card renderers ──────────────────────────────────────────────

const REQUEST_COLORS = requestColors;

function TpGpuCard({ gpuIdx, numGPUs }: { gpuIdx: number; numGPUs: number }) {
  const totalHeads = 16;
  const headsPerGpu = Math.floor(totalHeads / numGPUs);
  const headStart = gpuIdx * headsPerGpu;
  const headEnd = headStart + headsPerGpu - 1;

  return (
    <div style={gpuCardStyle(color.prefill)}>
      <div style={gpuHeaderStyle}>
        <Term tokenKey="gpu">GPU {gpuIdx}</Term>
      </div>
      <div style={gpuContentStyle}>
        <span style={{ fontSize: font.size.xs, color: color.textFaint }}>All layers — shard:</span>
        <span style={layerChipStyle()}>Attn heads {headStart}–{headEnd}</span>
        <span style={layerChipStyle()}>FFN cols {headStart * 64}–{headEnd * 64 + 63}</span>
        <div style={allReduceNoteStyle}>⇄ all-reduce per layer</div>
      </div>
    </div>
  );
}

function PpGpuCard({
  gpuIdx,
  numGPUs,
  totalLayers = 32,
}: {
  gpuIdx: number;
  numGPUs: number;
  totalLayers?: number;
}) {
  const layersPerGpu = Math.floor(totalLayers / numGPUs);
  const layerStart = gpuIdx * layersPerGpu;
  const layerEnd = layerStart + layersPerGpu - 1;
  const isFirst = gpuIdx === 0;
  const isLast = gpuIdx === numGPUs - 1;

  return (
    <div style={gpuCardStyle(color.waiting)}>
      <div style={gpuHeaderStyle}>
        <Term tokenKey="gpu">GPU {gpuIdx}</Term>
      </div>
      <div style={gpuContentStyle}>
        <span style={{ fontSize: font.size.xs, color: color.textFaint }}>Layers:</span>
        <span style={layerChipStyle(true)}>
          {layerStart}–{layerEnd}
        </span>
        <span style={{ fontSize: font.size.xs, color: color.textFaint, marginTop: 2 }}>
          {isFirst ? "← input" : `← GPU ${gpuIdx - 1}`}
          {" / "}
          {isLast ? "→ output" : `→ GPU ${gpuIdx + 1}`}
        </span>
        <span style={{ fontSize: font.size.xs, color: color.borderStrong }}>micro-batch pipeline</span>
      </div>
    </div>
  );
}

function EpGpuCard({
  gpuIdx,
  numGPUs,
  totalExperts = 8,
}: {
  gpuIdx: number;
  numGPUs: number;
  totalExperts?: number;
}) {
  const expertsPerGpu = Math.floor(totalExperts / numGPUs);
  const expertStart = gpuIdx * expertsPerGpu;

  return (
    <div style={gpuCardStyle(color.accent)}>
      <div style={gpuHeaderStyle}>
        <Term tokenKey="gpu">GPU {gpuIdx}</Term>
      </div>
      <div style={gpuContentStyle}>
        <span style={{ fontSize: font.size.xs, color: color.textFaint }}>Experts:</span>
        {Array.from({ length: expertsPerGpu }, (_, i) => expertStart + i).map((e) => (
          <span
            key={e}
            style={{
              ...expertChipBase,
              background: `${color.accent}22`,
              border: `1px solid ${color.accent}`,
              color: color.accent,
            }}
          >
            Expert {e}
          </span>
        ))}
        <span style={{ fontSize: font.size.xs, color: color.borderStrong, marginTop: 2 }}>
          ⇄ all-to-all dispatch
        </span>
      </div>
    </div>
  );
}

function DpGpuCard({
  gpuIdx,
  requests,
  numGPUs,
}: {
  gpuIdx: number;
  requests: { id: number; status: string }[];
  numGPUs: number;
}) {
  const myRequests = requests.filter((_, idx) => idx % numGPUs === gpuIdx);

  return (
    <div style={gpuCardStyle(color.decode)}>
      <div style={gpuHeaderStyle}>
        <Term tokenKey="gpu">GPU {gpuIdx}</Term>
      </div>
      <div style={gpuContentStyle}>
        <span style={{ fontSize: font.size.xs, color: color.textFaint }}>Full model replica</span>
        <span style={{ fontSize: font.size.xs, color: color.textFaint, marginTop: 2 }}>Requests:</span>
        {myRequests.length > 0 ? (
          myRequests.map((r) => (
            <span key={r.id} style={requestChipStyle(REQUEST_COLORS[r.id % REQUEST_COLORS.length])}>
              Req {r.id} ({r.status})
            </span>
          ))
        ) : (
          <span style={{ fontSize: font.size.sm, color: color.border, fontStyle: "italic" }}>none</span>
        )}
        <span style={{ fontSize: font.size.xs, color: color.borderStrong, marginTop: 2 }}>
          no cross-GPU comm
        </span>
      </div>
    </div>
  );
}

// ─── Mode info descriptions ───────────────────────────────────────────────────

const MODE_INFO: Record<
  ParMode,
  { label: string; accent: string; termKey: string; split: string; comm: string; useCase: string }
> = {
  tp: {
    label: "Tensor Parallel (TP)",
    accent: color.prefill,
    termKey: "tensorParallel",
    split: "Splits: weight matrices within each layer (attention heads, FFN columns)",
    comm: "Communication: all-reduce after every layer",
    useCase: "Use when: model is too large for one GPU and low-latency single-request serving matters",
  },
  pp: {
    label: "Pipeline Parallel (PP)",
    accent: color.waiting,
    termKey: "pipelineParallel",
    split: "Splits: contiguous subsets of layers across GPUs",
    comm: "Communication: activation tensors passed between adjacent pipeline stages",
    useCase: "Use when: model has many layers and you want to overlap compute across micro-batches",
  },
  ep: {
    label: "Expert Parallel (EP)",
    accent: color.accent,
    termKey: "expertParallel",
    split: "Splits: MoE expert blocks — each GPU owns a subset of experts (FFN only, not KV)",
    comm: "Communication: all-to-all dispatch of token activations to expert GPUs",
    useCase: "Use when: serving MoE models (Mixtral, DeepSeek). Often paired with DP-attention — see the SGLang stage",
  },
  dp: {
    label: "Data Parallel (DP)",
    accent: color.decode,
    termKey: "dataParallel",
    split: "Splits: the request batch — each GPU holds a full model replica",
    comm: "Communication: none in the forward pass (replicas are independent; just load-balanced)",
    useCase: "Use when: model fits on one GPU and you want to scale throughput linearly",
  },
};

// ─── Component ────────────────────────────────────────────────────────────────

export function Stage5Parallelism({ engine, config }: StageProps) {
  const numGPUs = config.numGPUs ?? 4;
  const [mode, setMode] = useState<ParMode>(
    (config.parallelism as ParMode | undefined) ?? "tp"
  );

  const info = MODE_INFO[mode];
  const activeRequests = engine.requests.filter(
    (r) => r.status === "waiting" || r.status === "running" || r.status === "swapped"
  );

  return (
    <div style={outerStyle} aria-label="Parallelism strategies visualization">
      {/* Mode selector */}
      <div>
        <p style={{ ...sectionLabel, marginBottom: space.md }}>
          Parallelism Mode ({numGPUs} GPUs)
        </p>
        <div style={modeSelectorStyle} role="group" aria-label="Select parallelism mode">
          {(["tp", "pp", "ep", "dp"] as ParMode[]).map((m) => (
            <button
              key={m}
              style={modeButtonStyle(mode === m, MODE_INFO[m].accent)}
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Mode info box */}
      <div style={infoBoxStyle}>
        <p style={{ margin: 0, fontWeight: font.weight.bold, color: color.textPrimary }}>
          <Term tokenKey={info.termKey}>{info.label}</Term>
        </p>
        <p style={{ margin: `${space.sm}px 0 2px` }}>{info.split}</p>
        <p style={{ margin: "2px 0" }}>{info.comm}</p>
        <p style={whenToUseStyle}>{info.useCase}</p>
      </div>

      {/* GPU cards grid */}
      <div>
        <p style={{ ...sectionLabel, marginBottom: 10 }}>
          GPU Layout
        </p>
        <div style={gpuGridStyle} aria-label={`${mode.toUpperCase()} GPU layout with ${numGPUs} GPUs`}>
          {Array.from({ length: numGPUs }, (_, i) => {
            if (mode === "tp") {
              return <TpGpuCard key={i} gpuIdx={i} numGPUs={numGPUs} />;
            } else if (mode === "pp") {
              return <PpGpuCard key={i} gpuIdx={i} numGPUs={numGPUs} />;
            } else if (mode === "ep") {
              return <EpGpuCard key={i} gpuIdx={i} numGPUs={numGPUs} />;
            } else {
              return (
                <DpGpuCard key={i} gpuIdx={i} requests={activeRequests} numGPUs={numGPUs} />
              );
            }
          })}
        </div>
      </div>
    </div>
  );
}
