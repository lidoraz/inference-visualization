import type { EngineState, Config } from "../engine/types";
import { Term } from "./Term";
import { color, space, radius, font, sectionLabel, palette } from "../theme";

interface HighwayPanelProps {
  engine: EngineState;
  config: Config;
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.md,
  fontFamily: font.sans,
};

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  fontSize: font.size.sm,
  color: color.textMuted,
};

const lanesContainerStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 6,
};

const laneRowStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
};

const barOuterStyle: React.CSSProperties = {
  flex: 1,
  height: 18,
  background: palette.surface0,
  borderRadius: radius.sm,
  overflow: "hidden",
  position: "relative",
};

const counterStyle: React.CSSProperties = {
  fontSize: font.size.xs,
  fontFamily: font.mono,
  color: color.textMuted,
  whiteSpace: "nowrap",
  minWidth: 72,
  textAlign: "right",
  flexShrink: 0,
};

const legendStyle: React.CSSProperties = {
  display: "flex",
  gap: space.lg,
  fontSize: font.size.xs,
  color: color.textFaint,
  alignItems: "center",
};

function dotStyle(hue: number): React.CSSProperties {
  return {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: `hsla(${hue}, 70%, 55%, 0.9)`,
    flexShrink: 0,
  };
}

function legendSwatch(bg: string): React.CSSProperties {
  return {
    display: "inline-block",
    width: 8,
    height: 8,
    background: bg,
    borderRadius: 2,
    marginRight: 3,
    verticalAlign: "middle",
  };
}

function idLabelStyle(hue: number): React.CSSProperties {
  return {
    fontSize: font.size.sm,
    fontFamily: font.mono,
    color: `hsla(${hue}, 70%, 65%, 1)`,
    width: 28,
    flexShrink: 0,
  };
}

// ─── Component ────────────────────────────────────────────────────────────────

export function HighwayPanel({ engine, config }: HighwayPanelProps) {
  const running = engine.requests.filter((r) => r.status === "running");

  if (running.length === 0) return null;

  const globalMax = Math.max(...running.map((r) => r.promptTokens.length + r.maxDecode)) || 1;

  return (
    <div style={containerStyle} aria-label="Batch highway visualization">
      <div style={headerStyle}>
        <Term tokenKey="sequenceProgress">
          <span style={sectionLabel}>Sequence Progress</span>
        </Term>
        <span style={{ color: color.textFaint, fontSize: font.size.xs }}>highway</span>
        <span>·</span>
        <span>{running.length} {running.length === 1 ? "lane" : "lanes"}</span>
        <span>·</span>
        <span>
          batch size:{" "}
          <Term tokenKey="maxBatchSize">{config.maxBatchSize}</Term>
        </span>
      </div>

      <div style={lanesContainerStyle}>
        {running.map((req) => {
          const hue = (req.id * 57) % 360;
          const promptLen = req.promptTokens.length;
          const decodedLen = req.decodedTokens.length;
          const totalLen = promptLen + req.maxDecode;
          const currentPos = promptLen + decodedLen;

          const promptPct = (promptLen / globalMax) * 100;
          const decodedPct = (decodedLen / globalMax) * 100;

          // Tick marks every blockSize tokens, expressed as % of the bar width.
          // Using a repeating-linear-gradient avoids extra DOM nodes.
          const tickSpacingPct = (config.blockSize / globalMax) * 100;
          const tickOverlay =
            tickSpacingPct > 1
              ? `repeating-linear-gradient(to right, transparent 0%, transparent calc(${tickSpacingPct}% - 1px), rgba(0,0,0,0.35) calc(${tickSpacingPct}% - 1px), rgba(0,0,0,0.35) ${tickSpacingPct}%)`
              : "none";

          return (
            <div
              key={req.id}
              style={laneRowStyle}
              aria-label={`Request ${req.id}: ${currentPos} of ${totalLen} tokens, phase: ${req.phase}`}
            >
              <div style={dotStyle(hue)} aria-hidden="true" />
              <span style={idLabelStyle(hue)}>#{req.id}</span>
              <div style={barOuterStyle} title={`${req.phase}: ${currentPos}/${totalLen} tokens`} aria-hidden="true">
                {promptPct > 0 && (
                  <div style={{ position: "absolute", left: 0, width: `${promptPct}%`, height: "100%", background: color.prefill }} />
                )}
                {decodedPct > 0 && (
                  <div
                    style={{
                      position: "absolute",
                      left: `${promptPct}%`,
                      width: `${decodedPct}%`,
                      height: "100%",
                      background: color.decode,
                      transition: "width 0.15s ease",
                    }}
                  />
                )}
                {/* tick overlay sits on top of fill segments */}
                <div style={{ position: "absolute", inset: 0, backgroundImage: tickOverlay }} />
              </div>
              <span style={counterStyle}>
                {currentPos} / {totalLen}
              </span>
            </div>
          );
        })}
      </div>

      <div style={legendStyle} aria-label="Color legend">
        <span><span style={legendSwatch(color.prefill)} />prompt</span>
        <span><span style={legendSwatch(color.decode)} />decoded</span>
        <span><span style={legendSwatch(palette.surface0)} />remaining</span>
      </div>
    </div>
  );
}
