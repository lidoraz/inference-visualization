/**
 * Controls — shared simulation control panel.
 *
 * Purely presentational: no engine imports, all state driven via props.
 * Local state is limited to the uncommitted add-request form fields.
 */

import { useState } from "react";
import type { Config } from "../engine/types";
import { randomSentence } from "../content/sampleSentences";
import { normalDecodeLength } from "../content/samplers";
import { color, space, radius, font, sectionLabel } from "../theme";
import { Term } from "./Term";

// Decode-length distribution for the 🎲 button (and load generator). Mean
// matches the default field value; sd gives a realistic spread of short/long
// generations. Clamped to >= 1 inside normalDecodeLength.
const DECODE_MEAN = 20;
const DECODE_SD = 12;

/**
 * Which control sections a stage needs. Live-sim stages enable most; the
 * diagram stages (5–7) drive their own in-view toggles and don't tick, so they
 * hide the simulation machinery.
 */
export interface StageCapabilities {
  /** Playback (step/play/reset) + speed. */
  simulation: boolean;
  /** Add Request controls (hidden on a pre-seeded onboarding stage). */
  addRequest: boolean;
  /** Engine Setup launch knobs (KV cache blocks, block size). */
  engineSetup: boolean;
  /** Load generator (auto-arrivals) — only meaningful for batching stages. */
  loadGenerator: boolean;
  /** Select-a-request + Cancel. */
  cancel: boolean;
  /** Scheduler knob (max batch size). */
  scheduler: boolean;
}

export interface ControlsProps {
  isPlaying: boolean;
  speed: number; // ms per tick
  config: Config;
  selectedRequestId: number | null;
  loadOn: boolean;
  loadIntensity: number; // 1 (sparse) .. 10 (bursty)
  /** Number of live requests — used to surface the "click a chip" cancel hint. */
  requestCount: number;
  caps: StageCapabilities;
  onStep(): void;
  onPlayToggle(): void;
  onReset(): void;
  onSpeedChange(ms: number): void;
  onAddRequest(prompt: string, maxDecode: number): void;
  onCancelRequest(requestId: number): void;
  /** Live scheduler knobs (max batch size) — applied on the next tick. */
  onConfigChange(patch: Partial<Config>): void;
  /** Engine-setup knobs (block size, KV-cache blocks) — re-initialize the pool. */
  onEngineSetupChange(patch: Partial<Config>): void;
  /** True once the sim has started; engine-setup knobs lock (pool is fixed). */
  engineSetupLocked: boolean;
  onLoadToggle(): void;
  onLoadIntensityChange(intensity: number): void;
}

// ─── styles ──────────────────────────────────────────────────────────────────

const panelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: space.lg,
  padding: `${space.lg}px ${space.xl}px`,
  background: color.panelBg,
  border: `1px solid ${color.border}`,
  borderRadius: radius.lg,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
};

const rowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: space.md,
  alignItems: "center",
};

const sectionDividerStyle: React.CSSProperties = {
  borderTop: `1px solid ${color.border}`,
  paddingTop: space.md + 2,
};

const btnStyle: React.CSSProperties = {
  padding: `${space.sm}px ${space.lg}px`,
  borderRadius: radius.sm + 1,
  border: `1px solid ${color.borderStrong}`,
  background: color.border,
  color: color.textPrimary,
  cursor: "pointer",
  fontSize: font.size.base,
  minHeight: 36,
};

const btnPrimaryStyle: React.CSSProperties = {
  ...btnStyle,
  background: color.prefill,
  color: color.panelBg,
  border: "none",
  fontWeight: font.weight.semibold,
};

const btnDangerStyle: React.CSSProperties = {
  ...btnStyle,
  background: color.danger,
  color: color.panelBg,
  border: "none",
};

const inputStyle: React.CSSProperties = {
  padding: `${space.xs}px ${space.md}px`,
  borderRadius: radius.sm,
  border: `1px solid ${color.borderStrong}`,
  background: color.panelBgInset,
  color: color.textPrimary,
  fontSize: font.size.base,
};

const labelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 2,
  fontSize: font.size.md,
  color: color.textMuted,
};

const inputDisabledStyle: React.CSSProperties = {
  ...inputStyle,
  opacity: 0.45,
  cursor: "not-allowed",
};

const lockNoteStyle: React.CSSProperties = {
  fontSize: font.size.sm,
  color: color.textFaint,
  margin: `${space.xs}px 0 0`,
  lineHeight: 1.5,
};

// ─── Speed helper ─────────────────────────────────────────────────────────────
// Slider is 1–10 left→right, matching the "Fast"→"Slow" labels: value 1 (left,
// Fast) = 100ms, value 10 (right, Slow) = 1000ms.
const SPEED_MIN = 100;
const SPEED_MAX = 1000;

function msToSlider(ms: number): number {
  return Math.round(1 + ((ms - SPEED_MIN) / (SPEED_MAX - SPEED_MIN)) * 9);
}

function sliderToMs(val: number): number {
  return Math.round(SPEED_MIN + ((val - 1) / 9) * (SPEED_MAX - SPEED_MIN));
}

// ─── Component ────────────────────────────────────────────────────────────────

export function Controls({
  isPlaying,
  speed,
  config,
  selectedRequestId,
  loadOn,
  loadIntensity,
  onStep,
  onPlayToggle,
  onReset,
  onSpeedChange,
  onAddRequest,
  onCancelRequest,
  requestCount,
  caps,
  onConfigChange,
  onEngineSetupChange,
  engineSetupLocked,
  onLoadToggle,
  onLoadIntensityChange,
}: ControlsProps) {
  const [prompt, setPrompt] = useState("");
  const [maxDecode, setMaxDecode] = useState(20);
  const [customizing, setCustomizing] = useState(false);

  function handleAdd() {
    // Empty field falls back to a random sample so Add is never a dead button.
    const text = prompt.trim() === "" ? randomSentence() : prompt.trim();
    onAddRequest(text, maxDecode);
    setPrompt("");
  }

  function handleRandom() {
    setPrompt(randomSentence());
  }

  function handleRandomDecode() {
    setMaxDecode(normalDecodeLength(DECODE_MEAN, DECODE_SD));
  }

  function handleAddKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleAdd();
  }

  return (
    <div style={panelStyle} aria-label="Simulation controls">
      {caps.simulation && (
        <>
          {/* ── Playback controls ── */}
          <div>
            <p style={{ ...sectionLabel, marginBottom: space.xs }}>Playback</p>
            <div style={rowStyle}>
              <button style={btnStyle} onClick={onStep} disabled={isPlaying} aria-label="Single step">
                Step ▶
              </button>
              <button
                style={btnPrimaryStyle}
                onClick={onPlayToggle}
                aria-label={isPlaying ? "Pause simulation" : "Play simulation"}
              >
                {isPlaying ? "⏸ Pause" : "▶ Play"}
              </button>
              <button style={btnStyle} onClick={onReset} aria-label="Reset simulation">
                ↺ Reset
              </button>
            </div>
          </div>

          {/* ── Speed slider ── */}
          <div style={sectionDividerStyle}>
            <p style={{ ...sectionLabel, marginBottom: space.xs }}>Speed</p>
            <label style={{ ...labelStyle, flexDirection: "row", alignItems: "center", gap: space.md }}>
              <span style={{ minWidth: 36, color: color.textPrimary }}>Fast</span>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={msToSlider(speed)}
                onChange={(e) => onSpeedChange(sliderToMs(Number(e.target.value)))}
                aria-label="Simulation speed"
                style={{ flex: 1 }}
              />
              <span style={{ minWidth: 36, color: color.textPrimary }}>Slow</span>
              <span style={{ minWidth: 48, color: color.textMuted, fontSize: font.size.md }}>
                {speed} ms
              </span>
            </label>
          </div>

          {/* ── Add request (one-click; Customize reveals fields) ── */}
          {caps.addRequest && (
          <div style={sectionDividerStyle}>
            <p style={{ ...sectionLabel, marginBottom: space.xs }}>Add Request</p>
            <div style={rowStyle}>
              <button
                style={btnPrimaryStyle}
                onClick={handleAdd}
                aria-label="Add a request (random sentence unless you customize the prompt)"
              >
                + Add Request
              </button>
              <button
                style={btnStyle}
                onClick={() => setCustomizing((c) => !c)}
                aria-expanded={customizing}
                aria-label="Customize the next request's prompt and length"
              >
                {customizing ? "▾ Customize" : "▸ Customize"}
              </button>
            </div>
            {customizing && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: space.sm, marginTop: space.md }}>
                <label style={{ ...labelStyle, flex: "1 1 100%" }}>
                  <span>Prompt</span>
                  <div style={{ display: "flex", gap: space.xs }}>
                    <input
                      type="text"
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      onKeyDown={handleAddKeyDown}
                      placeholder="Type a prompt or use 🎲…"
                      aria-label="Request prompt text"
                      style={{ ...inputStyle, flex: 1, minWidth: 0 }}
                    />
                    <button
                      style={btnStyle}
                      onClick={handleRandom}
                      aria-label="Fill prompt with a random sentence"
                      title="Random sentence"
                    >
                      🎲
                    </button>
                  </div>
                </label>
                <label style={labelStyle}>
                  <span>Max decode</span>
                  <div style={{ display: "flex", gap: space.xs }}>
                    <input
                      type="number"
                      value={maxDecode}
                      min={1}
                      max={512}
                      onChange={(e) => setMaxDecode(Number(e.target.value))}
                      aria-label="Maximum decode tokens"
                      style={{ ...inputStyle, width: 72, minWidth: 0 }}
                    />
                    <button
                      style={btnStyle}
                      onClick={handleRandomDecode}
                      aria-label="Set max decode to a random value from a normal distribution"
                      title="Sample a decode length from a normal distribution"
                    >
                      🎲
                    </button>
                  </div>
                </label>
              </div>
            )}
          </div>
          )}
        </>
      )}

      {/* ── Load generator (auto-arrivals) ── */}
      {caps.loadGenerator && (
        <div style={sectionDividerStyle}>
          <p style={{ ...sectionLabel, marginBottom: space.xs }}>Load Generator</p>
          <div style={rowStyle}>
            <button
              style={loadOn ? btnDangerStyle : btnPrimaryStyle}
              onClick={onLoadToggle}
              aria-pressed={loadOn}
              aria-label={loadOn ? "Stop auto-arrivals" : "Start auto-arrivals"}
              title="Auto-generate requests at randomly-spaced intervals (Poisson arrivals)"
            >
              {loadOn ? "⏹ Stop Load" : "🚦 Start Load"}
            </button>
          </div>
          <label
            style={{ ...labelStyle, flexDirection: "row", alignItems: "center", gap: space.md, marginTop: space.md }}
          >
            <span style={{ minWidth: 48, color: color.textPrimary }}>Light</span>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={loadIntensity}
              onChange={(e) => onLoadIntensityChange(Number(e.target.value))}
              aria-label="Load intensity"
              style={{ flex: 1 }}
            />
            <span style={{ minWidth: 48, color: color.textPrimary }}>Heavy</span>
          </label>
          <p style={{ fontSize: font.size.sm, color: color.textFaint, margin: `${space.xs}px 0 0` }}>
            Requests arrive at random intervals (exponential gaps), each with a normally-distributed
            decode length. Higher intensity = more frequent arrivals.
          </p>
        </div>
      )}

      {/* ── Cancel selected request ── */}
      {caps.cancel && (
        <div style={sectionDividerStyle}>
          <p style={{ ...sectionLabel, marginBottom: space.xs }}>Selected Request</p>
          {selectedRequestId !== null ? (
            <div style={rowStyle}>
              <span style={{ color: color.textMuted }}>#{selectedRequestId}</span>
              <button
                style={btnDangerStyle}
                onClick={() => onCancelRequest(selectedRequestId)}
                aria-label={`Cancel request ${selectedRequestId}`}
              >
                Cancel Request
              </button>
            </div>
          ) : (
            <p style={{ fontSize: font.size.sm, color: color.textFaint, margin: 0, lineHeight: 1.5 }}>
              {requestCount > 0
                ? "Click a request chip above to select it, then cancel it here to watch its KV blocks free instantly."
                : "Add requests, then click one to select and cancel it."}
            </p>
          )}
        </div>
      )}

      {/* ── Engine setup (launch-time; locked once the sim has started) ── */}
      {caps.engineSetup && (
      <div style={sectionDividerStyle}>
        <p style={{ ...sectionLabel, marginBottom: space.xs }}>Engine Setup</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: space.lg }}>
          <label style={labelStyle}>
            <span>
              <Term tokenKey="kvCache">KV Cache Blocks</Term>
            </span>
            <input
              type="number"
              value={config.kvCacheBlocks}
              min={1}
              max={256}
              disabled={engineSetupLocked}
              onChange={(e) =>
                onEngineSetupChange({ kvCacheBlocks: Number(e.target.value) })
              }
              aria-label="KV cache block count"
              style={{ ...(engineSetupLocked ? inputDisabledStyle : inputStyle), width: 72, minWidth: 0 }}
            />
          </label>
          <label style={labelStyle}>
            <span>
              <Term tokenKey="block">Block Size</Term>
            </span>
            <input
              type="number"
              value={config.blockSize}
              min={1}
              max={128}
              disabled={engineSetupLocked}
              onChange={(e) =>
                onEngineSetupChange({ blockSize: Number(e.target.value) })
              }
              aria-label="Block size (tokens per block)"
              style={{ ...(engineSetupLocked ? inputDisabledStyle : inputStyle), width: 72, minWidth: 0 }}
            />
          </label>
        </div>
        <p style={lockNoteStyle}>
          {engineSetupLocked
            ? "Locked while the sim is running — the block pool is allocated once at startup, like vLLM's launch flags. Press Reset to change these."
            : "Set before adding requests: these size the physical KV block pool (allocated once at startup)."}
        </p>
      </div>
      )}

      {/* ── Scheduler (live knobs) ── */}
      {caps.scheduler && (
      <div style={sectionDividerStyle}>
        <p style={{ ...sectionLabel, marginBottom: space.xs }}>Scheduler</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: space.lg }}>
          <label style={labelStyle}>
            <span>
              <Term tokenKey="maxBatchSize">Max Batch Size</Term>
            </span>
            <input
              type="number"
              value={config.maxBatchSize}
              min={1}
              max={64}
              onChange={(e) =>
                onConfigChange({ maxBatchSize: Number(e.target.value) })
              }
              aria-label="Maximum batch size"
              title="vLLM's max_num_seqs: how many requests run concurrently. Real servers use 256+; 1 serializes requests and wastes the GPU."
              style={{ ...inputStyle, width: 72, minWidth: 0 }}
            />
          </label>
          <label style={labelStyle}>
            <span>
              <Term tokenKey="tokenBudget">Token Budget</Term>
            </span>
            <input
              type="number"
              value={config.tokenBudget}
              min={1}
              max={2048}
              onChange={(e) =>
                onConfigChange({ tokenBudget: Number(e.target.value) })
              }
              aria-label="Token budget per step"
              title="vLLM's max_num_batched_tokens: total tokens processed in one step. Caps how much prefill work one step admits."
              style={{ ...inputStyle, width: 72, minWidth: 0 }}
            />
          </label>
        </div>
        <p style={lockNoteStyle}>
          Live knobs — apply on the next step. Hover a label for what it does.
        </p>
      </div>
      )}

      {/* Diagram stages have no live sim — point to the in-view toggles. */}
      {!caps.simulation && (
        <p style={{ fontSize: font.size.md, color: color.textFaint, margin: 0, lineHeight: 1.6 }}>
          This stage is a diagram — use the toggles in the panel to the left to explore it. The step
          guide above walks through each idea.
        </p>
      )}
    </div>
  );
}
