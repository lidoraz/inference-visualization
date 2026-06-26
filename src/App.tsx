/**
 * App — application shell for the vLLM Inference Visualizer.
 *
 * Responsibilities:
 *   - Own ALL state: engine state (via useReducer), config, UI state.
 *   - Wire the pure engine to the UI via a thin AppReducer.
 *   - Manage stage navigation (1–10) with per-stage config presets.
 *   - Drive the Play loop (setInterval dispatching STEP).
 *   - Render layout: header + stage nav, active stage, Controls, GuidePanel.
 *
 * Config ownership: config is NOT stored in EngineState. The AppReducer holds
 * config alongside engine and passes it into every engine.reduce / engine.tick
 * call. SET_CONFIG patches state.config and passes the merged config to the
 * engine (which ignores it, but that's intentional). SET_STAGE_CONFIG replaces
 * config wholesale and re-inits the engine.
 */

import { useReducer, useState, useEffect, useCallback, useRef, useSyncExternalStore } from "react";

function useIsMobile(breakpoint = 768): boolean {
  const query = `(max-width: ${breakpoint}px)`;
  return useSyncExternalStore(
    (cb) => {
      const mq = window.matchMedia(query);
      mq.addEventListener("change", cb);
      return () => mq.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false
  );
}
import { init, reduce as engineReduce } from "./engine/engine";
import type { Action, Config, EngineState } from "./engine/types";
import { Controls, type StageCapabilities } from "./components/Controls";
import { GuidePanel } from "./components/GuidePanel";
import { stageGuides } from "./content/stageGuides";
import { randomSentence } from "./content/sampleSentences";
import { exponentialDelay, normalDecodeLength } from "./content/samplers";
import { Stage1Lifecycle } from "./stages/Stage1Lifecycle";
import { StageModelConcepts } from "./stages/StageModelConcepts";
import { Stage2PagedKV } from "./stages/Stage2PagedKV";
import { Stage3Batching } from "./stages/Stage3Batching";
import { Stage4PDDisagg } from "./stages/Stage4PDDisagg";
import { Stage5Parallelism } from "./stages/Stage5Parallelism";
import { Stage6ModelFeatures } from "./stages/Stage6ModelFeatures";
import { StageAttention } from "./stages/StageAttention";
import { StageSpeculative } from "./stages/StageSpeculative";
import { Stage7SGLang } from "./stages/Stage7SGLang";
import { color, space, radius, font } from "./theme";

// ─── Stage config presets ─────────────────────────────────────────────────────

// Stage order (difficulty ramp): 1 Lifecycle, 2 Model Concepts (sampling),
// 3 Paged KV, 4 Batching, 5 Parallelism, 6 FP8/MoE, 7 Attention (sliding-window
// + KV-head variants), 8 Speculative Decoding, 9 PD Disaggregation, 10 SGLang.
const STAGE_CONFIGS: Record<number, Config> = {
  1: { maxBatchSize: 1, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 64 },
  2: { maxBatchSize: 1, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 64 },
  3: { maxBatchSize: 1, blockSize: 4, kvCacheBlocks: 12, tokenBudget: 64 },
  4: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32 },
  5: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32, numGPUs: 4, parallelism: "tp" },
  6: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32, quant: "fp16", moe: false },
  7: { maxBatchSize: 1, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 64 },
  8: { maxBatchSize: 1, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 64 },
  9: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32 },
  10: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32, numGPUs: 4, parallelism: "dp", moe: true },
};

// Which control sections each stage exposes. Live-sim stages (1, 3, 4, 9) show
// scheduler/queue controls; concept/diagram stages (2, 5, 6, 7, 8, 10) hide them.
const STAGE_CAPS: Record<number, StageCapabilities> = {
  // Stage 1 onboards with a single pre-seeded request: just step through it.
  1: { simulation: true, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  // Stage 2 (Model Concepts) is a self-contained illustrative panel — no sim.
  2: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  3: { simulation: true, addRequest: true, engineSetup: true, loadGenerator: true, cancel: true, scheduler: false },
  4: { simulation: true, addRequest: true, engineSetup: true, loadGenerator: true, cancel: true, scheduler: true },
  5: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  6: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  // Stage 7 (Attention) and 8 (Speculative) are self-contained diagrams.
  7: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  8: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
  9: { simulation: true, addRequest: true, engineSetup: true, loadGenerator: true, cancel: true, scheduler: true },
  10: { simulation: false, addRequest: false, engineSetup: false, loadGenerator: false, cancel: false, scheduler: false },
};

// How many sample requests to pre-seed so a stage lands ready to Play. Stage 1
// shows one lifecycle; the batching/PD stages want a few to fill the batch.
const STAGE_SEED_COUNT: Record<number, number> = {
  1: 1,
  2: 0, // Model Concepts: self-contained, no sim requests
  3: 3,
  4: 3,
  5: 4, // parallelism: a few requests so DP mode shows them distributed across GPUs
  6: 1, // one request so the MoE view routes a real prompt's tokens
  7: 0, // Attention: self-contained diagram
  8: 0, // Speculative decoding: self-contained diagram
  9: 3, // PD disaggregation: a few requests to populate prefill/decode workers
  10: 0,
};

// ─── Load generator tuning ──────────────────────────────────────────────────
// Intensity (1..10) maps to a mean inter-arrival gap. Light = long gaps,
// heavy = short gaps. Exponential sampling around the mean yields Poisson
// arrivals (natural bursts/lulls). Decode lengths come from a normal dist.
const LOAD_MEAN_GAP_MIN_MS = 400; // heaviest load: ~0.4s mean gap
const LOAD_MEAN_GAP_MAX_MS = 4000; // lightest load: ~4s mean gap
const LOAD_DECODE_MEAN = 20;
const LOAD_DECODE_SD = 12;

function intensityToMeanGapMs(intensity: number): number {
  // intensity 1 -> MAX gap (light), 10 -> MIN gap (heavy). Linear interpolation.
  const t = (intensity - 1) / 9; // 0..1
  return LOAD_MEAN_GAP_MAX_MS - t * (LOAD_MEAN_GAP_MAX_MS - LOAD_MEAN_GAP_MIN_MS);
}

// ─── App-level reducer ────────────────────────────────────────────────────────

interface AppState {
  engine: EngineState;
  config: Config;
}

/**
 * Build a fresh engine for a config with `seedCount` sample requests already
 * queued, so a stage lands ready to Play. Requests are waiting (hold no blocks),
 * so this does not "start" the sim — tick stays 0 and Engine Setup stays editable.
 */
function seededEngine(config: Config, seedCount: number): EngineState {
  let engine = init(config);
  for (let i = 0; i < seedCount; i++) {
    engine = engineReduce(
      engine,
      // min 8 so the decode phase is always visible — without a floor the normal
      // sampler can draw 1-2 and the request finishes the instant it reaches
      // decode (invisible decode phase on Stages 1 & 4).
      { type: "ADD_REQUEST", prompt: randomSentence(), maxDecode: normalDecodeLength(20, 12, 8) },
      config
    );
  }
  return engine;
}

type AppAction =
  | Action
  | { type: "SET_STAGE_CONFIG"; config: Config; seedCount: number }
  | { type: "RESEED"; seedCount: number };

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_CONFIG": {
      // Merge the config patch into the React-owned config. Engine ignores
      // SET_CONFIG but we still call it to stay consistent with the interface.
      const merged: Config = { ...state.config, ...action.config };
      return {
        engine: engineReduce(state.engine, action, merged),
        config: merged,
      };
    }

    case "SET_STAGE_CONFIG": {
      // Replace config wholesale and re-initialize with the stage's seed requests.
      const newConfig = action.config;
      return {
        engine: seededEngine(newConfig, action.seedCount),
        config: newConfig,
      };
    }

    case "RESEED":
      // Reset/reseed: rebuild the engine with the stage's seed requests so it
      // lands ready to Play again.
      return {
        engine: seededEngine(state.config, action.seedCount),
        config: state.config,
      };

    case "STEP":
    case "ADD_REQUEST":
    case "CANCEL_REQUEST":
    case "SET_ARRIVAL_RATE":
      return {
        ...state,
        engine: engineReduce(state.engine, action, state.config),
      };

    default:
      return state;
  }
}

// ─── Layout styles ────────────────────────────────────────────────────────────

const appStyle: React.CSSProperties = {
  minHeight: "100vh",
  background: color.pageBg,
  color: color.textPrimary,
  fontFamily: font.sans,
  display: "flex",
  flexDirection: "column",
};

const CONTENT_MAX_WIDTH = 1200;
const RAIL_WIDTH = 200;

const STAGE_SHORT_TITLE: Record<number, string> = {
  1: "Lifecycle",
  2: "Model Concepts",
  3: "Paged KV Cache",
  4: "Batching",
  5: "Parallelism",
  6: "FP8 & MoE",
  7: "Attention",
  8: "Spec Decoding",
  9: "PD Disagg",
  10: "SGLang",
};

const headerStyle: React.CSSProperties = {
  borderBottom: `1px solid ${color.border}`,
};

function headerInnerStyle(isMobile: boolean): React.CSSProperties {
  return {
    display: "flex",
    alignItems: "center",
    gap: space.xxl,
    flexWrap: "wrap",
    width: "100%",
    maxWidth: isMobile ? "100%" : CONTENT_MAX_WIDTH + RAIL_WIDTH + 48,
    margin: "0 auto",
    padding: isMobile ? `${space.md}px ${space.lg}px` : `${space.lg}px ${space.xxl}px`,
    boxSizing: "border-box",
  };
}

function titleStyle(isMobile: boolean): React.CSSProperties {
  return {
    margin: 0,
    fontSize: isMobile ? font.size.lg : font.size.xxl,
    fontWeight: font.weight.bold,
    color: color.accent,
    letterSpacing: "0.02em",
    whiteSpace: "nowrap",
  };
}

function bodyRowStyle(isMobile: boolean): React.CSSProperties {
  return {
    flex: 1,
    width: "100%",
    maxWidth: isMobile ? "100%" : CONTENT_MAX_WIDTH + RAIL_WIDTH + 48,
    margin: "0 auto",
    display: "flex",
    flexDirection: isMobile ? "column" : "row",
    alignItems: "flex-start",
    boxSizing: "border-box",
  };
}

// ─── Left nav rail (desktop) / horizontal tab strip (mobile) ─────────────────
function railStyle(isMobile: boolean): React.CSSProperties {
  if (isMobile) {
    return {
      width: "100%",
      display: "flex",
      flexDirection: "row",
      overflowX: "auto",
      gap: space.xs,
      padding: `${space.sm}px ${space.md}px`,
      boxSizing: "border-box",
      borderBottom: `1px solid ${color.border}`,
      WebkitOverflowScrolling: "touch" as React.CSSProperties["WebkitOverflowScrolling"],
      scrollbarWidth: "none" as React.CSSProperties["scrollbarWidth"],
    };
  }
  return {
    width: RAIL_WIDTH,
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    gap: space.xs,
    padding: `${space.xl}px ${space.md}px ${space.xl}px ${space.xxl}px`,
    boxSizing: "border-box",
    position: "sticky",
    top: 0,
  };
}

function stageTabStyle(active: boolean, isMobile: boolean): React.CSSProperties {
  if (isMobile) {
    return {
      display: "flex",
      alignItems: "center",
      gap: space.xs,
      flexShrink: 0,
      whiteSpace: "nowrap",
      padding: `${space.sm}px ${space.md}px`,
      borderRadius: radius.md,
      border: "none",
      borderBottom: `2px solid ${active ? color.accent : "transparent"}`,
      background: active ? color.panelBgInset : "transparent",
      color: active ? color.textPrimary : color.textMuted,
      cursor: "pointer",
      fontSize: font.size.base,
      transition: "background 0.15s, color 0.15s",
    };
  }
  return {
    display: "flex",
    alignItems: "flex-start",
    gap: space.md,
    width: "100%",
    textAlign: "left",
    padding: `${space.md}px ${space.lg}px`,
    borderRadius: radius.md,
    border: "none",
    borderLeft: `3px solid ${active ? color.accent : "transparent"}`,
    background: active ? color.panelBgInset : "transparent",
    color: active ? color.textPrimary : color.textMuted,
    cursor: "pointer",
    transition: "background 0.15s, color 0.15s",
  };
}

const stageNumStyle = (active: boolean): React.CSSProperties => ({
  fontSize: font.size.md,
  fontWeight: font.weight.bold,
  fontFamily: font.mono,
  color: active ? color.accent : color.textFaint,
  lineHeight: 1.3,
  flexShrink: 0,
});

const stageTitleStyle: React.CSSProperties = {
  fontSize: font.size.base,
  fontWeight: font.weight.semibold,
  lineHeight: 1.3,
};

function contentColStyle(isMobile: boolean): React.CSSProperties {
  return {
    flex: 1,
    minWidth: 0,
    maxWidth: isMobile ? "100%" : CONTENT_MAX_WIDTH,
    display: "flex",
    flexDirection: "column",
  };
}

function mainStyle(isMobile: boolean): React.CSSProperties {
  if (isMobile) {
    return {
      width: "100%",
      display: "flex",
      flexDirection: "column",
      gap: space.lg,
      padding: `0 ${space.md}px ${space.xl}px`,
      boxSizing: "border-box",
    };
  }
  return {
    width: "100%",
    display: "grid",
    gridTemplateColumns: "1fr 340px",
    gridTemplateRows: "1fr auto",
    gap: space.xl,
    padding: `0 ${space.md}px ${space.xl}px`,
    boxSizing: "border-box",
    alignItems: "start",
  };
}

const guideBandWrapStyle: React.CSSProperties = {
  width: "100%",
  padding: `${space.lg}px ${space.md}px ${space.md}px`,
  boxSizing: "border-box",
};

function guideBandHeaderStyle(isMobile: boolean): React.CSSProperties {
  return {
    display: "flex",
    flexDirection: isMobile ? "column" : "row",
    alignItems: isMobile ? "flex-start" : "center",
    justifyContent: "space-between",
    gap: space.md,
    marginBottom: space.md,
  };
}

const rejectionBannerStyle: React.CSSProperties = {
  padding: `10px 14px`,
  marginBottom: 14,
  borderRadius: radius.lg,
  background: `${color.danger}18`,
  border: `1px solid ${color.danger}`,
  color: color.danger,
  fontSize: font.size.base,
  lineHeight: 1.5,
};

function vizAreaStyle(isMobile: boolean): React.CSSProperties {
  return {
    background: color.panelBg,
    border: `1px solid ${color.border}`,
    borderRadius: radius.lg,
    minHeight: isMobile ? 0 : 320,
    padding: isMobile ? space.md : space.xl,
    gridColumn: "1",
    gridRow: "1",
    overflowX: "auto",
  };
}

function sidebarStyle(isMobile: boolean): React.CSSProperties {
  return {
    display: "flex",
    flexDirection: "column",
    gap: space.xl,
    gridColumn: isMobile ? undefined : "2",
    gridRow: isMobile ? undefined : "1 / 3",
  };
}

const guidePrevNextStyle: React.CSSProperties = {
  display: "flex",
  gap: space.md,
  alignItems: "center",
};

const navBtnStyle: React.CSSProperties = {
  padding: `${space.sm}px ${space.lg}px`,
  borderRadius: radius.sm + 1,
  border: `1px solid ${color.borderStrong}`,
  background: color.border,
  color: color.textPrimary,
  cursor: "pointer",
  fontSize: font.size.md,
  minHeight: 36,
};

const navBtnDisabledStyle: React.CSSProperties = {
  ...navBtnStyle,
  opacity: 0.35,
  cursor: "not-allowed",
};

const controlsRowStyle: React.CSSProperties = {
  gridColumn: "1",
  gridRow: "2",
};

// ─── Component ────────────────────────────────────────────────────────────────

export default function App() {
  const INITIAL_STAGE = 1;
  const INITIAL_SPEED = 500;
  const isMobile = useIsMobile();

  const [appState, dispatch] = useReducer(appReducer, undefined, () => ({
    engine: seededEngine(STAGE_CONFIGS[INITIAL_STAGE], STAGE_SEED_COUNT[INITIAL_STAGE]),
    config: STAGE_CONFIGS[INITIAL_STAGE],
  }));

  const [currentStage, setCurrentStage] = useState<number>(INITIAL_STAGE);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(INITIAL_SPEED);
  const [selectedRequestId, setSelectedRequestId] = useState<number | null>(null);
  const [stepIndex, setStepIndex] = useState<number>(0);
  const [loadOn, setLoadOn] = useState<boolean>(false);
  const [loadIntensity, setLoadIntensity] = useState<number>(5);

  // Work is "active" only while some request is waiting/running/swapped. Once all
  // requests are finished/cancelled (or there are none), ticking does nothing.
  const hasActiveWork = appState.engine.requests.some(
    (r) => r.status === "waiting" || r.status === "running" || r.status === "swapped"
  );

  // The sim has "started" once it has advanced past tick 0. Engine-setup knobs
  // lock here (the block pool is fixed at init). Pre-seeded *waiting* requests
  // hold no blocks and don't count as started, so the knobs stay editable until
  // the first Step/Play. Reset returns to the clean pre-start state.
  const simStarted = appState.engine.tick > 0;

  // ── Play loop ─────────────────────────────────────────────────────────────
  // Auto-pauses when there's no active work, so the tick doesn't run forever
  // after everything has finished or been rejected.
  useEffect(() => {
    if (!isPlaying) return;
    // Keep playing while the load generator is on — more requests are arriving
    // even if the queue is momentarily empty. Only auto-pause on idle when load
    // is off, so a finished manual run stops ticking on dead state.
    if (!hasActiveWork && !loadOn) {
      setIsPlaying(false);
      return;
    }
    const id = setInterval(() => {
      dispatch({ type: "STEP" });
    }, speed);
    return () => clearInterval(id);
  }, [isPlaying, speed, hasActiveWork, loadOn]);

  // ── Load generator (wall-clock auto-arrivals) ──────────────────────────────
  // Independent of the simulation tick: requests arrive on a real-time clock at
  // exponentially-distributed gaps (Poisson process), so the demo feels like a
  // live system under load. A ref holds the current intensity so changing the
  // slider re-times the NEXT gap without tearing down the timer.
  const loadIntensityRef = useRef(loadIntensity);
  loadIntensityRef.current = loadIntensity;

  useEffect(() => {
    if (!loadOn) return;
    let timer: ReturnType<typeof setTimeout>;
    const scheduleNext = () => {
      const meanGap = intensityToMeanGapMs(loadIntensityRef.current);
      const delay = exponentialDelay(meanGap);
      timer = setTimeout(() => {
        dispatch({
          type: "ADD_REQUEST",
          prompt: randomSentence(),
          maxDecode: normalDecodeLength(LOAD_DECODE_MEAN, LOAD_DECODE_SD),
        });
        scheduleNext();
      }, delay);
    };
    scheduleNext();
    return () => clearTimeout(timer);
  }, [loadOn]);

  // ── Stage switching ───────────────────────────────────────────────────────
  const handleStageSelect = useCallback(
    (stage: number) => {
      if (stage === currentStage) return;
      dispatch({ type: "SET_STAGE_CONFIG", config: STAGE_CONFIGS[stage], seedCount: STAGE_SEED_COUNT[stage] });
      setCurrentStage(stage);
      setIsPlaying(false);
      setLoadOn(false);
      setSelectedRequestId(null);
      setStepIndex(0);
    },
    [currentStage]
  );

  // ── Guide step navigation ─────────────────────────────────────────────────
  const stageGuide = stageGuides[currentStage];
  const maxStepIndex = stageGuide ? stageGuide.steps.length - 1 : 0;

  function handleGuidePrev() {
    setStepIndex((i) => Math.max(0, i - 1));
  }

  function handleGuideNext() {
    setStepIndex((i) => Math.min(maxStepIndex, i + 1));
  }

  // ── Controls callbacks ────────────────────────────────────────────────────
  function handleStep() {
    // No-op when nothing is left to process, so the tick doesn't climb on dead state.
    if (!hasActiveWork) return;
    dispatch({ type: "STEP" });
  }

  function handlePlayToggle() {
    // Don't start playing if there's nothing to do.
    if (!isPlaying && !hasActiveWork) return;
    setIsPlaying((p) => !p);
  }

  function handleReset() {
    dispatch({ type: "RESEED", seedCount: STAGE_SEED_COUNT[currentStage] });
    setIsPlaying(false);
    setLoadOn(false);
    setSelectedRequestId(null);
  }

  function handleLoadToggle() {
    setLoadOn((on) => {
      const next = !on;
      // Turning load on also starts the sim so arrivals get served immediately.
      if (next) setIsPlaying(true);
      return next;
    });
  }

  function handleLoadIntensityChange(intensity: number) {
    setLoadIntensity(intensity);
  }

  function handleSpeedChange(ms: number) {
    setSpeed(ms);
  }

  function handleAddRequest(prompt: string, maxDecode: number) {
    dispatch({ type: "ADD_REQUEST", prompt, maxDecode });
  }

  function handleCancelRequest(requestId: number) {
    dispatch({ type: "CANCEL_REQUEST", requestId });
    if (requestId === selectedRequestId) {
      setSelectedRequestId(null);
    }
  }

  // Live scheduler knobs (max batch, token budget): merged into config and read
  // by the scheduler on the next tick — safe to change while running.
  function handleConfigChange(patch: Partial<Config>) {
    dispatch({ type: "SET_CONFIG", config: patch });
  }

  // Engine-setup knobs (block size, KV-cache blocks) define the physical block
  // pool, which is allocated once at init — like vLLM server launch flags. They
  // can't change on a live pool, so applying them re-initializes the engine with
  // a fresh pool. Only allowed while the sim hasn't started (no requests, tick 0).
  function handleEngineSetupChange(patch: Partial<Config>) {
    dispatch({
      type: "SET_STAGE_CONFIG",
      config: { ...appState.config, ...patch },
      seedCount: STAGE_SEED_COUNT[currentStage],
    });
  }

  // ── Active stage component ────────────────────────────────────────────────
  const stageProps = {
    engine: appState.engine,
    config: appState.config,
    selectedRequestId,
    onSelectRequest: setSelectedRequestId,
  };

  let stageContent: React.ReactNode;
  if (currentStage === 1) {
    stageContent = <Stage1Lifecycle {...stageProps} />;
  } else if (currentStage === 2) {
    stageContent = <StageModelConcepts {...stageProps} />;
  } else if (currentStage === 3) {
    stageContent = <Stage2PagedKV {...stageProps} />;
  } else if (currentStage === 4) {
    stageContent = <Stage3Batching {...stageProps} />;
  } else if (currentStage === 5) {
    stageContent = <Stage5Parallelism {...stageProps} />;
  } else if (currentStage === 6) {
    stageContent = <Stage6ModelFeatures {...stageProps} />;
  } else if (currentStage === 7) {
    stageContent = <StageAttention {...stageProps} />;
  } else if (currentStage === 8) {
    stageContent = <StageSpeculative {...stageProps} />;
  } else if (currentStage === 9) {
    stageContent = <Stage4PDDisagg {...stageProps} />;
  } else if (currentStage === 10) {
    stageContent = <Stage7SGLang {...stageProps} />;
  } else {
    stageContent = <Stage1Lifecycle {...stageProps} />; // fallback (shouldn't happen)
  }

  // Requests rejected because they can never be served (e.g. prompt larger than
  // the whole KV cache). Surface the reason so the app never silently hangs.
  const rejected = appState.engine.requests.filter((r) => r.rejectionReason);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={appStyle}>
      {/* ── Header ── */}
      <header style={headerStyle}>
        <div style={headerInnerStyle(isMobile)}>
          <h1 style={titleStyle(isMobile)}>vLLM Inference Visualizer</h1>
        </div>
      </header>

      {/* ── Body: left nav rail + content column ── */}
      <div style={bodyRowStyle(isMobile)}>
        {/* Vertical (desktop) or horizontal scrollable (mobile) stage nav */}
        <nav style={railStyle(isMobile)} aria-label="Stage navigation">
          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((stage) => {
            const active = stage === currentStage;
            return (
              <button
                key={stage}
                style={stageTabStyle(active, isMobile)}
                onClick={() => handleStageSelect(stage)}
                aria-current={active ? "page" : undefined}
              >
                <span style={stageNumStyle(active)}>{stage}</span>
                <span style={stageTitleStyle}>{STAGE_SHORT_TITLE[stage]}</span>
              </button>
            );
          })}
        </nav>

        {/* Content column */}
        <div style={contentColStyle(isMobile)}>

      {/* ── Guide band (full-width, under the title, spans both panes) ── */}
      <div style={guideBandWrapStyle}>
        <div style={guideBandHeaderStyle(isMobile)}>
          <span style={{ fontSize: font.size.lg, fontWeight: font.weight.bold, color: color.textPrimary }}>
            {stageGuides[currentStage]?.title}
          </span>
          <div style={guidePrevNextStyle}>
            <span style={{ fontSize: font.size.md, color: color.textFaint }}>
              Step {stepIndex + 1} of {maxStepIndex + 1}
            </span>
            <button
              style={stepIndex === 0 ? navBtnDisabledStyle : navBtnStyle}
              onClick={handleGuidePrev}
              disabled={stepIndex === 0}
              aria-label="Previous guide step"
            >
              ← Prev
            </button>
            <button
              style={stepIndex >= maxStepIndex ? navBtnDisabledStyle : navBtnStyle}
              onClick={handleGuideNext}
              disabled={stepIndex >= maxStepIndex}
              aria-label="Next guide step"
            >
              Next →
            </button>
          </div>
        </div>
        <GuidePanel stageId={currentStage} stepIndex={stepIndex} />
      </div>

      {/* ── Main area ── */}
      <main style={mainStyle(isMobile)}>
        {/* Visualization */}
        <section style={vizAreaStyle(isMobile)} aria-label={`Stage ${currentStage} visualization`}>
          {rejected.length > 0 && (
            <div role="alert" style={rejectionBannerStyle}>
              <strong>⚠ {rejected.length} request{rejected.length !== 1 ? "s" : ""} stopped early.</strong>{" "}
              {rejected[0].rejectionReason}
            </div>
          )}
          {stageContent}
        </section>

        {/* Sidebar: Controls */}
        <aside style={sidebarStyle(isMobile)}>
          <Controls
            isPlaying={isPlaying}
            speed={speed}
            config={appState.config}
            selectedRequestId={selectedRequestId}
            loadOn={loadOn}
            loadIntensity={loadIntensity}
            requestCount={appState.engine.requests.length}
            caps={STAGE_CAPS[currentStage]}
            onStep={handleStep}
            onPlayToggle={handlePlayToggle}
            onReset={handleReset}
            onSpeedChange={handleSpeedChange}
            onAddRequest={handleAddRequest}
            onCancelRequest={handleCancelRequest}
            onConfigChange={handleConfigChange}
            onEngineSetupChange={handleEngineSetupChange}
            engineSetupLocked={simStarted}
            onLoadToggle={handleLoadToggle}
            onLoadIntensityChange={handleLoadIntensityChange}
          />
        </aside>

        {/* Spacer only needed in the desktop grid layout */}
        {!isMobile && <div style={controlsRowStyle} />}
      </main>
        </div>
      </div>
    </div>
  );
}
