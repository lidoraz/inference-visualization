# vLLM Inference Visualizer — Design

**Date:** 2026-06-23
**Status:** Draft for review

## Goal

An interactive, browser-based teaching demo that visualizes how vLLM serves LLM
inference — starting from a single request's prefill/decode lifecycle and
progressively layering on paged KV cache, continuous batching, preemption,
PD disaggregation, parallelism, and model features (FP8, MoE). A final stage
contrasts SGLang's architecture. The demo should *feel* like a live system
reacting to load, not a static diagram.

This is a prototype, not a framework. Scope is deliberately small: a toy
simulation with faithful mechanics, illustrative numbers, and clear visuals.

## Non-Goals

- No real model inference, GPU code, or ML compute.
- No real BPE tokenizer (naive word/subword split is sufficient; see Tokenization).
- No backend/server. Static SPA only.
- No router/multi-page app.

## Tech Stack

- **React + Vite**, TypeScript, single-page app, minimal dependencies.
- Deploys as a static build to a host (Vercel/Netlify/GitHub Pages) later.
- Rationale: the demo juggles substantial interactive state (token sequences
  stepping through prefill/decode, batched requests, KV blocks allocating and
  freeing, parallelism layouts). React's state model avoids hand-rolling a
  re-render loop in vanilla JS. Vite keeps the build trivial.

## Architecture

A pure, deterministic **toy simulation engine** (plain TS, no React) is the
single source of truth. React components are thin views that render engine
state and dispatch actions.

```
src/
  engine/                 # pure simulation, no React
    tokenizer.ts          # naive word/subword split
    types.ts              # Request, Block, BatchStep, EngineState, Config
    engine.ts             # init(config), tick(state, config) -> state
    kvcache.ts            # paged block allocation, block tables, free/evict
    scheduler.ts          # admission, batching, preemption/swapping
  content/
    glossary.ts           # keyed term -> { term, short, long }
    stageGuides.ts        # per-stage ordered narration referencing glossary keys
  components/             # shared visual primitives
    TokenStrip.tsx
    BlockGrid.tsx
    QueueLanes.tsx
    GuidePanel.tsx
    Term.tsx              # inline hoverable glossary term
    Controls.tsx          # step/play/reset/speed + load controls
  stages/                 # one component per learning stage
    Stage1Lifecycle.tsx
    Stage2PagedKV.tsx
    Stage3Batching.tsx
    Stage4PDDisagg.tsx
    Stage5Parallelism.tsx
    Stage6ModelFeatures.tsx
    Stage7SGLang.tsx
  App.tsx                 # stage nav + shared controls + reducer holding EngineState
```

The engine is **stage-aware via config flags**, not a different engine per
stage. Each stage mounts with a config preset and calls `init(config)`.

## Data Flow

```
User input (sentence, config sliders, Step/Play, Add/Cancel request)
        |
        v
  EngineState  --tick(state, config)-->  new EngineState   (pure, deterministic)
        |                                      |
   (useReducer at App level holds current EngineState)
        v                                      v
  Stage component reads state -> renders TokenStrip / BlockGrid / QueueLanes / GuidePanel
```

- **Single source of truth:** one `EngineState` in a reducer at `App` level.
- **`tick(state, config)` is pure:** same input always yields same output. The
  *Play* button is a `setInterval` calling `tick`. The engine is unit-testable
  with zero React.
- **Determinism with arrivals:** request arrivals (Stage 3+) are script-driven
  by default. If any arrival/jitter uses randomness, the RNG seed lives in
  `EngineState` (seeded RNG carried in state) so `tick` stays deterministic.
- **Config is per-stage:** presets for `maxBatchSize`, `blockSize`, `numGPUs`,
  `parallelism`, `quant`, `moe`, `kvCacheBlocks`, etc. Switching stages resets
  via `init(config)`. `Config` is a single wide type with optional
  stage-specific fields (not a discriminated union) — simpler for a prototype;
  the engine reads only the fields relevant to the active stage.
- **Views are pure functions of state:** components never mutate; they dispatch
  actions (`STEP`, `PLAY`, `RESET`, `SET_CONFIG`, `ADD_REQUEST`,
  `CANCEL_REQUEST`, `SET_ARRIVAL_RATE`).
- **Stage 7** runs *two* engine instances side by side (vLLM-config and
  SGLang-config) fed the same request stream for direct comparison.

## Tokenization

Naive whitespace + simple subword chunking. Not a real tokenizer, but visually
shows "one word becomes multiple tokens." Character-level was rejected: it
shrinks the conceptual output vocab to ~100 vs. a real model's ~141k, which
misrepresents the output projection / sampling step. Decode steps are labeled
"sampling from ~141k vocab" even though the toy sim picks the next token by
script.

## Simulation Fidelity

Toy simulation with faithful *mechanics* (not ML): real block sizes, cache
capacity limits, scheduler admission by queue state, batch token budgets,
preemption when the cache fills. Deterministic from config. Numbers are
plausible and hand-tuned for clarity.

## Stages

The demo is a guided progression from beginner entry point to advanced topics.
Each stage builds on the previous concept.

### Stage 1 — Single-request lifecycle
Type a sentence -> split into tokens. *Step*: prefill processes all prompt
tokens in one pass (highlighted together), then each step decodes one new token
appended to the sequence. A side panel shows the KV cache growing one slot per
token. Labels call out sampling from the large vocab. `maxBatchSize: 1`.

### Stage 2 — Paged KV cache
Same request, but the KV cache is shown as fixed-size **physical blocks** (e.g.
4 tokens/block) plus a **block table** mapping logical -> physical. Blocks
allocate as the sequence grows and free on completion. Introduces internal
fragmentation and why paging beats contiguous allocation.

### Stage 3 — Continuous batching
Multiple requests arrive over time into *waiting -> running* queues. Each tick
the scheduler admits requests up to a token budget; prefill and decode of
different requests interleave in one batch. A finished request frees its blocks
so a waiting one can start mid-flight.

### Stage 4 — PD disaggregation
Split into a *prefill* worker and a *decode* worker; show KV cache handed off
between them and why separating the two phases improves throughput.

### Stage 5 — Parallelism
Toggle TP / PP / EP / DP and show how model weights / layers / experts /
requests are split across simulated GPUs.

### Stage 6 — Model features
- **FP8:** memory footprint of weights + KV cache shrinking vs. higher precision.
- **MoE:** router picking top-k experts per token.

### Stage 7 — SGLang differences
Side-by-side comparison view. Two concrete, animatable comparisons:

1. **Shared-prefix caching.** vLLM uses a flat pool of blocks with a hash-based
   prefix cache; SGLang structures the KV cache as a **radix tree** of token
   sequences, so requests sharing a prefix (system prompt, few-shot examples,
   multi-turn history) share KV blocks with LRU eviction on the tree. Visual:
   two requests with a shared system prompt — vLLM stores/hash-matches it,
   SGLang shows one shared tree path branching at the first differing token.
2. **MoE serving.** Under tensor parallelism the (small, MLA-style) KV cache is
   replicated across every TP rank — wasteful. SGLang uses **data-parallel
   attention** (each rank holds distinct requests' KV) and reserves expert
   parallelism for the MoE FFN layers, plus optimized all-to-all
   (DeepEP-style) and compute/communication overlap. Visual: TP-replicated KV
   vs. DP-attention + EP.

Sidebar notes cover SGLang's programmable frontend and cache-aware,
overlapped scheduling.

**Punchline:** vLLM and SGLang share the fundamentals (Stages 1–4), but
SGLang's radix-tree cache + DP-attention/EP combo is what makes it strong for
large MoE serving (e.g. DeepSeek-class models).

## Interactive Controls (the "live system" feel)

vLLM is a system reacting to load, so controls let the user create and disturb
load, not just press Step.

**Request-level:**
- **Add request** on demand — inject a prompt into the waiting queue mid-run.
- **Arrival rate / burst** — flood the scheduler and watch the waiting queue
  back up.
- **Cancel/abort a request** — click a running request to kill it; its KV
  blocks free immediately and a waiting request gets admitted.
- **Per-request prompt length & max decode length** — requests finish at
  different times (realistic mixed workload).

**System-level:**
- **KV cache size** (total blocks) and **block size** — shrink the cache to
  trigger preemption/swapping under pressure.
- **Max batch size / token budget** per step — scheduler admission limit.
- **Tick speed / play-pause-step** — global clock.

**Emergent behaviors unlocked:** waiting-queue growth under load,
**preemption/swapping** (when the KV cache fills, vLLM evicts a running request
back to the waiting queue), block reclamation on completion/cancel admitting
new requests, and throughput shifting as batch size is tuned. These are
additional engine actions and config fields — no architectural change.

Preemption/swapping is **in scope** for the prototype.

## Explanations / Guide System

Teaching copy lives in data, not component code.

- **`content/glossary.ts`** — keyed entries: `{ term, short, long }`. Examples:
  `prefill`, `decode`, `kvCache`, `block`, `blockTable`, `preemption`,
  `continuousBatching`, `pdDisaggregation`, `tensorParallel`, `expertParallel`,
  `pipelineParallel`, `dataParallel`, `fp8`, `moe`, `radixAttention`,
  `dpAttention`.
- **`content/stageGuides.ts`** — per-stage ordered narration; each step
  references a glossary key plus an optional step-specific note. The guide
  panel for a stage is data, not hardcoded JSX.
- **`<GuidePanel>`** renders the current stage/step narration.
- **`<Term>`** renders an inline hoverable term showing its `short` text;
  reused across stages (e.g. "KV cache" appears in Stages 1, 2, 3...).

Benefits: one place to write/edit/proofread all explanations, term reuse, and
the guide stays in sync with the current sim step.

## Testing

- Unit-test the engine in isolation: `tokenizer`, `kvcache` (allocate/free/
  evict), `scheduler` (admission, preemption), and `tick` determinism (same
  state+config -> same next state).
- No heavy UI test harness for a prototype; manual browser verification of each
  stage.

## Open Questions / Deferred

- Exact visual styling and layout polish (per-stage).
- Stage 7 detailed interaction design (revisited after Stages 1–6 land).
- Whether arrival rate is a slider vs. discrete burst button (decide during
  implementation).
