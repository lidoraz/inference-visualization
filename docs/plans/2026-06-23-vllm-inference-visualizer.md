# vLLM Inference Visualizer Implementation Plan

> **For agentic workers:** REQUIRED: Use forge:subagent-driven-development (if subagents available) or forge:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive React+Vite teaching demo that visualizes vLLM serving mechanics, starting with a pure deterministic simulation engine and Stages 1–3 (single-request lifecycle, paged KV cache, continuous batching).

**Spec:** `docs/specs/2026-06-23-vllm-inference-visualizer-design.md`

**Architecture:** A pure, React-free TypeScript simulation engine (`init(config)`, `tick(state, config) -> state`) is the single source of truth. React components are thin views rendering engine state via a `useReducer` at `App`. Teaching copy lives in data files (glossary + stage guides). Implementation proceeds in three increments; this plan fully details Increment 1 and outlines 2–3.

**Tech Stack:** React 18, Vite, TypeScript, Vitest (engine unit tests). Minimal dependencies, no router, no backend.

**Verification Criteria:**
- [ ] `npm run dev` serves the app; `npm run build` produces a static bundle.
- [ ] Engine unit tests pass (`npm test`): tokenizer, kvcache alloc/free/evict, scheduler admission + preemption, `tick` determinism.
- [ ] Stage 1: typing a sentence tokenizes it; Step runs prefill then decodes one token/step; KV cache grows one slot/token.
- [ ] Stage 2: KV cache renders as fixed-size physical blocks + a logical→physical block table; blocks allocate as sequence grows and free on completion.
- [ ] Stage 3: multiple requests flow through waiting→running queues; scheduler admits by token budget; preemption evicts a running request when the cache fills; finishing/cancelling frees blocks and admits a waiting request.
- [ ] Controls work: Step/Play/Reset/speed, Add request, Cancel request, and the system sliders (KV cache blocks, block size, max batch size).
- [ ] Guide panel shows per-step narration from data; inline terms show glossary tooltips.

---

## File Structure

```
package.json, vite.config.ts, tsconfig.json, index.html   # scaffold
src/main.tsx                  # React entry
src/App.tsx                   # stage nav + reducer holding EngineState + shared controls
src/engine/
  types.ts                    # Request, Block, EngineState, Config, Action, RNG
  rng.ts                      # seeded deterministic RNG carried in state
  tokenizer.ts                # naive word/subword split
  kvcache.ts                  # paged block allocation / block table / free / evict
  scheduler.ts                # admission, batching, preemption/swapping
  engine.ts                   # init(config), reduce(state, action, config), tick(state, config)
src/content/
  glossary.ts                 # keyed term -> { term, short, long }
  stageGuides.ts              # per-stage ordered narration referencing glossary keys
src/components/
  Controls.tsx                # step/play/reset/speed + add/cancel + system sliders
  TokenStrip.tsx              # token sequence with prefill/decode highlighting
  BlockGrid.tsx               # physical KV blocks + block table
  QueueLanes.tsx              # waiting/running/(swapped) request lanes
  GuidePanel.tsx              # renders current stage/step narration
  Term.tsx                    # inline hoverable glossary term
src/stages/
  Stage1Lifecycle.tsx
  Stage2PagedKV.tsx
  Stage3Batching.tsx
tests/engine/                 # vitest specs mirroring src/engine
```

Files that change together live together; the engine is split by responsibility (tokenizer / kvcache / scheduler / orchestration) so each unit is independently testable and small enough to hold in context.

---

## Chunk 1: Scaffold + Engine Core Types & RNG

### Task 1: Project scaffold

**Files:**
- Create: `package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `src/main.tsx`, `src/App.tsx`, `vitest.config.ts`

- [ ] **Step 1: Scaffold Vite React-TS project**

Run: `npm create vite@latest . -- --template react-ts` (in the existing dir; keep `docs/`).
Then `npm install` and add Vitest: `npm install -D vitest`.

- [ ] **Step 2: Add test script**

In `package.json` scripts add: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 3: Minimal App placeholder**

`src/App.tsx` renders `<h1>vLLM Inference Visualizer</h1>` so dev server boots.

- [ ] **Step 4: Verify dev + build**

Run: `npm run dev` (expect server URL), then `npm run build` (expect `dist/` produced).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "chore: scaffold Vite React-TS app with vitest"
```

### Task 2: Core engine types + seeded RNG

**Files:**
- Create: `src/engine/types.ts`, `src/engine/rng.ts`, `tests/engine/rng.test.ts`

- [ ] **Step 1: Write failing RNG determinism test**

```ts
import { describe, it, expect } from "vitest";
import { makeRng } from "../../src/engine/rng";

describe("rng", () => {
  it("is deterministic for a given seed", () => {
    const a = makeRng(42); const b = makeRng(42);
    expect([a.next(), a.next(), a.next()]).toEqual([b.next(), b.next(), b.next()]);
  });
  it("advances state (carried, not global)", () => {
    const r = makeRng(1); const first = r.next();
    expect(r.next()).not.toBe(first); // extremely likely
  });
});
```

- [ ] **Step 2: Run test, expect FAIL** — Run: `npm test -- rng` (Expected: cannot find module).

- [ ] **Step 3: Implement `rng.ts`** — a small mulberry32-style PRNG returning `{ seed, next(): number }`, with seed stored so it can be carried in `EngineState`.

- [ ] **Step 4: Run test, expect PASS** — Run: `npm test -- rng`.

- [ ] **Step 5: Define `types.ts`** — `Phase = "prefill" | "decode"`; `Request` (id, prompt tokens, decoded tokens, maxDecode, status: waiting|running|swapped|finished|cancelled, blockTable); `Block` (id, requestId|null, tokenSlots); `Config` (single wide optional-field type: maxBatchSize, blockSize, kvCacheBlocks, tokenBudget, numGPUs?, parallelism?, quant?, moe?); `EngineState` (tick, requests, blocks, rng, config-derived counters); `Action` union (STEP, RESET, SET_CONFIG, ADD_REQUEST, CANCEL_REQUEST, SET_ARRIVAL_RATE). Two intentional divergences from the spec's type list: (a) the spec's `PLAY` action is **not** an engine action — Play is a UI concern handled by `setInterval` dispatching `STEP` (Task 9); (b) the spec's `BatchStep` type is **deferred** (not needed for Stages 1–3; revisit if a per-tick batch record is needed for visualization later).

- [ ] **Step 6: Commit**

```bash
git add src/engine/types.ts src/engine/rng.ts tests/engine/rng.test.ts
git commit -m "feat(engine): core types and seeded deterministic RNG"
```

---

## Chunk 2: Engine Units — Tokenizer, KV Cache, Scheduler

### Task 3: Tokenizer

**Files:** Create `src/engine/tokenizer.ts`, `tests/engine/tokenizer.test.ts`

- [ ] **Step 1: Failing test** — `tokenize("the quicker")` returns tokens where a long word splits into ≥2 subword tokens and whitespace is not a token; result is deterministic.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** naive split: split on whitespace, then chunk words longer than N chars into subword pieces; return `{ id, text }[]` with sequential ids.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(engine): naive word/subword tokenizer`.

### Task 4: Paged KV cache

**Files:** Create `src/engine/kvcache.ts`, `tests/engine/kvcache.test.ts`

- [ ] **Step 1: Failing tests** — allocate blocks for a sequence (blocks = ceil(tokens/blockSize)); block table maps logical→physical; freeing a request returns its blocks to the free pool; allocation fails (returns null/oom flag) when no free blocks remain.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** pure functions: `allocate(state, requestId, numTokens) -> {state, ok}`, `free(state, requestId) -> state`, `evict(state, requestId)` (free + mark swapped), `usage(state)`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(engine): paged KV cache with block tables`.

### Task 5: Scheduler (admission + preemption)

**Files:** Create `src/engine/scheduler.ts`, `tests/engine/scheduler.test.ts`

- [ ] **Step 1: Failing tests** — admits waiting requests up to `maxBatchSize` and `tokenBudget`; when cache is full and a new prefill needs blocks, preempts the lowest-priority running request (evict → swapped → back to waiting); freed blocks let a waiting request get admitted next tick. Include a **live capacity-shrink test**: after requests are running, apply a `SET_CONFIG` that lowers `kvCacheBlocks` below current usage and assert the next `schedule` preempts enough running requests to fit, without corrupting in-flight block tables. (De-risks the Stage 3 browser check.)
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `schedule(state, config) -> state` operating on queues by status; preemption uses kvcache.evict.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(engine): scheduler with admission and preemption`.

---

## Chunk 3: Engine Orchestration

### Task 6: init / reduce / tick

**Files:** Create `src/engine/engine.ts`, `tests/engine/engine.test.ts`

- [ ] **Step 1: Failing tests** —
  - `init(config)` returns empty queues, full free-block pool, seeded rng.
  - `reduce(state, {type:"ADD_REQUEST", prompt})` tokenizes and enqueues a waiting request.
  - `tick`: a single running request does prefill on the first tick (all prompt tokens, blocks allocated together), then decodes exactly one token per subsequent tick until `maxDecode`, then becomes finished and frees blocks.
  - **Determinism:** `tick(s, c)` twice from a cloned state yields deep-equal results.
  - `CANCEL_REQUEST` frees blocks immediately.
- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `engine.ts` composing tokenizer + kvcache + scheduler; `tick` = schedule → advance each running request one phase-step → free finished. RNG carried in state.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Commit** `feat(engine): init/reduce/tick orchestration`.

---

## Chunk 4: Content & Shared Components

### Task 7: Glossary + stage guides (data)

**Files:** Create `src/content/glossary.ts`, `src/content/stageGuides.ts`

- [ ] **Step 1:** Define glossary keyed entries for at least: prefill, decode, kvCache, block, blockTable, preemption, continuousBatching, tokenBudget, vocabSampling. Each `{ term, short, long }`.
- [ ] **Step 2:** Define `stageGuides` for stages 1–3: ordered steps, each `{ glossaryKey, note? }`.
- [ ] **Step 3: Commit** `feat(content): glossary and stage 1-3 guides`.

### Task 8: Shared components

**Files:** Create `Term.tsx`, `GuidePanel.tsx`, `Controls.tsx`, `TokenStrip.tsx`, `BlockGrid.tsx`, `QueueLanes.tsx` under `src/components/`.

- [ ] **Step 1:** `Term` — inline span showing glossary `short` on hover (title/tooltip).
- [ ] **Step 2:** `GuidePanel` — given stageId + current step index, render the guide entry's text and linked term.
- [ ] **Step 3:** `Controls` — Step/Play(interval)/Reset/speed; Add request (prompt input + length/maxDecode); Cancel (acts on selected request); sliders for kvCacheBlocks, blockSize, maxBatchSize → dispatch SET_CONFIG.
- [ ] **Step 4:** `TokenStrip` — render tokens; highlight prefill group vs. per-step decoded token.
- [ ] **Step 5:** `BlockGrid` — render physical blocks (filled/free, color by request) + block table.
- [ ] **Step 6:** `QueueLanes` — render waiting/running/swapped lanes of request chips; clicking a chip selects it (for cancel).
- [ ] **Step 7: Commit** `feat(components): shared visual primitives and controls`.

---

## Chunk 5: Stages 1–3 + App wiring

### Task 9: App shell + reducer

**Files:** Modify `src/App.tsx`

- [ ] **Step 1:** Hold `EngineState` in `useReducer` (delegates to engine `reduce`/`tick`); stage nav (tabs 1–3); render shared `Controls` + active stage; pass selected-request state for cancel.
- [ ] **Step 2:** Play loop via `setInterval` dispatching STEP at the speed setting; clear on pause/unmount.
- [ ] **Step 3: Commit** `feat(app): reducer-backed shell with stage nav and play loop`.

### Task 10: Stage 1 — lifecycle

**Files:** Create `src/stages/Stage1Lifecycle.tsx`

- [ ] **Step 1:** Mounts with config `{ maxBatchSize: 1, blockSize, kvCacheBlocks }`. Renders `TokenStrip` + a simple KV slot view + `GuidePanel`. Shows prefill-then-decode stepping; "sampling from ~141k vocab" label on decode.
- [ ] **Step 2:** Manual browser check: type sentence, Step through prefill + several decodes.
- [ ] **Step 3: Commit** `feat(stage1): single-request prefill/decode lifecycle`.

### Task 11: Stage 2 — paged KV cache

**Files:** Create `src/stages/Stage2PagedKV.tsx`

- [ ] **Step 1:** Reuses engine; renders `BlockGrid` (blocks + block table) prominently; shows allocation as sequence grows and free on completion; note internal fragmentation.
- [ ] **Step 2:** Browser check: blocks fill in `blockSize` chunks; block table updates; finishing frees blocks.
- [ ] **Step 3: Commit** `feat(stage2): paged KV cache with block table view`.

### Task 12: Stage 3 — continuous batching + preemption

**Files:** Create `src/stages/Stage3Batching.tsx`

- [ ] **Step 1:** Config raises `maxBatchSize` and a `tokenBudget`; renders `QueueLanes` + `BlockGrid`. Add multiple requests; watch interleaved prefill/decode, queue backup under load, preemption when KV cache shrinks, and block reclamation on finish/cancel admitting waiting requests.
- [ ] **Step 2:** Browser check: flood with Add requests; shrink kvCacheBlocks slider to trigger preemption; cancel a running request and see a waiting one admitted.
- [ ] **Step 3: Commit** `feat(stage3): continuous batching with preemption and cancel`.

### Task 13: Increment 1 verification

- [ ] **Step 1:** Run `npm test` (all engine tests pass) and `npm run build` (clean build).
- [ ] **Step 2:** Walk the Verification Criteria checklist above in the browser; check each box.
- [ ] **Step 3: Commit** any fixes; tag increment-1 complete in the progress file.

---

## Increment 2: Stages 4–6 (detailed)

**Design basis (decided):** *Reuse the live sim where natural, diagram the rest.*
Stage 4 reuses the real request lifecycle (a request prefills on one worker, then
its KV is handed to a decode worker — driven by the existing `tick` loop). Stages
5–6 are **config-driven diagrams**: parallelism layouts and FP8/MoE memory are
topology/memory concepts, not per-token behaviors, so they read engine state +
config and render computed diagrams rather than extending the engine's tick logic.
This keeps the engine simple and matches how Stages 1–3 were built.

**Guardrails for the implementing agent:**
- Do NOT change `tick`/`scheduler`/`kvcache` semantics for Stages 5–6. Those stages
  are presentational: read `engine` + `config`, compute a layout, render it. Any new
  per-stage knobs are added to `Config` as optional fields (the wide-optional-type
  convention already in `types.ts`) and consumed only by the stage component.
- Stage 4 MAY add a small amount of engine state (a per-request `worker: "prefill" | "decode"`
  tag and a transition on the prefill→decode boundary). If so, follow TDD in
  `tests/engine/` and keep it optional/back-compatible so Stages 1–3 are unaffected.
- Determinism rule still holds: no `Math.random()` in engine code; UI-only randomness
  (like Stage 5 request-to-GPU assignment for display) may use Math.random in the
  component, mirroring `sampleSentences.ts`.
- After each stage: `npx tsc --noEmit` clean, `npm run build` clean, `npm test` green
  (118 existing tests must keep passing), then commit. Verify visually in the browser.

### Task 14: Wire Stages 4–6 into the App shell

**Files:** Modify `src/App.tsx`; add `src/content/stageGuides.ts` entries (keys 4,5,6);
add `src/content/glossary.ts` entries (new terms listed per stage below).

- [ ] **Step 1:** Extend `STAGE_CONFIGS` (App.tsx:30) with presets:
  - `4: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32 }`
  - `5: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32, numGPUs: 4, parallelism: "tp" }`
  - `6: { maxBatchSize: 4, blockSize: 4, kvCacheBlocks: 16, tokenBudget: 32, quant: "fp16", moe: false }`
- [ ] **Step 2:** Change the stage-nav map `[1, 2, 3]` (App.tsx:314) to `[1, 2, 3, 4, 5, 6]`.
- [ ] **Step 3:** Extend the render switch (App.tsx:298-304) with `currentStage === 4/5/6`
  rendering `Stage4PDDisagg` / `Stage5Parallelism` / `Stage6ModelFeatures`, and import them.
- [ ] **Step 4:** Add `stageGuides` entries 4–6 (ordered narration referencing glossary keys,
  same shape as 1–3). Add glossary entries for new terms (see each task). Extend
  `tests/content/glossary.test.ts` is automatic (it already asserts every stageGuides key
  exists in glossary — just keep them in sync).
- [ ] **Step 5:** `npm test` green, `npm run build` clean. **Commit** `feat(app): register Stages 4-6 in nav and config`.

### Task 15: Stage 4 — PD disaggregation

**Files:** Create `src/stages/Stage4PDDisagg.tsx`. Optionally modify `src/engine/types.ts`
+ `src/engine/engine.ts` + `tests/engine/engine.test.ts` if adding the `worker` tag (TDD).
Add glossary keys: `pdDisaggregation`, `prefillWorker`, `decodeWorker`, `kvTransfer`.

- [ ] **Step 1 (optional engine, TDD):** If modeling the handoff in-engine, add optional
  `worker?: "prefill" | "decode"` to `Request`. Write a failing test: a request starts on
  the prefill worker during its prefill tick, and flips to the decode worker when phase
  transitions prefill→decode. Implement minimally in `tick` (set the tag at the existing
  phase-transition point). Keep it optional so Stages 1–3 (which ignore `worker`) are
  unaffected. Run `npm test -- engine`, confirm green. If you judge the tag unnecessary,
  derive worker purely from `phase` in the component instead and skip engine changes —
  document the choice in the component.
- [ ] **Step 2:** Build the component (StageProps). Render two labeled panels side by side:
  a **Prefill Worker** and a **Decode Worker**. Place each request in the panel matching
  its phase (prefill-phase running requests on the left; decode-phase + finished on the
  right). Reuse `QueueLanes` or simple request chips per panel; reuse `BlockGrid` to show
  each worker holding KV blocks. Draw an arrow / "KV transfer" indicator between panels;
  when a request crosses prefill→decode this tick, highlight the transfer. Use `<Term>` for
  `pdDisaggregation`, `prefillWorker`, `decodeWorker`, `kvTransfer`.
- [ ] **Step 3:** Explanatory copy: why splitting prefill (compute-bound, bursty) from
  decode (memory-bound, steady) lets each be scaled/optimized independently and improves
  throughput. Empty state when no requests.
- [ ] **Step 4:** Browser check: add requests, step, watch a request move from prefill→decode
  panel with the transfer indicator; confirm Stages 1–3 still behave unchanged.
- [ ] **Step 5:** tsc/build/test clean. **Commit** `feat(stage4): PD disaggregation view`.

### Task 16: Stage 5 — parallelism (TP/PP/EP/DP)

**Files:** Create `src/stages/Stage5Parallelism.tsx`. (No engine changes — config-driven.)
Add glossary keys: `tensorParallel`, `pipelineParallel`, `expertParallel`, `dataParallel`, `gpu`.

- [ ] **Step 1:** Component (StageProps) with a local mode toggle for `parallelism` value
  (`"tp" | "pp" | "ep" | "dp"`) — either local UI state or dispatched via `onConfigChange`
  if you thread it; simplest is local state seeded from `config.parallelism`. Read
  `config.numGPUs` (default 4 if undefined) and render a row/grid of N **GPU** cards.
- [ ] **Step 2:** For each mode, render how things split across the GPUs (diagrammatic):
  - **TP (tensor parallel):** each GPU holds a *shard of every layer's weights* (e.g. "Layer L: heads 0–7 on GPU0, 8–15 on GPU1…"); one request's compute spans all GPUs with an all-reduce per layer.
  - **PP (pipeline parallel):** each GPU holds a *contiguous subset of layers* (GPU0: layers 0–7, GPU1: 8–15…); a request flows GPU0→GPU1→… as a pipeline; show microbatches if simple.
  - **EP (expert parallel):** experts of MoE FFN layers are spread across GPUs; tokens route to the GPU holding their expert (ties into Stage 6 MoE).
  - **DP (data parallel):** each GPU holds a *full model replica* and processes a *different subset of requests*; assign the current `engine.requests` round-robin across GPUs for display.
  Keep these as clear labeled diagrams (boxes + captions), not animations.
- [ ] **Step 3:** `<Term>` for each parallelism term + `gpu`. Explanatory copy contrasting
  the four: what gets split (weights-within-layer / layers / experts / requests) and the
  communication cost of each. A one-line "when you'd use this" per mode.
- [ ] **Step 4:** Browser check: toggle all four modes, confirm the diagram updates and reads clearly.
- [ ] **Step 5:** tsc/build/test clean. **Commit** `feat(stage5): parallelism layouts TP/PP/EP/DP`.

### Task 17: Stage 6 — model features (FP8 + MoE)

**Files:** Create `src/stages/Stage6ModelFeatures.tsx`. (No engine changes — config-driven.)
Add glossary keys: `fp8`, `quantization`, `moe`, `expertRouting`, `activeParams`.

- [ ] **Step 1 (FP8 sub-view):** Local toggle for `quant` (`"fp16" | "fp8"`), seeded from
  `config.quant`. Render two memory bars (weights + KV cache) that **halve** when switching
  fp16→fp8 (fp8 = 1 byte/elem vs fp16 = 2 bytes/elem). Use a fixed illustrative model size
  (e.g. "8B params": weights 16 GB @ fp16 → 8 GB @ fp8; show KV cache bytes/token halving
  too, tying back to the Stage 2 block-size memory formula). `<Term>` for `fp8`, `quantization`.
- [ ] **Step 2 (MoE sub-view):** Local toggle for `moe` on/off. When on, render an MoE FFN
  layer: a **router** that, per token, selects **top-k of E experts** (e.g. top-2 of 8).
  Show a small set of tokens (reuse the active request's tokens, or a fixed demo set) each
  drawing arrows to their 2 chosen experts; gray out unselected experts. Surface the key
  insight via `<Term activeParams>`: total params are large but only top-k experts run per
  token, so *active* params (compute) are a fraction of *total* params (memory). Router
  choice can use Math.random in the component (UI-only) or a simple deterministic hash of
  token id — prefer the hash so it's stable across re-renders.
- [ ] **Step 3:** Explanatory copy for both: FP8 trades precision for ~2× memory/throughput;
  MoE scales capacity (more experts = more total params/knowledge) without scaling per-token
  compute. Note these combine (FP8 MoE models like DeepSeek) — forward reference to Stage 7.
- [ ] **Step 4:** Browser check: toggle fp16/fp8 (bars halve), toggle MoE (routing appears),
  confirm top-k selection renders clearly.
- [ ] **Step 5:** tsc/build/test clean. **Commit** `feat(stage6): FP8 memory + MoE expert routing`.

### Task 18: Increment 2 verification

- [ ] **Step 1:** `npm test` (118+ tests pass), `npx tsc --noEmit` clean, `npm run build` clean.
- [ ] **Step 2:** Browser walk: Stages 1–3 unchanged; Stage 4 shows prefill→decode handoff;
  Stage 5 toggles four parallelism layouts; Stage 6 toggles FP8 bars + MoE routing. Confirm
  every `<Term>` (hover + click popover) resolves and the guide Prev/Next works for stages 4–6.
- [ ] **Step 3:** Update the progress file (mark Tasks 14–18, add a session-log row). **Commit.**

**Note on review:** per the subagent-driven workflow, run a spec + code-quality review pass
at the end of this increment (as was done for the engine in Increment 1) before considering
it complete.

## Increment 3 (outline — expand when reached): Stage 7 SGLang

Two side-by-side engine instances (vLLM-config vs SGLang-config) fed the same request stream:
- **Shared-prefix caching:** flat block pool + hash prefix cache vs. radix-tree KV sharing with LRU eviction.
- **MoE serving:** TP-replicated KV vs. DP-attention + EP.
Plus sidebar notes on SGLang's programmable frontend and cache-aware overlapped scheduling.

---

## Notes for implementers

- TDD throughout the engine (`tests/engine/`); components are verified manually in the browser per the spec's testing section.
- Keep the engine free of React imports — it must run under Vitest standalone.
- Determinism is a hard requirement: all randomness goes through the seeded RNG carried in `EngineState`; never call `Math.random()` in engine code.
- DRY: stages reuse the same engine and the same shared components; differences are config presets + which components are emphasized.
