# Advanced Inference Topics — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. Engine changes follow TDD in `tests/engine/`; UI is verified in the browser. Determinism is a hard requirement — all randomness goes through the seeded RNG carried in `EngineState`, never `Math.random()` in engine code.

**Goal:** Extend the existing 7-stage vLLM/SGLang visualizer with the serving topics it currently omits — sampling, sliding-window / long-context attention, chunked prefill, latency/throughput metrics, and speculative decoding — and reorder stages so difficulty ramps monotonically.

**Builds on:** `docs/plans/2026-06-23-vllm-inference-visualizer.md` (the original 7 stages, all complete).

**Architecture (unchanged):** Pure React-free TS engine (`init`/`reduce`/`tick`) as source of truth; thin React views via `useReducer` in `App`; teaching copy in `glossary.ts` + `stageGuides.ts`; `[[term]]` inline markup in guide notes. New per-stage knobs go on the wide-optional `Config`. New stages added via `STAGE_CONFIGS` / `STAGE_CAPS` / `STAGE_SEED_COUNT` / `STAGE_SHORT_TITLE` / `stageGuides` + the render switch.

---

## Design decisions (agreed)

- **Mix of sub-views and new stages**, to keep the nav rail manageable:
  - **Sampling** → sub-view of Lifecycle (decode already names vocab sampling).
  - **Sliding-window / long-context** → sub-view of Paged KV Cache (it's about bounding KV growth — same block mental model).
  - **Chunked prefill** → sub-view of Continuous Batching (it's a scheduler behavior).
  - **Metrics (TTFT / ITL / throughput / goodput)** → a cross-cutting panel on the live stages (metrics are observed, not "visited").
  - **Speculative decoding** → its own new stage (distinct draft→verify→accept mechanism).
- **Stage reorder** so PD disaggregation (an advanced optimization) moves after the fundamentals.

### Final stage order

| # | Stage | New content |
|---|-------|-------------|
| 1 | Single-Request Lifecycle | + **Sampling** sub-view; metrics |
| 2 | Paged KV Cache | + **Sliding-window / long-context** sub-view |
| 3 | Continuous Batching | + **Chunked prefill** sub-view; metrics |
| 4 | Parallelism Strategies | (reviewed) |
| 5 | Model Features: FP8 & MoE | — |
| 6 | PD Disaggregation | (moved from 4) |
| 7 | Speculative Decoding | **new stage** |
| 8 | SGLang vs vLLM | (renumbered from 7) |

---

## Phases (in dependency / risk order)

### Phase A — Structural (no engine change)
- **Reorder stages**: Parallelism 5→4, FP8/MoE 6→5, PD 4→6; insert Spec Decoding as 7; SGLang 7→8. Renumber `STAGE_CONFIGS`, `STAGE_CAPS`, `STAGE_SEED_COUNT`, `STAGE_SHORT_TITLE`, `stageGuides` keys, the render switch in `App.tsx`. Fix every "Stage N" / "Stages 1–6" reference in copy (Stage 7/SGLang especially: it references "Stages 1–6").
- **Review Parallelism stage** for accuracy/clarity; align framing with the corrected Stage 7 (EP/DP-attention are not engine-exclusive, etc.).

### Phase B — Low-risk content
- **Sampling sub-view (Lifecycle):** illustrative next-token distribution that reshapes with temperature / top-k / top-p. UI-only (may use a deterministic illustrative distribution; no engine change). Glossary: `temperature`, `topK`, `topP`, `greedy`.
- **Metrics panel:** record per-request first-token tick in the engine (tiny, deterministic) to derive **TTFT**; derive **ITL** (ticks between decoded tokens), **throughput** (tokens/tick across batch), and **goodput**. Cross-cutting panel on Stages 1 & 3. Glossary: `ttft`, `itl`, `throughput`, `goodput`.

### Phase C — Engine mechanics (TDD)
- **Sliding-window attention:** optional `windowSize` on `Config`/`Request`; in `tick`, KV beyond the window is evictable so cache stops growing unbounded. Visualize as older blocks freeing as the window slides. The 1M-context payoff: window caps KV regardless of length. TDD in `tests/engine/`.
- **Chunked prefill:** a long prompt's prefill spans multiple ticks in `tokenBudget`-sized chunks, interleaved with ongoing decodes (vs. the current one-tick prefill). Scheduler/tick change; preserve determinism and existing tests. Glossary: `chunkedPrefill`.

### Phase D — Advanced new stage
- **Speculative decoding (new Stage 7):** a draft model proposes k tokens, the target verifies them in one pass, accepts the longest correct prefix. Uses the **seeded RNG** (currently carried but unused) for accept/reject. Visualize draft → verify → accept/reject + the speedup. Likely a presentational stage driven by config + RNG, mirroring Stages 5/6. Glossary: `speculativeDecoding`, `draftModel`, `verification`, `acceptanceRate`.

---

## Cross-cutting notes

- **RNG activation:** Phases B (sampling) and D (spec decoding) finally exercise the seeded RNG in `EngineState`. Any engine randomness must draw from it (not `Math.random`) so replays stay deterministic. UI-only illustrative randomness may use the `samplers.ts` pattern.
- **Glossary 90-char rule:** every new entry's `short` ≤ 90 chars (enforced by `glossary.test.ts`); every `[[term]]` key must exist (also tested).
- **Per-phase done = green:** `npx tsc --noEmit`, `npm test`, `npm run build` all clean, plus browser verification, then commit. Phases B–D each get sign-off before starting.
- **Verification criteria** (acceptance):
  - [ ] Stages reordered; no stale "Stage N" copy; all guides resolve.
  - [ ] Sampling sub-view reshapes the distribution with temp/top-k/top-p.
  - [ ] Metrics panel shows TTFT/ITL/throughput on a live run.
  - [ ] Sliding-window caps KV growth on long sequences (engine test).
  - [ ] Chunked prefill interleaves prefill chunks with decode (engine test).
  - [ ] Speculative decoding stage shows draft/verify/accept and a speedup.

---

## Status

- [x] Phase A — reorder (PD → later) + parallelism review
- [x] Phase B — sampling (extracted to a Model Concepts stage) + metrics panel
- [x] Phase C — sliding-window (engine + Attention stage) + chunked-prefill concept
- [x] Phase D — speculative decoding stage
- [x] Formal review pass — engine/quality/accuracy review across all 10 stages;
      fixed: engine block leak on decode-preemption failure, windowed-rejection cap,
      B100→B200, Mistral→Gemma2 SWA, illustrative-KV caveat, spec-decode formula
      bound, stage-number drift (aria-labels/comment/guide-key order), theme/DRY
      nits (shared requestColors, kv-pressure tokens, rejectionBanner/palette.yellow).
