# vLLM Inference Visualizer — Implementation Progress

**Plan:** `docs/plans/2026-06-23-vllm-inference-visualizer.md`
**Branch:** `main`
**Jira:** N/A

---

## Chunk 1: Scaffold + Engine Core Types & RNG

- [x] Task 1: Project scaffold
- [x] Task 2: Core engine types + seeded RNG

## Chunk 2: Engine Units — Tokenizer, KV Cache, Scheduler

- [x] Task 3: Tokenizer
- [x] Task 4: Paged KV cache
- [x] Task 5: Scheduler (admission + preemption)

## Chunk 3: Engine Orchestration

- [x] Task 6: init / reduce / tick (engine review passed: spec + quality)

## Chunk 4: Content & Shared Components

- [x] Task 7: Glossary + stage guides (data)
- [x] Task 8: Shared components

## Chunk 5: Stages 1–3 + App wiring

- [x] Task 9: App shell + reducer
- [x] Task 10: Stage 1 — lifecycle
- [x] Task 11: Stage 2 — paged KV cache
- [x] Task 12: Stage 3 — continuous batching + preemption
- [x] Task 13: Increment 1 verification (tests 118/118, build clean, user-verified in browser)

## Increment 2: Stages 4–6 (complete)

- [x] Task 14: Wire Stages 4–6 into App shell (nav, configs, guides, glossary)
- [x] Task 15: Stage 4 — PD disaggregation (prefill→decode worker handoff)
- [x] Task 16: Stage 5 — parallelism layouts (TP/PP/EP/DP, config-driven diagram)
- [x] Task 17: Stage 6 — model features (FP8 memory bars + MoE expert routing)
- [x] Task 18: Increment 2 verification (121 tests green, tsc clean, build clean)

## Increment 3: Stage 7 SGLang (complete)

- [x] Stage 7 glossary + guide content (prefixCache, radixAttention, lruEviction, dpAttention, programmableFrontend, overlappedScheduling)
- [x] Radix-tree prefix-cache builder + unit tests (src/content/radixTree.ts, 8 tests)
- [x] Stage7SGLang component — interactive prefix-cache comparison (vLLM flat vs SGLang radix tree) + MoE serving comparison (TP-replicated KV vs DP-attention + EP)
- [x] Wired into App (nav 1–7, STAGE_CONFIGS[7], render switch); 130 tests green, tsc + build clean, user-verified in browser

---

## Session Log

| Date | Tasks Completed | Notes |
|------|-----------------|-------|
| 2026-06-23 | Spec + plan written, reviewed, approved | Increment 1 fully detailed; 2–3 outlined |
| 2026-06-23 | Tasks 1–13 (Increment 1 complete) | Pure engine (111 engine tests) + Stages 1–3 UI. Engine passed spec+quality review. Added: random-sentence, clickable glossary, layout polish. 118 tests green, build clean, user-verified in browser. |
| 2026-06-23 | Tasks 14–18 (Increment 2 complete) | Stages 4–6 created. Worker assignment for Stage 4 derived from phase in component (no engine changes). 13 new glossary keys added (pdDisaggregation, prefillWorker, decodeWorker, kvTransfer, tensorParallel, pipelineParallel, expertParallel, dataParallel, gpu, fp8, quantization, moe, expertRouting, activeParams). No new Config fields needed (numGPUs, parallelism, quant, moe already existed in types.ts). 121 tests green, tsc clean, build clean. |
| 2026-06-23 | Increment 3 (Stage 7 SGLang complete) | Built as config-driven comparison diagrams (consistent with Stages 5/6), no engine changes. Interactive radix-tree prefix-cache builder (src/content/radixTree.ts, deterministic, 8 unit tests) contrasts vLLM flat-copy vs SGLang dedup; MoE serving panel contrasts TP-replicated KV vs DP-attention + EP. 6 new glossary keys. 130 tests green, tsc + build clean, browser-verified (toggling shared-prefix requests updates savings live). Note: "GLM 5.2" in Stage 2 preset is real (GLM-5.2 753B); earlier "GLM-5 doesn't exist" claim was a corrected error. |
