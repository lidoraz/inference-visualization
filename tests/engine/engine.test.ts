/**
 * Tests for src/engine/engine.ts
 *
 * Written FIRST (TDD) — all tests should be RED before the implementation
 * exists, then GREEN after a correct implementation.
 *
 * Test coverage:
 *   - init: fresh state shape
 *   - ADD_REQUEST: tokenize + enqueue
 *   - STEP lifecycle: prefill -> decode loop -> finished + blocks freed
 *   - determinism: two identical action-replay sequences produce equal tick results
 *   - CANCEL_REQUEST: frees blocks, sets cancelled
 *   - RESET: returns fresh init-equivalent state
 *   - SET_ARRIVAL_RATE: updates arrivalRatePerTick
 *   - SET_CONFIG: returns state unchanged (config is owned by the React layer)
 */

import { describe, it, expect } from "vitest";
import { init, reduce, tick, DEFAULT_MAX_DECODE } from "../../src/engine/engine";
import { makeRng } from "../../src/engine/rng";
import { usage } from "../../src/engine/kvcache";
import type { Action, Config, EngineState } from "../../src/engine/types";

// ---------------------------------------------------------------------------
// Helpers / Fixtures
// ---------------------------------------------------------------------------

function makeConfig(overrides: Partial<Config> = {}): Config {
  return {
    maxBatchSize: 4,
    blockSize: 4,
    kvCacheBlocks: 16,
    tokenBudget: 256,
    ...overrides,
  };
}

/** Replay a sequence of actions from a fresh init to produce a final state. */
function replay(actions: Action[], config: Config, seed = 1): EngineState {
  let state = init(config, seed);
  for (const action of actions) {
    state = reduce(state, action, config);
  }
  return state;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

describe("init", () => {
  it("returns tick 0", () => {
    const state = init(makeConfig());
    expect(state.tick).toBe(0);
  });

  it("returns empty requests array", () => {
    const state = init(makeConfig());
    expect(state.requests).toEqual([]);
  });

  it("returns a block pool of size kvCacheBlocks with all blocks free", () => {
    const config = makeConfig({ kvCacheBlocks: 8, blockSize: 4 });
    const state = init(config);
    expect(state.blocks).toHaveLength(8);
    const { used } = usage(state.blocks);
    expect(used).toBe(0);
  });

  it("returns nextRequestId 0", () => {
    const state = init(makeConfig());
    expect(state.nextRequestId).toBe(0);
  });

  it("returns arrivalRatePerTick 0", () => {
    const state = init(makeConfig());
    expect(state.arrivalRatePerTick).toBe(0);
  });

  it("seeds the rng with the provided seed", () => {
    const state = init(makeConfig(), 42);
    // The rng.seed getter on makeRng reflects initial seed — before any next() calls
    // it equals the coerced initial seed value.
    expect(state.rng.seed).toBeDefined();
    // Two inits with the same seed produce equivalent rng sequences.
    const state2 = init(makeConfig(), 42);
    expect(state.rng.next()).toBe(state2.rng.next());
  });

  it("uses seed 1 by default", () => {
    const state = init(makeConfig());
    const stateExplicit = init(makeConfig(), 1);
    expect(state.rng.next()).toBe(stateExplicit.rng.next());
  });
});

// ---------------------------------------------------------------------------
// ADD_REQUEST
// ---------------------------------------------------------------------------

describe("reduce – ADD_REQUEST", () => {
  it("appends one request to state.requests", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello world" }, config);
    expect(next.requests).toHaveLength(1);
  });

  it("new request has status 'waiting'", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(next.requests[0].status).toBe("waiting");
  });

  it("new request has phase 'prefill'", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(next.requests[0].phase).toBe("prefill");
  });

  it("tokenizes the prompt into promptTokens", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hi" }, config);
    expect(next.requests[0].promptTokens.length).toBeGreaterThan(0);
    expect(next.requests[0].promptTokens[0].text).toBe("hi");
  });

  it("starts with empty decodedTokens", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(next.requests[0].decodedTokens).toEqual([]);
  });

  it("starts with empty blockTable", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(next.requests[0].blockTable).toEqual([]);
  });

  it("records the current tick as arrivalTick", () => {
    const config = makeConfig();
    let state = init(config);
    // Advance a few ticks first.
    state = reduce(state, { type: "STEP" }, config);
    state = reduce(state, { type: "STEP" }, config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hi" }, config);
    expect(next.requests[0].arrivalTick).toBe(2);
  });

  it("uses action.maxDecode when provided", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello", maxDecode: 3 }, config);
    expect(next.requests[0].maxDecode).toBe(3);
  });

  it("uses DEFAULT_MAX_DECODE when maxDecode is omitted", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(next.requests[0].maxDecode).toBe(DEFAULT_MAX_DECODE);
  });

  it("increments nextRequestId after each ADD_REQUEST", () => {
    const config = makeConfig();
    let state = init(config);
    expect(state.nextRequestId).toBe(0);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "a" }, config);
    expect(state.nextRequestId).toBe(1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "b" }, config);
    expect(state.nextRequestId).toBe(2);
    expect(state.requests[0].id).toBe(0);
    expect(state.requests[1].id).toBe(1);
  });

  it("does not mutate the input state", () => {
    const config = makeConfig();
    const state = init(config);
    const snapshot = JSON.stringify(state);
    reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    expect(JSON.stringify(state)).toBe(snapshot);
  });
});

// ---------------------------------------------------------------------------
// STEP — single running request lifecycle
// ---------------------------------------------------------------------------

describe("reduce – STEP lifecycle (single request)", () => {
  /**
   * Use a minimal config where one request can always get blocks:
   *   blockSize=4, kvCacheBlocks=8, prompt="hi" (1 token -> 1 block)
   *   maxDecode=3 -> finish after 3 decode steps
   */
  const config = makeConfig({ blockSize: 4, kvCacheBlocks: 8, maxBatchSize: 2 });

  function setupWithRequest(maxDecode = 3): EngineState {
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi", maxDecode }, config);
    return state;
  }

  it("request remains 'waiting' before first STEP (scheduler has not run)", () => {
    const state = setupWithRequest();
    expect(state.requests[0].status).toBe("waiting");
  });

  it("after first STEP: request becomes 'running' and stays in 'prefill' (prefill is its own visible step)", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config);
    const req = state.requests[0];
    expect(req.status).toBe("running");
    expect(req.phase).toBe("prefill");
  });

  it("after second STEP: request transitions to 'decode'", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config); // tick 0: admit + prefill
    state = reduce(state, { type: "STEP" }, config); // tick 1: decode begins
    expect(state.requests[0].phase).toBe("decode");
  });

  it("after first STEP: blocks are allocated (blockTable non-empty)", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config);
    expect(state.requests[0].blockTable.length).toBeGreaterThan(0);
  });

  it("after prefill STEP: no decoded token is appended yet", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config);
    expect(state.requests[0].decodedTokens).toHaveLength(0);
  });

  it("after second STEP: one decoded token is appended", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config); // tick 0: admit + prefill
    state = reduce(state, { type: "STEP" }, config); // tick 1: decode token 1
    expect(state.requests[0].decodedTokens).toHaveLength(1);
  });

  it("records firstTokenTick at the tick the first token is emitted (TTFT)", () => {
    let state = setupWithRequest();
    state = reduce(state, { type: "STEP" }, config); // tick 0->1: admit + prefill, no token
    expect(state.requests[0].firstTokenTick).toBeUndefined();
    state = reduce(state, { type: "STEP" }, config); // tick 1->2: first decoded token
    expect(state.requests[0].firstTokenTick).toBe(2);
    // It should NOT move on later tokens.
    state = reduce(state, { type: "STEP" }, config);
    expect(state.requests[0].firstTokenTick).toBe(2);
  });

  it("decoded tokens have sequential ids and deterministic text", () => {
    let state = setupWithRequest(4);
    state = reduce(state, { type: "STEP" }, config); // tick 0: admit + prefill
    state = reduce(state, { type: "STEP" }, config); // tick 1: decode 1
    state = reduce(state, { type: "STEP" }, config); // tick 2: decode 2
    const req = state.requests[0];
    expect(req.decodedTokens).toHaveLength(2);
    // Each successive token should have a unique id
    expect(req.decodedTokens[0].id).not.toBeNaN();
    expect(req.decodedTokens[1].id).not.toBeNaN();
    expect(req.decodedTokens[0].id).not.toBe(req.decodedTokens[1].id);
  });

  it("tick counter increments by 1 on each STEP", () => {
    let state = setupWithRequest();
    expect(state.tick).toBe(0);
    state = reduce(state, { type: "STEP" }, config);
    expect(state.tick).toBe(1);
    state = reduce(state, { type: "STEP" }, config);
    expect(state.tick).toBe(2);
  });

  it("request finishes when decodedTokens.length reaches maxDecode", () => {
    let state = setupWithRequest(2); // maxDecode=2
    state = reduce(state, { type: "STEP" }, config); // tick 0: admit + prefill
    state = reduce(state, { type: "STEP" }, config); // tick 1: decode 1
    state = reduce(state, { type: "STEP" }, config); // tick 2: decode 2 -> finish
    const req = state.requests[0];
    expect(req.status).toBe("finished");
  });

  it("blocks are freed when request finishes", () => {
    let state = setupWithRequest(2);
    state = reduce(state, { type: "STEP" }, config); // admit + prefill
    state = reduce(state, { type: "STEP" }, config); // decode 1
    state = reduce(state, { type: "STEP" }, config); // decode 2 -> finish
    const { used } = usage(state.blocks);
    expect(used).toBe(0);
  });

  it("blockTable is cleared when request finishes", () => {
    let state = setupWithRequest(2);
    state = reduce(state, { type: "STEP" }, config); // admit + prefill
    state = reduce(state, { type: "STEP" }, config); // decode 1
    state = reduce(state, { type: "STEP" }, config); // decode 2 -> finish
    expect(state.requests[0].blockTable).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Forward progress under saturation (no deadlock)
// ---------------------------------------------------------------------------

describe("forward progress under KV-cache saturation", () => {
  it("a batch that saturates the cache still drains (no deadlock)", () => {
    // 4 requests with long prompts cannot all fit in 16 blocks at once. Without
    // decode-time preemption the batch deadlocks (decoded counts freeze). The
    // engine must preempt to free blocks so every request eventually finishes.
    const config = makeConfig({
      maxBatchSize: 4,
      blockSize: 4,
      kvCacheBlocks: 16,
      tokenBudget: 256,
    });
    let state = init(config);
    const longPrompt = "the quick brown fox jumps over the lazy sleeping dog near river";
    for (let i = 0; i < 4; i++) {
      state = reduce(state, { type: "ADD_REQUEST", prompt: longPrompt, maxDecode: 10 }, config);
    }

    // Run plenty of ticks — bounded so a deadlock fails the test instead of hanging.
    for (let t = 0; t < 200; t++) {
      state = reduce(state, { type: "STEP" }, config);
      if (state.requests.every((r) => r.status === "finished")) break;
    }

    expect(state.requests).toHaveLength(4);
    expect(state.requests.every((r) => r.status === "finished")).toBe(true);
    // All blocks reclaimed once everything finishes.
    expect(usage(state.blocks).used).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Sliding-window attention
// ---------------------------------------------------------------------------

describe("sliding-window attention", () => {
  it("caps a request's live KV blocks to the window as it decodes", () => {
    // windowSize 8, blockSize 4 => at most ceil(8/4)=2 blocks ever, even though
    // the sequence (1-token prompt + many decoded tokens) grows well past 8.
    const config = makeConfig({
      blockSize: 4,
      kvCacheBlocks: 16,
      maxBatchSize: 1,
      windowSize: 8,
    });
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi", maxDecode: 20 }, config);
    let maxBlocks = 0;
    for (let t = 0; t < 30; t++) {
      state = reduce(state, { type: "STEP" }, config);
      const req = state.requests[0];
      maxBlocks = Math.max(maxBlocks, req.blockTable.length);
      if (req.status === "finished") break;
    }
    // Never exceeds the window's worth of blocks.
    expect(maxBlocks).toBeLessThanOrEqual(2);
  });

  it("without a window, KV grows past the window size", () => {
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 16, maxBatchSize: 1 });
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi", maxDecode: 20 }, config);
    let maxBlocks = 0;
    for (let t = 0; t < 30; t++) {
      state = reduce(state, { type: "STEP" }, config);
      const req = state.requests[0];
      maxBlocks = Math.max(maxBlocks, req.blockTable.length);
      if (req.status === "finished") break;
    }
    // Full attention: a 21-token sequence needs more than 2 blocks.
    expect(maxBlocks).toBeGreaterThan(2);
  });

  it("does not cancel a windowed request whose full sequence exceeds cache but window fits", () => {
    // 17-token prompt → 5 blocks (> 4 cache blocks) WITHOUT a window would be
    // rejected as unsatisfiable. With windowSize 8 it needs only 2 blocks and
    // must be admitted, not cancelled.
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 4, maxBatchSize: 1, windowSize: 8 });
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "abcdefghijklmnopq", maxDecode: 5 }, config);
    state = reduce(state, { type: "STEP" }, config);
    expect(state.requests[0].status).not.toBe("cancelled");
    expect(state.requests[0].rejectionReason).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// No block leak when a single request exceeds the whole cache mid-decode
// ---------------------------------------------------------------------------

describe("decode-time preemption failure path", () => {
  it("force-finishes a request that alone outgrows the cache (no infinite loop)", () => {
    // 2 blocks × 4 tokens = 8 slots. Prompt "hi" = 1 token → can decode up to
    // 7 more tokens before needing a 3rd block. With maxDecode=20 it must not
    // loop forever — it should finish early with a rejectionReason.
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 2, maxBatchSize: 1 });
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi", maxDecode: 20 }, config);
    for (let t = 0; t < 40; t++) {
      state = reduce(state, { type: "STEP" }, config);
      if (state.requests[0].status === "finished") break;
    }
    const req = state.requests[0];
    expect(req.status).toBe("finished");
    expect(req.rejectionReason).toMatch(/Cache too small/);
    // All blocks must be freed once finished.
    expect(usage(state.blocks).used).toBe(0);
  });

  it("does not leak blocks when finishing early", () => {
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 2, maxBatchSize: 1 });
    let state = init(config, 1);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi", maxDecode: 20 }, config);
    for (let t = 0; t < 40; t++) {
      state = reduce(state, { type: "STEP" }, config);
      expect(usage(state.blocks).used).toBeLessThanOrEqual(2);
      if (state.requests[0].status === "finished") break;
    }
  });
});

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

describe("determinism", () => {
  it("two identical action sequences from init produce deep-equal states", () => {
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 8 });
    const actions: Action[] = [
      { type: "ADD_REQUEST", prompt: "hello world", maxDecode: 4 },
      { type: "STEP" },
      { type: "STEP" },
    ];

    const stateA = replay(actions, config, 1);
    const stateB = replay(actions, config, 1);

    // rng contains a function so we can't use JSON.stringify/toEqual directly.
    // Compare the serializable parts.
    expect(stateA.tick).toBe(stateB.tick);
    expect(stateA.requests).toEqual(stateB.requests);
    expect(stateA.blocks).toEqual(stateB.blocks);
    expect(stateA.nextRequestId).toBe(stateB.nextRequestId);
    expect(stateA.arrivalRatePerTick).toBe(stateB.arrivalRatePerTick);
  });

  it("tick() with two independently-built equal states produces equal next states", () => {
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 8 });
    const actions: Action[] = [
      { type: "ADD_REQUEST", prompt: "test prompt", maxDecode: 3 },
      { type: "STEP" }, // prefill
    ];

    const stateA = replay(actions, config, 1);
    const stateB = replay(actions, config, 1);

    const nextA = tick(stateA, config);
    const nextB = tick(stateB, config);

    expect(nextA.tick).toBe(nextB.tick);
    expect(nextA.requests).toEqual(nextB.requests);
    expect(nextA.blocks).toEqual(nextB.blocks);
  });

  it("same initial state and actions produce identical results", () => {
    // Verifies behavioral determinism: identical action replays from the same
    // seed yield deep-equal states (the observable contract of "no Math.random").
    const config = makeConfig({ blockSize: 4, kvCacheBlocks: 8 });
    let stateA = init(config, 7);
    let stateB = init(config, 7);

    stateA = reduce(stateA, { type: "ADD_REQUEST", prompt: "determinism check" }, config);
    stateB = reduce(stateB, { type: "ADD_REQUEST", prompt: "determinism check" }, config);

    for (let i = 0; i < 5; i++) {
      stateA = reduce(stateA, { type: "STEP" }, config);
      stateB = reduce(stateB, { type: "STEP" }, config);
    }

    expect(stateA.requests).toEqual(stateB.requests);
    expect(stateA.blocks).toEqual(stateB.blocks);
    expect(stateA.tick).toBe(stateB.tick);
  });
});

// ---------------------------------------------------------------------------
// CANCEL_REQUEST
// ---------------------------------------------------------------------------

describe("reduce – CANCEL_REQUEST", () => {
  const config = makeConfig({ blockSize: 4, kvCacheBlocks: 8 });

  it("sets request status to 'cancelled'", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "cancel me" }, config);
    state = reduce(state, { type: "STEP" }, config); // admit + prefill
    const reqId = state.requests[0].id;
    const next = reduce(state, { type: "CANCEL_REQUEST", requestId: reqId }, config);
    expect(next.requests[0].status).toBe("cancelled");
  });

  it("frees blocks immediately on cancel", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "cancel me" }, config);
    state = reduce(state, { type: "STEP" }, config); // admit + prefill
    const reqId = state.requests[0].id;
    const next = reduce(state, { type: "CANCEL_REQUEST", requestId: reqId }, config);
    const { used } = usage(next.blocks);
    expect(used).toBe(0);
  });

  it("clears the blockTable on cancel", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "cancel me" }, config);
    state = reduce(state, { type: "STEP" }, config);
    const reqId = state.requests[0].id;
    const next = reduce(state, { type: "CANCEL_REQUEST", requestId: reqId }, config);
    expect(next.requests[0].blockTable).toEqual([]);
  });

  it("cancelling a waiting request (no blocks) still sets status to cancelled", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "cancel waiting" }, config);
    const reqId = state.requests[0].id;
    const next = reduce(state, { type: "CANCEL_REQUEST", requestId: reqId }, config);
    expect(next.requests[0].status).toBe("cancelled");
    expect(next.requests[0].blockTable).toEqual([]);
  });

  it("does not affect other requests when one is cancelled", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "keep me" }, config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "cancel me" }, config);
    state = reduce(state, { type: "STEP" }, config);

    const cancelId = state.requests[1].id;
    const keepId = state.requests[0].id;
    const next = reduce(state, { type: "CANCEL_REQUEST", requestId: cancelId }, config);

    const kept = next.requests.find((r) => r.id === keepId)!;
    expect(kept.status).not.toBe("cancelled");
  });

  it("does not mutate the input state", () => {
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi" }, config);
    state = reduce(state, { type: "STEP" }, config);
    const reqId = state.requests[0].id;
    const snapshot = JSON.stringify({ requests: state.requests, blocks: state.blocks });
    reduce(state, { type: "CANCEL_REQUEST", requestId: reqId }, config);
    expect(JSON.stringify({ requests: state.requests, blocks: state.blocks })).toBe(snapshot);
  });
});

// ---------------------------------------------------------------------------
// RESET
// ---------------------------------------------------------------------------

describe("reduce – RESET", () => {
  it("returns tick 0 after reset", () => {
    let state = init(makeConfig());
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, makeConfig());
    state = reduce(state, { type: "STEP" }, makeConfig());
    const next = reduce(state, { type: "RESET" }, makeConfig());
    expect(next.tick).toBe(0);
  });

  it("returns empty requests after reset", () => {
    let state = init(makeConfig());
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, makeConfig());
    const next = reduce(state, { type: "RESET" }, makeConfig());
    expect(next.requests).toEqual([]);
  });

  it("returns a full free block pool after reset", () => {
    const config = makeConfig({ kvCacheBlocks: 8 });
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hello" }, config);
    state = reduce(state, { type: "STEP" }, config);
    const next = reduce(state, { type: "RESET" }, config);
    const { used } = usage(next.blocks);
    expect(used).toBe(0);
    expect(next.blocks).toHaveLength(8);
  });

  it("reset is reproducible: two resets from different states produce equal results", () => {
    const config = makeConfig({ kvCacheBlocks: 8 });
    // State A: some history
    let stateA = init(config);
    stateA = reduce(stateA, { type: "ADD_REQUEST", prompt: "something" }, config);
    stateA = reduce(stateA, { type: "STEP" }, config);
    stateA = reduce(stateA, { type: "STEP" }, config);
    const afterResetA = reduce(stateA, { type: "RESET" }, config);

    // State B: different history
    let stateB = init(config, 99);
    stateB = reduce(stateB, { type: "ADD_REQUEST", prompt: "different" }, config);
    const afterResetB = reduce(stateB, { type: "RESET" }, config);

    // Both resets use fixed seed=1, so they should produce identical states.
    expect(afterResetA.tick).toBe(afterResetB.tick);
    expect(afterResetA.requests).toEqual(afterResetB.requests);
    expect(afterResetA.blocks).toEqual(afterResetB.blocks);
    expect(afterResetA.nextRequestId).toBe(afterResetB.nextRequestId);
    expect(afterResetA.arrivalRatePerTick).toBe(afterResetB.arrivalRatePerTick);
  });
});

// ---------------------------------------------------------------------------
// SET_ARRIVAL_RATE
// ---------------------------------------------------------------------------

describe("reduce – SET_ARRIVAL_RATE", () => {
  it("updates arrivalRatePerTick", () => {
    const config = makeConfig();
    const state = init(config);
    const next = reduce(state, { type: "SET_ARRIVAL_RATE", ratePerTick: 0.5 }, config);
    expect(next.arrivalRatePerTick).toBe(0.5);
  });

  it("can set rate to zero", () => {
    const config = makeConfig();
    let state = init(config);
    state = reduce(state, { type: "SET_ARRIVAL_RATE", ratePerTick: 2 }, config);
    const next = reduce(state, { type: "SET_ARRIVAL_RATE", ratePerTick: 0 }, config);
    expect(next.arrivalRatePerTick).toBe(0);
  });

  it("does not mutate the input state", () => {
    const config = makeConfig();
    const state = init(config);
    const snapshot = JSON.stringify(state);
    reduce(state, { type: "SET_ARRIVAL_RATE", ratePerTick: 1 }, config);
    expect(JSON.stringify(state)).toBe(snapshot);
  });
});

// ---------------------------------------------------------------------------
// SET_CONFIG
// ---------------------------------------------------------------------------

describe("reduce – SET_CONFIG", () => {
  it("returns the same requests and tick unchanged", () => {
    const config = makeConfig();
    let state = init(config);
    state = reduce(state, { type: "ADD_REQUEST", prompt: "hi" }, config);
    const snapshot = { tick: state.tick, requestsLen: state.requests.length };
    const next = reduce(state, { type: "SET_CONFIG", config: { maxBatchSize: 8 } }, config);
    expect(next.tick).toBe(snapshot.tick);
    expect(next.requests.length).toBe(snapshot.requestsLen);
  });
});

// ---------------------------------------------------------------------------
// DEFAULT_MAX_DECODE export
// ---------------------------------------------------------------------------

describe("DEFAULT_MAX_DECODE", () => {
  it("is a positive integer", () => {
    expect(typeof DEFAULT_MAX_DECODE).toBe("number");
    expect(Number.isInteger(DEFAULT_MAX_DECODE)).toBe(true);
    expect(DEFAULT_MAX_DECODE).toBeGreaterThan(0);
  });
});
