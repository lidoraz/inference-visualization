/**
 * Tests for src/engine/scheduler.ts
 *
 * Written FIRST (TDD) — all tests should be RED before the implementation exists,
 * then GREEN after a correct implementation.
 */

import { describe, it, expect } from "vitest";
import { schedule } from "../../src/engine/scheduler";
import { createBlocks, allocate } from "../../src/engine/kvcache";
import type { Block, Config, EngineState, Request, Rng } from "../../src/engine/types";

// ---------------------------------------------------------------------------
// Helpers / Fixtures
// ---------------------------------------------------------------------------

const stubRng: Rng = { seed: 0, next: () => 0 };

function makeConfig(overrides: Partial<Config> = {}): Config {
  return {
    maxBatchSize: 4,
    blockSize: 4,
    kvCacheBlocks: 16,
    tokenBudget: 256,
    ...overrides,
  };
}

/** Build a minimal waiting Request. promptTokens count determines prefill cost. */
function makeWaitingRequest(
  id: number,
  arrivalTick: number,
  promptLen: number,
  maxDecode = 8
): Request {
  return {
    id,
    arrivalTick,
    status: "waiting",
    phase: "prefill",
    promptTokens: Array.from({ length: promptLen }, (_, i) => ({ id: i, text: `t${i}` })),
    decodedTokens: [],
    maxDecode,
    blockTable: [],
  };
}

/** Build a request already in "running" state with blocks allocated. */
function makeRunningRequest(
  id: number,
  arrivalTick: number,
  promptLen: number,
  decodedLen = 0,
  blockTable: number[] = []
): Request {
  return {
    id,
    arrivalTick,
    status: "running",
    phase: decodedLen > 0 ? "decode" : "prefill",
    promptTokens: Array.from({ length: promptLen }, (_, i) => ({ id: i, text: `t${i}` })),
    decodedTokens: Array.from({ length: decodedLen }, (_, i) => ({ id: 1000 + i, text: `d${i}` })),
    maxDecode: 16,
    blockTable,
  };
}

/** Build a swapped request (previously ran, blocks freed, decodedTokens preserved). */
function makeSwappedRequest(
  id: number,
  arrivalTick: number,
  promptLen: number,
  decodedLen: number
): Request {
  return {
    id,
    arrivalTick,
    status: "swapped",
    phase: "decode",
    promptTokens: Array.from({ length: promptLen }, (_, i) => ({ id: i, text: `t${i}` })),
    decodedTokens: Array.from({ length: decodedLen }, (_, i) => ({ id: 1000 + i, text: `d${i}` })),
    maxDecode: 16,
    blockTable: [],
  };
}

/** Wrap requests + blocks into a minimal EngineState. */
function makeState(
  requests: Request[],
  blocks: Block[],
  tick = 0
): EngineState {
  return {
    tick,
    requests,
    blocks,
    rng: stubRng,
    nextRequestId: 100,
    arrivalRatePerTick: 0,
  };
}

// ---------------------------------------------------------------------------
// Admission: maxBatchSize
// ---------------------------------------------------------------------------

describe("scheduler – rejects unsatisfiable prompts", () => {
  it("cancels a waiting request whose prompt exceeds total cache capacity", () => {
    // Cache holds 2 blocks * 4 = 8 token slots; prompt needs 17 tokens -> 5 blocks.
    // It can never be admitted, so the scheduler must reject it (not loop forever).
    const blocks = createBlocks(2, 4);
    const config = makeConfig({ kvCacheBlocks: 2, blockSize: 4 });
    const req = makeWaitingRequest(1, 0, 17);
    const next = schedule(makeState([req], blocks), config);

    const out = next.requests.find((r) => r.id === 1)!;
    expect(out.status).toBe("cancelled");
    expect(out.rejectionReason).toBeTruthy();
    // No blocks were consumed by the rejected request.
    expect(next.blocks.every((b) => b.requestId === null)).toBe(true);
  });

  it("does not reject a prompt that exactly fits the cache", () => {
    // 2 blocks * 4 = 8 token slots; prompt of 8 tokens -> exactly 2 blocks.
    const blocks = createBlocks(2, 4);
    const config = makeConfig({ kvCacheBlocks: 2, blockSize: 4, maxBatchSize: 1 });
    const req = makeWaitingRequest(1, 0, 8);
    const next = schedule(makeState([req], blocks), config);

    const out = next.requests.find((r) => r.id === 1)!;
    expect(out.status).toBe("running");
    expect(out.rejectionReason).toBeUndefined();
  });
});

describe("scheduler – admission up to maxBatchSize", () => {
  it("admits waiting requests up to maxBatchSize", () => {
    const blocks = createBlocks(16, 4);
    const requests = [
      makeWaitingRequest(1, 0, 4),
      makeWaitingRequest(2, 1, 4),
      makeWaitingRequest(3, 2, 4),
    ];
    const config = makeConfig({ maxBatchSize: 2 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const running = next.requests.filter((r) => r.status === "running");
    const waiting = next.requests.filter((r) => r.status === "waiting");

    expect(running).toHaveLength(2);
    expect(waiting).toHaveLength(1);
  });

  it("admits exactly maxBatchSize even when more are waiting", () => {
    const blocks = createBlocks(16, 4);
    const requests = [
      makeWaitingRequest(1, 0, 1),
      makeWaitingRequest(2, 1, 1),
      makeWaitingRequest(3, 2, 1),
      makeWaitingRequest(4, 3, 1),
      makeWaitingRequest(5, 4, 1),
    ];
    const config = makeConfig({ maxBatchSize: 3 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    expect(next.requests.filter((r) => r.status === "running")).toHaveLength(3);
    expect(next.requests.filter((r) => r.status === "waiting")).toHaveLength(2);
  });

  it("already-running requests count toward maxBatchSize", () => {
    // 2 already running + 3 waiting, maxBatchSize = 3  => only 1 more can be admitted
    let blocks = createBlocks(16, 4);
    const r1alloc = allocate(blocks, 10, 4, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 11, 4, 4);
    blocks = r2alloc.blocks;

    const runningReqs = [
      makeRunningRequest(10, 0, 4, 0, r1alloc.blockTable),
      makeRunningRequest(11, 1, 4, 0, r2alloc.blockTable),
    ];
    const waitingReqs = [
      makeWaitingRequest(1, 2, 4),
      makeWaitingRequest(2, 3, 4),
      makeWaitingRequest(3, 4, 4),
    ];
    const config = makeConfig({ maxBatchSize: 3 });
    const state = makeState([...runningReqs, ...waitingReqs], blocks);

    const next = schedule(state, config);
    expect(next.requests.filter((r) => r.status === "running")).toHaveLength(3);
    expect(next.requests.filter((r) => r.status === "waiting")).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// Admission: tokenBudget
// ---------------------------------------------------------------------------

describe("scheduler – tokenBudget enforcement", () => {
  it("defers a request whose prefill cost would exceed remaining budget", () => {
    const blocks = createBlocks(16, 4);
    // tokenBudget = 10; req1 costs 8 tokens (admitted), req2 costs 8 tokens (exceeds remaining 2)
    const requests = [
      makeWaitingRequest(1, 0, 8),
      makeWaitingRequest(2, 1, 8),
    ];
    const config = makeConfig({ maxBatchSize: 4, tokenBudget: 10 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const running = next.requests.filter((r) => r.status === "running");
    const waiting = next.requests.filter((r) => r.status === "waiting");

    expect(running).toHaveLength(1);
    expect(running[0].id).toBe(1);
    expect(waiting).toHaveLength(1);
    expect(waiting[0].id).toBe(2);
  });

  it("admits a request that exactly fits the remaining budget", () => {
    const blocks = createBlocks(16, 4);
    const requests = [
      makeWaitingRequest(1, 0, 6),
      makeWaitingRequest(2, 1, 4), // 6 + 4 = 10 == tokenBudget
    ];
    const config = makeConfig({ maxBatchSize: 4, tokenBudget: 10 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    expect(next.requests.filter((r) => r.status === "running")).toHaveLength(2);
  });

  it("skips a request exceeding budget and admits a later smaller one", () => {
    // Design choice: we process candidates in FIFO order; if a candidate doesn't fit
    // the budget we skip it (don't break — check remaining candidates).
    // This matches the "skip on budget exceeded" strategy.
    const blocks = createBlocks(16, 4);
    const requests = [
      makeWaitingRequest(1, 0, 9),  // cost 9 — admitted (budget: 16 - 9 = 7 left)
      makeWaitingRequest(2, 1, 8),  // cost 8 — exceeds remaining 7, skip
      makeWaitingRequest(3, 2, 4),  // cost 4 — fits remaining 7, admit
    ];
    const config = makeConfig({ maxBatchSize: 4, tokenBudget: 16 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const running = next.requests.filter((r) => r.status === "running");
    const runningIds = running.map((r) => r.id).sort();
    expect(runningIds).toContain(1);
    expect(runningIds).toContain(3);
    expect(next.requests.find((r) => r.id === 2)?.status).toBe("waiting");
  });
});

// ---------------------------------------------------------------------------
// Admission: block allocation failure
// ---------------------------------------------------------------------------

describe("scheduler – block allocation gating", () => {
  it("does not admit a waiting request when blocks are temporarily exhausted", () => {
    // Cache has 4 blocks total (enough for the 2-block prompt in principle), but a
    // running request currently occupies 3, leaving only 1 free. The waiting req
    // needs 2 -> can't be admitted now, but is satisfiable later, so it WAITS
    // (not cancelled).
    let blocks = createBlocks(4, 4);
    const alloc = allocate(blocks, 0, 12, 4); // running req owns 3 blocks
    blocks = alloc.blocks;
    const running = makeRunningRequest(0, 0, 12, 0, alloc.blockTable);
    const waiting = makeWaitingRequest(1, 1, 8); // needs 2 blocks
    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4 });
    const state = makeState([running, waiting], blocks);

    const next = schedule(state, config);
    expect(next.requests.find((r) => r.id === 1)?.status).toBe("waiting");
  });

  it("admits as many requests as blocks allow", () => {
    // 2 blocks total, blockSize 4; req1 needs 1 block (4 tokens), req2 needs 1 block
    const blocks = createBlocks(2, 4);
    const requests = [
      makeWaitingRequest(1, 0, 4),
      makeWaitingRequest(2, 1, 4),
      makeWaitingRequest(3, 2, 4), // no blocks left for this one
    ];
    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 2 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    expect(next.requests.filter((r) => r.status === "running")).toHaveLength(2);
    expect(next.requests.find((r) => r.id === 3)?.status).toBe("waiting");
  });

  it("updates state.blocks when a request is admitted", () => {
    const blocks = createBlocks(4, 4);
    const requests = [makeWaitingRequest(1, 0, 4)];
    const config = makeConfig({ maxBatchSize: 4 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const usedBlocks = next.blocks.filter((b) => b.requestId === 1);
    expect(usedBlocks).toHaveLength(1); // ceil(4/4) = 1 block
  });
});

// ---------------------------------------------------------------------------
// Admission: FIFO ordering
// ---------------------------------------------------------------------------

describe("scheduler – FIFO ordering (arrivalTick then id)", () => {
  it("admits the earliest-arriving requests first", () => {
    const blocks = createBlocks(16, 4);
    // Requests arrive out of order by tick
    const requests = [
      makeWaitingRequest(3, 5, 4),
      makeWaitingRequest(1, 1, 4),
      makeWaitingRequest(2, 3, 4),
    ];
    const config = makeConfig({ maxBatchSize: 2 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const running = next.requests.filter((r) => r.status === "running");
    const runningIds = running.map((r) => r.id);
    expect(runningIds).toContain(1);
    expect(runningIds).toContain(2);
    expect(next.requests.find((r) => r.id === 3)?.status).toBe("waiting");
  });

  it("breaks ties in arrivalTick by lower id first", () => {
    const blocks = createBlocks(16, 4);
    // Same arrivalTick — lower id should be preferred
    const requests = [
      makeWaitingRequest(5, 0, 4),
      makeWaitingRequest(2, 0, 4),
      makeWaitingRequest(8, 0, 4),
    ];
    const config = makeConfig({ maxBatchSize: 2 });
    const state = makeState(requests, blocks);

    const next = schedule(state, config);
    const running = next.requests.filter((r) => r.status === "running");
    const runningIds = running.map((r) => r.id);
    expect(runningIds).toContain(2);
    expect(runningIds).toContain(5);
    expect(next.requests.find((r) => r.id === 8)?.status).toBe("waiting");
  });
});

// ---------------------------------------------------------------------------
// Swapped request re-admission
// ---------------------------------------------------------------------------

describe("scheduler – swapped request re-admission", () => {
  it("re-admits a swapped request once capacity is available", () => {
    const blocks = createBlocks(8, 4);
    const swappedReq = makeSwappedRequest(1, 0, 4, 3); // 4 prompt + 3 decoded = 7 tokens => 2 blocks

    const config = makeConfig({ maxBatchSize: 4 });
    const state = makeState([swappedReq], blocks);

    const next = schedule(state, config);
    const req = next.requests.find((r) => r.id === 1)!;
    expect(req.status).toBe("running");
  });

  it("preserves decodedTokens when re-admitting a swapped request", () => {
    const blocks = createBlocks(8, 4);
    const swappedReq = makeSwappedRequest(1, 0, 4, 3);

    const config = makeConfig({ maxBatchSize: 4 });
    const state = makeState([swappedReq], blocks);

    const next = schedule(state, config);
    const req = next.requests.find((r) => r.id === 1)!;
    expect(req.decodedTokens).toHaveLength(3);
    expect(req.decodedTokens.map((t) => t.text)).toEqual(["d0", "d1", "d2"]);
  });

  it("allocates blocks for a re-admitted swapped request", () => {
    const blocks = createBlocks(8, 4);
    const swappedReq = makeSwappedRequest(1, 0, 4, 4); // 8 tokens => 2 blocks

    const config = makeConfig({ maxBatchSize: 4 });
    const state = makeState([swappedReq], blocks);

    const next = schedule(state, config);
    const req = next.requests.find((r) => r.id === 1)!;
    expect(req.blockTable.length).toBeGreaterThan(0);
    const ownedBlocks = next.blocks.filter((b) => b.requestId === 1);
    expect(ownedBlocks).toHaveLength(req.blockTable.length);
  });

  it("swapped request stays swapped when blocks are temporarily exhausted", () => {
    // 4-block cache (enough in principle), but a running request occupies 3,
    // leaving 1 free. The swapped req needs 2 -> stays swapped (satisfiable later).
    let blocks = createBlocks(4, 4);
    const alloc = allocate(blocks, 0, 12, 4); // running req owns 3 blocks
    blocks = alloc.blocks;
    const running = makeRunningRequest(0, 0, 12, 0, alloc.blockTable);
    const swappedReq = makeSwappedRequest(1, 1, 4, 4); // 8 tokens => 2 blocks

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4 });
    const state = makeState([running, swappedReq], blocks);

    const next = schedule(state, config);
    expect(next.requests.find((r) => r.id === 1)?.status).toBe("swapped");
  });

  it("swapped requests are treated as FIFO together with waiting requests", () => {
    const blocks = createBlocks(4, 4); // 4 blocks
    // Swapped request arrived at tick 0, waiting arrived at tick 1
    // maxBatchSize 1 => only earliest should be admitted
    const swappedReq = makeSwappedRequest(1, 0, 4, 0); // 4 tokens, 1 block
    const waitingReq = makeWaitingRequest(2, 1, 4);    // 4 tokens, 1 block

    const config = makeConfig({ maxBatchSize: 1 });
    const state = makeState([swappedReq, waitingReq], blocks);

    const next = schedule(state, config);
    expect(next.requests.find((r) => r.id === 1)?.status).toBe("running");
    expect(next.requests.find((r) => r.id === 2)?.status).toBe("waiting");
  });
});

// ---------------------------------------------------------------------------
// Preemption under live capacity shrink
// ---------------------------------------------------------------------------

describe("scheduler – preemption under live capacity shrink", () => {
  it("evicts newest running requests when used blocks exceed kvCacheBlocks", () => {
    // Set up 3 running requests, each using 2 blocks = 6 blocks total.
    // Then config reports kvCacheBlocks = 4 (can hold 2 requests x 2 blocks).
    // Newest request (highest arrivalTick) should be swapped first.
    let blocks = createBlocks(6, 4);

    const r1alloc = allocate(blocks, 1, 8, 4); // 2 blocks
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 2, 8, 4); // 2 blocks
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 3, 8, 4); // 2 blocks
    blocks = r3alloc.blocks;

    const runningReqs = [
      makeRunningRequest(1, 0, 4, 4, r1alloc.blockTable), // oldest
      makeRunningRequest(2, 1, 4, 4, r2alloc.blockTable),
      makeRunningRequest(3, 2, 4, 4, r3alloc.blockTable), // newest
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    // r3 (newest) should be swapped
    expect(next.requests.find((r) => r.id === 3)?.status).toBe("swapped");
    // r1 and r2 should remain running
    expect(next.requests.find((r) => r.id === 1)?.status).toBe("running");
    expect(next.requests.find((r) => r.id === 2)?.status).toBe("running");
  });

  it("frees blocks of preempted requests", () => {
    let blocks = createBlocks(6, 4);

    const r1alloc = allocate(blocks, 1, 8, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 2, 8, 4);
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 3, 8, 4);
    blocks = r3alloc.blocks;

    const runningReqs = [
      makeRunningRequest(1, 0, 4, 4, r1alloc.blockTable),
      makeRunningRequest(2, 1, 4, 4, r2alloc.blockTable),
      makeRunningRequest(3, 2, 4, 4, r3alloc.blockTable),
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    // Preempted request's blocks must be freed
    const r3Blocks = next.blocks.filter((b) => b.requestId === 3);
    expect(r3Blocks).toHaveLength(0);
  });

  it("clears blockTable of preempted requests", () => {
    let blocks = createBlocks(6, 4);

    const r1alloc = allocate(blocks, 1, 8, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 2, 8, 4);
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 3, 8, 4);
    blocks = r3alloc.blocks;

    const runningReqs = [
      makeRunningRequest(1, 0, 4, 4, r1alloc.blockTable),
      makeRunningRequest(2, 1, 4, 4, r2alloc.blockTable),
      makeRunningRequest(3, 2, 4, 4, r3alloc.blockTable),
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    const preemptedReq = next.requests.find((r) => r.id === 3)!;
    expect(preemptedReq.blockTable).toEqual([]);
  });

  it("surviving (older) requests keep intact blockTables after preemption", () => {
    let blocks = createBlocks(6, 4);

    const r1alloc = allocate(blocks, 1, 8, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 2, 8, 4);
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 3, 8, 4);
    blocks = r3alloc.blocks;

    const runningReqs = [
      makeRunningRequest(1, 0, 4, 4, r1alloc.blockTable),
      makeRunningRequest(2, 1, 4, 4, r2alloc.blockTable),
      makeRunningRequest(3, 2, 4, 4, r3alloc.blockTable),
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 4, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    const r1 = next.requests.find((r) => r.id === 1)!;
    const r2 = next.requests.find((r) => r.id === 2)!;

    // blockTables must be preserved
    expect(r1.blockTable).toEqual(r1alloc.blockTable);
    expect(r2.blockTable).toEqual(r2alloc.blockTable);
  });

  it("evicts multiple requests if necessary until usage fits", () => {
    // 4 running requests each taking 1 block = 4 blocks total.
    // Shrink to kvCacheBlocks = 2, so must evict 2 newest.
    let blocks = createBlocks(4, 4);

    const r1alloc = allocate(blocks, 1, 4, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 2, 4, 4);
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 3, 4, 4);
    blocks = r3alloc.blocks;
    const r4alloc = allocate(blocks, 4, 4, 4);
    blocks = r4alloc.blocks;

    const runningReqs = [
      makeRunningRequest(1, 0, 4, 0, r1alloc.blockTable),
      makeRunningRequest(2, 1, 4, 0, r2alloc.blockTable),
      makeRunningRequest(3, 2, 4, 0, r3alloc.blockTable),
      makeRunningRequest(4, 3, 4, 0, r4alloc.blockTable),
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 2, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    // r3 and r4 (newest two) should be swapped
    expect(next.requests.find((r) => r.id === 3)?.status).toBe("swapped");
    expect(next.requests.find((r) => r.id === 4)?.status).toBe("swapped");
    // r1, r2 should remain running
    expect(next.requests.find((r) => r.id === 1)?.status).toBe("running");
    expect(next.requests.find((r) => r.id === 2)?.status).toBe("running");
  });

  it("preemption tie-breaks by highest id when arrivalTick is equal", () => {
    let blocks = createBlocks(3, 4);

    const r1alloc = allocate(blocks, 10, 4, 4);
    blocks = r1alloc.blocks;
    const r2alloc = allocate(blocks, 20, 4, 4);
    blocks = r2alloc.blocks;
    const r3alloc = allocate(blocks, 30, 4, 4);
    blocks = r3alloc.blocks;

    // All have same arrivalTick — highest id should be preempted first
    const runningReqs = [
      makeRunningRequest(10, 5, 4, 0, r1alloc.blockTable),
      makeRunningRequest(20, 5, 4, 0, r2alloc.blockTable),
      makeRunningRequest(30, 5, 4, 0, r3alloc.blockTable),
    ];

    const config = makeConfig({ maxBatchSize: 4, kvCacheBlocks: 2, blockSize: 4 });
    const state = makeState(runningReqs, blocks);

    const next = schedule(state, config);

    expect(next.requests.find((r) => r.id === 30)?.status).toBe("swapped");
    expect(next.requests.find((r) => r.id === 10)?.status).toBe("running");
    expect(next.requests.find((r) => r.id === 20)?.status).toBe("running");
  });
});

// ---------------------------------------------------------------------------
// Purity
// ---------------------------------------------------------------------------

describe("scheduler – purity (no mutation of inputs)", () => {
  it("does not mutate the input state object", () => {
    const blocks = createBlocks(8, 4);
    const requests = [makeWaitingRequest(1, 0, 4)];
    const state = makeState(requests, blocks);
    const stateCopy = JSON.stringify(state);

    schedule(state, makeConfig());

    expect(JSON.stringify(state)).toBe(stateCopy);
  });

  it("does not mutate the input requests array", () => {
    const blocks = createBlocks(8, 4);
    const requests = [makeWaitingRequest(1, 0, 4), makeWaitingRequest(2, 1, 4)];
    const state = makeState(requests, blocks);
    const reqSnapshot = JSON.stringify(state.requests);

    schedule(state, makeConfig());

    expect(JSON.stringify(state.requests)).toBe(reqSnapshot);
  });

  it("does not mutate the input blocks array", () => {
    const blocks = createBlocks(8, 4);
    const requests = [makeWaitingRequest(1, 0, 4)];
    const state = makeState(requests, blocks);
    const blockSnapshot = JSON.stringify(state.blocks);

    schedule(state, makeConfig());

    expect(JSON.stringify(state.blocks)).toBe(blockSnapshot);
  });

  it("returns a new EngineState object (not the same reference)", () => {
    const blocks = createBlocks(8, 4);
    const requests = [makeWaitingRequest(1, 0, 4)];
    const state = makeState(requests, blocks);

    const next = schedule(state, makeConfig());

    expect(next).not.toBe(state);
    expect(next.requests).not.toBe(state.requests);
    expect(next.blocks).not.toBe(state.blocks);
  });
});

// ---------------------------------------------------------------------------
// Finished/cancelled requests are left untouched
// ---------------------------------------------------------------------------

describe("scheduler – finished and cancelled requests are not touched", () => {
  it("does not change status of finished or cancelled requests", () => {
    const blocks = createBlocks(8, 4);
    const requests: Request[] = [
      { ...makeWaitingRequest(1, 0, 4), status: "finished" },
      { ...makeWaitingRequest(2, 1, 4), status: "cancelled" },
      makeWaitingRequest(3, 2, 4),
    ];
    const state = makeState(requests, blocks);

    const next = schedule(state, makeConfig());

    expect(next.requests.find((r) => r.id === 1)?.status).toBe("finished");
    expect(next.requests.find((r) => r.id === 2)?.status).toBe("cancelled");
    expect(next.requests.find((r) => r.id === 3)?.status).toBe("running");
  });
});
