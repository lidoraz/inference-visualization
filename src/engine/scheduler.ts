/**
 * Scheduler for the vLLM Inference Visualizer.
 *
 * Pure function — never mutates its inputs.  Returns a new EngineState with
 * updated request statuses and block ownership after running one scheduling
 * pass (admission + preemption).
 *
 * Called once per tick BEFORE token advancement (engine.ts calls schedule()
 * then advances running requests).
 *
 * Design decisions (noted where spec was silent):
 *   - "Skip on budget exceeded" strategy: if a candidate's prefill cost would
 *     exceed the remaining tokenBudget we skip it and check the next candidate
 *     rather than stopping admission entirely. This maximises GPU utilisation.
 *   - Block capacity check for preemption uses config.kvCacheBlocks as the
 *     *authoritative* capacity, which may be less than blocks.length when the
 *     user shrinks the slider live. We consider only blocks 0..kvCacheBlocks-1
 *     in scope; blocks beyond that range are treated as non-existent for usage
 *     counting purposes.  Because createBlocks always produces ids 0..N-1 and
 *     kvcache.free/allocate operate on the full blocks array, we reduce
 *     kvCacheBlocks by slicing/filtering rather than rearranging physical ids.
 *   - Preemption uses a virtual capacity window: we count used blocks whose id
 *     < config.kvCacheBlocks as the effective "used" count and compare against
 *     config.kvCacheBlocks.  Any running request that owns blocks at ids >=
 *     kvCacheBlocks is also treated as over-capacity and is a preemption
 *     candidate.
 *
 * NOTE: This implementation assumes the physical blocks array passed in
 * state.blocks has already been set up by createBlocks (ids are 0-based
 * sequential integers).  The scheduler does NOT resize the pool; it only
 * treats blocks with id >= kvCacheBlocks as inaccessible.
 */

import type { Block, Config, EngineState, Request } from "./types";
import { allocate, free } from "./kvcache";

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Total order: primary sort key = arrivalTick ASC, secondary = id ASC (FIFO). */
function fifoCompare(a: Request, b: Request): number {
  if (a.arrivalTick !== b.arrivalTick) return a.arrivalTick - b.arrivalTick;
  return a.id - b.id;
}

/**
 * Return the effective number of blocks actually usable given the current
 * config capacity.  We compare against config.kvCacheBlocks as the capacity
 * limit regardless of how many Block objects are in the array.
 */
function effectiveUsage(blocks: Block[], kvCacheBlocks: number): number {
  // Count blocks that are allocated AND whose id falls within the live capacity.
  return blocks.filter(
    (b) => b.requestId !== null && b.id < kvCacheBlocks
  ).length;
}

/**
 * Return true if a running request owns any block whose id >= kvCacheBlocks
 * (i.e. a block that has been "removed" from the live pool by a SET_CONFIG).
 */
function requestExceedsCapacity(req: Request, kvCacheBlocks: number): boolean {
  return req.blockTable.some((physId) => physId >= kvCacheBlocks);
}

// ---------------------------------------------------------------------------
// schedule
// ---------------------------------------------------------------------------

/**
 * Run one scheduling pass:
 *   1. Preempt running requests that exceed the live KV-cache capacity.
 *   2. Admit waiting/swapped requests (FIFO) subject to maxBatchSize,
 *      tokenBudget, and block availability.
 *
 * Pure — returns a new EngineState; inputs are never mutated.
 */
export function schedule(state: EngineState, config: Config): EngineState {
  const { maxBatchSize, blockSize, kvCacheBlocks, tokenBudget } = config;

  // Work on shallow copies so we never mutate the originals.
  let currentBlocks: Block[] = state.blocks.slice();
  // Map from request id -> updated Request (start with identity copies).
  const reqMap = new Map<number, Request>(
    state.requests.map((r) => [r.id, { ...r, blockTable: r.blockTable.slice() }])
  );

  // ---------------------------------------------------------------------------
  // Phase 1 – Preemption
  // ---------------------------------------------------------------------------
  // Identify running requests that need to be evicted.  We evict newest-first
  // (highest arrivalTick, then highest id) until effective usage fits within
  // kvCacheBlocks.

  // Candidates for preemption: all currently-running requests.
  const preemptCandidates = Array.from(reqMap.values())
    .filter((r) => r.status === "running")
    .sort((a, b) => {
      // Reverse FIFO: highest arrivalTick first, then highest id first.
      if (a.arrivalTick !== b.arrivalTick) return b.arrivalTick - a.arrivalTick;
      return b.id - a.id;
    });

  // Track which request ids were preempted this tick so they are NOT
  // immediately re-admitted during Phase 2 admission.
  const preemptedThisTick = new Set<number>();

  // Evict newest-first until we are within capacity.
  // We must keep evicting as long as:
  //   (a) effective usage exceeds kvCacheBlocks, OR
  //   (b) some still-running request owns blocks at ids >= kvCacheBlocks.
  // INVARIANT this loop relies on: older requests own lower-id blocks. This holds
  // because admission is FIFO (oldest first) and kvcache.allocate() always claims
  // the lowest-id free blocks. So evicting newest-first (which we do here) frees
  // the highest-id blocks first, and once the current (older) candidate has no
  // out-of-range blocks, no still-running older request can either. If the
  // allocation strategy ever stops being lowest-id-first, revisit this break.
  for (const candidate of preemptCandidates) {
    const currentUsed = effectiveUsage(currentBlocks, kvCacheBlocks);
    const overCapacityBlocks = requestExceedsCapacity(candidate, kvCacheBlocks);

    if (currentUsed <= kvCacheBlocks && !overCapacityBlocks) {
      // Usage is within limits and this candidate (older than already-evicted
      // ones) has no out-of-range blocks — stop evicting.
      break;
    }

    // Preempt this request.
    currentBlocks = free(currentBlocks, candidate.id);
    preemptedThisTick.add(candidate.id);
    reqMap.set(candidate.id, {
      ...candidate,
      status: "swapped",
      blockTable: [],
    });
  }

  // ---------------------------------------------------------------------------
  // Phase 2 – Admission
  // ---------------------------------------------------------------------------
  // Candidates are waiting + swapped requests, sorted FIFO.
  // Exclude requests that were just preempted this tick — they cannot be
  // re-admitted until a future tick when capacity is available.
  const admissionCandidates = Array.from(reqMap.values())
    .filter(
      (r) =>
        (r.status === "waiting" || r.status === "swapped") &&
        !preemptedThisTick.has(r.id)
    )
    .sort(fifoCompare);

  const currentRunningCount = Array.from(reqMap.values()).filter(
    (r) => r.status === "running"
  ).length;

  let runningCount = currentRunningCount;
  let budgetUsed = 0;

  for (const candidate of admissionCandidates) {
    // Reject the unsatisfiable: a prompt that needs more blocks than the entire
    // cache can ever hold will never be admitted. Cancel it with a reason rather
    // than letting it wait forever. (vLLM likewise errors on prompts exceeding
    // KV capacity.) Checked against total capacity, independent of batch limits.
    // Apply the sliding window so a long sequence that fits within the window
    // isn't spuriously cancelled (the allocation path below is windowed too).
    const seqNeeded = candidate.promptTokens.length + candidate.decodedTokens.length;
    const totalNeeded = config.windowSize ? Math.min(seqNeeded, config.windowSize) : seqNeeded;
    if (Math.ceil(totalNeeded / blockSize) > kvCacheBlocks) {
      reqMap.set(candidate.id, {
        ...candidate,
        status: "cancelled",
        blockTable: [],
        rejectionReason: `Prompt needs ${Math.ceil(totalNeeded / blockSize)} blocks but the KV cache holds only ${kvCacheBlocks}.`,
      });
      continue;
    }

    // Budget: cost is the request's prompt length (prefill cost).
    const prefillCost = candidate.promptTokens.length;

    // Check batch size limit.
    if (runningCount >= maxBatchSize) {
      // Batch is full — record why and stop admitting (all remaining candidates
      // are blocked by the same limit, so set the reason on each).
      reqMap.set(candidate.id, {
        ...candidate,
        waitReason: `Batch full: ${runningCount}/${maxBatchSize} requests already running.`,
      });
      continue;
    }

    if (budgetUsed + prefillCost > tokenBudget) {
      // Skip this candidate but continue checking smaller ones that may still fit.
      reqMap.set(candidate.id, {
        ...candidate,
        waitReason: `Token budget: ${budgetUsed}/${tokenBudget} used this step; this prompt needs ${prefillCost} more.`,
      });
      continue;
    }

    // Block allocation: need blocks for all tokens (prompt + already decoded),
    // capped to the sliding window when configured.
    const seqTokens =
      candidate.promptTokens.length + candidate.decodedTokens.length;
    const totalTokens = config.windowSize
      ? Math.min(seqTokens, config.windowSize)
      : seqTokens;

    const allocResult = allocate(currentBlocks, candidate.id, totalTokens, blockSize);
    if (!allocResult.ok) {
      // Not enough free blocks right now — stays waiting; record why.
      reqMap.set(candidate.id, {
        ...candidate,
        waitReason: `KV cache full: not enough free blocks for ${Math.ceil(totalTokens / blockSize)} block(s).`,
      });
      continue;
    }

    // Admit the request.
    currentBlocks = allocResult.blocks;
    runningCount += 1;
    budgetUsed += prefillCost;

    reqMap.set(candidate.id, {
      ...candidate,
      status: "running",
      // A newly-admitted waiting request always starts in prefill phase.
      // A re-admitted swapped request keeps its decodedTokens (already in
      // candidate) but we also set phase to prefill so the engine re-prefills
      // the full prompt.  Design choice: swapped requests resume as "prefill"
      // since their KV state was discarded when blocks were freed.
      phase: "prefill",
      blockTable: allocResult.blockTable,
      waitReason: undefined, // admitted — clear any prior wait reason
    });
  }

  // ---------------------------------------------------------------------------
  // Build and return the new EngineState.
  // ---------------------------------------------------------------------------
  return {
    ...state,
    requests: state.requests.map((r) => reqMap.get(r.id) ?? r),
    blocks: currentBlocks,
  };
}
