/**
 * Engine orchestration for the vLLM Inference Visualizer.
 *
 * Public surface consumed by App.tsx's useReducer.
 * NO React imports — pure TypeScript, no side effects, no global mutable state.
 *
 * Design choices (documented where spec gave latitude):
 *
 *   RESET seed: RESET always calls init(config, 1) with a fixed seed so that
 *   clicking Reset in the UI always produces a reproducible starting state
 *   regardless of how far the simulation had advanced.  The seed=1 constant is
 *   intentional and documented here.
 *
 *   SET_CONFIG: config is owned by the React reducer layer; the engine receives
 *   the current config on every call.  SET_CONFIG therefore returns state
 *   unchanged — any capacity change is enforced on the NEXT tick via the
 *   scheduler's preemption logic.  This keeps the engine stateless w.r.t. config.
 *
 *   Reallocation during decode: when a running request in decode phase needs one
 *   more token slot we first FREE its existing blocks, then ALLOCATE the new
 *   total (prompt + decoded_so_far + 1 new).  This all-or-nothing approach is
 *   simpler than an incremental "grow" and consistent with vLLM's block-manager
 *   semantics.  If the new allocation fails (no free blocks) we skip appending
 *   the token this tick and leave the request running — the scheduler's
 *   preemption will resolve the pressure on the next tick.
 *
 *   Automatic arrivals: auto-injection based on arrivalRatePerTick is deferred
 *   to a later increment.  Arrivals in Increment 1 come only from explicit
 *   ADD_REQUEST actions.  See TODO below.
 *
 *   Decoded token representation: token text is "t<index>" where index is the
 *   0-based position in decodedTokens.  Token id is promptTokens.length +
 *   decodedTokens.length (sequential across the full sequence).  Both are
 *   fully deterministic given the request state.
 */

import type { Action, Config, EngineState, Request } from "./types";
import { makeRng } from "./rng";
import { tokenize } from "./tokenizer";
import { createBlocks, free, allocate } from "./kvcache";
import { schedule } from "./scheduler";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default maximum number of decode tokens if ADD_REQUEST doesn't specify one. */
export const DEFAULT_MAX_DECODE = 8;

/** Fixed seed used by RESET to guarantee a reproducible start state. */
const RESET_SEED = 1;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

/**
 * Create a fresh EngineState.
 *
 * tick=0, no requests, full free-block pool, seeded rng, nextRequestId=0,
 * arrivalRatePerTick=0.
 */
export function init(config: Config, seed = 1): EngineState {
  return {
    tick: 0,
    requests: [],
    blocks: createBlocks(config.kvCacheBlocks, config.blockSize),
    rng: makeRng(seed),
    nextRequestId: 0,
    arrivalRatePerTick: 0,
  };
}

// ---------------------------------------------------------------------------
// reduce
// ---------------------------------------------------------------------------

/**
 * Pure action handler.  Returns a new EngineState; never mutates inputs.
 *
 * All structural simulation is delegated to tick() / schedule() / kvcache
 * helpers so that this function stays thin.
 */
export function reduce(
  state: EngineState,
  action: Action,
  config: Config
): EngineState {
  switch (action.type) {
    case "RESET":
      // Restart deterministically from fixed seed=1 regardless of prior history.
      return init(config, RESET_SEED);

    case "SET_CONFIG":
      // Config is merged and owned by the React layer; nothing to do here.
      // The scheduler's preemption enforces any capacity reduction on the next tick.
      return state;

    case "ADD_REQUEST": {
      const promptTokens = tokenize(action.prompt);
      const newRequest: Request = {
        id: state.nextRequestId,
        promptText: action.prompt,
        promptTokens,
        decodedTokens: [],
        maxDecode: action.maxDecode ?? DEFAULT_MAX_DECODE,
        status: "waiting",
        phase: "prefill",
        blockTable: [],
        arrivalTick: state.tick,
      };
      return {
        ...state,
        requests: [...state.requests, newRequest],
        nextRequestId: state.nextRequestId + 1,
      };
    }

    case "CANCEL_REQUEST": {
      const updatedBlocks = free(state.blocks, action.requestId);
      const updatedRequests = state.requests.map((r) => {
        if (r.id !== action.requestId) return r;
        // Only cancel if the request is in a cancellable status.
        if (
          r.status === "finished" ||
          r.status === "cancelled"
        ) {
          return r;
        }
        return { ...r, status: "cancelled" as const, blockTable: [] };
      });
      return { ...state, requests: updatedRequests, blocks: updatedBlocks };
    }

    case "SET_ARRIVAL_RATE":
      return { ...state, arrivalRatePerTick: action.ratePerTick };

    case "STEP":
      return tick(state, config);

    default:
      // TypeScript exhaustiveness guard — should never reach here.
      return state;
  }
}

// ---------------------------------------------------------------------------
// Decode-time preemption helper
// ---------------------------------------------------------------------------

interface DecodeVictim {
  id: number;
  /** True if the victim was already emitted into updatedRequests this tick. */
  alreadyEmitted: boolean;
}

/**
 * Pick the newest running request (highest arrivalTick, then highest id) to
 * preempt so an older starving request can allocate. Excludes the requester
 * itself and any already-preempted request. Considers both requests already
 * emitted this tick (in `updatedRequests`, still "running") and requests still
 * ahead in the loop (in `scheduled`). Newest-first preserves progress on older
 * requests. Returns null when no eligible victim remains.
 */
function pickNewestRunningVictim(
  requesterId: number,
  scheduled: Request[],
  updatedRequests: Request[],
  preemptedIds: Set<number>
): DecodeVictim | null {
  const emittedIds = new Set(updatedRequests.map((r) => r.id));
  const candidates: { req: Request; alreadyEmitted: boolean }[] = [];

  for (const r of updatedRequests) {
    if (r.id === requesterId || r.status !== "running") continue;
    if (r.blockTable.length === 0) continue; // nothing to free
    candidates.push({ req: r, alreadyEmitted: true });
  }
  for (const r of scheduled) {
    if (r.id === requesterId) continue;
    if (emittedIds.has(r.id)) continue; // counted above
    if (preemptedIds.has(r.id)) continue;
    if (r.status !== "running") continue;
    if (r.blockTable.length === 0) continue;
    candidates.push({ req: r, alreadyEmitted: false });
  }

  if (candidates.length === 0) return null;

  candidates.sort((a, b) => {
    if (a.req.arrivalTick !== b.req.arrivalTick) return b.req.arrivalTick - a.req.arrivalTick;
    return b.req.id - a.req.id;
  });

  return { id: candidates[0].req.id, alreadyEmitted: candidates[0].alreadyEmitted };
}

// ---------------------------------------------------------------------------
// tick
// ---------------------------------------------------------------------------

/**
 * Advance the simulation by one step.  Pure and deterministic.
 *
 * Steps:
 *   1. TODO (deferred): auto-inject arrivals based on arrivalRatePerTick.
 *      For Increment 1 all arrivals are explicit ADD_REQUEST actions.
 *      When implemented this must use state.rng (not Math.random) to
 *      preserve determinism.
 *   2. schedule() — admission + preemption.
 *   3. Advance each running request by one phase-step.
 *   4. Increment tick.
 */
export function tick(state: EngineState, config: Config): EngineState {
  // Step 1 – auto-arrival injection (DEFERRED — see TODO above).

  // Requests that were already running BEFORE this tick's scheduling pass. Used
  // to give prefill its own visible tick: a request admitted this tick stays in
  // prefill (prompt processed, blocks allocated, no token) and only transitions
  // to decode on the NEXT tick. Without this, prefill is consumed in the same
  // tick it's admitted and never observable in the UI (e.g. Stage 4's prefill
  // worker, Stage 1's prefill explainer).
  const previouslyRunning = new Set(
    state.requests.filter((r) => r.status === "running").map((r) => r.id)
  );

  // Step 2 – scheduling pass (admission + preemption).
  const scheduled = schedule(state, config);

  // Step 3 – advance running requests.
  // Block updates accumulate sequentially across requests, so we use an explicit
  // loop (not .map) to make the order dependency obvious: each request's block
  // reallocation must observe the prior request's changes to currentBlocks.
  let currentBlocks = scheduled.blocks;
  const updatedRequests: Request[] = [];
  // Requests preempted mid-tick to free blocks for an older request. When the
  // loop reaches one of these, it is emitted as "swapped" instead of decoding.
  const preemptedIds = new Set<number>();
  for (const req of scheduled.requests) {
    if (preemptedIds.has(req.id)) {
      // Preempted earlier this tick by a starving older request — swap it out.
      updatedRequests.push({ ...req, status: "swapped" as const, blockTable: [] });
      continue;
    }

    if (req.status !== "running") {
      updatedRequests.push(req);
      continue;
    }

    if (req.phase === "prefill" && !previouslyRunning.has(req.id)) {
      // Prefill phase for a request admitted THIS tick: the prompt is processed
      // in one pass and its KV is materialised in the allocated blocks. No token
      // is emitted, and the request stays in prefill for this committed step so
      // prefill is observable in the UI (Stage 4 prefill worker, Stage 1
      // explainer). It transitions to decode on the next tick (below).
      updatedRequests.push(req);
      continue;
    }

    // From here the request decodes one token. A request still tagged "prefill"
    // here was admitted on a PRIOR tick (its prefill step is complete), so we
    // flip it to decode and emit its first token in the same step — keeping
    // token/finish timing identical to a pure-decode request.
    let decoding: Request = req.phase === "prefill" ? { ...req, phase: "decode" } : req;

    // Decode phase: generate one new token.
    const newTokenIndex = decoding.decodedTokens.length; // 0-based position in decoded
    const newTokenId = req.promptTokens.length + newTokenIndex; // sequential global id
    const newToken = { id: newTokenId, text: `t${newTokenIndex}` };
    const newDecodedTokens = [...req.decodedTokens, newToken];

    // Record TTFT: the tick this request emits its very first token.
    if (decoding.firstTokenTick === undefined) {
      decoding = { ...decoding, firstTokenTick: scheduled.tick + 1 };
    }

    // Check if done.
    if (newDecodedTokens.length >= decoding.maxDecode) {
      // Finished — free blocks immediately.
      currentBlocks = free(currentBlocks, decoding.id);
      updatedRequests.push({
        ...decoding,
        decodedTokens: newDecodedTokens,
        status: "finished" as const,
        blockTable: [],
      });
      continue;
    }

    // Still decoding — reallocate blocks for the new total token count.
    // Strategy: free existing blocks then allocate the new total all-or-nothing.
    // This is simpler than a grow-only allocation and ensures the block table
    // is always accurate.
    // Sliding-window attention caps the live KV to the last `windowSize` tokens,
    // so the cache stops growing on long sequences (older blocks are evicted).
    const seqTokens = decoding.promptTokens.length + newDecodedTokens.length;
    const totalTokens = config.windowSize ? Math.min(seqTokens, config.windowSize) : seqTokens;
    let workingBlocks = free(currentBlocks, decoding.id);
    let allocResult = allocate(workingBlocks, decoding.id, totalTokens, config.blockSize);

    // Decode-time preemption: if the cache is full, evict the newest OTHER running
    // request to free blocks so this (older) request can grow. Without this the
    // batch deadlocks once every running request needs one more block but none is
    // free. Mirrors vLLM, which preempts to keep older requests progressing.
    // Evict newest-first (highest arrivalTick, then id) until the alloc succeeds.
    while (!allocResult.ok) {
      const victim = pickNewestRunningVictim(
        decoding.id,
        scheduled.requests,
        updatedRequests,
        preemptedIds
      );
      if (victim === null) break; // nobody left to evict
      workingBlocks = free(workingBlocks, victim.id);
      if (victim.alreadyEmitted) {
        // Victim was processed earlier this tick — rewrite its emitted entry.
        const idx = updatedRequests.findIndex((r) => r.id === victim.id);
        if (idx !== -1) {
          updatedRequests[idx] = {
            ...updatedRequests[idx],
            status: "swapped",
            blockTable: [],
          };
        }
      } else {
        // Victim is still ahead in the loop — mark it so it's swapped when reached.
        preemptedIds.add(victim.id);
      }
      allocResult = allocate(workingBlocks, decoding.id, totalTokens, config.blockSize);
    }

    if (!allocResult.ok) {
      // This request alone cannot grow to the next token — the cache is
      // permanently full for it. Finish it at the current decoded count and
      // record why, rather than looping forever in a freed-blocks state.
      // Commit workingBlocks so any victims freed above don't stay orphaned.
      currentBlocks = free(workingBlocks, decoding.id);
      const blocksNeededForNext = Math.ceil(
        (decoding.promptTokens.length + decoding.decodedTokens.length + 1) / config.blockSize
      );
      updatedRequests.push({
        ...decoding,
        status: "finished" as const,
        blockTable: [],
        rejectionReason: `Cache too small to continue: next token needs ${blocksNeededForNext} blocks but only ${config.kvCacheBlocks} exist. Generated ${decoding.decodedTokens.length} of ${decoding.maxDecode} tokens.`,
      });
      continue;
    }

    // Commit the reallocation.
    currentBlocks = allocResult.blocks;
    updatedRequests.push({
      ...decoding,
      decodedTokens: newDecodedTokens,
      blockTable: allocResult.blockTable,
    });
  }

  // Step 4 – increment tick.
  return {
    ...scheduled,
    requests: updatedRequests,
    blocks: currentBlocks,
    tick: scheduled.tick + 1,
  };
}
