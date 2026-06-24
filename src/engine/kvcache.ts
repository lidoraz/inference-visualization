/**
 * Paged KV-cache block management for the vLLM Inference Visualizer.
 *
 * All functions are pure — inputs are never mutated; updated copies are
 * returned. This makes the engine deterministic and the functions trivially
 * unit-testable in isolation.
 */

import type { Block } from "./types";

// ---------------------------------------------------------------------------
// createBlocks
// ---------------------------------------------------------------------------

/**
 * Allocate a fresh free pool of `numBlocks` blocks, each with capacity
 * `blockSize` token slots.
 */
export function createBlocks(numBlocks: number, blockSize: number): Block[] {
  const blocks: Block[] = [];
  for (let i = 0; i < numBlocks; i++) {
    blocks.push({ id: i, requestId: null, tokenSlots: blockSize, usedSlots: 0 });
  }
  return blocks;
}

// ---------------------------------------------------------------------------
// blocksNeeded
// ---------------------------------------------------------------------------

/**
 * How many blocks are required to hold `numTokens` tokens given `blockSize`
 * slots per block?  Uses ceiling division so partial blocks are counted.
 */
export function blocksNeeded(numTokens: number, blockSize: number): number {
  if (numTokens === 0) return 0;
  return Math.ceil(numTokens / blockSize);
}

// ---------------------------------------------------------------------------
// allocate
// ---------------------------------------------------------------------------

export interface AllocateResult {
  blocks: Block[];
  blockTable: number[];
  ok: boolean;
}

/**
 * Claim `blocksNeeded(numTokens, blockSize)` free blocks for `requestId`.
 *
 * - Fills every claimed block fully except the last, which receives only the
 *   remainder tokens (partial fill).
 * - If there are not enough free blocks, returns `ok: false` with the
 *   original blocks array unchanged and an empty blockTable.  All-or-nothing:
 *   nothing is allocated on failure.
 * - Never mutates the input array.
 */
export function allocate(
  blocks: Block[],
  requestId: number,
  numTokens: number,
  blockSize: number
): AllocateResult {
  const needed = blocksNeeded(numTokens, blockSize);
  const freeBlocks = blocks.filter((b) => b.requestId === null);

  if (freeBlocks.length < needed) {
    return { blocks, blockTable: [], ok: false };
  }

  const chosen = freeBlocks.slice(0, needed);
  const chosenIds = new Set(chosen.map((b) => b.id));
  const blockTable: number[] = chosen.map((b) => b.id);

  const updatedBlocks = blocks.map((b) => {
    if (!chosenIds.has(b.id)) return b;

    // Determine how many token slots this block uses.
    const logicalIndex = blockTable.indexOf(b.id);
    const isLast = logicalIndex === needed - 1;
    const remainder = numTokens % blockSize;
    const usedSlots = isLast && remainder !== 0 ? remainder : blockSize;

    return { ...b, requestId, usedSlots };
  });

  return { blocks: updatedBlocks, blockTable, ok: true };
}

// ---------------------------------------------------------------------------
// free
// ---------------------------------------------------------------------------

/**
 * Return all blocks owned by `requestId` to the free pool.
 * Only the specified request's blocks are touched; all other blocks are
 * returned unchanged.  Never mutates the input array.
 */
export function free(blocks: Block[], requestId: number): Block[] {
  return blocks.map((b) =>
    b.requestId === requestId
      ? { ...b, requestId: null, usedSlots: 0 }
      : b
  );
}

// ---------------------------------------------------------------------------
// usage
// ---------------------------------------------------------------------------

export interface UsageResult {
  used: number;
  total: number;
}

/**
 * Count how many blocks are currently allocated (requestId != null) vs the
 * total pool size.
 */
export function usage(blocks: Block[]): UsageResult {
  const used = blocks.filter((b) => b.requestId !== null).length;
  return { used, total: blocks.length };
}
