import { describe, it, expect } from "vitest";
import {
  createBlocks,
  blocksNeeded,
  allocate,
  free,
  usage,
} from "../../src/engine/kvcache";
import type { Block } from "../../src/engine/types";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function freshPool(numBlocks: number, blockSize: number): Block[] {
  return createBlocks(numBlocks, blockSize);
}

// ---------------------------------------------------------------------------
// createBlocks
// ---------------------------------------------------------------------------

describe("createBlocks", () => {
  it("produces the requested number of blocks", () => {
    const blocks = createBlocks(8, 16);
    expect(blocks).toHaveLength(8);
  });

  it("assigns sequential ids starting at 0", () => {
    const blocks = createBlocks(4, 16);
    expect(blocks.map((b) => b.id)).toEqual([0, 1, 2, 3]);
  });

  it("initialises every block as free", () => {
    const blocks = createBlocks(4, 16);
    for (const b of blocks) {
      expect(b.requestId).toBeNull();
      expect(b.usedSlots).toBe(0);
      expect(b.tokenSlots).toBe(16);
    }
  });

  it("uses the supplied blockSize as tokenSlots", () => {
    const blocks = createBlocks(3, 32);
    for (const b of blocks) {
      expect(b.tokenSlots).toBe(32);
    }
  });
});

// ---------------------------------------------------------------------------
// blocksNeeded
// ---------------------------------------------------------------------------

describe("blocksNeeded", () => {
  it("exact multiple -> no extra block", () => {
    expect(blocksNeeded(16, 16)).toBe(1);
    expect(blocksNeeded(32, 16)).toBe(2);
  });

  it("partial fill -> rounds up", () => {
    expect(blocksNeeded(17, 16)).toBe(2);
    expect(blocksNeeded(1, 16)).toBe(1);
  });

  it("zero tokens -> zero blocks", () => {
    expect(blocksNeeded(0, 16)).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// allocate
// ---------------------------------------------------------------------------

describe("allocate", () => {
  it("claims ceil(tokens/blockSize) blocks", () => {
    const blocks = freshPool(8, 16);
    const result = allocate(blocks, 42, 25, 16); // needs ceil(25/16)=2
    expect(result.ok).toBe(true);
    expect(result.blockTable).toHaveLength(2);
    const claimed = result.blocks.filter((b) => b.requestId === 42);
    expect(claimed).toHaveLength(2);
  });

  it("builds a valid blockTable (logical index -> physical block id)", () => {
    const blocks = freshPool(8, 16);
    const result = allocate(blocks, 1, 16, 16);
    expect(result.ok).toBe(true);
    expect(result.blockTable).toHaveLength(1);
    const physId = result.blockTable[0];
    const physBlock = result.blocks.find((b) => b.id === physId);
    expect(physBlock).toBeDefined();
    expect(physBlock!.requestId).toBe(1);
  });

  it("fully fills all blocks except the last when tokens are an exact multiple", () => {
    const blocks = freshPool(8, 16);
    const result = allocate(blocks, 7, 32, 16); // 2 full blocks
    expect(result.ok).toBe(true);
    const claimed = result.blocks.filter((b) => b.requestId === 7);
    expect(claimed).toHaveLength(2);
    for (const b of claimed) {
      expect(b.usedSlots).toBe(16);
    }
  });

  it("last block is partially filled when tokens not a multiple of blockSize", () => {
    const blocks = freshPool(8, 16);
    const result = allocate(blocks, 3, 20, 16); // 16 full + 4 partial
    expect(result.ok).toBe(true);
    const claimed = result.blocks.filter((b) => b.requestId === 3);
    expect(claimed).toHaveLength(2);

    // Find the blocks in blockTable order so we can check last
    const lastBlockId = result.blockTable[result.blockTable.length - 1];
    const lastBlock = result.blocks.find((b) => b.id === lastBlockId)!;
    const firstBlockId = result.blockTable[0];
    const firstBlock = result.blocks.find((b) => b.id === firstBlockId)!;

    expect(firstBlock.usedSlots).toBe(16);
    expect(lastBlock.usedSlots).toBe(4);
  });

  it("returns ok:false when not enough free blocks", () => {
    const blocks = freshPool(2, 16);
    const result = allocate(blocks, 5, 50, 16); // needs 4 blocks, only 2 free
    expect(result.ok).toBe(false);
  });

  it("leaves blocks completely unchanged on failure (all-or-nothing)", () => {
    const blocks = freshPool(2, 16);
    const result = allocate(blocks, 5, 50, 16);
    expect(result.ok).toBe(false);
    // Every block should still be free
    for (const b of result.blocks) {
      expect(b.requestId).toBeNull();
      expect(b.usedSlots).toBe(0);
    }
    // The returned blocks array should be the original (or structurally identical)
    expect(result.blocks).toEqual(blocks);
  });

  it("returns an empty blockTable on failure", () => {
    const blocks = freshPool(1, 16);
    const result = allocate(blocks, 9, 50, 16);
    expect(result.ok).toBe(false);
    expect(result.blockTable).toEqual([]);
  });

  it("does NOT mutate the input blocks array", () => {
    const blocks = freshPool(8, 16);
    const snapshot = JSON.stringify(blocks);
    allocate(blocks, 1, 32, 16);
    expect(JSON.stringify(blocks)).toBe(snapshot);
  });

  it("blockTable entries are valid physical block ids present in returned blocks", () => {
    const blocks = freshPool(8, 16);
    const result = allocate(blocks, 2, 35, 16); // 3 blocks
    expect(result.ok).toBe(true);
    const ids = result.blocks.map((b) => b.id);
    for (const physId of result.blockTable) {
      expect(ids).toContain(physId);
    }
  });
});

// ---------------------------------------------------------------------------
// free
// ---------------------------------------------------------------------------

describe("free", () => {
  it("returns a request's blocks to the free pool", () => {
    const blocks = freshPool(8, 16);
    const { blocks: afterAlloc } = allocate(blocks, 10, 32, 16);
    const afterFree = free(afterAlloc, 10);
    for (const b of afterFree) {
      expect(b.requestId).toBeNull();
      expect(b.usedSlots).toBe(0);
    }
  });

  it("only frees blocks belonging to the specified requestId", () => {
    const blocks = freshPool(8, 16);
    const { blocks: after1 } = allocate(blocks, 11, 16, 16);
    const { blocks: after2 } = allocate(after1, 12, 16, 16);
    const afterFree = free(after2, 11);

    const req12Blocks = afterFree.filter((b) => b.requestId === 12);
    expect(req12Blocks).toHaveLength(1);

    const req11Blocks = afterFree.filter((b) => b.requestId === 11);
    expect(req11Blocks).toHaveLength(0);
  });

  it("does NOT mutate the input blocks array", () => {
    const blocks = freshPool(8, 16);
    const { blocks: afterAlloc } = allocate(blocks, 20, 32, 16);
    const snapshot = JSON.stringify(afterAlloc);
    free(afterAlloc, 20);
    expect(JSON.stringify(afterAlloc)).toBe(snapshot);
  });

  it("is a no-op for a requestId that owns no blocks", () => {
    const blocks = freshPool(4, 16);
    const afterFree = free(blocks, 99);
    expect(afterFree).toEqual(blocks);
  });
});

// ---------------------------------------------------------------------------
// usage
// ---------------------------------------------------------------------------

describe("usage", () => {
  it("reports zero used blocks for a fresh pool", () => {
    const blocks = freshPool(8, 16);
    const { used, total } = usage(blocks);
    expect(used).toBe(0);
    expect(total).toBe(8);
  });

  it("reports correct used count after allocation", () => {
    const blocks = freshPool(8, 16);
    const { blocks: after } = allocate(blocks, 1, 48, 16); // 3 blocks
    const { used, total } = usage(after);
    expect(used).toBe(3);
    expect(total).toBe(8);
  });

  it("reports correct used count after free", () => {
    const blocks = freshPool(8, 16);
    const { blocks: after } = allocate(blocks, 1, 48, 16);
    const freed = free(after, 1);
    const { used, total } = usage(freed);
    expect(used).toBe(0);
    expect(total).toBe(8);
  });

  it("total equals the full pool size regardless of allocation state", () => {
    const blocks = freshPool(10, 8);
    const { total } = usage(blocks);
    expect(total).toBe(10);
  });
});
