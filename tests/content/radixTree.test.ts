import { describe, it, expect } from "vitest";
import {
  buildRadixTree,
  flatTokenCount,
  radixTokenCount,
  isShared,
  type RadixInput,
  type RadixNode,
} from "../../src/content/radixTree";

/** Collect every node (excluding root) for assertions. */
function allNodes(root: RadixNode): RadixNode[] {
  const out: RadixNode[] = [];
  const walk = (n: RadixNode) => {
    out.push(n);
    n.children.forEach(walk);
  };
  root.children.forEach(walk);
  return out;
}

describe("buildRadixTree", () => {
  it("single request becomes one leaf holding the whole sequence", () => {
    const inputs: RadixInput[] = [{ requestId: 0, tokens: ["a", "b", "c"] }];
    const root = buildRadixTree(inputs);
    expect(root.children).toHaveLength(1);
    expect(root.children[0].segment).toEqual(["a", "b", "c"]);
    expect(root.children[0].requestIds).toEqual([0]);
  });

  it("two requests with a shared prefix share one node, branching at divergence", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["sys", "you", "are", "X"] },
      { requestId: 1, tokens: ["sys", "you", "are", "Y"] },
    ];
    const root = buildRadixTree(inputs);
    // Root has a single branch (the shared "sys you are").
    expect(root.children).toHaveLength(1);
    const shared = root.children[0];
    expect(shared.segment).toEqual(["sys", "you", "are"]);
    expect(shared.requestIds.sort()).toEqual([0, 1]);
    expect(isShared(shared)).toBe(true);
    // It branches into the two distinct tails.
    expect(shared.children).toHaveLength(2);
    const tails = shared.children.map((c) => c.segment).sort();
    expect(tails).toEqual([["X"], ["Y"]]);
  });

  it("identical requests fully share the path (no branching)", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["a", "b"] },
      { requestId: 1, tokens: ["a", "b"] },
    ];
    const root = buildRadixTree(inputs);
    expect(root.children).toHaveLength(1);
    const node = root.children[0];
    expect(node.segment).toEqual(["a", "b"]);
    expect(node.requestIds.sort()).toEqual([0, 1]);
    expect(node.children).toHaveLength(0);
  });

  it("disjoint requests produce separate top-level branches", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["a", "b"] },
      { requestId: 1, tokens: ["x", "y"] },
    ];
    const root = buildRadixTree(inputs);
    expect(root.children).toHaveLength(2);
    expect(allNodes(root).every((n) => n.requestIds.length === 1)).toBe(true);
  });

  it("splits an existing leaf when a later request shares only part of it", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["a", "b", "c", "d"] },
      { requestId: 1, tokens: ["a", "b", "z"] },
    ];
    const root = buildRadixTree(inputs);
    const shared = root.children[0];
    expect(shared.segment).toEqual(["a", "b"]);
    expect(shared.requestIds.sort()).toEqual([0, 1]);
    const tails = shared.children.map((c) => c.segment).sort();
    expect(tails).toEqual([["c", "d"], ["z"]]);
  });

  it("is deterministic: same input yields structurally equal trees", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["sys", "hello"] },
      { requestId: 1, tokens: ["sys", "world"] },
    ];
    expect(JSON.stringify(buildRadixTree(inputs))).toBe(
      JSON.stringify(buildRadixTree(inputs))
    );
  });
});

describe("token accounting", () => {
  it("flatTokenCount sums all sequence lengths (vLLM-style duplication)", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["a", "b", "c"] },
      { requestId: 1, tokens: ["a", "b", "d"] },
    ];
    expect(flatTokenCount(inputs)).toBe(6);
  });

  it("radixTokenCount counts shared segments once (SGLang dedup)", () => {
    const inputs: RadixInput[] = [
      { requestId: 0, tokens: ["a", "b", "c"] },
      { requestId: 1, tokens: ["a", "b", "d"] },
    ];
    const root = buildRadixTree(inputs);
    // Shared "a b" (2) + tail "c" (1) + tail "d" (1) = 4, vs flat 6.
    expect(radixTokenCount(root)).toBe(4);
    expect(radixTokenCount(root)).toBeLessThan(flatTokenCount(inputs));
  });
});
