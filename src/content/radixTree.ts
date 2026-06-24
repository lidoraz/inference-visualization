/**
 * Radix-tree builder for the SGLang prefix-cache visualization (Stage 7).
 *
 * Pure, deterministic UI helper — no engine state, no Math.random. Given a set
 * of labeled token sequences (requests), it builds a compressed trie (radix
 * tree) where sequences sharing a prefix share one path and branch at the first
 * differing token. This is exactly how SGLang's RadixAttention reuses KV across
 * requests with a common prefix.
 *
 * The tree is built over token *text* (the demo tokenizer is deterministic, so
 * identical prompts produce identical token streams). Each node carries the
 * token segment on the edge leading into it plus the ids of every request whose
 * path passes through it, so the UI can highlight shared vs. unique segments.
 */

export interface RadixNode {
  /** Stable id for React keys (assigned in build order). */
  id: number;
  /** The run of token texts on the edge leading into this node (compressed). */
  segment: string[];
  /** Ids of requests whose path passes through this node. */
  requestIds: number[];
  children: RadixNode[];
}

export interface RadixInput {
  requestId: number;
  tokens: string[];
}

/** Length of the common leading run between two token arrays. */
function commonPrefixLen(a: string[], b: string[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

/**
 * Build a radix tree from a list of tokenized requests. Returns a root node
 * whose `segment` is empty and whose children are the distinct first-token
 * branches. Insertion order follows the input order for determinism.
 */
export function buildRadixTree(inputs: RadixInput[]): RadixNode {
  let nextId = 0;
  const root: RadixNode = { id: nextId++, segment: [], requestIds: [], children: [] };

  function makeNode(segment: string[], requestId: number): RadixNode {
    return { id: nextId++, segment, requestIds: [requestId], children: [] };
  }

  for (const { requestId, tokens } of inputs) {
    root.requestIds.push(requestId);
    let node = root;
    let pos = 0; // index into tokens not yet matched

    while (pos < tokens.length) {
      const rest = tokens.slice(pos);
      // Find a child whose segment starts with the same token.
      const child = node.children.find((c) => c.segment[0] === rest[0]);

      if (!child) {
        // No matching branch — attach the remainder as a new leaf.
        node.children.push(makeNode(rest, requestId));
        break;
      }

      const shared = commonPrefixLen(child.segment, rest);

      if (shared === child.segment.length) {
        // Entire child segment is shared — descend and continue matching.
        child.requestIds.push(requestId);
        node = child;
        pos += shared;
        continue;
      }

      // Partial overlap — split the child at the divergence point.
      const splitNode: RadixNode = {
        id: nextId++,
        segment: child.segment.slice(0, shared),
        requestIds: [...child.requestIds, requestId],
        children: [],
      };
      // The existing child keeps its tail and its original requestIds.
      child.segment = child.segment.slice(shared);
      splitNode.children.push(child);
      // The new request's diverging tail becomes a sibling under the split.
      const tail = rest.slice(shared);
      if (tail.length > 0) {
        splitNode.children.push(makeNode(tail, requestId));
      }
      // Replace child with splitNode in the parent's children list.
      const idx = node.children.indexOf(child);
      node.children[idx] = splitNode;
      break;
    }
  }

  return root;
}

/** True when more than one request shares this node (a reused prefix). */
export function isShared(node: RadixNode): boolean {
  return node.requestIds.length > 1;
}

/**
 * Total token slots a flat (vLLM-style) cache would store if every request kept
 * its own copy — i.e. the sum of all sequence lengths. Used to contrast against
 * the radix tree's deduplicated total.
 */
export function flatTokenCount(inputs: RadixInput[]): number {
  return inputs.reduce((sum, r) => sum + r.tokens.length, 0);
}

/** Total token slots the radix tree stores (shared segments counted once). */
export function radixTokenCount(root: RadixNode): number {
  let total = 0;
  const walk = (n: RadixNode) => {
    total += n.segment.length;
    n.children.forEach(walk);
  };
  walk(root);
  return total;
}
