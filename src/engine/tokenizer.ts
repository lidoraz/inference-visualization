/**
 * Naive word/subword tokenizer for the vLLM Inference Visualizer.
 *
 * NOT a real BPE — purely for visual teaching purposes.
 * Splits on whitespace, then chunks each word into pieces of at most
 * MAX_CHARS characters so long words visibly become multiple tokens.
 */

import type { Token } from "./types";

/** Maximum characters per subword chunk. */
const MAX_CHARS = 4;

/**
 * Tokenize a string into an array of Token objects.
 *
 * Rules:
 * - Whitespace is used as a delimiter only; no whitespace tokens are emitted.
 * - Each word is chunked into subword pieces of at most MAX_CHARS characters.
 * - Token ids are assigned sequentially starting at 0 across the whole sequence.
 * - Empty / whitespace-only input returns [].
 */
export function tokenize(text: string): Token[] {
  const words = text.split(/\s+/).filter((w) => w.length > 0);
  const tokens: Token[] = [];

  for (const word of words) {
    for (let start = 0; start < word.length; start += MAX_CHARS) {
      const piece = word.slice(start, start + MAX_CHARS);
      tokens.push({ id: tokens.length, text: piece });
    }
  }

  return tokens;
}
