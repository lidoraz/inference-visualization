import { describe, it, expect } from "vitest";
import { tokenize } from "../../src/engine/tokenizer";

describe("tokenize", () => {
  it("returns [] for empty string", () => {
    expect(tokenize("")).toEqual([]);
  });

  it("returns [] for whitespace-only string", () => {
    expect(tokenize("   ")).toEqual([]);
    expect(tokenize("\t\n  ")).toEqual([]);
  });

  it("does not emit whitespace as a token", () => {
    const tokens = tokenize("hi there");
    expect(tokens.every((t) => t.text.trim().length > 0)).toBe(true);
  });

  it("short word stays as one token", () => {
    const tokens = tokenize("the");
    expect(tokens).toHaveLength(1);
    expect(tokens[0]).toEqual({ id: 0, text: "the" });
  });

  it("long word splits into >=2 subword tokens", () => {
    // "quicker" (7 chars) → ["quic", "ker"] with MAX_CHARS=4
    const tokens = tokenize("quicker");
    expect(tokens.length).toBeGreaterThanOrEqual(2);
    expect(tokens.map((t) => t.text).join("")).toBe("quicker");
  });

  it("each subword chunk is at most MAX_CHARS characters", () => {
    const tokens = tokenize("extraordinarily");
    tokens.forEach((t) => expect(t.text.length).toBeLessThanOrEqual(4));
  });

  it("sequential ids start at 0 across multiple words", () => {
    const tokens = tokenize("the quick fox");
    expect(tokens.map((t) => t.id)).toEqual(
      tokens.map((_, i) => i)
    );
    expect(tokens[0].id).toBe(0);
  });

  it("reconstructed text equals original word characters (no chars dropped)", () => {
    // Joining all token texts (no separator) should give all word characters
    // concatenated in order. Whitespace between words is not preserved since
    // it is used only as a delimiter and not emitted as tokens.
    const tokens = tokenize("hello world");
    expect(tokens.map((t) => t.text).join("")).toBe("helloworld");
  });

  it("single long word splits ids sequentially", () => {
    // "quicker" → 2 tokens, ids 0 and 1
    const tokens = tokenize("quicker");
    expect(tokens[0].id).toBe(0);
    expect(tokens[1].id).toBe(1);
  });

  it("ids are sequential across word-boundary subwords", () => {
    // "ab cde" with MAX_CHARS=4: ["ab"(0), "cde"(1)]
    // "abcde fg" → ["abcd"(0), "e"(1), "fg"(2)]
    const tokens = tokenize("abcde fg");
    expect(tokens.map((t) => t.id)).toEqual([0, 1, 2]);
    expect(tokens.map((t) => t.text)).toEqual(["abcd", "e", "fg"]);
  });

  it("is deterministic — same input gives deep-equal results", () => {
    const input = "the quick brown fox jumps over the lazy dog";
    const first = tokenize(input);
    const second = tokenize(input);
    expect(first).toEqual(second);
  });
});
