import { describe, it, expect } from "vitest";
import { softmax, computeDistribution, CANDIDATES } from "./sampling";

describe("softmax", () => {
  it("returns probabilities that sum to 1 at normal temperature", () => {
    const logits = [1, 2, 3];
    const probs = softmax(logits, 1.0);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    expect(probs.length).toBe(3);
  });

  it("higher logit → higher probability", () => {
    const probs = softmax([1, 2, 3], 1.0);
    expect(probs[2]).toBeGreaterThan(probs[1]);
    expect(probs[1]).toBeGreaterThan(probs[0]);
  });

  it("temperature 0 puts all mass on the max", () => {
    const probs = softmax([1, 5, 2], 0);
    expect(probs[1]).toBe(1);
    expect(probs[0]).toBe(0);
    expect(probs[2]).toBe(0);
  });

  it("temperature 0.01 is treated as greedy", () => {
    const probs = softmax([1, 5, 2], 0.01);
    expect(probs[1]).toBe(1);
  });

  it("high temperature flattens the distribution", () => {
    const low = softmax([0, 5], 0.1);
    const high = softmax([0, 5], 10);
    // With high temp the gap shrinks
    expect(high[1] - high[0]).toBeLessThan(low[1] - low[0]);
  });

  it("is numerically stable for large logits", () => {
    const probs = softmax([1000, 1001], 1.0);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
  });
});

describe("computeDistribution", () => {
  it("returns an entry for every candidate", () => {
    const dist = computeDistribution(1, CANDIDATES.length, 1.0);
    expect(dist.length).toBe(CANDIDATES.length);
  });

  it("kept probabilities renormalize to 1", () => {
    const dist = computeDistribution(1, CANDIDATES.length, 1.0);
    const keptSum = dist.filter((d) => d.kept).reduce((a, d) => a + d.prob, 0);
    expect(keptSum).toBeCloseTo(1, 5);
  });

  it("top-k = 1 keeps only the highest-logit token", () => {
    const dist = computeDistribution(1, 1, 1.0);
    const kept = dist.filter((d) => d.kept);
    expect(kept.length).toBe(1);
    expect(kept[0].text).toBe("the"); // highest logit in CANDIDATES
    expect(kept[0].prob).toBeCloseTo(1, 5);
  });

  it("top-k restricts to at most k tokens", () => {
    const k = 3;
    const dist = computeDistribution(1, k, 1.0);
    expect(dist.filter((d) => d.kept).length).toBeLessThanOrEqual(k);
  });

  it("top-p = 0.1 keeps only the top token(s) covering 10% mass", () => {
    // With temperature 1 and the given logits, 'the' already holds ~30%+ alone,
    // so top-p 0.1 should keep just 1 token.
    const dist = computeDistribution(1, CANDIDATES.length, 0.1);
    expect(dist.filter((d) => d.kept).length).toBe(1);
  });

  it("top-p = 1 keeps all tokens (no nucleus cutoff)", () => {
    const dist = computeDistribution(1, CANDIDATES.length, 1.0);
    expect(dist.every((d) => d.kept)).toBe(true);
  });

  it("excluded tokens have prob 0", () => {
    const dist = computeDistribution(1, 1, 1.0);
    dist.filter((d) => !d.kept).forEach((d) => expect(d.prob).toBe(0));
  });

  it("greedy (temperature 0) puts all mass on 'the' (highest logit)", () => {
    const dist = computeDistribution(0, CANDIDATES.length, 1.0);
    const kept = dist.filter((d) => d.kept);
    expect(kept.length).toBe(1);
    expect(kept[0].text).toBe("the");
    expect(kept[0].prob).toBeCloseTo(1, 5);
  });

  it("top-k and top-p together use the more restrictive bound", () => {
    // k=8 (all), p=0.1 → nucleus wins and keeps only 1
    const distP = computeDistribution(1, CANDIDATES.length, 0.1);
    // k=1, p=1.0 → top-k wins and keeps only 1
    const distK = computeDistribution(1, 1, 1.0);
    expect(distP.filter((d) => d.kept).length).toBe(1);
    expect(distK.filter((d) => d.kept).length).toBe(1);
  });

  it("accepts custom candidates", () => {
    const custom = [
      { text: "yes", logit: 2 },
      { text: "no", logit: 1 },
    ];
    const dist = computeDistribution(1, 2, 1.0, custom);
    expect(dist.length).toBe(2);
    expect(dist.every((d) => d.kept)).toBe(true);
  });
});
