import { describe, it, expect } from "vitest";
import { softmax, computeDistribution, CANDIDATES } from "../../src/content/sampling";

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

describe("softmax", () => {
  it("produces a normalized distribution", () => {
    const p = softmax([1, 2, 3], 1);
    expect(sum(p)).toBeCloseTo(1, 6);
    expect(p.every((x) => x >= 0)).toBe(true);
  });

  it("is monotonic in the logits (higher logit → higher prob)", () => {
    const p = softmax([1, 2, 3], 1);
    expect(p[2]).toBeGreaterThan(p[1]);
    expect(p[1]).toBeGreaterThan(p[0]);
  });

  it("temperature 0 collapses to greedy (all mass on the argmax)", () => {
    const p = softmax([1, 3, 2], 0);
    expect(p).toEqual([0, 1, 0]);
  });

  it("lower temperature sharpens, higher flattens", () => {
    const sharp = softmax([1, 2, 3], 0.5);
    const flat = softmax([1, 2, 3], 2);
    // The top token holds more mass when sharper.
    expect(sharp[2]).toBeGreaterThan(flat[2]);
  });
});

describe("computeDistribution", () => {
  const N = CANDIDATES.length;

  it("renormalizes kept probabilities to sum to 1", () => {
    const dist = computeDistribution(1, N, 1);
    expect(sum(dist.map((d) => d.prob))).toBeCloseTo(1, 6);
  });

  it("top-k keeps exactly k tokens", () => {
    const dist = computeDistribution(1, 3, 1);
    expect(dist.filter((d) => d.kept)).toHaveLength(3);
    // Excluded tokens carry zero probability.
    expect(dist.filter((d) => !d.kept).every((d) => d.prob === 0)).toBe(true);
  });

  it("top-p keeps the smallest nucleus reaching p", () => {
    // Very small p keeps just the top token; loose top-k so p is the binding cut.
    const dist = computeDistribution(1, N, 0.01);
    expect(dist.filter((d) => d.kept)).toHaveLength(1);
  });

  it("greedy (temperature 0) keeps a single token at prob 1", () => {
    const dist = computeDistribution(0, N, 1);
    const kept = dist.filter((d) => d.kept);
    expect(kept).toHaveLength(1);
    expect(kept[0].prob).toBeCloseTo(1, 6);
  });

  it("is deterministic for the same inputs", () => {
    expect(JSON.stringify(computeDistribution(1.3, 5, 0.9))).toBe(
      JSON.stringify(computeDistribution(1.3, 5, 0.9))
    );
  });
});
