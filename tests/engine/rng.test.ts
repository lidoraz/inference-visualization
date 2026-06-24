import { describe, it, expect } from "vitest";
import { makeRng } from "../../src/engine/rng";

describe("rng", () => {
  it("is deterministic for a given seed", () => {
    const a = makeRng(42); const b = makeRng(42);
    expect([a.next(), a.next(), a.next()]).toEqual([b.next(), b.next(), b.next()]);
  });
  it("advances state (carried, not global)", () => {
    const r = makeRng(1); const first = r.next();
    expect(r.next()).not.toBe(first);
  });
  it("returns floats in [0,1)", () => {
    const r = makeRng(7);
    for (let i=0;i<100;i++){ const v=r.next(); expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1); }
  });
});
