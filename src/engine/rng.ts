import type { Rng } from "./types";

/**
 * Mulberry32 seeded PRNG.
 * All state is carried on the returned object — no global/module-level mutable
 * state, no Math.random(). Two RNGs created with the same seed are guaranteed
 * to produce identical sequences (determinism requirement).
 */
export function makeRng(seed: number): Rng {
  let state = seed >>> 0; // coerce to uint32

  return {
    // Returns CURRENT internal state (advances with next()), so an EngineState
    // can be snapshotted/restored faithfully via makeRng(state.rng.seed).
    // Caveat: a shallow copy of EngineState shares this closure — to branch the
    // simulation, rebuild rng with makeRng(seed). Moot in Increment 1 since
    // next() is never called (auto-arrivals deferred).
    get seed() {
      return state;
    },
    next(): number {
      // mulberry32 step
      state = (state + 0x6d2b79f5) >>> 0;
      let z = state;
      z = Math.imul(z ^ (z >>> 15), z | 1);
      z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
      z = ((z ^ (z >>> 14)) >>> 0);
      return z / 0x100000000;
    },
  };
}
