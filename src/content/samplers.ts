/**
 * UI-only random samplers for the load generator and random-decode button.
 *
 * These use Math.random by design — they stand in for user input (request
 * arrivals and decode lengths), exactly like randomSentence(). The simulation
 * engine stays deterministic; randomness only enters where a human otherwise
 * would (typing a prompt, clicking Add). Never import these into src/engine.
 */

/**
 * Draw an inter-arrival delay (ms) from an exponential distribution with the
 * given mean. Exponential gaps make arrivals a Poisson process — the standard
 * model for independent request traffic, giving natural bursts and lulls.
 */
export function exponentialDelay(meanMs: number): number {
  // Inverse-CDF sampling: -mean * ln(U), U in (0,1].
  const u = 1 - Math.random(); // shift to (0,1] so ln is finite
  return -meanMs * Math.log(u);
}

/**
 * Draw a decode length from a normal distribution, rounded and clamped to a
 * minimum. Uses the Box–Muller transform.
 */
export function normalDecodeLength(mean: number, sd: number, min = 1): number {
  const u1 = 1 - Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return Math.max(min, Math.round(mean + sd * z));
}
