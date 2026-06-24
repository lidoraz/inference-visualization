/**
 * Latency / throughput metrics derived from engine request state.
 *
 * Pure, React-free — extracted from the Stage 3 view so it can be unit-tested.
 * Ticks are the demo's time unit:
 *   TTFT       = firstTokenTick − arrivalTick, averaged over requests that have
 *                produced their first token.
 *   ITL        = decode ticks since first token / tokens produced after it,
 *                averaged over requests with >1 decoded token.
 *   Throughput = total decoded tokens so far / elapsed ticks.
 */

import type { Request } from "../engine/types";

export interface LatencyMetrics {
  avgTtft: number | null;
  avgItl: number | null;
  throughput: number;
}

export function latencyMetrics(requests: Request[], tick: number): LatencyMetrics {
  const started = requests.filter((r) => r.firstTokenTick !== undefined);
  let ttftSum = 0;
  let itlSum = 0;
  let itlCount = 0;
  let totalDecoded = 0;
  for (const r of requests) {
    totalDecoded += r.decodedTokens.length;
  }
  for (const r of started) {
    ttftSum += (r.firstTokenTick as number) - r.arrivalTick;
    // Decode ticks elapsed since the first token, vs tokens produced after it.
    const decodeTicks = tick - (r.firstTokenTick as number);
    const tokensAfterFirst = r.decodedTokens.length - 1;
    if (tokensAfterFirst > 0 && decodeTicks > 0) {
      itlSum += decodeTicks / tokensAfterFirst;
      itlCount += 1;
    }
  }
  return {
    avgTtft: started.length > 0 ? ttftSum / started.length : null,
    avgItl: itlCount > 0 ? itlSum / itlCount : null,
    throughput: tick > 0 ? totalDecoded / tick : 0,
  };
}
