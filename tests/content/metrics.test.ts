import { describe, it, expect } from "vitest";
import { latencyMetrics } from "../../src/content/metrics";
import type { Request } from "../../src/engine/types";

/** Minimal Request factory for metrics tests (only the fields metrics reads). */
function req(overrides: Partial<Request> = {}): Request {
  return {
    id: 0,
    promptTokens: [],
    decodedTokens: [],
    maxDecode: 20,
    status: "running",
    phase: "decode",
    blockTable: [],
    arrivalTick: 0,
    ...overrides,
  };
}

function tokens(n: number) {
  return Array.from({ length: n }, (_, i) => ({ id: i, text: `t${i}` }));
}

describe("latencyMetrics", () => {
  it("returns nulls/zero when no request has started", () => {
    const m = latencyMetrics([req()], 5);
    expect(m.avgTtft).toBeNull();
    expect(m.avgItl).toBeNull();
    expect(m.throughput).toBe(0); // tick is non-zero but no decoded tokens
  });

  it("TTFT = firstTokenTick − arrivalTick, averaged over started requests", () => {
    const m = latencyMetrics(
      [
        req({ id: 0, arrivalTick: 0, firstTokenTick: 2, decodedTokens: tokens(1) }),
        req({ id: 1, arrivalTick: 1, firstTokenTick: 5, decodedTokens: tokens(1) }),
      ],
      6
    );
    // (2-0) and (5-1) => average of 2 and 4 = 3.
    expect(m.avgTtft).toBe(3);
  });

  it("ignores not-yet-started requests in the TTFT average", () => {
    const m = latencyMetrics(
      [
        req({ id: 0, arrivalTick: 0, firstTokenTick: 4, decodedTokens: tokens(1) }),
        req({ id: 1, arrivalTick: 0 }), // never produced a token
      ],
      6
    );
    expect(m.avgTtft).toBe(4);
  });

  it("ITL = decode ticks since first token / tokens after the first", () => {
    // first token at tick 2, now tick 8 => 6 decode ticks; 4 decoded tokens =>
    // 3 tokens after the first => 6/3 = 2 ticks/token.
    const m = latencyMetrics(
      [req({ arrivalTick: 0, firstTokenTick: 2, decodedTokens: tokens(4) })],
      8
    );
    expect(m.avgItl).toBe(2);
  });

  it("ITL is null until a request has more than one decoded token", () => {
    const m = latencyMetrics(
      [req({ arrivalTick: 0, firstTokenTick: 2, decodedTokens: tokens(1) })],
      5
    );
    expect(m.avgItl).toBeNull();
  });

  it("throughput = total decoded tokens / tick", () => {
    const m = latencyMetrics(
      [
        req({ id: 0, decodedTokens: tokens(3) }),
        req({ id: 1, decodedTokens: tokens(5) }),
      ],
      4
    );
    expect(m.throughput).toBe(2); // (3 + 5) / 4
  });

  it("throughput is 0 at tick 0 (avoids divide-by-zero)", () => {
    const m = latencyMetrics([req({ decodedTokens: tokens(3) })], 0);
    expect(m.throughput).toBe(0);
  });
});
