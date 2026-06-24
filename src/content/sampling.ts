/**
 * Illustrative next-token sampling math (Stage 2 Model Concepts).
 *
 * Pure, React-free — extracted from SamplingView so it can be unit-tested. A
 * fixed set of candidate tokens with base logits is reshaped by temperature,
 * then truncated by top-k / top-p, then renormalized over the kept set.
 */

export interface Candidate {
  text: string;
  logit: number;
}

export interface Cand {
  text: string;
  prob: number;
  kept: boolean;
}

// A plausible peaked distribution of next-token candidates (logits).
export const CANDIDATES: Candidate[] = [
  { text: "the", logit: 3.2 },
  { text: "a", logit: 2.6 },
  { text: "cache", logit: 2.1 },
  { text: "memory", logit: 1.7 },
  { text: "GPU", logit: 1.2 },
  { text: "model", logit: 0.8 },
  { text: "tokens", logit: 0.3 },
  { text: "banana", logit: -1.5 },
];

export function softmax(logits: number[], temperature: number): number[] {
  // temperature 0 → greedy (all mass on the max).
  if (temperature <= 0.01) {
    const max = Math.max(...logits);
    return logits.map((l) => (l === max ? 1 : 0));
  }
  const scaled = logits.map((l) => l / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Apply temperature → softmax → top-k → top-p truncation → renormalize. */
export function computeDistribution(
  temperature: number,
  topK: number,
  topP: number,
  candidates: Candidate[] = CANDIDATES
): Cand[] {
  const probs = softmax(
    candidates.map((c) => c.logit),
    temperature
  );
  // Sort indices by probability desc.
  const order = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);

  const keep = new Set<number>();
  let cumulative = 0;
  for (let rank = 0; rank < order.length; rank++) {
    const { p, i } = order[rank];
    if (rank >= topK) break; // top-k cutoff
    keep.add(i);
    cumulative += p;
    if (cumulative >= topP) break; // top-p (nucleus) cutoff — include this one then stop
  }

  const keptMass = order.reduce((sum, o) => (keep.has(o.i) ? sum + o.p : sum), 0);
  return candidates.map((c, i) => ({
    text: c.text,
    prob: keep.has(i) && keptMass > 0 ? probs[i] / keptMass : 0,
    kept: keep.has(i),
  }));
}
