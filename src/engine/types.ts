/**
 * Core engine types for the vLLM Inference Visualizer.
 * Pure declarations only — no logic, no React imports.
 * These types are consumed by every engine unit (tokenizer, kvcache, scheduler,
 * engine.ts) and must remain stable across stages.
 */

export type Phase = "prefill" | "decode";

export type RequestStatus =
  | "waiting"
  | "running"
  | "swapped"
  | "finished"
  | "cancelled";

export interface Token {
  id: number;
  text: string;
}

export interface Request {
  id: number;
  /** The original prompt string before tokenization (for display/onboarding). */
  promptText?: string;
  promptTokens: Token[];
  decodedTokens: Token[];
  maxDecode: number;
  status: RequestStatus;
  phase: Phase;
  /** logical block index -> physical block id */
  blockTable: number[];
  arrivalTick: number;
  /** Tick at which this request emitted its first decoded token. Used to derive
   *  TTFT (time to first token = firstTokenTick - arrivalTick). */
  firstTokenTick?: number;
  /** Set when a request is cancelled because it can never be served (e.g. its
   *  prompt needs more KV blocks than the whole cache holds). */
  rejectionReason?: string;
  /** Set by the scheduler each tick a request stays waiting/swapped, explaining
   *  why it wasn't admitted (token budget, batch full, or no free blocks).
   *  Cleared when the request is admitted. Purely informational for the UI. */
  waitReason?: string;
}

export interface Block {
  id: number;
  requestId: number | null;
  /** capacity */
  tokenSlots: number;
  usedSlots: number;
}

export interface Config {
  maxBatchSize: number;
  blockSize: number;
  kvCacheBlocks: number;
  tokenBudget: number;
  numGPUs?: number;
  parallelism?: "tp" | "pp" | "ep" | "dp" | "none";
  quant?: "fp16" | "fp8";
  moe?: boolean;
  /** Sliding-window attention: cap a request's live KV to its last `windowSize`
   *  tokens so the cache stops growing unbounded on long sequences. Undefined =
   *  full (global) attention, the default. */
  windowSize?: number;
}

/** Matches the shape returned by makeRng in rng.ts. */
export interface Rng {
  seed: number;
  next(): number;
}

export interface EngineState {
  tick: number;
  requests: Request[];
  blocks: Block[];
  rng: Rng;
  nextRequestId: number;
  arrivalRatePerTick: number;
}

/**
 * Engine actions.
 *
 * Intentional divergences from the spec:
 *   (a) No PLAY action — Play is a UI concern (setInterval dispatching STEP),
 *       not an engine responsibility.
 *   (b) BatchStep type is deferred — not needed for Stages 1–3.
 */
export type Action =
  | { type: "STEP" }
  | { type: "RESET" }
  // SET_CONFIG carries the merged config for the React layer to own; the engine
  // is stateless w.r.t. config (it reads config on every call), so engine.reduce
  // returns state unchanged. Capacity changes take effect via scheduler
  // preemption on the next tick.
  | { type: "SET_CONFIG"; config: Partial<Config> }
  | { type: "ADD_REQUEST"; prompt: string; maxDecode?: number }
  | { type: "CANCEL_REQUEST"; requestId: number }
  | { type: "SET_ARRIVAL_RATE"; ratePerTick: number };
