/**
 * Pure state machine logic for the MTP (Multi-Token Prediction) visualizer.
 * Extracted so it can be unit-tested without a DOM/React environment.
 */

export type Mode = "standard" | "mtp1" | "mtp3";

export interface StagingToken {
  id: number;
  text: string;
}

export interface MtpState {
  confirmed: string[];
  staging: StagingToken[];
  passes: number;
  idCounter: number;
}

export function draftSizeForMode(mode: Mode): number {
  if (mode === "mtp1") return 1;
  if (mode === "mtp3") return 3;
  return 0;
}

export function initialState(): MtpState {
  return { confirmed: [], staging: [], passes: 0, idCounter: 0 };
}

export function nextPass(state: MtpState, mode: Mode, targetPhrase: string[]): MtpState {
  const draftSize = draftSizeForMode(mode);
  let nextConfirmed = [...state.confirmed];
  let nextIdCounter = state.idCounter;

  // Step 1: accept all staging tokens (deterministic demo — always correct)
  for (const st of state.staging) {
    if (nextConfirmed.length < targetPhrase.length) {
      nextConfirmed = [...nextConfirmed, st.text];
    }
  }

  // Step 2: main head emits 1 token
  if (nextConfirmed.length < targetPhrase.length) {
    nextConfirmed = [...nextConfirmed, targetPhrase[nextConfirmed.length]];
  }

  // Step 3: MTP heads predict draftSize more tokens into staging
  const nextStaging: StagingToken[] = [];
  for (let i = 0; i < draftSize; i++) {
    const idx = nextConfirmed.length + i;
    if (idx < targetPhrase.length) {
      nextStaging.push({ id: nextIdCounter++, text: targetPhrase[idx] });
    }
  }

  return {
    confirmed: nextConfirmed,
    staging: nextStaging,
    passes: state.passes + 1,
    idCounter: nextIdCounter,
  };
}

export function isDone(state: MtpState, targetPhrase: string[]): boolean {
  return (
    state.confirmed.length >= targetPhrase.length &&
    state.staging.length === 0 &&
    state.passes > 0
  );
}
