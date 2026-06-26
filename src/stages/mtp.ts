/**
 * Pure state machine logic for the MTP (Multi-Token Prediction) visualizer.
 * Extracted so it can be unit-tested without a DOM/React environment.
 */

export type Mode = "standard" | "mtp1" | "mtp3";

export type VerificationStatus = "accepted" | "rejected" | "discarded";

export interface StagingToken {
  id: number;
  text: string;   // what the MTP module predicted (may be wrong)
}

export interface VerifiedToken {
  id: number;
  predicted: string;  // what MTP said
  truth: string;      // what the target model knows is correct
  status: VerificationStatus;
}

export interface MtpState {
  confirmed: string[];
  staging: StagingToken[];
  passes: number;
  idCounter: number;
  lastVerification: VerifiedToken[];
}

export function draftSizeForMode(mode: Mode): number {
  if (mode === "mtp1") return 1;
  if (mode === "mtp3") return 3;
  return 0;
}

export function initialState(): MtpState {
  return { confirmed: [], staging: [], passes: 0, idCounter: 0, lastVerification: [] };
}

/**
 * How many staging tokens will be accepted on the next pass, given the
 * current state. Used by the caller to know what positions to predict for.
 */
export function previewAcceptedCount(state: MtpState, targetPhrase: string[]): number {
  let accepted = 0;
  for (let i = 0; i < state.staging.length; i++) {
    if (state.staging[i].text === targetPhrase[state.confirmed.length + i]) {
      accepted++;
    } else {
      break;
    }
  }
  return accepted;
}

/**
 * Advance the simulation by one forward pass.
 *
 * newStagingTokens — the MTP module's predictions for the NEXT pass,
 * built by the caller (component) so it can inject wrong predictions.
 * The caller must target positions starting at (confirmed + accepted + 1).
 */
export function nextPass(
  state: MtpState,
  mode: Mode,
  targetPhrase: string[],
  newStagingTokens: StagingToken[]
): MtpState {
  if (mode === "standard") {
    const nextConfirmed = [...state.confirmed];
    if (nextConfirmed.length < targetPhrase.length) {
      nextConfirmed.push(targetPhrase[nextConfirmed.length]);
    }
    return {
      confirmed: nextConfirmed,
      staging: [],
      passes: state.passes + 1,
      idCounter: state.idCounter,
      lastVerification: [],
    };
  }

  // Step 1: verify existing staging against the target phrase
  const verification: VerifiedToken[] = [];
  let rejectionSeen = false;

  for (let i = 0; i < state.staging.length; i++) {
    const st = state.staging[i];
    const truth = targetPhrase[state.confirmed.length + i] ?? "";

    if (rejectionSeen) {
      verification.push({ id: st.id, predicted: st.text, truth, status: "discarded" });
    } else if (st.text === truth) {
      verification.push({ id: st.id, predicted: st.text, truth, status: "accepted" });
    } else {
      verification.push({ id: st.id, predicted: st.text, truth, status: "rejected" });
      rejectionSeen = true;
    }
  }

  // Step 2: build confirmed from accepted staging, then main head adds 1 correct token
  const nextConfirmed = [...state.confirmed];
  for (const v of verification) {
    if (v.status === "accepted") nextConfirmed.push(v.predicted);
  }
  if (nextConfirmed.length < targetPhrase.length) {
    nextConfirmed.push(targetPhrase[nextConfirmed.length]);
  }

  return {
    confirmed: nextConfirmed,
    staging: newStagingTokens,
    passes: state.passes + 1,
    idCounter: state.idCounter + newStagingTokens.length,
    lastVerification: verification,
  };
}

export function isDone(state: MtpState, targetPhrase: string[]): boolean {
  return (
    state.confirmed.length >= targetPhrase.length &&
    state.staging.length === 0 &&
    state.passes > 0
  );
}
