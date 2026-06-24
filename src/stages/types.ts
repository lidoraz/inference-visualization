/**
 * Shared prop contract for all stage visualization components.
 * Import this in stage components and in App.tsx.
 */

import type { Config, EngineState } from "../engine/types";

export interface StageProps {
  engine: EngineState;
  config: Config;
  selectedRequestId: number | null;
  onSelectRequest(id: number): void;
}
