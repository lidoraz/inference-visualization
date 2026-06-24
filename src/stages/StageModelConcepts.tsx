/**
 * Model Concepts — model/decoding-side topics that sit adjacent to (not inside)
 * the vLLM serving mechanics. Currently: token sampling. These shape *what* the
 * model emits per step, independent of how the server batches/schedules it.
 */

import type { StageProps } from "./types";
import { Term } from "../components/Term";
import { SamplingView } from "../components/SamplingView";
import { color, space, font, sectionLabel } from "../theme";

const outerStyle: React.CSSProperties = {
  padding: `${space.xl}px ${space.md}px`,
  display: "flex",
  flexDirection: "column",
  gap: space.xxl,
  fontFamily: font.sans,
  color: color.textPrimary,
};

const sectionHeadingStyle: React.CSSProperties = {
  ...sectionLabel,
  margin: `0 0 ${space.md}px`,
};

export function StageModelConcepts(_props: StageProps) {
  return (
    <div style={outerStyle} aria-label="Model Concepts visualization">
      <div>
        <h3 style={sectionHeadingStyle}>
          Token <Term tokenKey="vocabSampling">Sampling</Term>
        </h3>
        <SamplingView />
      </div>
    </div>
  );
}
