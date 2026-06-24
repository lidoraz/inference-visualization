/**
 * GuidePanel — stage guide reader.
 *
 * Props:
 *   stageId   — which stage's guide to display
 *   stepIndex — current step (0-based, clamped defensively)
 *
 * Looks up stageGuides[stageId], renders:
 *   - stage title
 *   - "Step N of M" counter
 *   - glossary term + long description for the current step
 *   - optional note
 */

import type { ReactNode } from "react";
import { stageGuides } from "../content/stageGuides";
import { glossary } from "../content/glossary";
import type { GlossaryEntry } from "../content/glossary";
import type { StageGuide } from "../content/stageGuides";
import { Term } from "./Term";
import { color, space, radius, font } from "../theme";

/**
 * Parse inline glossary markup into React nodes. Supports [[key]] (renders the
 * glossary term's display name) and [[key|label]] (custom label). Anything that
 * isn't markup is passed through as plain text. Unknown keys fall back to the
 * label/key text via the Term component's own fallback.
 */
function renderWithTerms(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const key = m[1].trim();
    const label = m[2]?.trim();
    nodes.push(
      <Term key={`t${i++}`} tokenKey={key}>
        {label ?? glossary[key]?.term ?? key}
      </Term>
    );
    last = re.lastIndex;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

interface GuidePanelProps {
  stageId: number;
  stepIndex: number;
}

const panelStyle: React.CSSProperties = {
  padding: `${space.lg}px ${space.xl}px`,
  background: color.panelBg,
  border: `1px solid ${color.border}`,
  borderRadius: radius.lg,
  color: color.textPrimary,
  fontFamily: font.sans,
  fontSize: font.size.lg,
};

const eyebrowStyle: React.CSSProperties = {
  margin: `0 0 ${space.xs}px`,
  fontSize: font.size.sm,
  fontWeight: font.weight.bold,
  textTransform: "uppercase",
  letterSpacing: "0.07em",
  color: color.accent,
};


const longStyle: React.CSSProperties = {
  margin: `0 0 ${space.md}px`,
  lineHeight: 1.5,
};

const noteStyle: React.CSSProperties = {
  margin: 0,
  padding: `${space.sm}px ${space.md + 2}px`,
  background: color.border,
  borderLeft: `3px solid ${color.waiting}`,
  borderRadius: radius.sm,
  color: color.waiting,
  fontSize: font.size.base,
  fontStyle: "italic",
};

const fallbackStyle: React.CSSProperties = {
  ...panelStyle,
  color: color.textMuted,
  fontStyle: "italic",
};

export function GuidePanel({ stageId, stepIndex }: GuidePanelProps) {
  let guide: StageGuide | undefined;
  try {
    guide = (stageGuides as Record<number, StageGuide>)[stageId];
  } catch {
    guide = undefined;
  }

  if (!guide || !guide.steps || guide.steps.length === 0) {
    return (
      <div style={fallbackStyle} role="complementary" aria-label="Stage guide">
        <p style={{ margin: 0 }}>No guide available for stage {stageId}.</p>
      </div>
    );
  }

  const clampedIndex = Math.max(0, Math.min(stepIndex, guide.steps.length - 1));
  const step = guide.steps[clampedIndex];

  let entry: GlossaryEntry | undefined;
  if (step.glossaryKey) {
    try {
      entry = (glossary as Record<string, GlossaryEntry>)[step.glossaryKey];
    } catch {
      entry = undefined;
    }
  }

  // Step label (the blue eyebrow): the glossary term, or an explicit title for
  // keyless steps (e.g. "Overview"). The stage title now lives in the band header.
  const heading = entry?.term ?? step.title ?? step.glossaryKey;

  return (
    <div style={panelStyle} role="complementary" aria-label="Stage guide">
      {heading && <p style={eyebrowStyle}>{heading}</p>}
      {entry && <p style={longStyle}>{entry.long}</p>}
      {step.note &&
        (entry ? (
          // With a glossary definition above, the note is a short aside (yellow box).
          <p style={noteStyle}>{renderWithTerms(step.note)}</p>
        ) : (
          // No definition: the note IS the main body — render as normal prose.
          <p style={longStyle}>{renderWithTerms(step.note)}</p>
        ))}
    </div>
  );
}
