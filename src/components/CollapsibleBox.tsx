/**
 * CollapsibleBox — a titled panel that toggles its content open/closed.
 *
 * Used for stage explainer text so it can be tucked away to give the
 * visualization (and any alert banners) more room. Collapsed by default.
 */

import { useState, type ReactNode } from "react";
import { color, space, radius, font } from "../theme";

interface CollapsibleBoxProps {
  title: string;
  children: ReactNode;
  /** Whether the box starts expanded. Defaults to collapsed. */
  defaultOpen?: boolean;
}

const boxStyle: React.CSSProperties = {
  background: color.panelBg,
  border: `1px solid ${color.border}`,
  borderRadius: radius.lg,
  padding: `${space.lg}px ${space.xl}px`,
  display: "flex",
  flexDirection: "column",
  gap: space.md,
};

const toggleStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: space.md,
  background: "none",
  border: "none",
  color: color.textPrimary,
  fontSize: font.size.md,
  fontWeight: font.weight.bold,
  textTransform: "uppercase",
  letterSpacing: "0.06em",
  cursor: "pointer",
  padding: 0,
  fontFamily: font.sans,
};

const bodyStyle: React.CSSProperties = {
  fontSize: font.size.base,
  lineHeight: 1.65,
  color: color.textMuted,
  display: "flex",
  flexDirection: "column",
  gap: space.md,
};

export function CollapsibleBox({ title, children, defaultOpen = false }: CollapsibleBoxProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={boxStyle}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        style={toggleStyle}
      >
        <span aria-hidden="true">{open ? "▾" : "▸"}</span>
        {title}
      </button>
      {open && <div style={bodyStyle}>{children}</div>}
    </div>
  );
}
