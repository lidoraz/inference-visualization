/**
 * Term — inline glossary reference.
 *
 * Usage:
 *   <Term tokenKey="prefill" />
 *   <Term tokenKey="prefill">prefill phase</Term>
 *
 * Hover: native title tooltip with the entry's `short` description.
 * Click: toggles an inline popover showing the full `long` explanation.
 * Falls back to plain text when the key is absent from the glossary.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";
import type { GlossaryEntry } from "../content/glossary";
import { glossary } from "../content/glossary";
import { color, radius, font, palette } from "../theme";

interface TermProps {
  tokenKey: string;
  children?: ReactNode;
}

const termStyle: React.CSSProperties = {
  borderBottom: "1px dotted currentColor",
  cursor: "pointer",
  color: "inherit",
  background: "none",
  border: "none",
  borderBottomStyle: "dotted",
  borderBottomWidth: 1,
  borderBottomColor: "currentColor",
  padding: 0,
  font: "inherit",
  // A term reads in its natural case even inside an uppercase, letter-spaced
  // heading — don't inherit those transforms.
  textTransform: "none",
  letterSpacing: 0,
};

const popoverStyle: React.CSSProperties = {
  position: "absolute",
  zIndex: 50,
  top: "calc(100% + 6px)",
  left: 0,
  width: 320,
  maxWidth: "80vw",
  padding: "10px 12px",
  borderRadius: radius.lg,
  background: palette.crust,
  border: `1px solid ${color.borderStrong}`,
  color: color.textPrimary,
  fontSize: 12.5,
  lineHeight: 1.55,
  fontWeight: font.weight.normal,
  textAlign: "left",
  boxShadow: "0 6px 20px rgba(0,0,0,0.45)",
  cursor: "default",
  // The popover renders inside the term, which may sit in an uppercase,
  // letter-spaced heading — don't let the tooltip text inherit those.
  textTransform: "none",
  letterSpacing: 0,
  fontFamily: font.sans,
};

function getEntry(key: string): GlossaryEntry | undefined {
  try {
    return (glossary as Record<string, GlossaryEntry>)[key];
  } catch {
    return undefined;
  }
}

export function Term({ tokenKey, children }: TermProps) {
  // `open` (click) shows the full `long` text; `hovered` shows the short blurb.
  // Both use the same styled popover so hover and click look identical — and
  // hover is instant (no native `title` delay).
  const [open, setOpen] = useState(false);
  const [hovered, setHovered] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const entry = getEntry(tokenKey);
  const label = children ?? entry?.term ?? tokenKey;

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    function onEsc(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onEsc);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onEsc);
    };
  }, [open]);

  if (!entry) {
    // Key missing — render plainly with no crash.
    return <span>{label}</span>;
  }

  // Click takes priority (full text); otherwise hover shows the short blurb.
  const showPopover = open || hovered;

  return (
    <span
      ref={ref}
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        type="button"
        style={termStyle}
        aria-label={`${entry.term}: ${entry.short}`}
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        {label}
      </button>
      {showPopover && (
        <span role="tooltip" style={popoverStyle} onClick={(e) => e.stopPropagation()}>
          <span
            style={{
              display: "block",
              fontWeight: font.weight.bold,
              marginBottom: 4,
              color: color.accent,
            }}
          >
            {entry.term}
          </span>
          {open ? entry.long : entry.short}
          {!open && (
            <span
              style={{
                display: "block",
                marginTop: 6,
                fontSize: font.size.xs,
                fontStyle: "italic",
                color: color.textFaint,
              }}
            >
              Click for more →
            </span>
          )}
        </span>
      )}
    </span>
  );
}
