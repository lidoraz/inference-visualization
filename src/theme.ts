/**
 * Shared design tokens for the vLLM Inference Visualizer.
 *
 * The UI uses the Catppuccin Mocha palette. These tokens name the colors and a
 * small spacing/typography scale so components stop hard-coding hex values and
 * stay visually consistent. Prefer these over inline literals.
 *
 * Pure data, no imports — safe to use from any component.
 */

/** Catppuccin Mocha named colors (the palette already in use across the app). */
export const palette = {
  // Surfaces (darkest → lightest)
  crust: "#11111b",
  mantle: "#181825",
  base: "#1e1e2e",
  surface0: "#313244",
  surface1: "#45475a",
  surface2: "#585b70",
  // Text
  text: "#cdd6f4",
  subtext0: "#a6adc8",
  overlay0: "#6c7086",
  // Accents
  blue: "#89b4fa",
  mauve: "#cba6f7",
  green: "#a6e3a1",
  yellow: "#f9e2af",
  red: "#f38ba8",
  peach: "#fab387",
  sky: "#89dceb",
} as const;

/**
 * Semantic aliases — use these when the *meaning* is what matters, so intent
 * survives a future palette change.
 */
export const color = {
  // Backgrounds
  pageBg: palette.base,
  panelBg: palette.base,
  panelBgInset: palette.mantle,
  panelBgDeep: palette.crust,
  border: palette.surface0,
  borderStrong: palette.surface2,
  // Text
  textPrimary: palette.text,
  textMuted: palette.subtext0,
  textFaint: palette.overlay0,
  // Status / phase accents
  accent: palette.mauve, // brand / SGLang / highlights
  prefill: palette.blue,
  decode: palette.green,
  waiting: palette.yellow,
  swapped: palette.mauve,
  danger: palette.red,
  warn: palette.peach,
  info: palette.sky,
} as const;

/**
 * Distinct accent colors for coloring per-request / per-token items (cycled by
 * index). Shared so stages don't each re-declare the same array.
 */
export const requestColors: string[] = [
  color.prefill,
  color.danger,
  color.decode,
  color.warn,
  color.accent,
  color.waiting,
];

/** Spacing scale (px). Keep gaps/padding on these steps. */
export const space = {
  xs: 4,
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
  xxl: 24,
} as const;

/** Border radius scale (px). */
export const radius = {
  sm: 4,
  md: 6,
  lg: 8,
  pill: 12,
} as const;

/** Font sizes (px) and weights used across the UI. */
export const font = {
  mono: "monospace",
  sans: "sans-serif",
  size: {
    xs: 10,
    sm: 11,
    md: 12,
    base: 13,
    lg: 14,
    xl: 16,
    xxl: 18,
  },
  weight: {
    normal: 400,
    semibold: 600,
    bold: 700,
  },
} as const;

/** Standard uppercase section-label style fragment (spread into a style object). */
export const sectionLabel: React.CSSProperties = {
  fontSize: font.size.sm,
  fontWeight: font.weight.bold,
  textTransform: "uppercase",
  letterSpacing: "0.07em",
  color: color.textMuted,
  fontFamily: font.sans,
};

/** Standard bordered card/panel surface (the recurring section container). */
export const panel: React.CSSProperties = {
  background: color.panelBg,
  border: `1px solid ${color.border}`,
  borderRadius: radius.lg,
  padding: `${space.xl}px`,
};

/** A quiet, neutral inline note line (muted text on a subtle panel). */
export const notePanel: React.CSSProperties = {
  padding: `${space.md}px ${space.lg}px`,
  borderRadius: radius.lg,
  background: color.panelBg,
  fontSize: font.size.base,
  color: color.textMuted,
  lineHeight: 1.6,
};

/**
 * A status/category chip tinted from a single accent color: translucent fill,
 * matching text, semi-opaque border. Keeps colored chips consistent everywhere.
 */
export function statusTint(accent: string): React.CSSProperties {
  return {
    display: "inline-block",
    padding: `3px 10px`,
    borderRadius: radius.pill,
    fontSize: font.size.md,
    fontWeight: font.weight.semibold,
    fontFamily: font.sans,
    background: `${accent}22`,
    color: accent,
    border: `1px solid ${accent}66`,
  };
}
