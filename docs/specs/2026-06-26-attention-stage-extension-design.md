# Attention Stage Extension Design

**Date:** 2026-06-26
**File:** `src/stages/StageAttention.tsx`
**Scope:** Add Q/K/V computation step-through and Flash Attention tiling panels; reorder all panels to flow fundamentals → optimizations → variants.

---

## Goal

Extend the Attention stage with visual-first, tooltip-driven content that teaches the fundamentals of attention computation before showing architectural variants. No explanatory body text — every concept lives in a `Term` tooltip or glossary entry.

**Note on existing panels:** The Sliding Window and KV Variants panels contain existing `<p>` body text. Removing that text is out of scope for this change — it can be addressed in a follow-up cleanup pass.

---

## Panel Order (after change)

1. **Q/K/V Computation** (new) — what attention does
2. **Flash Attention Tiling** (new) — how it's computed efficiently
3. **Sliding Window** (existing, unchanged including body text)
4. **KV Head Variants** (existing, unchanged including body text)

---

## Panel 1 — Q/K/V Computation Step-Through

### Layout

A step-through navigator with Prev/Next buttons and a step indicator (e.g. "2 / 5"). Each step reveals or highlights a portion of a fixed diagram showing the full attention pipeline.

### Steps

| Step | What's shown | Label (Term-wrapped) |
|------|-------------|----------------------|
| 1 | Input token rows → three colored matrix blocks: **Q** (blue/`color.prefill`), **K** (green/`color.decode`), **V** (orange/`color.accent`) | `<Term tokenKey="queryMatrix">`, `<Term tokenKey="keyMatrix">`, `<Term tokenKey="valueMatrix">` |
| 2 | Q × Kᵀ → score matrix (N×N), uniform gray cells | `<Term tokenKey="attentionScore">` + formula label `QKᵀ / √d_k` |
| 3 | Score matrix with heatmap coloring (row-wise softmax applied; each row visibly sums to 1 via intensity) | `<Term tokenKey="softmax">` |
| 4 | Softmax × V → output matrix (N × d_v), rows are weighted blends shown with mixed color fill | `<Term tokenKey="scaledDotProduct">` + label `Attention(Q,K,V)` |
| 5 | Output matrix dims labeled with residual stream tooltip; no arrow needed | `<Term tokenKey="residualStream">` |

### Visual conventions

- Matrices rendered as small CSS grids (5 tokens × 8 dims for Q/K/V; 5×5 for score matrix)
- Color intensity = relative value (no raw numbers shown)
- Active matrix highlighted with `color.accent` border; inactive matrices dimmed to 40% opacity (`color.textPrimary` at `opacity: 0.4`)
- All matrix labels are `<Term>` components — no plain text for named concepts
- Formula labels (e.g. `QKᵀ / √d_k`) rendered as plain-text strings inside `Term.short` — `GlossaryEntry` fields are typed as `string`, not `ReactNode`, so no HTML tags in glossary values

### Matrix cell values (reproducible)

Use hardcoded constant arrays so the visualization is deterministic and visually illustrative (not random noise):

- **Q, K, V cells:** fixed 5×8 float arrays normalized to [0, 1], designed to show variation across tokens and dims
- **Score matrix:** derived as `softmax(Q_fixed @ K_fixed_T / sqrt(8))` computed once at module level — ensures the heatmap shows realistic attending patterns (e.g. diagonal + a few off-diagonal hot spots)
- **Output matrix:** derived as `score_fixed @ V_fixed` — shows blended colors reflecting the attention pattern

All fixed arrays defined as `const` outside the component at the top of the file.

---

## Panel 2 — Flash Attention Tiling

### Layout

Two clickable cards (reusing the KV variant card pattern from `StageAttention.tsx` lines 182–222): **Standard** | **Flash**. Selected card highlighted with `color.accent` border.

### Standard card view

- Full N×N attention matrix shown as a single block
- `<Term tokenKey="hbm">` badge on the block with a warm/red-tinted border (`color.danger`)
- No animation

### Flash card view

- Same N×N footprint divided into a 3×3 tile grid
- Active tile highlighted in `color.accent`; rest at 40% opacity
- Animated active tile: driven by `useEffect` + `setInterval` (700ms tick, active tile index increments 0→8 then loops), interval cleared on component unmount and when `flashMode !== 'flash'`
- `<Term tokenKey="sram">` badge on the active tile; `<Term tokenKey="hbm">` badge below the grid
- Legend: two horizontal bars — HBM (wide, muted) vs SRAM (narrow, accent) — no body text, `Term`-wrapped labels only

### Term links required in headings/labels

- Panel heading: `<Term tokenKey="flashAttention">Flash Attention</Term>`
- Standard label: `<Term tokenKey="scaledDotProduct">Standard Attention</Term>`
- Flash label: `<Term tokenKey="flashAttention">Flash Attention</Term>`

---

## Glossary Additions

New entries to add to `src/content/glossary.ts`:

| key | term | short | long |
|-----|------|-------|------|
| `queryMatrix` | Query (Q) | Token's "question" projected into attention space | Q = input × W_Q. Each row is what a token is looking for. Dot-producted against all keys to produce attention scores. |
| `keyMatrix` | Key (K) | Token's "label" projected into attention space | K = input × W_K. Each row is what a token advertises about itself. High Q·K score = strong attention. |
| `valueMatrix` | Value (V) | Token's content projected into attention space | V = input × W_V. The actual information retrieved. Output = weighted sum of V rows using softmax attention weights. |
| `attentionScore` | Attention Score | Raw similarity between a query and a key | Computed as Q·Kᵀ / √d_k. Large dot products push softmax into regions of near-zero gradient — dividing by √d_k keeps scores in a well-behaved range. |
| `softmax` | Softmax | Normalizes scores into a probability distribution | Converts raw scores into weights that sum to 1 per row. Ensures each output token attends to a valid distribution over all positions. |
| `scaledDotProduct` | Scaled Dot-Product Attention | The core attention operation | Attention(Q,K,V) = softmax(QKᵀ / √d_k) × V. Dividing by √d_k prevents large dot products from pushing softmax into vanishing-gradient territory. |
| `flashAttention` | Flash Attention | IO-efficient attention that avoids materializing the full N×N matrix | Tiles Q, K, V into blocks that fit in SRAM. Computes attention tile-by-tile, accumulating the output without ever writing the full N×N score matrix to HBM. O(N) HBM reads vs O(N²) for standard attention. |
| `hbm` | HBM (High Bandwidth Memory) | The main GPU memory — large but slow relative to on-chip cache | All model weights and activations live here. HBM bandwidth (not FLOPS) is the bottleneck for attention at large sequence lengths. |
| `sram` | SRAM (On-Chip Cache) | Fast on-chip memory on the GPU — small but very fast | Flash Attention keeps active tiles in SRAM to avoid repeated HBM round-trips. Typically 20–40 MB on modern GPUs. |
| `residualStream` | Residual Stream | The running hidden state that accumulates information across layers | Each transformer layer reads from and writes back to the residual stream via the attention output projection. Enables gradient flow through deep networks. |

**Note:** `attentionScore` and `scaledDotProduct` both explain the `√d_k` scaling — they are harmonized to the same root cause (vanishing gradient via softmax saturation), with `attentionScore` giving the short version and `scaledDotProduct` giving the full formula context.

---

## Implementation Notes

### Component Reuse

- **`<Term>`** (`src/components/Term.tsx`) — wrap every matrix label, formula label, legend item, and panel heading; never use raw text for named concepts
- **KV variant card pattern** (`StageAttention.tsx` lines 182–222) — reuse the same clickable card + fill-bar layout for the Flash toggle (Standard / Flash cards)
- **Theme tokens** (`src/theme.ts`) — all colors from `color.*`, spacing from `space.*`, font from `font.size.*`; no raw hex or px values
- **Dimming pattern** — `color.textPrimary` at `opacity: 0.4` for inactive matrices (not `color.text`, which does not exist in the theme)

### State

- `const [qkvStep, setQkvStep] = useState(1)` — 1-indexed, 1–5
- `const [flashMode, setFlashMode] = useState<'standard' | 'flash'>('standard')`
- `const [activeTile, setActiveTile] = useState(0)` — 0–8, driven by interval when `flashMode === 'flash'`

### Matrix Grid

- CSS `display: grid`, `grid-template-columns: repeat(8, 1fr)` for Q/K/V; `repeat(5, 1fr)` for score/output
- Cells are `div`s with `backgroundColor` set from the fixed data arrays, interpolated between a base color and the matrix's accent color

### Flash Tile Animation

```ts
useEffect(() => {
  if (flashMode !== 'flash') return;
  const id = setInterval(() => setActiveTile(t => (t + 1) % 9), 700);
  return () => clearInterval(id);
}, [flashMode]);
```

Interval clears automatically on unmount and when mode switches to Standard.

### Feature Branch

`feature/attention-fundamentals`

---

## Out of Scope

- Multi-head parallelism (already covered in KV head variants)
- Causal masking diagram (can be added later as a step variant)
- Actual numeric values in matrices
- Removing existing body text from Sliding Window / KV Variants panels
