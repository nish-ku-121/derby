from __future__ import annotations
"""Publication-oriented plotting helpers (clean rebuild).

Provides:
    * `paper_style` context manager for lightweight, reproducible styling.
    * Dataclass configs (`AxesConfig`, `HighlightConfig`, `SmoothingConfig`, `LegendConfig`).
    * `plot_epochs_simple` – epoch vs. reward with optional smoothing + top-k highlight.
    * `make_legend_figure` – Phase 1: header-only grouped legend figure for spacing sanity.

Deliberately removed: prior experimental panel legend heuristics & complex spacing logic.
Future steps (after header layout is approved):
    1. Add per‑group item listing below headers.
    2. Composition helper to vertically stack plot + legend figure.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Sequence, Dict, Tuple, Optional

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette (Okabe–Ito colorblind-safe)
# ---------------------------------------------------------------------------
PALETTE_OKABE_ITO: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # gray
]

# ---------------------------------------------------------------------------
# Dataclass Configs
# ---------------------------------------------------------------------------
@dataclass
class AxesConfig:
    x_col: str = 'epoch'
    y_col: str = 'mean_reward'
    x_label: str | None = None
    y_label: str | None = None
    xlim: Tuple[float, float] | None = None
    ylim: Tuple[float, float] | None = None
    x_scale: str = 'linear'
    y_scale: str = 'linear'
    invert_x: bool = False
    invert_y: bool = False
    grid: bool = True

@dataclass
class HighlightConfig:
    top_k: int = 5
    window: int = 0              # if >0 use tail average over last N epochs for ranking
    outline: bool = True
    markers: bool = True
    marker_every: int | None = None  # if set, place marker every N epochs

@dataclass
class SmoothingConfig:
    window: int = 5
    use_for_ordering: bool = True  # if True rank by smoothed series; else raw

@dataclass
class LegendConfig:
    label_col: str | None = 'label'
    max_entries: int | None = None
    group_cols: Sequence[str] | None = None
    n_rows: int | None = None
    row_gap_frac: float = 0.0
    wrap: bool = True
    max_chars_per_line: int = 28

# ---------------------------------------------------------------------------
# Style Context
# ---------------------------------------------------------------------------
@contextmanager
def paper_style(figsize: Tuple[float, float] | None = None, base_font: float = 9.0):
    orig = mpl.rcParams.copy()
    mpl.rcParams.update({
        'figure.dpi': 110,
        'font.size': base_font,
        'axes.titlesize': base_font + 1.5,
        'axes.labelsize': base_font + 1,
        'legend.fontsize': base_font - 0.5,
        'xtick.labelsize': base_font - 0.5,
        'ytick.labelsize': base_font - 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    try:
        yield
    finally:
        mpl.rcParams.update(orig)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_grouped_legend_on_axis(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    id_cols: Sequence[str],
    label_col: str | None,
    group_cols: Sequence[str] | None,
    group_sep: str,
    base_font: float,
    max_entries: int | None,
    wrap: bool,
    max_chars_per_line: int,
    n_rows: int | None,
    n_cols: int | None,
    row_gap_frac: float,
    plotted_labels=None,
    plotted_colors=None,
):
    # Logic copied from make_legend_figure, but draws on provided axis
    if df.empty:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    if not group_cols:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No group columns', ha='center', va='center')
        return
    def _group_key(row) -> str:
        parts: list[str] = []
        for c in group_cols:
            val = row.get(c, '')
            parts.append('' if pd.isna(val) else str(val))
        if all(p == '' for p in parts):
            return ''
        return group_sep.join(parts)
    keys = df.apply(_group_key, axis=1)
    order: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)
    order = sorted(order)
    if '' in order:
        order = [g for g in order if g != '']
    if not order:
        ax.axis('off')
        ax.text(0.5, 0.5, 'No groups', ha='center', va='center')
        return
    group_items: Dict[str, list[str]] = {}
    if label_col and label_col in df.columns:
        item_series = df[label_col].astype(str)
    else:
        item_series = df[id_cols].astype(str).agg('_'.join, axis=1) if isinstance(id_cols, (list, tuple)) else df[id_cols].astype(str)
    tmp = pd.DataFrame({'__gk__': keys, '__lab__': item_series})
    for gk, sub in tmp.groupby('__gk__'):
        uniq = sorted(dict.fromkeys(sub['__lab__'].tolist()))
        if max_entries is not None:
            uniq = uniq[:max_entries]
        group_items[gk] = uniq
    n = len(order)
    # Always grid mode with items
    if (n_rows is None) and (n_cols is None):
        import math
        n_cols = max(1, int(math.ceil(n ** 0.5)))
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    elif n_cols is None and n_rows is not None:
        n_cols = max(1, (n + n_rows - 1) // n_rows)
    elif n_rows is None and n_cols is not None:
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    assert n_cols is not None and n_rows is not None
    ax.axis('off')
    eff_font = base_font * 1.06
    item_font = eff_font * 0.80
    left_pad = 0.04
    right_pad = 0.04
    top_pad = 0.25 if n_rows > 1 else 0.30
    bottom_pad = 0.10
    usable_w = 1 - left_pad - right_pad
    usable_h = 1 - top_pad - bottom_pad
    rg = max(0.0, min(0.25, row_gap_frac))
    gap_h = rg * usable_h if n_rows > 1 else 0.0
    total_gap_h = gap_h * (n_rows - 1)
    effective_h = max(1e-6, usable_h - total_gap_h)
    cell_w = usable_w / max(1, n_cols)
    cell_h = effective_h / max(1, n_rows)
    # Build a mapping from label to color for swatches
    label_to_color = dict(zip(plotted_labels, plotted_colors)) if plotted_labels and plotted_colors else {}

    for idx, gk in enumerate(order):
        row = idx // n_cols
        col = idx % n_cols
        x0 = left_pad + col * cell_w
        y0 = 1.0 - top_pad - row * (cell_h + gap_h)
        # Header
        ax.text(x0, y0, gk if gk else '(other)', ha='left', va='top', fontsize=eff_font, fontweight='bold', transform=ax.transAxes)
        items = group_items.get(gk, [])
        for i, lab in enumerate(items):
            y_item = y0 - (i + 1) * 0.07
            color = label_to_color.get(lab, 'gray')
            # Draw color swatch
            ax.add_patch(
                mpl.patches.Rectangle((x0, y_item - 0.012), 0.018, 0.018, color=color, transform=ax.transAxes, clip_on=False)
            )
            # Draw label (no bullet prefix)
            ax.text(x0 + 0.024, y_item, lab, ha='left', va='center', fontsize=item_font, color='black', transform=ax.transAxes)

# ---------------------------------------------------------------------------
# Core Plot Function (minimal, stable baseline)
# ---------------------------------------------------------------------------
def plot_epochs_simple(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str] = ('policy_class', 'run_id'),
    axes: AxesConfig | None = None,
    highlight: HighlightConfig | None = None,
    smoothing: SmoothingConfig | None = None,
    legend: LegendConfig | None = None,
    figsize: Tuple[float, float],
    title: str | None = None,
    show_metadata: bool = True,
    metadata_keys: Sequence[str] = ('learning_rate', 'setup'),
    smoothing_disclaimer_loc: str = 'lower right',
    legend_height: float = 1.2,
) -> tuple[plt.Figure, plt.Axes]:
    if axes is None:
        axes = AxesConfig()
    if highlight is None:
        highlight = HighlightConfig()
    if legend is None:
        legend = LegendConfig()
    # figsize is now mandatory

    x_col = axes.x_col
    y_col = axes.y_col
    required = set(id_cols) | {x_col, y_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    if df.empty:
        with paper_style(figsize=figsize):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return fig, ax

    work = df.copy()
    work['_x_num'] = pd.to_numeric(work[x_col], errors='coerce')

    # Compute unique time series keys (ordered)
    unique_keys = []
    seen_keys = set()
    for key_vals, _ in work.sort_values(list(id_cols) + ['_x_num']).groupby(list(id_cols)):
        key_tuple = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        if key_tuple not in seen_keys:
            seen_keys.add(key_tuple)
            unique_keys.append(key_tuple)

    # Smoothing
    applied_smoothing = False
    smoothing_window = 0
    if smoothing and smoothing.window > 1:
        # Apply smoothing logic here if needed
        applied_smoothing = True
        smoothing_window = smoothing.window

    # Color assignment by performance order
    color_cycle = PALETTE_OKABE_ITO
    color_map: Dict[Tuple, str] = {}
    plotted_labels = []
    plotted_colors = []
    legend_label_source = None
    highlighted_labels = []
    highlighted_colors = []
    # Compute last_rows and top_keys for highlighting
    last_rows = work.sort_values('_x_num').groupby(list(id_cols)).tail(1)
    # Top keys: highlight top_k by mean reward
    top_k = highlight.top_k if highlight and hasattr(highlight, 'top_k') else 8
    top_keys = set(
        last_rows.sort_values(y_col, ascending=False)
        .head(top_k)
        .apply(lambda r: tuple(r[c] for c in id_cols), axis=1)
    )

    # Legend integration flag and height
    integrate_grouped = legend is not None
    legend_height = legend_height if integrate_grouped else 0.0

    with paper_style(figsize=figsize):
        # Create main plot and legend axes in one figure
        import matplotlib.gridspec as gridspec
        total_height = figsize[1]
        plot_height = total_height - legend_height
        fig = plt.figure(figsize=figsize)
        # If legend_height is 0, only plot main axis
        if legend_height > 0:
            gs = gridspec.GridSpec(2, 1, height_ratios=[plot_height, legend_height], hspace=0.08)
            ax = fig.add_subplot(gs[0, 0])
            legend_ax = fig.add_subplot(gs[1, 0])
            legend_ax.axis('off')
        else:
            ax = fig.add_subplot(1, 1, 1)
            legend_ax = None

        # Plot each series and collect labels/colors
        for key_tuple in unique_keys:
            g = work[(work[list(id_cols)] == pd.Series(key_tuple, index=id_cols)).all(axis=1)]
            is_top = key_tuple in top_keys
            if '_display_label' in g.columns:
                lbl = g['_display_label'].iloc[0]
                legend_label_source = '_display_label'
            elif legend and legend.label_col and legend.label_col in g.columns:
                lbl = g[legend.label_col].iloc[0]
                legend_label_source = legend.label_col
            else:
                lbl = str(key_tuple)
                legend_label_source = None
            x = g['_x_num']
            y = g[y_col] if y_col in g.columns else g['plot_reward']
            color = color_map.get(key_tuple, color_cycle[len(color_map) % len(color_cycle)])
            color_map[key_tuple] = color
            lw = 2.2 if is_top else 1.0
            alpha = 1.0 if is_top else 0.25
            z = 3 if is_top else 1
            line, = ax.plot(x, y, label=lbl, color=color, lw=lw, alpha=alpha, zorder=z)
            plotted_labels.append(lbl)
            plotted_colors.append(color)
            if is_top:
                highlighted_labels.append(lbl)
                highlighted_colors.append(color)
            if is_top and highlight.outline:
                line.set_path_effects([
                    mpl.patheffects.Stroke(linewidth=lw + 2.4, foreground='white', alpha=alpha),
                    mpl.patheffects.Normal(),
                ])
            if is_top and highlight.markers and highlight.marker_every and highlight.marker_every > 1:
                mask = []
                x_min = x.min()
                for idx_i, xv in enumerate(x):
                    if (xv - x_min) % highlight.marker_every == 0 or xv == x.max():
                        mask.append(idx_i)
                ax.plot(x.iloc[mask], y.iloc[mask], linestyle='None', marker='o',
                        markersize=4, markerfacecolor=color, markeredgecolor='white',
                        markeredgewidth=0.7, alpha=alpha, zorder=z)

        # Axis config
        if axes.x_label:
            ax.set_xlabel(axes.x_label)
        if axes.y_label:
            ax.set_ylabel(axes.y_label)
        if axes.xlim:
            ax.set_xlim(*axes.xlim)
        if axes.ylim:
            ax.set_ylim(*axes.ylim)
        ax.set_xscale(axes.x_scale)
        ax.set_yscale(axes.y_scale)
        if axes.invert_x:
            ax.invert_xaxis()
        if axes.invert_y:
            ax.invert_yaxis()

        # Metadata box
        if show_metadata:
            meta_lines: list[str] = []
            for key in metadata_keys:
                if key in work.columns:
                    vals = pd.unique(work[key].dropna())
                    if len(vals) == 1:
                        meta_lines.append(f"{key}: {vals[0]}")
                    elif len(vals) > 1:
                        meta_lines.append(f"{key}: mixed")
            max_x = pd.to_numeric(work[x_col], errors='coerce').max()
            if pd.notna(max_x):
                if x_col == 'epoch':
                    meta_lines.append(f"epochs: {int(max_x)}")
                else:
                    meta_lines.append(f"{x_col}_max: {max_x:.3g}")
            if meta_lines:
                ax.text(0.995, 0.02, '\n'.join(meta_lines), ha='right', va='bottom', transform=ax.transAxes,
                        fontsize=7.2, bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.55,
                        edgecolor='none'))

        # Smoothing disclaimer
        if applied_smoothing:
            disclaim = f"Smoothed (window={smoothing_window})"
            loc_map = {
                'lower right': (0.995, 0.005, 'right', 'bottom'),
                'lower left': (0.005, 0.005, 'left', 'bottom'),
                'upper right': (0.995, 0.995, 'right', 'top'),
                'upper left': (0.005, 0.995, 'left', 'top'),
            }
            x_pos, y_pos, ha, va = loc_map.get(smoothing_disclaimer_loc.lower(), (0.995, 0.005, 'right', 'bottom'))
            ax.text(x_pos, y_pos, disclaim, transform=ax.transAxes, ha=ha, va=va,
                    fontsize=6.5, color='0.25',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', alpha=0.4, edgecolor='none'),
                    zorder=5)

        # Grid
        ax.grid(axes.grid, which='both', alpha=0.3 if axes.grid else 0.0)

        if title:
            ax.set_title(title)

        # If integrated grouped legend: render legend directly on legend_ax
    if integrate_grouped and legend_ax is not None:
        # Build legend handles and labels for all plotted lines
        handles, labels = [], []
        for line in ax.get_lines():
            handles.append(line)
            labels.append(line.get_label())

        # Optionally group legend items by group_cols
        group_cols = legend.group_cols if legend and legend.group_cols else None
        group_sep = ' / '
        base_font = mpl.rcParams.get('font.size', 9.0)
        n_rows = legend.n_rows if legend and legend.n_rows else 1
        n_cols = None
        row_gap_frac = legend.row_gap_frac if legend else 0.0

        # If grouping is requested, build groupings
        if group_cols and all(c in df.columns for c in group_cols):
            # Map label to group key
            label_to_group = {}
            for idx, row in df.iterrows():
                if legend_label_source and legend_label_source in df.columns:
                    lbl = str(row[legend_label_source])
                elif legend and legend.label_col and legend.label_col in df.columns:
                    lbl = str(row[legend.label_col])
                else:
                    lbl = str(tuple(row[c] for c in id_cols))
                gk = group_sep.join(str(row[c]) for c in group_cols)
                label_to_group[lbl] = gk
            # Build groupings
            group_map = {}
            for h, l in zip(handles, labels):
                gk = label_to_group.get(l, '')
                group_map.setdefault(gk, []).append((h, l))
            # Sort groups
            group_order = sorted(group_map.keys())
            # Layout grid
            legend_ax.axis('off')
            eff_font = base_font * 1.06
            item_font = eff_font * 0.80
            left_pad = 0.04
            right_pad = 0.04
            top_pad = 0.25 if n_rows > 1 else 0.30
            bottom_pad = 0.10
            usable_w = 1 - left_pad - right_pad
            usable_h = 1 - top_pad - bottom_pad
            rg = max(0.0, min(0.25, row_gap_frac))
            gap_h = rg * usable_h if n_rows > 1 else 0.0
            total_gap_h = gap_h * (n_rows - 1)
            effective_h = max(1e-6, usable_h - total_gap_h)
            cell_w = usable_w / max(1, len(group_order))
            cell_h = effective_h / max(1, n_rows)
            for idx, gk in enumerate(group_order):
                row = idx // len(group_order)
                col = idx % len(group_order)
                x0 = left_pad + col * cell_w
                y0 = 1.0 - top_pad - row * (cell_h + gap_h)
                # Header
                legend_ax.text(x0, y0, gk if gk else '(other)', ha='left', va='top', fontsize=eff_font, fontweight='bold', transform=legend_ax.transAxes)
                items = group_map[gk]
                for i, (h, l) in enumerate(items):
                    y_item = y0 - (i + 1) * 0.07
                    # Draw legend line
                    legend_ax.plot([x0 + 0.024, x0 + 0.09], [y_item, y_item], color=h.get_color(), lw=h.get_linewidth(), alpha=h.get_alpha())
                    legend_ax.text(x0 + 0.10, y_item, l, ha='left', va='center', fontsize=item_font, color='black', transform=legend_ax.transAxes)
        else:
            # No grouping: just list all lines
            legend_ax.axis('off')
            base_font = mpl.rcParams.get('font.size', 9.0)
            item_font = base_font * 0.80
            left_pad = 0.04
            top_pad = 0.30
            for i, (h, l) in enumerate(zip(handles, labels)):
                y_item = 1.0 - top_pad - i * 0.07
                legend_ax.plot([left_pad + 0.024, left_pad + 0.09], [y_item, y_item], color=h.get_color(), lw=h.get_linewidth(), alpha=h.get_alpha())
                legend_ax.text(left_pad + 0.10, y_item, l, ha='left', va='center', fontsize=item_font, color='black', transform=legend_ax.transAxes)
    return fig, ax

        # (Removed unreachable, mis-indented code after return)

# ---------------------------------------------------------------------------
# Header-only grouped legend figure (Phase 1)
# ---------------------------------------------------------------------------
def make_legend_figure(
    df: pd.DataFrame,
    *,
    id_cols: Sequence[str] = ('policy_class', 'run_id'),
    label_col: str | None = 'label',
    group_cols: Sequence[str] | None = None,
    group_sep: str = ' / ',
    title: str | None = None,
    figsize: Tuple[float, float] = (6.5, 1.2),
    base_font: float = 9.0,
    max_entries: int | None = None,   # cap items per group (after unique + sort)
    wrap: bool = True,
    max_chars_per_line: int = 22,
    n_rows: int | None = None,        # if None: auto near-square; else fixed rows
    n_cols: int | None = None,        # optional explicit columns; else derived
    row_gap_frac: float = 0.0,        # vertical gap between grid rows (0–0.25)
) -> plt.Figure:
    """Build a grouped legend figure with headers and per-group item lists.

    Simplified API (reduced params):
      * Always shows per-group items (previous show_items=True).
      * Always sorts group headers alphabetically (previous sort='alpha').
      * Item labels are unique, alphabetically sorted, and bullet-prefixed.
      * Removed parameters: shorten, item_sort, item_font_scale, item_prefix,
        show_items, fill, auto_resize, header_font_delta, min_gap_frac,
        drop_blank (now always True), palette (unused), ncols (legacy alias).

    Remaining tunables: font size, wrapping, grid row count, row gap, figsize.
    """
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig
    if not group_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No group columns', ha='center', va='center')
        return fig

    def _group_key(row) -> str:
        parts: list[str] = []
        for c in group_cols:
            val = row.get(c, '')
            parts.append('' if pd.isna(val) else str(val))
        if all(p == '' for p in parts):
            return ''  # collapse fully empty group to empty string (cleaner than ' /  / ')
        return group_sep.join(parts)

    keys = df.apply(_group_key, axis=1)
    order: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)
    # Always alpha sort for deterministic ordering
    order = sorted(order)
    # Always drop blank groups
    if '' in order:
        order = [g for g in order if g != '']
    if not order:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        ax.text(0.5, 0.5, 'No groups', ha='center', va='center')
        return fig

    # Build per-group item lists if requested
    group_items: Dict[str, list[str]] = {}
    # Determine item labels: prefer label_col if present else join id_cols
    if label_col and label_col in df.columns:
        item_series = df[label_col].astype(str)
    else:
        item_series = df[id_cols].astype(str).agg('_'.join, axis=1) if isinstance(id_cols, (list, tuple)) else df[id_cols].astype(str)
    tmp = pd.DataFrame({'__gk__': keys, '__lab__': item_series})
    for gk, sub in tmp.groupby('__gk__'):
        uniq = sorted(dict.fromkeys(sub['__lab__'].tolist()))  # stable order then alpha sort
        if max_entries is not None:
            uniq = uniq[:max_entries]
        group_items[gk] = uniq

    n = len(order)

    # Always grid mode with items
    if (n_rows is None) and (n_cols is None):
        import math
        n_cols = max(1, int(math.ceil(n ** 0.5)))
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    elif n_cols is None and n_rows is not None:
        n_cols = max(1, (n + n_rows - 1) // n_rows)
    elif n_rows is None and n_cols is not None:
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    assert n_cols is not None and n_rows is not None
    capacity = n_rows * n_cols
    if capacity < n:
        extra_rows = (n + n_cols - 1) // n_cols
        n_rows = extra_rows
    # Auto figure height heuristic (always on)
    if figsize is not None:
        base_h = figsize[1]
        per_row = 0.85  # empirical height per row when items present
        new_h = max(base_h, 0.55 + (n_rows - 1) * per_row)
        if abs(new_h - base_h) > 1e-3:
            figsize = (figsize[0], new_h)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=base_font * 1.06 + 0.5,
                     y=0.93 if (n_rows <= 3) else 0.96)

    eff_font = base_font * 1.06  # fixed modest increase for headers

    def _wrap_text(txt: str) -> str:
        if not wrap or len(txt) <= max_chars_per_line:
            return txt
        # Prefer breaking at group separators first
        if group_sep.strip() and group_sep in txt:
            parts = txt.split(group_sep)
            lines: list[str] = []
            cur = ''
            for p in parts:
                seg = (p if cur == '' else group_sep + p)
                if len(cur.replace('\n', '')) + len(seg) > max_chars_per_line and cur:
                    lines.append(cur.strip())
                    cur = p
                else:
                    if cur:
                        cur += group_sep + p
                    else:
                        cur = p
            if cur:
                lines.append(cur.strip())
            return '\n'.join(lines)
        # Fallback simple chunking
        chunks = [txt[i:i + max_chars_per_line] for i in range(0, len(txt), max_chars_per_line)]
        return '\n'.join(chunks)

    def _estimate_width(txt: str) -> float:
        # Approx width in arbitrary units: characters * width_factor,
        # add small penalty for line breaks (multi-line takes vertical space but keep width of longest line)
        if '\n' in txt:
            lines = txt.split('\n')
            longest = max(lines, key=len)
            return len(longest) * 0.62 * eff_font
        return max(1, len(txt)) * 0.62 * eff_font

    processed: list[tuple[str, str, float]] = []  # (original, wrapped, est_width)
    for g in order:
        raw = g if g.strip() else '(blank)'
        wrapped = _wrap_text(raw)
        w_est = _estimate_width(wrapped)
        processed.append((raw, wrapped, w_est))

    # Build row-major coordinates
    _render_grouped_layout_on_axis(
        ax=ax,
        processed=processed,
        group_items=group_items,
        n_rows=n_rows,
        n_cols=n_cols,
        eff_font=eff_font,
        row_gap_frac=row_gap_frac,
    )

    return fig

__all__ = [
    'PALETTE_OKABE_ITO',
    'AxesConfig', 'HighlightConfig', 'SmoothingConfig', 'LegendConfig',
    'paper_style', 'plot_epochs_simple', 'make_legend_figure',
]

# ---------------------------------------------------------------------------
# Internal reusable legend rendering helpers
# ---------------------------------------------------------------------------
def _render_grouped_layout_on_axis(
    *,
    ax: plt.Axes,
    processed: list[tuple[str, str, float]],
    group_items: Dict[str, list[str]],
    n_rows: int,
    n_cols: int,
    eff_font: float,
    row_gap_frac: float,
    bullet_prefix: str = '• ',
    item_scale: float = 0.80,
):
    """Render headers + items given processed metadata onto an axis (axis off)."""
    n = len(processed)
    indices = list(range(n))
    coords = []
    r = c = 0
    for _i in indices:
        coords.append((r, c))
        c += 1
        if c >= n_cols:
            c = 0
            r += 1
    left_pad = 0.04
    right_pad = 0.04
    top_pad = 0.25 if n_rows > 1 else 0.30
    bottom_pad = 0.10
    usable_w = 1 - left_pad - right_pad
    usable_h = 1 - top_pad - bottom_pad
    rg = max(0.0, min(0.25, row_gap_frac))
    gap_h = rg * usable_h if n_rows > 1 else 0.0
    total_gap_h = gap_h * (n_rows - 1)
    effective_h = max(1e-6, usable_h - total_gap_h)
    cell_w = usable_w / max(1, n_cols)
    cell_h = effective_h / max(1, n_rows)
    item_font = eff_font * item_scale
    for (idx, (r, c)) in enumerate(coords):
        if idx >= len(processed):
            break
        gk_raw, wrapped, _ = processed[idx]
        cx = left_pad + cell_w * (c + 0.5)
        row_top_start = 1 - top_pad
        cy = row_top_start - (cell_h + gap_h) * r - cell_h / 2.0
        header_lines = wrapped.split('\n')
        items = group_items.get(gk_raw, [])
        display_items = [bullet_prefix + it for it in items]
        all_lines = header_lines + display_items
        if not all_lines:
            continue
        n_lines = len(all_lines)
        inner_pad = 0.08 * cell_h
        inner_height = cell_h - 2 * inner_pad
        line_step = inner_height / (n_lines - 1) if n_lines > 1 else 0
        top_y = cy + (inner_height / 2.0)
        for j, ln in enumerate(all_lines):
            is_header = j < len(header_lines)
            fsz = eff_font if is_header else item_font
            y_line = top_y - j * line_step
            ax.text(cx, y_line, ln, ha='center', va='center', fontsize=fsz)

def _render_grouped_legend_on_axis(
    ax: plt.Axes,
    *,
    df: pd.DataFrame,
    id_cols: Sequence[str],
    label_col: str | None,
    group_cols: Sequence[str] | None,
    group_sep: str,
    base_font: float,
    max_entries: int | None,
    wrap: bool,
    max_chars_per_line: int,
    n_rows: int | None,
    n_cols: int | None,
    row_gap_frac: float,
    plotted_labels: Optional[list[str]] = None,
    plotted_colors: Optional[list[str]] = None,
):
    if group_cols is None:
        ax.text(0.5, 0.5, 'No grouped legend cols', ha='center', va='center')
        return
    if df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return
    # Build keys
    def _group_key(row) -> str:
        parts = []
        for c in group_cols:
            val = row.get(c, '')
            parts.append('' if pd.isna(val) else str(val))
        if all(p == '' for p in parts):
            return ''
        return group_sep.join(parts)
    keys = df.apply(_group_key, axis=1)
    order = []
    seen: set[str] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            order.append(k)
    order = sorted(order)
    if '' in order:
        order = [g for g in order if g != '']
    if not order:
        ax.text(0.5, 0.5, 'No groups', ha='center', va='center')
        return
    # Items
    # If plotted_labels provided, filter items to only those
    if label_col and label_col in df.columns:
        item_series = df[label_col].astype(str)
    else:
        item_series = df[id_cols].astype(str).agg('_'.join, axis=1) if isinstance(id_cols, (list, tuple)) else df[id_cols].astype(str)
    tmp = pd.DataFrame({'__gk__': keys, '__lab__': item_series})
    group_items: Dict[str, list[str]] = {}
    allowed = set(plotted_labels) if plotted_labels is not None else None
    for gk, sub in tmp.groupby('__gk__'):
        uniq = sorted(dict.fromkeys(sub['__lab__'].tolist()))
        if allowed is not None:
            uniq = [u for u in uniq if u in allowed]
        if max_entries is not None:
            uniq = uniq[:max_entries]
        group_items[gk] = uniq

    # Build a mapping from label to color for swatches
    label_to_color = dict(zip(plotted_labels, plotted_colors)) if plotted_labels and plotted_colors else {}

    # Render grid of group headers and items
    ax.axis('off')
    import math
    n_groups = len(group_items)
    if n_rows is None and n_cols is None:
        n_cols = max(1, int(math.ceil(n_groups ** 0.5)))
        n_rows = max(1, (n_groups + n_cols - 1) // n_cols)
    elif n_cols is None and n_rows is not None:
        n_cols = max(1, (n_groups + n_rows - 1) // n_rows)
    elif n_rows is None and n_cols is not None:
        n_rows = max(1, (n_groups + n_cols - 1) // n_cols)
    assert n_cols is not None and n_rows is not None

    # Layout grid
    x_pad = 0.04
    y_pad = 0.08
    cell_w = (1.0 - 2 * x_pad) / n_cols
    cell_h = (1.0 - 2 * y_pad - (n_rows - 1) * row_gap_frac) / n_rows

    for idx, (gk, items) in enumerate(group_items.items()):
        row = idx // n_cols
        col = idx % n_cols
        x0 = x_pad + col * cell_w
        y0 = 1.0 - y_pad - row * (cell_h + row_gap_frac)
        # Header
        ax.text(x0, y0, gk if gk else '(other)', ha='left', va='top', fontsize=base_font + 1.2, fontweight='bold', transform=ax.transAxes)
        # Items: render as bulleted list with color swatch
        for i, lab in enumerate(items):
            y_item = y0 - (i + 1) * 0.07
            color = label_to_color.get(lab, 'gray')
            # Draw color swatch
            ax.add_patch(
                mpl.patches.Rectangle((x0, y_item - 0.012), 0.018, 0.018, color=color, transform=ax.transAxes, clip_on=False)
            )
            # Draw label
            ax.text(x0 + 0.024, y_item, f"• {lab}", ha='left', va='center', fontsize=base_font, color='black', transform=ax.transAxes)
    # Wrap + width estimation
    eff_font = base_font * 1.06
    def _wrap_text(txt: str) -> str:
        if not wrap or len(txt) <= max_chars_per_line:
            return txt
        if group_sep.strip() and group_sep in txt:
            parts = txt.split(group_sep)
            lines = []
            cur = ''
            for p in parts:
                seg = p if cur == '' else group_sep + p
                if len(cur.replace('\n', '')) + len(seg) > max_chars_per_line and cur:
                    lines.append(cur.strip())
                    cur = p
                else:
                    if cur:
                        cur += group_sep + p
                    else:
                        cur = p
            if cur:
                lines.append(cur.strip())
            return '\n'.join(lines)
        chunks = [txt[i:i + max_chars_per_line] for i in range(0, len(txt), max_chars_per_line)]
        return '\n'.join(chunks)
    processed: list[tuple[str, str, float]] = []
    for g in order:
        raw = g if g.strip() else '(blank)'
        wrapped = _wrap_text(raw)
        processed.append((raw, wrapped, 0.0))
    # Determine grid dims
    n = len(processed)
    if (n_rows is None) and (n_cols is None):
        import math
        n_cols = max(1, int(math.ceil(n ** 0.5)))
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    elif n_cols is None and n_rows is not None:
        n_cols = max(1, (n + n_rows - 1) // n_rows)
    elif n_rows is None and n_cols is not None:
        n_rows = max(1, (n + n_cols - 1) // n_cols)
    assert n_cols is not None and n_rows is not None
    _render_grouped_layout_on_axis(
        ax=ax,
        processed=processed,
        group_items=group_items,
        n_rows=n_rows,
        n_cols=n_cols,
        eff_font=eff_font,
        row_gap_frac=row_gap_frac,
    )
