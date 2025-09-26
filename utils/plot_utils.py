"""
Utilities for loading results, filtering, extracting fields, and plotting.
"""
from __future__ import annotations

import os
import json
from typing import Iterable, Optional, Dict, Any, Tuple, Union, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import re
from matplotlib.lines import Line2D

from utils.epoch_agg_loader import load_epoch_agg


def load_epoch_agg_multi(paths: Iterable[str] | str) -> pd.DataFrame:
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    dfs = []
    for p in paths:
        try:
            dfi = load_epoch_agg(p)
            if dfi is not None and not dfi.empty:
                dfi = dfi.copy()
                dfi["source_dir"] = str(p)
                dfs.append(dfi)
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def filter_epoch_df(
    df: pd.DataFrame,
    *,
    setup: Optional[str] = None,
    agent_name: Optional[str] = None,
    label: Optional[str] = None,
    sort: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy()
    mask = pd.Series(True, index=df.index)
    if setup is not None and 'setup' in df.columns:
        mask &= (df['setup'] == setup)
    if agent_name is not None and 'agent_name' in df.columns:
        mask &= (df['agent_name'] == agent_name)
    if label is not None and 'label' in df.columns:
        mask &= (df['label'] == label)
    out = df[mask].copy()
    if sort:
        sort_cols = [c for c in ['policy_class', 'run_id', 'epoch'] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def extract_fields(
    df: pd.DataFrame,
    *,
    nested_column: str,
    fields: Union[str, Iterable[str]],
    new_column_name: Optional[str] = None,
    new_column_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if nested_column not in df.columns:
        raise KeyError(f"extract_fields: nested column '{nested_column}' not found in DataFrame columns: {list(df.columns)}")

    out = df.copy()
    if isinstance(fields, str):
        fields_list = [fields]
        col_names = [new_column_name or fields]
    else:
        fields_list = list(fields)
        if new_column_names is not None:
            if len(new_column_names) != len(fields_list):
                raise ValueError(
                    f"extract_fields: new_column_names length ({len(new_column_names)}) does not match number of fields ({len(fields_list)})."
                )
            col_names = list(new_column_names)
        else:
            col_names = fields_list

    values_per_field = {k: [] for k in fields_list}
    missing_counts = {k: 0 for k in fields_list}

    cache: Dict[Any, Dict[str, Any]] = {}
    lookup_cache: Dict[Tuple[Any, str], Any] = {}

    def _key_for_cache(x: Any) -> Any:
        if isinstance(x, str):
            return ("s", x)
        if isinstance(x, dict):
            return ("d", id(x))
        return ("o", str(x))

    for v in out[nested_column].values:
        base_key = _key_for_cache(v)
        if base_key in cache:
            d = cache[base_key]
        else:
            if isinstance(v, dict):
                d = v
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                d = {}
            elif isinstance(v, str):
                try:
                    d = json.loads(v)
                except Exception:
                    d = {}
            else:
                d = {}
            cache[base_key] = d

        for k in fields_list:
            lk = (base_key, k)
            if lk in lookup_cache:
                val = lookup_cache[lk]
            else:
                val = d.get(k) if isinstance(d, dict) else None
                lookup_cache[lk] = val
            if val is None and (not isinstance(d, dict) or k not in d):
                missing_counts[k] += 1
            values_per_field[k].append(val)

    for key, col_name in zip(fields_list, col_names):
        out[col_name] = values_per_field[key]
        if missing_counts[key]:
            print(f"Warning: extract_fields: {missing_counts[key]} rows missing key '{key}' in column '{nested_column}'. Extracted None for those rows.")

    return out


def last_epoch_table(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("policy_class", "run_id"),
    epoch_col: str = "epoch",
    value_col: str = "mean_reward",
    sort_desc: bool = True,
    extra_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a table of last-epoch values per group from a standard epoch-aggregate DataFrame.

    Parameters
    - df: DataFrame with standard schema including group columns, an epoch column, and a value column.
    - group_cols: Columns to group by (default: (policy_class, run_id)).
    - epoch_col: Name of the epoch column (default: 'epoch').
    - value_col: Name of the value column to extract at last epoch (default: 'mean_reward').
    - sort_desc: Sort the output by the last-epoch value descending (default: True).

    - extra_cols: Optional additional columns to include from the selected last rows (e.g., 'policy_params_json').

    Returns
    - DataFrame with columns [*group_cols, 'last_epoch', f'last_epoch_{value_col}', *extra_cols] sorted by the last-epoch value.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[*group_cols, 'last_epoch', f'last_epoch_{value_col}'])
    for c in list(group_cols) + [epoch_col, value_col]:
        if c not in df.columns:
            raise KeyError(f"last_epoch_table: required column '{c}' not found in DataFrame.")

    tmp = df.copy()
    tmp['_epoch_num'] = pd.to_numeric(tmp[epoch_col], errors='coerce')
    # Sort to ensure tail(1) picks the max epoch row per group
    tmp = tmp.sort_values([*group_cols, '_epoch_num'])
    sel_cols = [*group_cols, value_col, '_epoch_num']
    if extra_cols:
        sel_cols.extend([c for c in extra_cols if c in tmp.columns and c not in sel_cols])
    last_rows = (
        tmp.groupby(list(group_cols), as_index=False)
           .tail(1)
           [sel_cols]
           .rename(columns={value_col: f'last_epoch_{value_col}', '_epoch_num': 'last_epoch'})
    )
    if sort_desc:
        last_rows = last_rows.sort_values(f'last_epoch_{value_col}', ascending=False).reset_index(drop=True)
    else:
        last_rows = last_rows.reset_index(drop=True)
    return last_rows


def summarize_experiment_metadata(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = ("learning_rate", "setup"),
) -> Dict[str, str]:
    """Summarize minimal experiment metadata from a plotting DataFrame.

    Attempts to extract the following when requested in `keys`:
    - 'setup': from df['setup'] if present (single value or 'mixed').
    - 'learning_rate': from df['learning_rate'] if present (single value or 'mixed').
    - 'epochs': max epoch number from df['epoch'] (numeric), formatted as an integer.
    - 'days': from any of [num_days, n_days, days] if present (single value or 'mixed').
    - 'trajs': from any of [num_trajs, n_trajs, trajectories, n_trajectories, num_trajectories].

    Returns a dict of key -> string value for keys that could be resolved.
    """
    if df is None or df.empty:
        return {}

    def _unique_or_mixed(series: pd.Series) -> Optional[str]:
        vals = pd.unique(series.dropna())
        if len(vals) == 0:
            return None
        if len(vals) == 1:
            v = vals[0]
            # pretty-format numbers
            try:
                fv = float(v)
                # compact formatting: up to 5 significant digits
                return f"{fv:.5g}"
            except Exception:
                return str(v)
        return "mixed"

    out: Dict[str, str] = {}

    if "setup" in keys and 'setup' in df.columns:
        v = _unique_or_mixed(df['setup'])
        if v is not None:
            out['setup'] = v

    if "learning_rate" in keys and 'learning_rate' in df.columns:
        v = _unique_or_mixed(df['learning_rate'])
        if v is not None:
            out['learning_rate'] = v

    if "epochs" in keys and 'epoch' in df.columns:
        e = pd.to_numeric(df['epoch'], errors='coerce')
        if not e.dropna().empty:
            out['epochs'] = str(int(e.max()))

    if "days" in keys:
        for c in ['num_days', 'n_days', 'days']:
            if c in df.columns:
                v = _unique_or_mixed(df[c])
                if v is not None:
                    out['days'] = v
                    break

    if "trajs" in keys:
        for c in ['num_trajs', 'n_trajs', 'trajectories', 'n_trajectories', 'num_trajectories']:
            if c in df.columns:
                v = _unique_or_mixed(df[c])
                if v is not None:
                    out['trajs'] = v
                    break

    return out


def plot_epoch_rewards(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    # y-axis bounds
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    # x-axis bounds
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    # plot downsampling: 1 = plot every epoch, 2 = every other, etc.
    plot_every_epoch: int = 1,
    # legend placement: 'bottom' (default), 'right', or any matplotlib loc string
    legend_position: str = 'bottom',
    # whether to include run_id suffix only when needed to distinguish within same policy
    show_run_id_when_needed: bool = True,
    # whether to strip a common suffix from policy names for readability
    clean_policy_names: bool = True,
    # highlighting & styling
    highlight_best: bool = True,
    top_k: int = 1,
    dim_others: bool = True,
    others_use_gray: bool = True,
    differentiate_runs: bool = True,
    use_end_annotation_for_best: bool = False,
    best_tiebreak_window: int = 50,
    legend_title: str = 'Policy',
    palette: Union[str, Sequence[str]] = 'okabe-ito',
    # halo/outline for highlighted curves
    top_k_outline: bool = True,
    outline_color: str = 'white',
    outline_width: float = 2.5,
    # markers for highlighted curves (placed on plotted points, which already respect plot_every_epoch)
    top_k_markers: bool = True,
    top_k_marker_styles: Sequence[str] = ('o', 's', '^', 'D', 'P', 'X'),
    top_k_marker_size: float = 4.0,
    # optional separate cadence for markers (epochs). If >1, markers will be shown roughly every N epochs
    # based on the actual epochs present for each plotted line. If None or <=1, markers appear on all plotted points.
    top_k_marker_every_epoch: Optional[int] = None,
    # distinct linestyles for highlighted curves to handle exact overlaps
    top_k_linestyles: Sequence[str] = ('-', '--', ':', '-.'),
    # legend grouping: provide a regex to extract a group token from policy names (e.g., r"(v\d+(?:_\d+)?)").
    # If provided, legend entries will be grouped by the first capture group. Unmatched entries go to 'Other'.
    legend_group_regex: Optional[str] = None,
    # if True and legend_group_regex is set, arrange groups side-by-side as columns with headers.
    legend_group_side_by_side: bool = False,
    # minimal metadata caption
    show_metadata: bool = False,
    metadata_keys: Sequence[str] = ("learning_rate", "setup"),
    metadata_loc: str = 'top-left',  # one of: top-left, top-right, bottom-left, bottom-right
    metadata_fontsize: float = 9.0,
    metadata_box_alpha: float = 0.6,
):
    """Plot epoch vs. mean reward lines for each (policy_class, run_id).

    Parameters
    - df: DataFrame with at least columns {policy_class, run_id, epoch, mean_reward}.
           Optionally may contain std_reward for the std band overlay.
    - title: Optional chart title.
    - ax: Optional matplotlib Axes to draw into; creates one if None.
    - y_min, y_max: Optional y-axis bounds.
    - x_min, x_max: Optional x-axis bounds (epoch range).
    - plot_every_epoch: Downsampling factor for epochs. 1 plots all epochs, 2 plots every other, etc.
    - legend_position: Where to place the legend. 'bottom' (default) arranges a multi-column legend below
                       the plot; 'right' places it to the right; any other matplotlib loc string is accepted.
    - show_run_id_when_needed: Only append a run_id suffix in legend labels when a policy has multiple runs.
    - clean_policy_names: If True, strips suffixes like '_MarketEnv_Continuous' from policy names for readability.
    - highlight_best: If True, highlight the best-performing curve(s) and dim the rest for context.
    - top_k: Number of top curves to highlight (default 1).
    - dim_others: If True, non-top curves are drawn thinner and with reduced alpha.
    - others_use_gray: If True, draw non-top curves in gray; otherwise keep their assigned color but dimmed.
    - differentiate_runs: If True, vary line style for runs when a policy has multiple runs.
    - use_end_annotation_for_best: If True, annotate the final point for top curves.
    - best_tiebreak_window: Window size for tie-break using mean of last-N epochs.
        - legend_title: Title to use for the legend (default 'Policy').
    - palette: Color palette for highlighted curves (top_k). 'okabe-ito' or a sequence of color codes.
        - top_k_outline: If True, draw a white halo under highlighted lines for better separation.
        - outline_color, outline_width: Appearance of the halo.
            - top_k_markers: If True, add distinct markers to highlighted lines. Markers will appear on plotted points
            (which are already downsampled by plot_every_epoch), avoiding extra density parameters.
        - top_k_marker_styles, top_k_marker_size: Marker styling for highlighted curves.
        - top_k_marker_every_epoch: Optional separate cadence for markers (in epochs). If provided and >1, markers are
            drawn at approximately every N epochs using the epochs present in the plotted line; otherwise on all plotted points.
            - top_k_linestyles: Linestyles assigned to highlighted curves in rank order to keep overlapping lines visible.
        - legend_group_regex: Optional regex to extract a subgroup token from policy names for legend grouping.
            Provide a pattern with one capturing group (e.g., r"(v\\d+(?:_\\d+)?)"). Entries that don't match are grouped
            under 'Other'. If None, legend is flat (no subgroup headers).
        - legend_group_side_by_side: If True and grouping is enabled, lay out groups as columns with bold headers
            (side-by-side) instead of a single stacked list.
            - show_metadata: If True, draw a small caption with minimal metadata inside the axes.
            - metadata_keys: Which fields to include (subset of: learning_rate, setup, epochs, days, trajs).
            - metadata_loc: Corner to place metadata box: top-left, top-right, bottom-left, bottom-right.
            - metadata_fontsize: Font size for metadata text.
            - metadata_box_alpha: Background box transparency for metadata.

    Returns
    - (fig, ax): The matplotlib Figure and Axes used for plotting.
    """
    required_cols = {"policy_class", "run_id", "epoch", "mean_reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if df is None or df.empty:
        ax.set_title(title or 'No data to plot')
        return fig, ax

    df2 = df.copy()
    df2['run_id'] = df2['run_id'].astype(str)

    # Optionally downsample epochs per (policy, run)
    if isinstance(plot_every_epoch, (int,)) and plot_every_epoch > 1:
        def _downsample(grp: pd.DataFrame) -> pd.DataFrame:
            g = grp.sort_values('epoch')
            e = pd.to_numeric(g['epoch'], errors='coerce')
            if e.isna().all():
                return g
            min_e = int(e.min())
            max_e = e.max()
            # Keep first and last, and every Nth relative to min
            mask = ((e - min_e) % plot_every_epoch == 0) | (e == max_e) | (e == min_e)
            return g[mask]

        df2 = (
            df2.groupby(['policy_class', 'run_id'], group_keys=False)
               .apply(_downsample)
               .reset_index(drop=True)
        )

    dup_points = (
        df2.groupby(['policy_class', 'run_id', 'epoch']).size().reset_index(name='count')
    )
    bad = dup_points[dup_points['count'] > 1]
    if not bad.empty:
        offenders = bad.head(10).to_dict(orient='records')
        raise ValueError(
            "Duplicate rows detected for (policy_class, run_id, epoch). "
            f"Please deduplicate before plotting. Examples: {offenders}"
        )

    # Helper to pretty policy labels
    def _clean_policy_name(name: str) -> str:
        if not clean_policy_names or not isinstance(name, str):
            return str(name)
        return name.replace('_MarketEnv_Continuous', '')

    # Determine how many runs per policy to optionally hide run_id in legend
    runs_per_policy = (
        df2.groupby('policy_class')['run_id'].nunique().to_dict()
        if 'policy_class' in df2.columns else {}
    )

    # Determine top-k best curves by last-epoch mean reward (tie-break by last-N mean)
    best_keys: set[Tuple[str, str]] = set()
    best_ordered_list: list[Tuple[str, str]] = []
    if highlight_best and not df2.empty:
        dfl = df2.copy()
        dfl['epoch_num'] = pd.to_numeric(dfl['epoch'], errors='coerce')
        stats_rows = []
        for (policy, run), grp in dfl.groupby(['policy_class', 'run_id']):
            g = grp.sort_values('epoch_num')
            if g['epoch_num'].isna().all():
                continue
            idx_last = g['epoch_num'].idxmax()
            last_val = float(g.loc[idx_last, 'mean_reward'])
            if best_tiebreak_window and best_tiebreak_window > 1:
                tail = g.tail(best_tiebreak_window)
                lastn_mean = float(pd.to_numeric(tail['mean_reward'], errors='coerce').mean())
            else:
                lastn_mean = last_val
            stats_rows.append((policy, run, last_val, lastn_mean))
        if stats_rows:
            stats_df = pd.DataFrame(stats_rows, columns=['policy_class', 'run_id', 'last_mean', 'lastn_mean'])
            stats_df = stats_df.sort_values(['last_mean', 'lastn_mean'], ascending=[False, False])
            # Clip top_k to available rows
            k = max(1, min(int(top_k), len(stats_df))) if top_k is not None else 1
            best_ordered_list = list(stats_df[['policy_class', 'run_id']].head(k).itertuples(index=False, name=None))
            best_keys = set(best_ordered_list)

    # Prepare palette for highlighted curves
    if isinstance(palette, str):
        pal_name = palette.lower()
        if pal_name in ['okabe-ito', 'okabe_ito', 'okabeito', 'oi']:
            # Okabe–Ito color-blind friendly palette
            highlight_colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        elif pal_name in ['tab10', 'tableau10']:
            highlight_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']
        elif pal_name in ['tab20']:
            # Use Matplotlib tab20 colors
            highlight_colors = [
                '#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c','#98df8a','#d62728','#ff9896',
                '#9467bd','#c5b0d5','#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f','#c7c7c7',
                '#bcbd22','#dbdb8d','#17becf','#9edae5'
            ]
        else:
            # Fallback to Matplotlib default cycle colors later by not forcing colors
            highlight_colors = None
    else:
        highlight_colors = list(palette)

    # Build a mapping from best keys to colors cycling through palette
    best_color_map: Dict[Tuple[str, str], Optional[str]] = {}
    best_marker_map: Dict[Tuple[str, str], Optional[str]] = {}
    best_ls_map: Dict[Tuple[str, str], str] = {}
    if best_keys:
        if highlight_colors is None or len(highlight_colors) == 0:
            cols = [None] * len(best_keys)  # let mpl choose
        else:
            cols = [highlight_colors[i % len(highlight_colors)] for i in range(len(best_keys))]
        # assign colors and marker styles in ranked order for consistency
        for i, key in enumerate(best_ordered_list):
            col = cols[i] if i < len(cols) else None
            best_color_map[key] = col
            if top_k_markers:
                best_marker_map[key] = top_k_marker_styles[i % len(top_k_marker_styles)]
            best_ls_map[key] = top_k_linestyles[i % len(top_k_linestyles)]

    # Linestyle cycle for differentiating runs
    linestyle_cycle = ['-', '--', ':', '-.']
    def _linestyle_for_run(policy: str, run: str) -> str:
        if not differentiate_runs:
            return '-'
        n = runs_per_policy.get(policy, 1)
        if n <= 1:
            return '-'
        # stable mapping by run id
        idx = abs(hash(run)) % len(linestyle_cycle)
        return linestyle_cycle[idx]

    # Plot lines with highlighting logic
    line_for_group: Dict[Tuple[str, str], Any] = {}
    label_for_line: Dict[Any, str] = {}
    group_for_line: Dict[Any, Optional[str]] = {}

    def _legend_group_for_policy(policy: str) -> Optional[str]:
        if not legend_group_regex:
            return None
        pat = legend_group_regex
        try:
            m = re.search(pat, policy)
        except re.error:
            m = None
        if m:
            return m.group(1)
        return 'Other'
    for (policy, run), grp in df2.sort_values(['policy_class', 'run_id', 'epoch']).groupby(['policy_class', 'run_id']):
        base_label = _clean_policy_name(policy)
        if show_run_id_when_needed and runs_per_policy.get(policy, 1) > 1:
            label = f"{base_label} — {run[:8]}"
        else:
            label = base_label

        is_best = (policy, run) in best_keys if highlight_best else False
        if is_best:
            # Distinct, bright colors for highlighted curves
            color = best_color_map.get((policy, run))
            lw = 2.6
            alpha = 1.0
            ls = best_ls_map.get((policy, run), '-')
            z = 3
            marker = best_marker_map.get((policy, run)) if top_k_markers else None
            marker_kwargs = {
                'marker': marker,
                'markersize': top_k_marker_size,
                'markeredgecolor': 'white',
                'markeredgewidth': 0.75,
            } if marker else {}
            # Optional separate marker cadence based on epoch values in this group
            if marker and top_k_marker_every_epoch and top_k_marker_every_epoch > 1:
                e_num = pd.to_numeric(grp['epoch'], errors='coerce')
                if not e_num.isna().all():
                    try:
                        min_e = int(e_num.min())
                        max_e = e_num.max()
                        positions = []
                        for idx, ev in enumerate(e_num.values):
                            if pd.isna(ev):
                                continue
                            iv = int(ev)
                            if (iv - min_e) % int(top_k_marker_every_epoch) == 0 or ev == min_e or ev == max_e:
                                positions.append(idx)
                        if positions:
                            marker_kwargs['markevery'] = positions
                    except Exception:
                        pass
        else:
            ls = _linestyle_for_run(policy, run)
            if dim_others:
                lw = 1.1
                alpha = 0.28
            else:
                lw = 1.4
                alpha = 0.8
            color = '0.65' if others_use_gray else None  # None lets matplotlib pick distinct colors
            z = 2 if others_use_gray else 1

        (line,) = ax.plot(
            grp['epoch'], grp['mean_reward'],
            label=label, lw=lw, alpha=alpha, linestyle=ls, color=color, zorder=z,
            **(marker_kwargs if is_best else {})
        )
        line_for_group[(policy, run)] = line
        label_for_line[line] = label
        group_for_line[line] = _legend_group_for_policy(policy)

        # Add halo/outline under highlighted lines to improve separation
        if is_best and top_k_outline:
            line.set_path_effects([
                pe.Stroke(linewidth=lw + outline_width, foreground=outline_color, alpha=alpha),
                pe.Normal(),
            ])

        # Endpoint annotation for best
        if is_best and use_end_annotation_for_best:
            gsort = grp.sort_values('epoch')
            xe = gsort['epoch'].iloc[-1]
            ye = float(gsort['mean_reward'].iloc[-1])
            ax.scatter([xe], [ye], s=36, color=line.get_color(), zorder=4)
            ax.annotate(f"{base_label}\n{ye:.3f}", xy=(xe, ye), xytext=(5, 5), textcoords='offset points', fontsize=9, color=line.get_color(), ha='left', va='bottom')

    ax.set_title(title or 'Epoch vs Mean Reward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Reward')
    ax.grid(True, alpha=0.3)

    # y-axis limits
    if y_min is not None or y_max is not None:
        ymin_current, ymax_current = ax.get_ylim()
        ymin_new = y_min if y_min is not None else ymin_current
        ymax_new = y_max if y_max is not None else ymax_current
        ax.set_ylim(ymin_new, ymax_new)

    # x-axis limits
    if x_min is not None or x_max is not None:
        xmin_current, xmax_current = ax.get_xlim()
        xmin_new = x_min if x_min is not None else xmin_current
        xmax_new = x_max if x_max is not None else xmax_current
        ax.set_xlim(xmin_new, xmax_new)

    # Legend improvements: by default put legend below the plot to avoid squashing width
    handles, labels = ax.get_legend_handles_labels()

    # Build grouped legend if a grouping regex is provided
    any_groups = bool(legend_group_regex) and any(group_for_line.get(h) is not None for h in handles)
    if any_groups:
        # Pair each entry with its group
        items = []  # (group, label, handle)
        for h, lab in zip(handles, labels):
            grp = group_for_line.get(h)
            # Only include entries with labels (mpl already filters _nolegend_)
            items.append((grp or 'Other', lab, h))

        # Sort by group name, then by label
        items.sort(key=lambda t: (t[0], t[1]))

        grouped: Dict[str, list[Tuple[str, Any]]] = {}
        for grp, lab, h in items:
            grouped.setdefault(grp, []).append((lab, h))

        group_names = sorted(grouped.keys())

        if legend_group_side_by_side:
            # Arrange as columns in column-major order expected by Matplotlib's legend when ncol > 1.
            # Build a contiguous block per group: [HEADER, entry1, entry2, ..., padding]
            # and then concatenate blocks for all groups. With ncol = number of groups,
            # the legend will place each block as a separate column.
            max_rows = max(1 + len(entries) for entries in grouped.values())  # header + entries
            new_handles: list[Any] = []
            new_labels: list[str] = []
            header_positions: set[int] = set()

            for g_idx, grp in enumerate(group_names):
                entries = grouped[grp]
                # header at the top of the column
                header = Line2D([], [], linestyle='None', marker=None, color='none')
                new_handles.append(header)
                new_labels.append(str(grp))
                header_positions.add(g_idx * max_rows)  # position after concatenation across groups

                # add entries for this group
                for lab, h in entries:
                    new_handles.append(h)
                    new_labels.append(lab)

                # pad to max_rows with placeholders
                pad_needed = max_rows - (1 + len(entries))
                for _ in range(pad_needed):
                    placeholder = Line2D([], [], linestyle='None', marker=None, color='none')
                    new_handles.append(placeholder)
                    new_labels.append(" ")

            handles, labels = new_handles, new_labels

            # Place legend with columns = number of groups
            if legend_position == 'bottom':
                fig.subplots_adjust(bottom=0.26)
                leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=len(group_names), frameon=True, title=legend_title)
            elif legend_position == 'right':
                leg = ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc='upper left', ncol=len(group_names), title=legend_title)
            else:
                leg = ax.legend(handles, labels, loc=legend_position, ncol=len(group_names), title=legend_title)

            # Bold headers
            for i, txt in enumerate(leg.get_texts()):
                if i in header_positions:
                    txt.set_weight('bold')
        else:
            # Stacked grouped legend (single column with headers)
            new_handles: list[Any] = []
            new_labels: list[str] = []
            header_labels: set[str] = set()
            for grp in group_names:
                entries = grouped[grp]
                header = Line2D([], [], linestyle='None', marker=None, color='none')
                new_handles.append(header)
                new_labels.append(str(grp))
                header_labels.add(str(grp))
                for lab, h in entries:
                    new_handles.append(h)
                    new_labels.append(lab)

            handles, labels = new_handles, new_labels

            if legend_position == 'bottom':
                fig.subplots_adjust(bottom=0.26)
                leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=True, title=legend_title)
            elif legend_position == 'right':
                leg = ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc='upper left', ncol=1, title=legend_title)
            else:
                leg = ax.legend(handles, labels, loc=legend_position, ncol=1, title=legend_title)

            for txt in leg.get_texts():
                if txt.get_text() in header_labels:
                    txt.set_weight('bold')
    else:
        # Flat legend (no grouping)
        ordered = sorted(zip(handles, labels), key=lambda x: x[1])
        if ordered:
            handles, labels = zip(*ordered)
        else:
            handles, labels = (handles, labels)

        if legend_position == 'bottom':
            # Place legend at the bottom with multiple columns
            n_items = len(labels)
            ncol = 1 if n_items <= 1 else 2 if n_items <= 8 else 3 if n_items <= 18 else 4
            # Leave space at the bottom for the legend
            fig.subplots_adjust(bottom=0.22)
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol, frameon=True, title=legend_title)
        elif legend_position == 'right':
            ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc='upper left', title=legend_title)
        else:
            # Use whatever matplotlib understands for loc
            ax.legend(handles, labels, loc=legend_position, title=legend_title)

    plt.tight_layout()

    # Optional minimal metadata caption
    if show_metadata:
        meta = summarize_experiment_metadata(df2, keys=metadata_keys)
        if meta:
            # build multi-line label with short keys
            key_map = {
                'learning_rate': 'lr',
                'setup': 'setup',
                'epochs': 'epochs',
                'days': 'days',
                'trajs': 'trajs',
            }
            lines = []
            for k in metadata_keys:
                if k in meta:
                    lines.append(f"{key_map.get(k, k)}: {meta[k]}")
            if lines:
                text = "\n".join(lines)
                loc_map = {
                    'top-left': (0.01, 0.99, 'left', 'top'),
                    'top-right': (0.99, 0.99, 'right', 'top'),
                    'bottom-left': (0.01, 0.01, 'left', 'bottom'),
                    'bottom-right': (0.99, 0.01, 'right', 'bottom'),
                }
                x, y, ha, va = loc_map.get(metadata_loc, (0.01, 0.99, 'left', 'top'))
                ax.text(
                    x, y, text,
                    transform=ax.transAxes,
                    fontsize=metadata_fontsize,
                    ha=ha, va=va,
                    zorder=5,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=metadata_box_alpha, boxstyle='round,pad=0.2'),
                )
    return fig, ax
