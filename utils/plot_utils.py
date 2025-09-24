"""
Utilities for loading results, filtering, extracting fields, and plotting.
"""
from __future__ import annotations

import os
import json
from typing import Iterable, Optional, Dict, Any, Tuple, Union, Sequence

import pandas as pd
import matplotlib.pyplot as plt

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


def plot_epoch_rewards(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
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

    for (policy, run), grp in df2.sort_values(['policy_class', 'run_id', 'epoch']).groupby(['policy_class', 'run_id']):
        label = f"{policy} — {run[:8]}"
        ax.plot(grp['epoch'], grp['mean_reward'], label=label, lw=1.6, alpha=0.85)

    ax.set_title(title or 'Epoch vs Mean Reward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Reward')
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    ordered = sorted(zip(handles, labels), key=lambda x: x[1])
    if ordered:
        handles, labels = zip(*ordered)
        ax.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc='upper left', title='Trace')
    else:
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', title='Trace')

    plt.tight_layout()
    return fig, ax
