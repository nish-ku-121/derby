from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd


def list_parquet_files(root: str | os.PathLike, pattern: str = "epoch_agg__*.parquet") -> List[Path]:
    """Return a sorted list of epoch aggregate parquet files below ``root``.

    Lookup strategy:
      1. Look only at the top level (``root/epoch_agg__*.parquet``). If any
         matches are found, return those (legacy single-run / flat layout).
      2. If none are found at the top level, fall back to a recursive search
         (``rglob``) for the same filename pattern at any depth under ``root``.

    This supports historical layouts as well as newer sweep outputs such as:
        root/run_0001/epoch_agg__<uuid>.parquet
        root/run_0002/epoch_agg__<uuid>.parquet
    and also more deeply nested or sharded forms (date shards, worker shards, etc.).

    Notes:
      - The recursive phase is unconditional with respect to directory names; it no
        longer assumes a ``run_*`` prefix.
      - Returned paths are lexicographically sorted for stable concatenation order.
      - If ``root`` does not exist or no files are found, an empty list is returned.

    Parameters
    ----------
    root : str | os.PathLike
        Directory to search.
    pattern : str, default "epoch_agg__*.parquet"
        Glob pattern for epoch aggregate files.

    Returns
    -------
    List[Path]
        Sorted list of matching parquet file paths (may be empty).
    """
    p = Path(root)
    if not p.exists():
        return []
    flat = sorted(p.glob(pattern))
    if flat:
        return flat
    # Recursive fallback: search any depth beneath root for matching parquet files.
    # This replaces the previous one-level 'run_*' heuristic so new directory layouts
    # (e.g., nested date prefixes or shard folders) are automatically supported.
    # We still avoid returning the root twice by only doing this if no flat files were found.
    return sorted(p.rglob(pattern))


def load_epoch_agg(root: str | os.PathLike, pattern: str = "epoch_agg__*.parquet") -> pd.DataFrame:
    files = list_parquet_files(root, pattern)
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def basic_summary_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = df.groupby(["setup", "policy_class", "epoch"], as_index=False).agg(
        mean_of_means=("mean_reward", "mean"),
        runs=("run_id", "nunique"),
    )
    return grouped
