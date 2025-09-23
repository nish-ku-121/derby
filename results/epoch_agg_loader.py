from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def list_parquet_files(root: str | os.PathLike, pattern: str = "epoch_agg__*.parquet") -> List[Path]:
    """List run-level parquet files under a directory.

    Args:
        root: Directory containing run parquet files.
        pattern: Glob pattern to match files. Defaults to "epoch_agg__*.parquet".

    Returns:
        List of matching file paths.
    """
    p = Path(root)
    return sorted(p.glob(pattern)) if p.exists() else []


def load_epoch_agg(root: str | os.PathLike, pattern: str = "epoch_agg__*.parquet") -> pd.DataFrame:
    """Load and combine epoch aggregate parquet files from a directory.

    Each parquet is expected to be a per-run file produced by the YAML runner.

    Columns include: run_id, config_hash, label, setup, num_days, num_trajs, num_epochs, epoch,
    agent_name, policy_class, policy_params_json, mean_reward, std_reward, n_trajs

    Args:
        root: Directory containing run parquet files.
        pattern: Glob pattern to match files.

    Returns:
        Pandas DataFrame concatenating all matching files (empty DataFrame if none).
    """
    files = list_parquet_files(root, pattern)
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def basic_summary_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a simple cross-run summary per policy per epoch.

    Aggregates mean of mean_reward across runs and attaches count of runs.
    Note: mean_reward here is already averaged over trajectories within each run.

    Returns a DataFrame with columns: setup, policy_class, epoch, mean_of_means, runs
    """
    if df.empty:
        return df
    grouped = df.groupby(["setup", "policy_class", "epoch"], as_index=False).agg(
        mean_of_means=("mean_reward", "mean"),
        runs=("run_id", "nunique"),
    )
    return grouped
