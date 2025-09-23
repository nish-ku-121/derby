from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def list_parquet_files(root: str | os.PathLike, pattern: str = "epoch_agg__*.parquet") -> List[Path]:
	p = Path(root)
	return sorted(p.glob(pattern)) if p.exists() else []


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
