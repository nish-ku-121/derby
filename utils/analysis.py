from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Mapping, Sequence

import pandas as pd

from utils.epoch_agg_loader import list_parquet_files, load_epoch_agg


def load_epoch_rewards(
    paths: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    pattern: str = "epoch_agg__*.parquet",
    source_col: str | None = "source_dir",
) -> pd.DataFrame:
    """Load one or more epoch-aggregate result roots.

    The returned DataFrame is still the canonical Parquet row shape. This helper
    only adds an optional source column when multiple roots are provided.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = load_epoch_agg(path, pattern=pattern)
        if df.empty:
            continue
        if source_col:
            df = df.copy()
            df[source_col] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def expand_policy_params(
    df: pd.DataFrame,
    *,
    column: str = "policy_params_json",
    fields: Sequence[str] | None = None,
    prefix: str = "param_",
) -> pd.DataFrame:
    """Expand JSON policy params into scalar columns.

    Missing params become NA without warnings. This keeps baseline rows quiet
    when learner-only fields such as learning_rate are requested.
    """
    if df.empty:
        return df.copy()
    if column not in df.columns:
        raise KeyError(f"policy params column not found: {column}")

    parsed: list[Mapping[str, object]] = []
    discovered: set[str] = set()
    cache: dict[object, Mapping[str, object]] = {}

    for value in df[column].values:
        key = value if isinstance(value, str) else id(value)
        if key in cache:
            params = cache[key]
        elif isinstance(value, Mapping):
            params = value
            cache[key] = params
        elif value is None or (isinstance(value, float) and pd.isna(value)):
            params = {}
            cache[key] = params
        elif isinstance(value, str):
            try:
                loaded = json.loads(value)
                params = loaded if isinstance(loaded, Mapping) else {}
            except json.JSONDecodeError:
                params = {}
            cache[key] = params
        else:
            params = {}
            cache[key] = params

        parsed.append(params)
        discovered.update(str(k) for k in params.keys())

    selected = list(fields) if fields is not None else sorted(discovered)
    out = df.copy()
    for field in selected:
        out[f"{prefix}{field}"] = [params.get(field, pd.NA) for params in parsed]
    return out


def filter_epoch_rewards(
    df: pd.DataFrame,
    *,
    agent_name: str | None = None,
    agent_label_prefix: str | None = None,
    policy_class: str | None = None,
    setup: str | None = None,
) -> pd.DataFrame:
    """Apply common exact-match filters to an epoch reward DataFrame."""
    out = df
    if agent_name is not None and "agent_name" in out.columns:
        out = out[out["agent_name"] == agent_name]
    if policy_class is not None and "policy_class" in out.columns:
        out = out[out["policy_class"] == policy_class]
    if setup is not None and "setup" in out.columns:
        out = out[out["setup"] == setup]
    if agent_label_prefix is not None and "agent_label" in out.columns:
        out = out[out["agent_label"].astype(str).str.startswith(agent_label_prefix, na=False)]
    return out.copy()


def last_epoch_table(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("agent_label", "run_id", "agent_name"),
    epoch_col: str = "epoch",
    value_col: str = "mean_reward",
    include_cols: Sequence[str] = ("agent_name", "policy_class"),
    sort_desc: bool = True,
) -> pd.DataFrame:
    """Return one final-epoch row per group, sorted by final reward."""
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, "last_epoch", f"last_epoch_{value_col}", *include_cols]
        )

    required = set(group_cols) | {epoch_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    work = df.copy()
    work["_epoch_num"] = pd.to_numeric(work[epoch_col], errors="coerce")
    keep_cols = [*group_cols, "_epoch_num", value_col]
    keep_cols.extend(c for c in include_cols if c in work.columns and c not in keep_cols)

    rows = (
        work.sort_values([*group_cols, "_epoch_num"])
        .groupby(list(group_cols), dropna=False, as_index=False)
        .tail(1)[keep_cols]
        .rename(columns={"_epoch_num": "last_epoch", value_col: f"last_epoch_{value_col}"})
        .reset_index(drop=True)
    )
    if sort_desc:
        rows = rows.sort_values(f"last_epoch_{value_col}", ascending=False).reset_index(drop=True)
    return rows


def inspect_epoch_rewards(root: str | os.PathLike) -> dict[str, object]:
    """Collect a compact, printable summary for a result root."""
    files = list_parquet_files(root)
    df = load_epoch_agg(root)
    summary: dict[str, object] = {
        "root": str(root),
        "files": [str(path) for path in files],
        "rows": int(len(df)),
        "columns": list(df.columns),
    }
    if df.empty:
        return summary

    for col in ("agent_name", "agent_label", "policy_class", "run_id", "setup"):
        if col in df.columns:
            vals = sorted(str(v) for v in df[col].dropna().unique())
            summary[col] = vals
    if "epoch" in df.columns:
        epochs = pd.to_numeric(df["epoch"], errors="coerce")
        summary["epoch_min"] = None if epochs.dropna().empty else int(epochs.min())
        summary["epoch_max"] = None if epochs.dropna().empty else int(epochs.max())
    return summary


def _print_inspection(root: str | os.PathLike, *, head: int) -> None:
    summary = inspect_epoch_rewards(root)
    print(f"root: {summary['root']}")
    print(f"parquet_files: {len(summary['files'])}")
    for path in summary["files"]:
        print(f"  {path}")
    print(f"rows: {summary['rows']}")
    print(f"columns: {', '.join(summary['columns'])}")

    for col in ("setup", "agent_name", "agent_label", "policy_class", "run_id"):
        vals = summary.get(col)
        if vals:
            print(f"{col}s: {len(vals)}")
            for value in vals[:head]:
                print(f"  {value}")
            if len(vals) > head:
                print(f"  ... {len(vals) - head} more")

    if "epoch_min" in summary:
        print(f"epoch_range: {summary['epoch_min']}..{summary['epoch_max']}")

    df = load_epoch_agg(root)
    if not df.empty:
        group_cols = tuple(c for c in ("agent_label", "run_id", "agent_name") if c in df.columns)
        table = last_epoch_table(df, group_cols=group_cols)
        if not table.empty:
            print("last_epoch_table:")
            print(table.head(head).to_string(index=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect Derby epoch aggregate Parquet outputs.")
    parser.add_argument("root", help="Result root or run directory containing epoch_agg__*.parquet files")
    parser.add_argument("--head", type=int, default=12, help="Number of values/table rows to print")
    args = parser.parse_args(argv)
    _print_inspection(args.root, head=args.head)


if __name__ == "__main__":
    main()
