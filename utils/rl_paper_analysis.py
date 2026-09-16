"""Reusable analysis helpers for the RL paper result set.

These functions keep run-level evidence intact until a caller explicitly aggregates
across seeds. They intentionally do not classify numerical health: that requires
the retained audit/configuration evidence for each experiment group.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_RUN_COLS: tuple[str, ...] = ("run_id",)


def require_unique_run_epochs(
    df: pd.DataFrame,
    *,
    run_cols: Sequence[str] = DEFAULT_RUN_COLS,
    epoch_col: str = "epoch",
) -> None:
    """Raise when a run contains more than one row for an epoch."""
    required = set(run_cols) | {epoch_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    duplicates = df.duplicated([*run_cols, epoch_col], keep=False)
    if duplicates.any():
        examples = df.loc[duplicates, [*run_cols, epoch_col]].head(8).to_dict("records")
        raise ValueError(f"duplicate run/epoch rows: {examples}")


def run_outcomes(
    df: pd.DataFrame,
    *,
    run_cols: Sequence[str] = DEFAULT_RUN_COLS,
    epoch_col: str = "epoch",
    reward_col: str = "mean_reward",
    expected_epochs: int = 500,
    final_window: int = 50,
) -> pd.DataFrame:
    """Summarize observed reward and completion evidence for each run.

    A trailing-zero collapse requires a complete expected run, earlier nonzero
    reward, and exactly zero reward in the final expected window. Partial runs do
    not receive a final-window score or collapse classification.
    """
    if expected_epochs <= 0:
        raise ValueError("expected_epochs must be positive")
    if final_window <= 0 or final_window > expected_epochs:
        raise ValueError("final_window must be in 1..expected_epochs")
    required = set(run_cols) | {epoch_col, reward_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    require_unique_run_epochs(df, run_cols=run_cols, epoch_col=epoch_col)

    rows: list[dict[str, object]] = []
    final_start = expected_epochs - final_window
    grouper: str | list[str] = run_cols[0] if len(run_cols) == 1 else list(run_cols)
    for keys, group in df.groupby(grouper, dropna=False, sort=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        work = group.copy()
        work["_epoch"] = pd.to_numeric(work[epoch_col], errors="coerce")
        work["_reward"] = pd.to_numeric(work[reward_col], errors="coerce")
        work = work.sort_values("_epoch")
        observed_epochs = set(work["_epoch"].dropna().astype(int))
        complete = observed_epochs == set(range(expected_epochs))
        expected_final = work[work["_epoch"].between(final_start, expected_epochs - 1)]
        has_final_window = len(expected_final) == final_window and set(
            expected_final["_epoch"].dropna().astype(int)
        ) == set(range(final_start, expected_epochs))
        rewards = work["_reward"]
        nonzero = rewards.ne(0) & rewards.notna()
        nonzero_epochs = work.loc[nonzero, "_epoch"].dropna()
        first_reward = work.loc[work["_epoch"].eq(0), "_reward"]
        final_rewards = expected_final["_reward"] if has_final_window else pd.Series(dtype=float)
        all_observed_zero = bool(rewards.notna().all() and rewards.eq(0).all())
        zero_from_initialization = all_observed_zero
        trailing_zero = bool(
            complete
            and has_final_window
            and final_rewards.notna().all()
            and final_rewards.eq(0).all()
            and nonzero.any()
        )
        row: dict[str, object] = dict(zip(run_cols, key_values))
        row.update(
            epochs_observed=int(len(work)),
            epoch_min=None if work["_epoch"].dropna().empty else int(work["_epoch"].min()),
            epoch_max=None if work["_epoch"].dropna().empty else int(work["_epoch"].max()),
            complete=complete,
            reward_epoch_0=None if first_reward.empty else float(first_reward.iloc[0]),
            reward_min=None if rewards.dropna().empty else float(rewards.min()),
            reward_max=None if rewards.dropna().empty else float(rewards.max()),
            nonzero_epochs=int(nonzero.sum()),
            last_nonzero_epoch=None if nonzero_epochs.empty else int(nonzero_epochs.max()),
            zero_from_initialization=zero_from_initialization,
            trailing_zero_collapse=trailing_zero,
            final_window_available=has_final_window,
            final_window_mean_reward=(
                float(final_rewards.mean()) if has_final_window else None
            ),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seed_curves(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    seed_col: str = "global_seed",
    epoch_col: str = "epoch",
    reward_col: str = "mean_reward",
) -> pd.DataFrame:
    """Return equal-seed reward summaries for each group and epoch.

    Every seed contributes exactly one row per epoch. This is deliberately separate
    from rollout-level `std_reward`, which measures within-run trajectory variation.
    """
    required = set(group_cols) | {seed_col, epoch_col, reward_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    keys = [*group_cols, seed_col]
    require_unique_run_epochs(df, run_cols=keys, epoch_col=epoch_col)
    summary = (
        df.groupby([*group_cols, epoch_col], as_index=False, dropna=False)[reward_col]
        .agg(["mean", "min", "max", "count"])
        .rename(
            columns={
                "mean": "seed_mean",
                "min": "seed_min",
                "max": "seed_max",
                "count": "seed_count",
            }
        )
        .reset_index()
    )
    return summary.sort_values([*group_cols, epoch_col]).reset_index(drop=True)
