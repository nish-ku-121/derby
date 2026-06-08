from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd


PALETTE_OKABE_ITO: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
)


LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")


@dataclass(frozen=True)
class AxesConfig:
    x_col: str = "epoch"
    y_col: str = "mean_reward"
    x_label: str = "Epoch"
    y_label: str = "Mean Reward"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    grid: bool = True


@dataclass(frozen=True)
class SmoothingConfig:
    window: int | None = None
    min_periods: int = 1


@dataclass(frozen=True)
class VarianceConfig:
    column: str | None = "std_reward"
    scale: float = 1.0
    alpha: float = 0.14
    smooth: bool = True


@dataclass(frozen=True)
class LegendConfig:
    title: str = "Policy"
    loc: str = "best"
    show_run_id: bool = True
    max_label_chars: int | None = 80


@dataclass(frozen=True)
class StyleConfig:
    palette: Sequence[str] = PALETTE_OKABE_ITO
    linewidth: float = 2.0
    alpha: float = 0.95
    outline: bool = False
    outline_width: float = 2.5
    vary_linestyle_for_repeated_color: bool = True


def _smooth(series: pd.Series, config: SmoothingConfig | None) -> pd.Series:
    if config is None or config.window is None or config.window <= 1:
        return series
    return (
        pd.to_numeric(series, errors="coerce")
        .rolling(window=int(config.window), min_periods=config.min_periods)
        .mean()
    )


def _curve_label(values: tuple[object, ...], curve_cols: Sequence[str], legend: LegendConfig) -> str:
    parts = []
    for col, value in zip(curve_cols, values):
        if col == "run_id" and not legend.show_run_id:
            continue
        text = "" if pd.isna(value) else str(value)
        if col == "run_id":
            text = text[:8]
        parts.append(text)
    label = " | ".join(p for p in parts if p)
    if legend.max_label_chars is not None and len(label) > legend.max_label_chars:
        return label[: max(0, legend.max_label_chars - 1)] + "..."
    return label


def _normalize_curve_cols(curve_cols: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(curve_cols, str):
        return (curve_cols,)
    return tuple(curve_cols)


def plot_learning_curves(
    df: pd.DataFrame,
    *,
    curve_cols: Sequence[str] | str = ("agent_label", "run_id"),
    color_col: str | None = "agent_label",
    agent_name: str | None = "learner",
    axes: AxesConfig | None = None,
    smoothing: SmoothingConfig | int | None = None,
    variance: VarianceConfig | None = None,
    legend: LegendConfig | None = None,
    style: StyleConfig | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 4.8),
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot paper-oriented learning curves from epoch aggregate rows.

    The default curve identity is one line per (agent_label, run_id). This function
    does not aggregate across runs; repeated rows for the same curve and epoch
    are treated as an input error.
    """
    curve_cols = _normalize_curve_cols(curve_cols)
    axes = axes or AxesConfig()
    legend = legend or LegendConfig()
    style = style or StyleConfig()
    if isinstance(smoothing, int):
        smoothing = SmoothingConfig(window=smoothing)

    required = set(curve_cols) | {axes.x_col, axes.y_col}
    if agent_name is not None and "agent_name" in df.columns:
        df = df[df["agent_name"] == agent_name]
    if color_col is not None:
        required.add(color_col)
    if variance is not None and variance.column is not None:
        required.add(variance.column)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    work = df.copy()
    if work.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.set_title(title or "No data")
        ax.set_xlabel(axes.x_label)
        ax.set_ylabel(axes.y_label)
        return fig, ax

    duplicate_counts = (
        work.groupby([*curve_cols, axes.x_col], dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicates = duplicate_counts[duplicate_counts["count"] > 1]
    if not duplicates.empty:
        examples = duplicates.head(8).to_dict(orient="records")
        raise ValueError(
            "Duplicate rows for the same curve and x value. "
            "Filter the DataFrame or include another column in curve_cols. "
            f"Examples: {examples}"
        )

    work["_x_num"] = pd.to_numeric(work[axes.x_col], errors="coerce")
    work["_y_num"] = pd.to_numeric(work[axes.y_col], errors="coerce")

    color_keys = list(dict.fromkeys(work[color_col].astype(str))) if color_col else []
    color_cycle = cycle(style.palette)
    color_map = {key: next(color_cycle) for key in color_keys}
    linestyle_cycle = cycle(LINESTYLES)
    linestyle_map: dict[tuple[object, ...], str] = {}
    color_curve_cols = list(dict.fromkeys([color_col, *curve_cols])) if color_col else []
    curves_per_color = (
        work[color_curve_cols]
        .drop_duplicates()
        .groupby(color_col)
        .size()
        .to_dict()
        if color_col
        else {}
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sort_cols = [*curve_cols, "_x_num"]
    for key_values, group in work.sort_values(sort_cols).groupby(list(curve_cols), dropna=False):
        key_tuple = key_values if isinstance(key_values, tuple) else (key_values,)
        group = group.sort_values("_x_num")
        color_key = str(group[color_col].iloc[0]) if color_col else str(key_tuple)
        color = color_map[color_key] if color_key in color_map else next(color_cycle)
        if style.vary_linestyle_for_repeated_color and curves_per_color.get(color_key, 1) > 1:
            linestyle = linestyle_map.setdefault(key_tuple, next(linestyle_cycle))
        else:
            linestyle = "-"
        label = _curve_label(key_tuple, curve_cols, legend)

        x = group["_x_num"]
        y = _smooth(group["_y_num"], smoothing)

        line, = ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=style.linewidth,
            alpha=style.alpha,
            label=label,
        )
        if style.outline:
            line.set_path_effects([
                pe.Stroke(linewidth=style.linewidth + style.outline_width, foreground="white"),
                pe.Normal(),
            ])
        if variance is not None and variance.column is not None:
            spread = pd.to_numeric(group[variance.column], errors="coerce") * variance.scale
            if variance.smooth:
                spread = _smooth(spread, smoothing)
            lower = y - spread
            upper = y + spread
            ax.fill_between(x, lower, upper, color=color, alpha=variance.alpha, linewidth=0)

    ax.set_xlabel(axes.x_label)
    ax.set_ylabel(axes.y_label)
    if title:
        ax.set_title(title)
    if axes.xlim is not None:
        ax.set_xlim(*axes.xlim)
    if axes.ylim is not None:
        ax.set_ylim(*axes.ylim)
    if axes.grid:
        ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title=legend.title, loc=legend.loc, frameon=True)
    fig.tight_layout()
    return fig, ax


# Backward-compatible name for the first clean paper plotting surface.
plot_epochs_simple = plot_learning_curves


__all__ = [
    "PALETTE_OKABE_ITO",
    "AxesConfig",
    "SmoothingConfig",
    "VarianceConfig",
    "LegendConfig",
    "StyleConfig",
    "plot_learning_curves",
    "plot_epochs_simple",
]
