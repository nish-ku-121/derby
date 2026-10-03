"""Build the publication plotting pass from existing audited RL results only.

Outputs are unsmoothed vector PDFs. Individual runs are never extended past their
last recorded epoch. The control and smooth figures use a disclosed focus scale:
out-of-range segments are omitted from that view and exact seed extrema are marked
at its boundary, rather than silently clipping them. The compact outcome table uses
a disclosed qualitative classification so numerical failure remains separate from
reward collapse.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils.analysis import aggregate_seed_curves

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "zero_collapse"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
TRACE_DATA = DATA / "reward_traces.parquet"

SEEDS = [123, 456, 789]
SEED_COLORS = {123: "#0072B2", 456: "#D55E00", 789: "#009E73"}
PAIR_STYLES = {
    ("relu", "relu"): ("ReLU / ReLU", "#4D4D4D"),
    ("relu", "softplus"): ("ReLU / softplus", "#0072B2"),
    ("elu", "softplus"): ("ELU / softplus", "#D55E00"),
}
FOCUS_REWARD_LIMITS = (-2.0, 25.0)

CONTROL_ROOTS = {
    "REINFORCE": "control_reinforce",
    "ActorCritic": "control_actor_critic",
}
ADAPTIVE_ROOTS = {
    "REINFORCE": "adaptive_reinforce",
    "ActorCritic": "adaptive_actor_critic",
}
SMOOTH_ROOTS = {
    "REINFORCE control": CONTROL_ROOTS["REINFORCE"],
    "ActorCritic control": CONTROL_ROOTS["ActorCritic"],
    "REINFORCE smooth": "smooth_reinforce",
    "ActorCritic smooth": "smooth_actor_critic",
}
COMBINED_ROOTS = {
    "REINFORCE": "combined_reinforce",
    "ActorCritic": "combined_actor_critic",
}
UNTREATED_ROOTS = {
    "REINFORCE seed 123": "untreated_reinforce_123",
    "REINFORCE seeds 456/789": "untreated_reinforce_456_789",
    "ActorCritic": "untreated_actor_critic",
}
BIAS_ROOTS = {
    "REINFORCE": "bias_reinforce",
    "ActorCritic": "bias_actor_critic",
}
FIXED_SGD_ROOTS = {
    "REINFORCE": "fixed_sgd_reinforce",
    "ActorCritic": "fixed_sgd_actor_critic",
}
FIXED_ADAM_ROOTS = {
    "REINFORCE": "fixed_adam_reinforce",
    "ActorCritic": "fixed_adam_actor_critic",
}
MATCHED_FIXED_OPTIMIZER_AUDIT = DATA / "run_outcomes.csv"

PARAMS = (
    "learning_rate", "use_baseline", "actor_hidden_activation", "actor_final_activation",
    "param_kernel_initializer", "optimizer", "adaptive_learning_rate",
    "adaptive_lr_epsilon", "adaptive_lr_eta", "init_action_center",
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "figure.titlesize": 10.5,
        "axes.linewidth": .75,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    })


def load_roots(roots: dict[str, str]) -> pd.DataFrame:
    """Load a named subset of the committed normalized trace bundle."""
    if not TRACE_DATA.exists():
        raise FileNotFoundError(f"missing publication trace bundle: {TRACE_DATA}")
    data = pd.read_parquet(TRACE_DATA)
    data = data[data.source_key.isin(roots.values())].copy()
    if data.empty:
        raise RuntimeError(f"no trace rows for source keys: {sorted(roots.values())}")
    for col in ("epoch", "mean_reward", "global_seed", "param_learning_rate", "param_adaptive_lr_epsilon"):
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def numeric_setting_match(values: pd.Series, target: float) -> np.ndarray:
    """Match serialized numeric settings without merging adjacent small values."""
    numeric = pd.to_numeric(values, errors="coerce")
    return np.isclose(numeric, target, rtol=1e-12, atol=0.0, equal_nan=False)


def condition(data: pd.DataFrame, policy: str, baseline, *, rate=None, epsilon=None, pair=None) -> pd.DataFrame:
    out = data[data.policy_class.eq(policy)]
    if baseline is None:
        out = out[out.param_use_baseline.isna()]
    else:
        out = out[out.param_use_baseline.eq(baseline)]
    if rate is not None:
        out = out[numeric_setting_match(out.param_learning_rate, rate)]
    if epsilon is not None:
        out = out[numeric_setting_match(out.param_adaptive_lr_epsilon, epsilon)]
    if pair is not None:
        out = out[out.param_actor_hidden_activation.eq(pair[0]) & out.param_actor_final_activation.eq(pair[1])]
    if out.empty:
        raise RuntimeError(f"missing data: policy={policy}, baseline={baseline}, rate={rate}, epsilon={epsilon}, pair={pair}")
    if rate is not None and out.param_learning_rate.dropna().nunique() != 1:
        raise RuntimeError(f"ambiguous learning-rate cell for requested rate={rate}")
    if epsilon is not None and out.param_adaptive_lr_epsilon.dropna().nunique() != 1:
        raise RuntimeError(f"ambiguous epsilon cell for requested epsilon={epsilon}")
    return out.copy()


def group_cols(baseline, *, rate=False, epsilon=False, pair=False) -> list[str]:
    cols = ["policy_class"]
    if baseline is not None:
        cols.append("param_use_baseline")
    if rate:
        cols.append("param_learning_rate")
    if epsilon:
        cols.append("param_adaptive_lr_epsilon")
    if pair:
        cols.extend(["param_actor_hidden_activation", "param_actor_final_activation"])
    return cols


def draw_seed_curves(
    ax,
    subset,
    *,
    mean_color="#202020",
    # Near-opaque seed colors prevent dense ActorCritic oscillations from reading
    # as an uncertainty ribbon. They remain raw connected trajectories, not a
    # filled interval or a smoothed band.
    seed_alpha=.94,
    seed_width=.62,
    mean_width=1.75,
    focus_limits: tuple[float, float] | None = None,
):
    """Draw raw seed traces, preserving out-of-range excursions as labeled markers.

    A focus scale deliberately leaves an out-of-range segment blank rather than
    drawing it along the plot boundary. The marker labels the exact per-seed
    extreme, so the figure does not silently turn a blow-up into a bounded trace.
    """
    for seed, curve in subset.groupby("global_seed", sort=True):
        curve = curve.sort_values("epoch")
        values = pd.to_numeric(curve.mean_reward, errors="coerce")
        color = SEED_COLORS.get(int(seed), "#888")
        if focus_limits is None:
            visible = values
        else:
            ymin, ymax = focus_limits
            visible = values.where(values.between(ymin, ymax))
            below = values[values < ymin]
            above = values[values > ymax]
            if not below.empty:
                idx = below.idxmin()
                x = float(curve.loc[idx, "epoch"])
                value = float(values.loc[idx])
                ax.scatter(x, ymin, marker="v", color=color, s=20, zorder=5, clip_on=False)
                ax.annotate(f"{value:.1f}", (x, ymin), xytext=(2, 3), textcoords="offset points", fontsize=5.6, color=color, clip_on=False)
            if not above.empty:
                idx = above.idxmax()
                x = float(curve.loc[idx, "epoch"])
                value = float(values.loc[idx])
                ax.scatter(x, ymax, marker="^", color=color, s=20, zorder=5, clip_on=False)
                ax.annotate(f"{value:.1f}", (x, ymax), xytext=(2, -9), textcoords="offset points", fontsize=5.6, color=color, clip_on=False)
        ax.plot(curve.epoch, visible, color=color, alpha=seed_alpha, lw=seed_width)
    varying = [c for c in ("param_learning_rate", "param_adaptive_lr_epsilon", "param_actor_hidden_activation", "param_actor_final_activation", "param_use_baseline") if c in subset and subset[c].nunique(dropna=False) > 1]
    summary = aggregate_seed_curves(subset, group_cols=["policy_class", *varying])
    complete = summary[summary.seed_count.eq(subset.global_seed.nunique())]
    mean_values = complete.seed_mean
    if focus_limits is not None:
        mean_values = mean_values.where(mean_values.between(*focus_limits))
    ax.plot(
        complete.epoch, mean_values, color=mean_color, lw=mean_width,
        linestyle=(0, (3.0, 1.6)), zorder=4,
    )


def finish_axis(ax, *, zero=True):
    # Publication seed panels are line-only. A future fill_between call would add
    # a PolyCollection and should fail loudly instead of silently changing the
    # visual encoding.
    if any(isinstance(artist, PolyCollection) for artist in ax.collections):
        raise RuntimeError("Area-fill artist detected in a line-only reward panel")
    if zero:
        ax.axhline(0, color="#B5B5B5", lw=.55, linestyle="-", zorder=0)
    ax.grid(True, color="#D0D0D0", alpha=.45, lw=.45)
    ax.spines[["top", "right"]].set_visible(False)


def trace_metrics(trace: pd.DataFrame, horizon: int) -> dict[str, object]:
    trace = trace.sort_values("epoch")
    reward = pd.to_numeric(trace.mean_reward, errors="coerce")
    finite_reward = reward[np.isfinite(reward)]
    last_epoch = int(trace.epoch.max())
    complete = set(trace.epoch.astype(int)) == set(range(horizon))
    metric_nonfinite = False
    for metric in ("grad_norm", "effective_learning_rate"):
        if metric in trace:
            observed = pd.to_numeric(trace[metric], errors="coerce").dropna()
            metric_nonfinite = metric_nonfinite or np.isinf(observed).any()
    numerical = (not complete) or np.isinf(reward).any() or (finite_reward.abs().max() > 1e6) or metric_nonfinite
    all_zero = bool(not finite_reward.empty and finite_reward.eq(0).all())
    peak = float(finite_reward.max()) if not finite_reward.empty else np.nan
    final_reward = float(finite_reward.iloc[-1]) if not finite_reward.empty else np.nan
    final = trace[trace.epoch.between(horizon - 50, horizon - 1)].mean_reward
    final50 = float(final.mean()) if len(final) == 50 else np.nan
    material_epochs = trace.loc[reward.ge(5), "epoch"]
    last_material = int(material_epochs.max()) if not material_epochs.empty else None
    if numerical:
        outcome = "numerical failure / blow-up"
    elif all_zero:
        outcome = "zero from initialization"
    elif peak < 1:
        outcome = "failure to reach informative reward"
    elif peak >= 5 and ((np.isfinite(final50) and final50 < 1) or final_reward < 1):
        # "Delayed" separates a run that learned for a meaningful portion of its
        # horizon from collapse during initial takeoff. It is not a claim about a
        # universal phase boundary.
        outcome = "delayed reward collapse" if (last_material or 0) >= horizon * .2 else "early reward collapse"
    elif np.isfinite(final50) and final50 >= 5:
        outcome = "sustained high reward"
    else:
        outcome = "limited / low reward"
    return {
        "outcome": outcome,
        "peak_reward": peak,
        "final50_reward": final50,
        "last_epoch": last_epoch,
        "last_material_epoch": last_material,
    }


def outcome_codes(subset: pd.DataFrame, horizon: int) -> str:
    code = {
        "sustained high reward": "SH", "delayed reward collapse": "DC",
        "early reward collapse": "EC", "zero from initialization": "ZI",
        "failure to reach informative reward": "FI", "numerical failure / blow-up": "NF",
        "limited / low reward": "LL",
    }
    parts = []
    for seed, trace in subset.groupby("global_seed", sort=True):
        parts.append(f"{int(seed)}:{code[trace_metrics(trace, horizon)['outcome']]}")
    return "  ".join(parts)


def seed_legend_handles():
    handles = [Line2D([], [], color=SEED_COLORS[s], lw=1.2, label=f"seed {s}") for s in SEEDS]
    handles.append(Line2D([], [], color="black", lw=2.0, linestyle=(0, (3.0, 1.6)), label="equal-seed mean"))
    return handles


OUTCOME_KEY = (
    "Outcome tags: SH sustained high, DC delayed collapse, EC early collapse, "
    "ZI zero from initialization, FI failed to reach informative reward, "
    "LL limited/low reward, NF numerical failure."
)


def figure_control() -> None:
    data = load_roots(CONTROL_ROOTS)
    rows = [
        ("REINFORCE", False, [1e-7, 1e-6, 1e-5], "REINFORCE, baseline off"),
        ("REINFORCE", True, [1e-7, 1e-6, 1e-5], "REINFORCE, baseline on"),
        ("ActorCritic", None, [1e-9, 1e-8, 1e-7], "ActorCritic"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.6), sharex=True)
    for r, (policy, baseline, rates, label) in enumerate(rows):
        for c, rate in enumerate(rates):
            ax = axes[r, c]
            sub = condition(data, policy, baseline, rate=rate)
            draw_seed_curves(ax, sub, focus_limits=FOCUS_REWARD_LIMITS)
            finish_axis(ax)
            ax.set_xlim(0, 499)
            ax.set_ylim(*FOCUS_REWARD_LIMITS)
            ax.set_title(f"LR = {rate:.0e}\n{outcome_codes(sub, 500)}", fontsize=7.7)
            if c == 0:
                ax.set_ylabel(f"{label}\nMean reward")
            if r == 2:
                ax.set_xlabel("Training epoch")
    fig.legend(handles=seed_legend_handles(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .968))
    fig.suptitle("Fixed-rate control with centered initialization", y=.997)
    fig.text(.5, .012, "Focus scale [-2, 25]. Triangles label exact out-of-range seed extrema; out-of-range segments are omitted rather than clipped. " + OUTCOME_KEY, ha="center", fontsize=6.25)
    fig.tight_layout(rect=(0, .035, 1, .935))
    fig.savefig(FIG / "fig2_fixed_rate_controls.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_epsilon_atlas(*, combined: bool) -> None:
    roots = COMBINED_ROOTS if combined else ADAPTIVE_ROOTS
    epsilons = [3e-6, 1e-5, 3e-5] if combined else [1e-8, 1e-7, 1e-6]
    data = load_roots(roots)
    rows = [
        ("REINFORCE", False, "REINFORCE, baseline off"),
        ("REINFORCE", True, "REINFORCE, baseline on"),
        ("ActorCritic", None, "ActorCritic"),
    ]
    ymin = float(data.mean_reward.min())
    ymax = float(data.mean_reward.max())
    pad = .04 * (ymax - ymin)
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.6), sharex=True, sharey=True)
    for r, (policy, baseline, label) in enumerate(rows):
        for c, epsilon in enumerate(epsilons):
            ax = axes[r, c]
            sub = condition(data, policy, baseline, epsilon=epsilon)
            draw_seed_curves(ax, sub)
            finish_axis(ax)
            ax.set_xlim(0, 999)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_title(rf"$\epsilon={epsilon:.0e}$" + f"\n{outcome_codes(sub, 1000)}", fontsize=7.7)
            if c == 0:
                ax.set_ylabel(f"{label}\nMean reward")
            if r == 2:
                ax.set_xlabel("Training epoch")
    fig.legend(handles=seed_legend_handles(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .968))
    title = "Combined adaptive SGD (ReLU / softplus)" if combined else "Isolated adaptive step size (ReLU / ReLU)"
    fig.suptitle(title, y=.997)
    fig.text(.5, .012, "Shared reward scale; raw unsmoothed trajectories. " + OUTCOME_KEY, ha="center", fontsize=6.25)
    fig.tight_layout(rect=(0, .035, 1, .935))
    name = "fig5_combined_treatment.pdf" if combined else "fig3_adaptive_step_size.pdf"
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def compact_smooth_cells() -> list[dict[str, object]]:
    """The pre-specified representative matched ablations for the main text."""
    return [
        {"policy": "REINFORCE", "baseline": False, "label": "REINFORCE, baseline off", "rate": 1e-6},
        {"policy": "REINFORCE", "baseline": True, "label": "REINFORCE, baseline on", "rate": 1e-6},
        {"policy": "ActorCritic", "baseline": None, "label": "ActorCritic", "rate": 1e-7},
    ]


def draw_pair_panel(ax, data, policy, baseline, rate, *, focus_limits: tuple[float, float] | None = None):
    for pair, (name, color) in PAIR_STYLES.items():
        sub = condition(data, policy, baseline, rate=rate, pair=pair)
        for seed, curve in sub.groupby("global_seed", sort=True):
            curve = curve.sort_values("epoch")
            values = pd.to_numeric(curve.mean_reward, errors="coerce")
            if focus_limits is None:
                visible = values
            else:
                ymin, ymax = focus_limits
                visible = values.where(values.between(ymin, ymax))
                below = values[values < ymin]
                if not below.empty:
                    idx = below.idxmin()
                    x = float(curve.loc[idx, "epoch"])
                    value = float(values.loc[idx])
                    ax.scatter(x, ymin, marker="v", color=color, s=16, zorder=5, clip_on=False)
                    ax.annotate(f"{value:.1f}", (x, ymin), xytext=(2, 3), textcoords="offset points", fontsize=5.3, color=color, clip_on=False)
            ax.plot(curve.epoch, visible, color=color, alpha=.23, lw=.65)
        summary = aggregate_seed_curves(sub, group_cols=group_cols(baseline, rate=True, pair=True))
        complete = summary[summary.seed_count.eq(3)]
        mean_values = complete.seed_mean if focus_limits is None else complete.seed_mean.where(complete.seed_mean.between(*focus_limits))
        ax.plot(complete.epoch, mean_values, color=color, lw=1.75, label=name)
    finish_axis(ax)
    ax.set_xlim(0, 499)


def figure_smooth() -> list[dict[str, object]]:
    data = load_roots(SMOOTH_ROOTS)
    rows = [
        ("REINFORCE", False, [1e-7, 1e-6, 1e-5], "REINFORCE, baseline off"),
        ("REINFORCE", True, [1e-7, 1e-6, 1e-5], "REINFORCE, baseline on"),
        ("ActorCritic", None, [1e-9, 1e-8, 1e-7], "ActorCritic"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 6.6), sharex=True)
    for r, (policy, baseline, rates, label) in enumerate(rows):
        for c, rate in enumerate(rates):
            ax = axes[r, c]
            draw_pair_panel(ax, data, policy, baseline, rate, focus_limits=FOCUS_REWARD_LIMITS)
            ax.set_ylim(*FOCUS_REWARD_LIMITS)
            ax.set_title(f"LR = {rate:.0e}")
            if c == 0:
                ax.set_ylabel(f"{label}\nMean reward")
            if r == 2:
                ax.set_xlabel("Training epoch")
    handles = [Line2D([], [], color=color, lw=2, label=name) for name, color in PAIR_STYLES.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .968))
    fig.suptitle("Smooth output parameterization: full fixed-rate factorial", y=.997)
    fig.text(.5, .012, "Focus scale [-2, 25]. Triangles label exact out-of-range seed minima; omitted segments are not clipped. Faint lines are seeds; heavy lines are equal-seed means.", ha="center", fontsize=6.45)
    fig.tight_layout(rect=(0, .035, 1, .935))
    fig.savefig(FIG / "fig7_smooth_parameterization_full.pdf", bbox_inches="tight")
    plt.close(fig)

    selected = compact_smooth_cells()
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.75), sharex=True)
    for ax, cell in zip(axes, selected):
        draw_pair_panel(ax, data, cell["policy"], cell["baseline"], cell["rate"], focus_limits=FOCUS_REWARD_LIMITS)
        ax.set_ylim(*FOCUS_REWARD_LIMITS)
        ax.set_title(f"{cell['label']}\nLR = {cell['rate']:.0e}")
        ax.set_xlabel("Training epoch")
    axes[0].set_ylabel("Mean reward")
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
    fig.text(.5, -.015, "Representative matched fixed-rate ablations. Focus scale [-2, 25]; triangle labels preserve out-of-range minima.", ha="center", fontsize=6.55)
    fig.tight_layout(rect=(0, .055, 1, .91))
    fig.savefig(FIG / "fig4_smooth_parameterization.pdf", bbox_inches="tight")
    plt.close(fig)
    return selected


def figure_matched_fixed_optimizer() -> None:
    """Render paper Figure 8 from the committed matched-study trace bundle."""
    traces = pd.read_parquet(TRACE_DATA)
    traces = traces[traces.source_key.eq("matched_fixed_optimizer")].copy()
    audits = pd.read_csv(MATCHED_FIXED_OPTIMIZER_AUDIT)
    if traces.empty or audits.empty:
        raise RuntimeError("missing matched fixed-optimizer traces or audit")

    panels = [
        ("REINFORCE", "off", "sgd"), ("REINFORCE", "off", "adam"),
        ("REINFORCE", "on", "sgd"), ("REINFORCE", "on", "adam"),
        ("ActorCritic", "state-value critic", "sgd"),
        ("ActorCritic", "state-value critic", "adam"),
    ]
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(3, 2, figsize=(12.2, 10.65), sharex=True, sharey=True)
    for ax, (algorithm, baseline, optimizer) in zip(axes.reshape(-1), panels):
        panel = traces[
            traces.policy_class.eq(algorithm)
            & traces.baseline.eq(baseline)
            & traces.optimizer.eq(optimizer)
        ]
        panel_audit = audits[
            audits.algorithm.eq(algorithm)
            & audits.baseline.eq(baseline)
            & audits.optimizer.eq(optimizer)
        ]
        rates = sorted(pd.to_numeric(panel.param_learning_rate, errors="coerce").dropna().unique())
        for color, rate in zip(colors, rates):
            cell = panel[numeric_setting_match(panel.param_learning_rate, rate)]
            for seed, trace in cell.groupby("global_seed", sort=True):
                trace = trace.sort_values("epoch")
                rewards = pd.to_numeric(trace.mean_reward, errors="coerce")
                ax.plot(trace.epoch, rewards, color=color, alpha=.22, linewidth=.75)
                if rewards.abs().gt(25).any():
                    extreme_idx = rewards.abs().idxmax()
                    edge = -0.45 if float(rewards.loc[extreme_idx]) < 0 else 24.45
                    ax.scatter(int(trace.loc[extreme_idx, "epoch"]), edge, marker="X", s=28, color=color, zorder=5)
                record = panel_audit[
                    numeric_setting_match(panel_audit.learning_rate, rate)
                    & panel_audit.seed.eq(seed)
                ]
                if len(record) != 1:
                    raise RuntimeError(f"ambiguous matched audit row: {algorithm}/{baseline}/{optimizer}/{rate}/{seed}")
                if record.outcome.iloc[0] == "numerical_failure":
                    ax.scatter(int(trace.epoch.max()), -0.75, marker="v", s=32, color=color, zorder=6)
            mean = cell.groupby("epoch", as_index=False).agg(mean_reward=("mean_reward", "mean"), seed_count=("global_seed", "nunique"))
            mean = mean[mean.seed_count.eq(len(SEEDS))]
            ax.plot(mean.epoch, mean.mean_reward, color=color, linewidth=2.15, label=f"{rate:.0e}")
        ax.axhline(0, color="#777", linewidth=.55)
        ax.set_xlim(0, 999)
        ax.set_ylim(-1, 25)
        ax.grid(alpha=.24, linewidth=.45)
        title = f"REINFORCE, baseline {baseline}" if algorithm == "REINFORCE" else "ActorCritic, TD state-value"
        ax.set_title(f"{title} - {optimizer.upper()}")
        ax.legend(title="Fixed rate", fontsize=7, title_fontsize=7, loc="upper left", frameon=True)
    for ax in axes[-1]:
        ax.set_xlabel("Training epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean reward")
    fig.suptitle("Matched fixed-optimizer comparison", y=.995)
    fig.text(.5, .012, "Faint lines: individual seeds. Thick lines: equal-seed mean, drawn only where all three seeds have observations. X: reward outside shared [-1, 25] display range. Down-triangle: numerical-failure run. Curves are raw, unsmoothed epoch means.", ha="center", fontsize=7.1)
    fig.tight_layout(rect=(0, .05, 1, .96))
    fig.savefig(FIG / "fig8_matched_fixed_optimizer.pdf", bbox_inches="tight")
    plt.close(fig)


def outcome_code(outcome: str) -> str:
    return {
        "sustained high reward": "SH",
        "delayed reward collapse": "DC",
        "early reward collapse": "EC",
        "zero from initialization": "ZI",
        "failure to reach informative reward": "FI",
        "limited / low reward": "LL",
        "numerical failure / blow-up": "NF",
    }[outcome]


def table_setting_rows(name, roots, initialization, activation, optimizer, setting_kind, rates_or_eps, horizon, *, pair=None):
    """Return one outcome-table row per real fixed setting in ``roots``.

    ``pair`` is used only for the full smooth-parameterization factorial.  It
    keeps the three output parameterizations separate instead of assuming that
    a rate identifies an activation pair.
    """
    data = load_roots(roots)
    rows = []
    conditions = [("REINFORCE", False, "off"), ("REINFORCE", True, "on"), ("ActorCritic", None, "state-value critic")]
    for policy, baseline, baseline_label in conditions:
        for value in rates_or_eps[policy]:
            kwargs = {"rate": value} if setting_kind == "LR" else {"epsilon": value}
            sub = condition(data, policy, baseline, pair=pair, **kwargs)
            observed_seeds = set(pd.to_numeric(sub.global_seed, errors="coerce").dropna().astype(int))
            if observed_seeds != set(SEEDS):
                raise RuntimeError(
                    f"table setting does not have the intended seed set {SEEDS}: "
                    f"{name}, policy={policy}, baseline={baseline}, value={value}, pair={pair}; "
                    f"observed={sorted(observed_seeds)}"
                )
            seed_codes = {}
            source_roots = []
            for seed in SEEDS:
                trace = sub[sub.global_seed.eq(seed)]
                if trace.empty:
                    seed_codes[seed] = "missing"
                else:
                    seed_codes[seed] = outcome_code(trace_metrics(trace, horizon)["outcome"])
                    source_roots.append(str(trace.source_key.iloc[0]))
            rows.append({
            "experiment": name, "algorithm": policy, "baseline": baseline_label,
            "initialization": initialization, "activation": activation,
            "optimizer": optimizer, "setting": f"{setting_kind}={value:.0e}",
            "n_seeds": len(SEEDS), "horizon": horizon,
            "seed_123": seed_codes[123], "seed_456": seed_codes[456], "seed_789": seed_codes[789],
            "source_root": "; ".join(sorted(set(source_roots))),
            })
    return rows


def matched_optimizer_outcome(trace: pd.DataFrame, audit_row: pd.Series) -> str:
    """Classify a Fig. 6 raw reward trace with the table's established codes.

    The published Fig. 6 audit is also consulted for completion/configuration
    failures, while reward outcomes are classified from the recorded trajectory
    using the same thresholds as ``trace_metrics``.  This avoids treating the
    Fig. 6 aggregate audit labels as a substitute for seed-level evidence.
    """
    reward = pd.to_numeric(trace.mean_reward, errors="coerce")
    finite = reward[np.isfinite(reward)]
    complete = set(pd.to_numeric(trace.epoch, errors="coerce").dropna().astype(int)) == set(range(1000))
    audit_ok = (
        bool(audit_row.completion_record)
        and not bool(audit_row.failure_record)
        and int(audit_row.parquet_count) == 1
        and bool(audit_row.complete_epochs)
        and bool(audit_row.finite_metrics)
        and bool(audit_row.fixed_rate_identity)
        and not str(audit_row.config_mismatches if pd.notna(audit_row.config_mismatches) else "").strip()
    )
    numerical = (
        not complete or not audit_ok or finite.empty or not np.isfinite(reward).all()
        or float(finite.abs().max()) > 1e6 or str(audit_row.outcome) == "numerical_failure"
    )
    if numerical:
        return "NF"
    if finite.eq(0).all():
        return "ZI"
    peak = float(finite.max())
    final50 = reward.iloc[-50:]
    final50_mean = float(final50.mean()) if len(final50) == 50 else np.nan
    last_reward = float(finite.iloc[-1])
    material = trace.loc[reward.ge(5), "epoch"]
    last_material = int(material.max()) if not material.empty else None
    if peak < 1:
        return "FI"
    if peak >= 5 and ((np.isfinite(final50_mean) and final50_mean < 1) or last_reward < 1):
        return "DC" if (last_material or 0) >= 200 else "EC"
    if np.isfinite(final50_mean) and final50_mean >= 5:
        return "SH"
    return "LL"


def matched_fixed_optimizer_rows() -> list[dict[str, object]]:
    """Read the configuration-audited Fig. 6 runs and produce 18 setting rows.

    This family was intentionally calibrated per optimizer; its nominal rates
    are displayed independently and are never merged into a shared scale.
    """
    if not MATCHED_FIXED_OPTIMIZER_AUDIT.exists() or not TRACE_DATA.exists():
        raise RuntimeError("missing matched fixed-optimizer audit or trace bundle")
    audit = pd.read_csv(MATCHED_FIXED_OPTIMIZER_AUDIT)
    traces = pd.read_parquet(TRACE_DATA)
    traces = traces[traces.source_key.eq("matched_fixed_optimizer")].copy()
    required_audit = {
        "group", "algorithm", "baseline", "optimizer", "run", "seed", "learning_rate",
        "result_dir", "completion_record", "failure_record", "parquet_count", "config_mismatches",
        "complete_epochs", "finite_metrics", "fixed_rate_identity", "outcome",
    }
    if missing := required_audit - set(audit.columns):
        raise RuntimeError(f"Fig. 6 audit is missing columns: {sorted(missing)}")
    keys = ["group", "algorithm", "baseline", "optimizer", "learning_rate"]
    rows = []
    for key, setting_audit in audit.groupby(keys, sort=True, dropna=False):
        group, algorithm, baseline, optimizer, learning_rate = key
        if len(setting_audit) != len(SEEDS) or set(setting_audit.seed.astype(int)) != set(SEEDS):
            raise RuntimeError(f"Fig. 6 setting lacks the three intended seeds: {key}")
        seed_codes: dict[int, str] = {}
        source_roots = []
        for audit_row in setting_audit.itertuples(index=False):
            trace = traces[
                (traces.group.eq(audit_row.group))
                & (traces.run.eq(audit_row.run))
                & (traces.global_seed.eq(audit_row.seed))
            ]
            if trace.empty:
                raise RuntimeError(f"Fig. 6 trace missing for {audit_row.group}/{audit_row.run}/seed={audit_row.seed}")
            seed_codes[int(audit_row.seed)] = matched_optimizer_outcome(trace, pd.Series(audit_row._asdict()))
            source_roots.append("matched_fixed_optimizer")
        baseline_label = "state-value critic" if algorithm == "ActorCritic" else str(baseline)
        rows.append({
            "experiment": "Matched fixed-rate SGD vs. Adam",
            "algorithm": algorithm,
            "baseline": baseline_label,
            "initialization": "Centered zero head",
            "activation": "ReLU / softplus",
            "optimizer": "Adam" if str(optimizer).lower() == "adam" else "SGD",
            "setting": f"LR={float(learning_rate):.0e}",
            "n_seeds": len(SEEDS), "horizon": 1000,
            "seed_123": seed_codes[123], "seed_456": seed_codes[456], "seed_789": seed_codes[789],
            "source_root": "; ".join(sorted(set(source_roots))),
        })
    if len(rows) != 18:
        raise RuntimeError(f"expected 18 Fig. 6 matched optimizer settings, found {len(rows)}")
    return rows


def build_outcome_table() -> pd.DataFrame:
    fixed_rates = {"REINFORCE": [3e-4, 1e-3], "ActorCritic": [3e-4, 1e-3, 3e-3]}
    smooth_rates = {"REINFORCE": [1e-7, 1e-6, 1e-5], "ActorCritic": [1e-9, 1e-8, 1e-7]}
    rows = []
    rows += table_setting_rows("Untreated initialization", UNTREATED_ROOTS, "Glorot, uncentered", "ReLU / ReLU", "SGD", "LR", {"REINFORCE": [1e-7,1e-6,1e-5], "ActorCritic": [1e-9,1e-8,1e-7]}, 500)
    rows += table_setting_rows("Bias-centered Glorot", BIAS_ROOTS, "Glorot with centered bias", "ReLU / ReLU", "SGD", "LR", {"REINFORCE": [1e-7,1e-6,1e-5], "ActorCritic": [1e-9,1e-8,1e-7]}, 500)
    rows += table_setting_rows("Control 2", CONTROL_ROOTS, "Centered zero head", "ReLU / ReLU", "SGD", "LR", {"REINFORCE": [1e-7,1e-6,1e-5], "ActorCritic": [1e-9,1e-8,1e-7]}, 500)
    # Full, one-setting-per-row smooth-parameterization factorial.  ReLU/ReLU
    # lives in Control 2; the two softplus variants live in the treatment roots.
    for pair, (activation, _color) in PAIR_STYLES.items():
        rows += table_setting_rows("Smooth parameterization", SMOOTH_ROOTS, "Centered zero head", activation, "SGD", "LR", smooth_rates, 500, pair=pair)
    rows += table_setting_rows("Fixed high-rate SGD", FIXED_SGD_ROOTS, "Centered zero head", "ReLU / softplus", "SGD", "LR", fixed_rates, 200)
    rows += table_setting_rows("Fixed high-rate Adam", FIXED_ADAM_ROOTS, "Centered zero head", "ReLU / softplus", "Adam", "LR", fixed_rates, 1000)
    rows += matched_fixed_optimizer_rows()
    rows += table_setting_rows("Isolated adaptive step size", ADAPTIVE_ROOTS, "Centered zero head", "ReLU / ReLU", "adaptive SGD", "epsilon", {"REINFORCE": [1e-8,1e-7,1e-6], "ActorCritic": [1e-8,1e-7,1e-6]}, 1000)
    rows += table_setting_rows("Combined treatment", COMBINED_ROOTS, "Centered zero head", "ReLU / softplus", "adaptive SGD", "epsilon", {"REINFORCE": [3e-6,1e-5,3e-5], "ActorCritic": [3e-6,1e-5,3e-5]}, 1000)
    table = pd.DataFrame(rows)
    expected_rows = 104
    if len(table) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} prescribed outcome-table cells, found {len(table)}")
    table.to_csv(TAB / "experiment_outcomes.csv", index=False)

    display_cols = ["experiment","algorithm","baseline","initialization","activation","optimizer","setting","n_seeds","horizon","seed_123","seed_456","seed_789"]
    # CSV is the canonical table source and LaTeX is consumed by the manuscript.
    # Deliberately do not emit a redundant Markdown rendering.
    latex_names = {
        "experiment": "Experiment", "algorithm": "Algorithm", "baseline": "Baseline / critic",
        "initialization": "Initialization", "activation": "Activation", "optimizer": "Optimizer",
        "setting": "Setting", "n_seeds": "$n$", "horizon": "Epochs",
        "seed_123": "Seed 123", "seed_456": "Seed 456", "seed_789": "Seed 789",
    }
    def tex_escape(value):
        text = str(value)
        for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
            text = text.replace(old, new)
        return text
    lines = [
        r"% Requires \usepackage{longtable,pdflscape,array}",
        r"\begin{landscape}",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{1.90cm}>{\raggedright\arraybackslash}p{1.85cm}>{\raggedright\arraybackslash}p{1.70cm}>{\raggedright\arraybackslash}p{2.05cm}>{\raggedright\arraybackslash}p{1.35cm}>{\raggedright\arraybackslash}p{1.25cm}>{\raggedright\arraybackslash}p{2.10cm}>{\centering\arraybackslash}p{.45cm}>{\centering\arraybackslash}p{.80cm}>{\centering\arraybackslash}p{.65cm}>{\centering\arraybackslash}p{.65cm}>{\centering\arraybackslash}p{.65cm}}",
        r"\caption{Qualitative outcomes by experiment configuration and seed. Each row is one hyperparameter setting. Codes: SH sustained high; DC delayed collapse; EC early collapse; ZI zero from initialization; FI failed to reach informative reward; LL limited/low reward; NF numerical failure. In the matched fixed-rate SGD--Adam family, nominal rates are optimizer-specific calibrated settings, not equivalent update scales.}\label{tab:rl_experiment_outcomes}\\",
        " & ".join(latex_names[c] for c in display_cols) + r" \\",
        r"\hline",
        r"\endfirsthead",
        " & ".join(latex_names[c] for c in display_cols) + r" \\",
        r"\hline",
        r"\endhead",
    ]
    for row in table[display_cols].itertuples(index=False, name=None):
        lines.append(" & ".join(tex_escape(v) for v in row) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", r"\end{landscape}"])
    latex = "\n".join(lines) + "\n"
    (TAB / "experiment_outcomes.tex").write_text(latex, encoding="utf-8")
    return table


def main() -> None:
    configure_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    figure_control()
    figure_epsilon_atlas(combined=False)
    figure_smooth()
    figure_epsilon_atlas(combined=True)
    figure_matched_fixed_optimizer()
    build_outcome_table()


if __name__ == "__main__":
    if "--tables-only" in sys.argv:
        configure_style()
        FIG.mkdir(parents=True, exist_ok=True)
        TAB.mkdir(parents=True, exist_ok=True)
        table = build_outcome_table()
    elif "--figures-only" in sys.argv:
        configure_style()
        FIG.mkdir(parents=True, exist_ok=True)
        figure_control()
        figure_epsilon_atlas(combined=False)
        figure_smooth()
        figure_epsilon_atlas(combined=True)
        figure_matched_fixed_optimizer()
    else:
        main()
