"""Rerun one ActorCritic Control 2 cell and plot a TD-V action-value diagnostic.

This is deliberately a standalone evidence-generation script.  The normal runner
does not retain model checkpoints, so the diagnostic reruns a pre-specified cell
and evaluates the learned state-value critic immediately after epoch 224.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
import yaml

from derby.core.agents import Agent
from derby.core.environments import train
from derby.experiments.one_camp_n_days.experiment import OneCampNDaysExperiment
from derby.experiments.one_camp_n_days.runner import (
    _derive_policy_seed,
    _prepare_runtime_policy_params,
    _seed_everything,
)
from derby.policies.actor_critic import ActorCritic
from derby.policies.baselines import FixedBidPolicy
from pipeline.make_config_grid import generate_configs


ROOT = Path(__file__).resolve().parents[3]
CONFIG_SPEC = ROOT / "configs/actor_critic_sweep_control_2.yaml"
RETAINED = ROOT / "results/staging/control/actor_critic/actor_critic_sweep_control_2/run_0005"
OUT = ROOT / "paper/zero_collapse"
FIGURE = OUT / "figures/fig6_td_value_critic_diagnostic.pdf"
DATA = OUT / "data/td_value_critic_diagnostic.csv"
META = OUT / "data/td_value_critic_diagnostic.json"

SELECTED_EPOCH = 224  # zero-based, selected from the retained reward dynamics
MC_SAMPLES = 50_000
# Reachable after three one-item auction steps: the learner won one item at
# price 5 and the fixed-bid competitor won two at price 5.
FIXED_LEARNER_STATE = np.array([10.0, 100.0, 2.0, 5.0, 1.0, 3.0], dtype=np.float32)
FIXED_BASELINE_STATE = np.array([10.0, 100.0, 2.0, 10.0, 2.0, 3.0], dtype=np.float32)
FIXED_RAW_STATE = np.concatenate([FIXED_LEARNER_STATE, FIXED_BASELINE_STATE])
FIXED_TOTAL_LIMIT = 100.0
COMPETITOR_BID = 5.0


def load_selected_config() -> dict:
    """Resolve the tracked sweep spec to the fixed diagnostic cell."""
    with tempfile.TemporaryDirectory(prefix="derby_td_value_") as directory:
        generate_configs(str(CONFIG_SPEC), directory)
        for candidate in sorted(Path(directory).glob("run_*.yaml")):
            config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            params = config["agents"][0]["params"]
            if (
                int(config["seed"]) == 456
                and float(params["learning_rate"]) == 1e-8
                and config["agents"][0]["policy"] == "ActorCritic"
            ):
                return config
    raise RuntimeError("diagnostic cell (seed=456, learning_rate=1e-8) was not generated")


def build_experiment(config: dict):
    """Construct the exact current Control 2 learner/baseline pair."""
    seed = int(config["seed"])
    _seed_everything(seed)
    experiment = OneCampNDaysExperiment(seed=seed)
    env, spec_ids = experiment.build_one_segment_setup()
    scale_states, actions_scaler, scale_actions, descale_actions = experiment.build_env_transforms(env)

    learner_cfg = config["agents"][0]
    params = dict(learner_cfg["params"])
    params["auction_item_spec_ids"] = spec_ids
    params["seed"] = _derive_policy_seed(seed, "ActorCritic", learner_cfg["name"])
    for key in ("learning_rate", "init_action_center", "init_action_stddev", "min_action_stddev"):
        if key in params:
            params[key] = float(params[key])
    params = _prepare_runtime_policy_params(params, actions_scaler)
    learner = Agent(
        learner_cfg["name"],
        ActorCritic(**params),
        scale_states,
        scale_actions,
        descale_actions,
    )

    baseline_cfg = config["agents"][1]
    baseline = Agent(baseline_cfg["name"], FixedBidPolicy(**baseline_cfg["params"]))
    env.vectorize = True
    env.init([learner, baseline], int(config["num_days"]))
    return env, learner, scale_states


def retained_rewards() -> dict[int, float]:
    paths = list(RETAINED.glob("*.parquet"))
    if len(paths) != 1:
        return {}
    table = pq.read_table(paths[0], columns=["epoch", "agent_name", "mean_reward"])
    frame = table.to_pydict()
    return {
        int(epoch): float(reward)
        for epoch, name, reward in zip(frame["epoch"], frame["agent_name"], frame["mean_reward"])
        if name == "learner"
    }


def branch_estimates(
    learner: Agent,
    scale_states,
    first_win: bool,
    first_bid: float = 0.0,
) -> dict[str, float]:
    """Estimate return and bootstrap for one deterministic side of the bid cliff."""
    state = FIXED_RAW_STATE.copy()
    if first_win:
        state[3] += float(first_bid)
        state[4] += 1.0
    else:
        state[6 + 3] += COMPETITOR_BID
        state[6 + 4] += 1.0
    state[5] = state[6 + 5] = 4.0

    # The terminal continuation return depends on impressions and the sampled
    # final-step policy action. Spend does not enter the terminal payout formula.
    states = np.broadcast_to(state, (MC_SAMPLES, 1, state.size)).copy()
    scaled = tf.convert_to_tensor(scale_states(states), dtype=tf.float32)
    sampled_scaled_actions = learner.policy.choose_actions(learner.policy.call(scaled)).numpy()
    sampled_actions = learner.actions_descaler(sampled_scaled_actions)
    future_bid = sampled_actions[:, 0, 0, 1].astype(np.float64)
    future_limit = sampled_actions[:, 0, 0, 2].astype(np.float64)
    future_win = (future_bid > COMPETITOR_BID) & (future_bid <= future_limit)
    terminal_reward = (
        -np.where(future_win, future_bid, 0.0)
        + np.minimum(100.0, 10.0 * (float(state[4]) + future_win.astype(np.float64)))
    )
    return {
        "continuation_mean": float(np.mean(terminal_reward)),
        "continuation_se": float(np.std(terminal_reward, ddof=1) / math.sqrt(MC_SAMPLES)),
    }


def critic_value(learner: Agent, scale_states, raw_state: np.ndarray) -> float:
    scaled = tf.convert_to_tensor(scale_states(raw_state[None, None, :]), dtype=tf.float32)
    return float(learner.policy.value_function(scaled).numpy().reshape(-1)[0])


def evaluate_curves(learner: Agent, scale_states):
    # ``choose_actions`` caches an auction-item ID tensor while traced. Training
    # uses batch size 1; clear that trace-side cache before the MC batch size.
    learner.policy._ais_tensor = None
    lose_ref = branch_estimates(learner, scale_states, first_win=False)

    lose_next = FIXED_RAW_STATE.copy()
    lose_next[6 + 3] += COMPETITOR_BID
    lose_next[6 + 4] += 1.0
    lose_next[5] = lose_next[6 + 5] = 4.0
    v_lose = critic_value(learner, scale_states, lose_next)

    # Include exact threshold and dense support on both sides.
    bids = np.unique(np.concatenate([np.linspace(0.0, 10.0, 401), [COMPETITOR_BID]]))
    rows = []
    win_refs = {}
    for bid in bids:
        if bid < COMPETITOR_BID:
            branches = [(1.0, False)]
        elif bid > COMPETITOR_BID:
            branches = [(1.0, True)]
        else:
            branches = [(0.5, False), (0.5, True)]

        q_true = 0.0
        q_td = 0.0
        q_true_var = 0.0
        for probability, won in branches:
            immediate = -float(bid) if won else 0.0
            if won:
                win_next = FIXED_RAW_STATE.copy()
                win_next[3] += float(bid)
                win_next[4] += 1.0
                win_next[5] = win_next[6 + 5] = 4.0
                v_next = critic_value(learner, scale_states, win_next)
                if float(bid) not in win_refs:
                    win_refs[float(bid)] = branch_estimates(
                        learner, scale_states, first_win=True, first_bid=float(bid)
                    )
                ref = win_refs[float(bid)]
            else:
                v_next = v_lose
                ref = lose_ref
            q_true += probability * (immediate + ref["continuation_mean"])
            q_td += probability * (immediate + v_next)
            q_true_var += (probability * ref["continuation_se"]) ** 2
        rows.append((float(bid), q_true, math.sqrt(q_true_var), q_td))
    return rows, {
        "lose": lose_ref,
        "win_at_cliff": win_refs[COMPETITOR_BID],
        "win_at_bid_10": win_refs[10.0],
        "v_lose": v_lose,
    }


def configure_plotting() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def write_outputs(rows, metadata: dict) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bid", "q_true_mc", "q_true_mc_se", "q_td_implied"])
        writer.writerows(rows)

    bids = np.array([row[0] for row in rows])
    q_true = np.array([row[1] for row in rows])
    q_td = np.array([row[3] for row in rows])
    configure_plotting()
    fig, ax = plt.subplots(figsize=(5.15, 3.15), constrained_layout=True)
    ax.plot(bids, q_true, color="#202020", lw=1.8, label=r"Reference $Q^\pi_{\mathrm{ref}}(s,a)$")
    ax.plot(bids, q_td, color="#0072B2", lw=1.8, ls="--", label=r"TD-$V$ implied $r_t + V_w(s_{t+1})$")
    ax.axvline(COMPETITOR_BID, color="#777777", lw=0.8, ls=":", zorder=0)
    ax.text(
        COMPETITOR_BID + 0.08,
        0.78,
        "cliff at bid = 5",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color="#666666",
        zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.set(xlabel="Bid", ylabel="Action value", xlim=(0.0, 10.0))
    ax.set_title("TD–$V$ critic diagnostic near collapse (epoch 224, seed 456)", pad=8)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="out", length=3)
    fig.savefig(
        FIGURE,
        metadata={
            "Title": "Current TD-V critic-implied action values near collapse",
            "Subject": "ActorCritic Control 2; seed 456; epoch 224; fixed joint bidder state",
            "Author": "Derby diagnostic rerun",
        },
    )
    plt.close(fig)
    META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_selected_config()
    assert config["seed"] == 456
    assert float(config["agents"][0]["params"]["learning_rate"]) == 1e-8
    assert config["agents"][0]["policy"] == "ActorCritic"
    assert config["agents"][0]["params"]["critic_type"] == "td"
    assert SELECTED_EPOCH < int(config["num_epochs"])

    env, learner, scale_states = build_experiment(config)
    historical = retained_rewards()
    rerun_rewards = []
    for epoch in range(SELECTED_EPOCH + 1):
        train(env, int(config["num_trajs"]), 100)
        rewards = np.asarray(learner.cumulative_rewards[-int(config["num_trajs"]):], dtype=float)
        rerun_rewards.append(float(np.mean(rewards)))
        for agent in env.agents:
            agent.cumulative_rewards = np.empty(0, dtype=np.float32)
        if epoch % 25 == 0 or epoch == SELECTED_EPOCH:
            print(f"epoch={epoch} mean_reward={rerun_rewards[-1]:.8g}", flush=True)

    rows, estimates = evaluate_curves(learner, scale_states)
    comparison_epochs = sorted(set([0, SELECTED_EPOCH]) & set(historical))
    reproduction = {
        str(epoch): {
            "retained": historical[epoch],
            "rerun": rerun_rewards[epoch],
            "absolute_difference": abs(historical[epoch] - rerun_rewards[epoch]),
        }
        for epoch in comparison_epochs
    }
    metadata = {
        "experiment": "actor_critic_sweep_control_2",
        "source_config": str(CONFIG_SPEC.relative_to(ROOT)).replace("\\", "/"),
        "source_config_selection": {"seed": 456, "learning_rate": 1e-8},
        "algorithm": "current ActorCritic one-step TD state-value critic",
        "optimizer": "fixed-rate SGD",
        "learning_rate": 1e-8,
        "seed": 456,
        "selected_epoch_zero_based": SELECTED_EPOCH,
        "updates_completed": SELECTED_EPOCH + 1,
        "selection_rationale": "pre-specified near the retained run's last nonzero epoch 247",
        "fixed_raw_joint_state": {
            "field_order_per_bidder": ["reach", "budget", "target_spec_id", "spend", "impressions", "timestep"],
            "learner": FIXED_LEARNER_STATE.tolist(),
            "baseline": FIXED_BASELINE_STATE.tolist(),
        },
        "fixed_swept_action": {"total_limit": FIXED_TOTAL_LIMIT, "bid_range": [0.0, 10.0]},
        "competitor": {"policy": "FixedBidPolicy", "bid": 5.0, "total_limit": 5.0},
        "discount_factor": 1.0,
        "reference_formula": "Q^pi(s,a) = E[r_t + r_{t+1} | s_t=s, a_t=a, then learned policy pi]",
        "critic_implied_formula": "Qhat_TD(s,a) = E[r_t + V_w(s_{t+1}) | s_t=s, a_t=a]",
        "terminal_bootstrap": 0.0,
        "reference_estimator": "analytic current transition plus Monte Carlo final learned-policy action",
        "monte_carlo_samples_per_next_state": MC_SAMPLES,
        "tie_at_bid_5": "equal-probability allocation between the two tied bidders",
        "branch_estimates": estimates,
        "retained_rerun_reward_comparison": reproduction,
        "outputs": {
            "figure": str(FIGURE.relative_to(ROOT)).replace("\\", "/"),
            "data": str(DATA.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    write_outputs(rows, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        with DATA.open(newline="", encoding="utf-8") as handle:
            rows = [
                (
                    float(row["bid"]),
                    float(row["q_true_mc"]),
                    float(row["q_true_mc_se"]),
                    float(row["q_td_implied"]),
                )
                for row in csv.DictReader(handle)
            ]
        write_outputs(rows, json.loads(META.read_text(encoding="utf-8")))
    else:
        main()
