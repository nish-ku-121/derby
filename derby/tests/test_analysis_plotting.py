from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml
from unittest.mock import patch

matplotlib.use("Agg")

from derby.experiments.one_camp_n_days import runner as one_camp_runner
from pipeline.make_config_grid import generate_configs
from utils.analysis import expand_policy_params, last_epoch_table
from utils.paper_plot import VarianceConfig, plot_learning_curves

import matplotlib.pyplot as plt  # noqa: E402


def _epoch_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "agent_label": "policy-a",
                "run_id": "run-1",
                "agent_name": "learner",
                "policy_class": "REINFORCE",
                "epoch": 0,
                "mean_reward": 1.0,
                "std_reward": 0.2,
                "policy_params_json": '{"learning_rate": 0.1, "dist_type": "gaussian"}',
            },
            {
                "agent_label": "policy-a",
                "run_id": "run-1",
                "agent_name": "learner",
                "policy_class": "REINFORCE",
                "epoch": 1,
                "mean_reward": 2.0,
                "std_reward": 0.3,
                "policy_params_json": '{"learning_rate": 0.1, "dist_type": "gaussian"}',
            },
            {
                "agent_label": "policy-a",
                "run_id": "run-2",
                "agent_name": "learner",
                "policy_class": "REINFORCE",
                "epoch": 0,
                "mean_reward": 3.0,
                "std_reward": 0.4,
                "policy_params_json": '{"learning_rate": 0.1, "dist_type": "gaussian"}',
            },
            {
                "agent_label": "policy-a",
                "run_id": "run-2",
                "agent_name": "learner",
                "policy_class": "REINFORCE",
                "epoch": 1,
                "mean_reward": 4.0,
                "std_reward": 0.5,
                "policy_params_json": '{"learning_rate": 0.1, "dist_type": "gaussian"}',
            },
            {
                "agent_label": "fixed-bid",
                "run_id": "run-1",
                "agent_name": "baseline",
                "policy_class": "FixedBidPolicy",
                "epoch": 0,
                "mean_reward": 0.5,
                "std_reward": 0.1,
                "policy_params_json": '{"bid_per_item": 5}',
            },
        ]
    )


def test_expand_policy_params_is_quiet_for_missing_baseline_fields() -> None:
    expanded = expand_policy_params(
        _epoch_df(),
        fields=("learning_rate", "dist_type", "bid_per_item"),
    )

    assert expanded.loc[0, "param_learning_rate"] == 0.1
    assert pd.isna(expanded.loc[4, "param_learning_rate"])
    assert expanded.loc[4, "param_bid_per_item"] == 5


def test_last_epoch_table_keeps_agent_identity_by_default() -> None:
    table = last_epoch_table(_epoch_df())

    learner = table[
        (table["agent_label"] == "policy-a")
        & (table["run_id"] == "run-1")
        & (table["agent_name"] == "learner")
    ]
    baseline = table[
        (table["agent_label"] == "fixed-bid")
        & (table["run_id"] == "run-1")
        & (table["agent_name"] == "baseline")
    ]

    assert learner["last_epoch_mean_reward"].iloc[0] == 2.0
    assert baseline["last_epoch_mean_reward"].iloc[0] == 0.5


def test_plot_learning_curves_keeps_one_line_per_agent_label_run_id() -> None:
    fig, ax = plot_learning_curves(_epoch_df(), variance=VarianceConfig())

    try:
        assert len(ax.lines) == 2
        assert {line.get_label() for line in ax.lines} == {
            "policy-a | run-1",
            "policy-a | run-2",
        }
        assert len(ax.collections) == 2
    finally:
        plt.close(fig)


def test_plot_learning_curves_rejects_duplicate_curve_epoch_rows() -> None:
    df = pd.concat([_epoch_df(), _epoch_df().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate rows"):
        plot_learning_curves(df)


def test_make_config_grid_renders_agent_label_templates(tmp_path) -> None:
    base_path = tmp_path / "base.yaml"
    spec_path = tmp_path / "sweep.yaml"
    out_dir = tmp_path / "configs"

    base_path.write_text(
        yaml.safe_dump(
            {
                "num_days": 1,
                "num_trajs": 2,
                "num_epochs": 1,
                "setup": "one_segment",
                "agents": [
                    {
                        "name": "learner",
                        "label": "REINFORCE",
                        "policy": "REINFORCE",
                        "params": {
                            "learning_rate": 1e-5,
                            "use_baseline": False,
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "sweep_name": "demo",
                "base_config": str(base_path),
                "override": {
                    "agents.0.label": (
                        "REINFORCE|LR={agents.0.params.learning_rate}|"
                        "Baseline={agents.0.params.use_baseline}"
                    )
                },
                "grid": {
                    "agents.0.params.learning_rate": [1e-5, 1e-6],
                    "agents.0.params.use_baseline": [True, False],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    generate_configs(str(spec_path), str(out_dir))

    labels = []
    for path in sorted(out_dir.glob("run_*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels.append(cfg["agents"][0]["label"])

    assert labels == [
        "REINFORCE|LR=1e-05|Baseline=True",
        "REINFORCE|LR=1e-05|Baseline=False",
        "REINFORCE|LR=1e-06|Baseline=True",
        "REINFORCE|LR=1e-06|Baseline=False",
    ]


def test_runner_writes_agent_label_without_experiment_label(tmp_path) -> None:
    def fake_train(env, num_of_trajs, horizon_cutoff, **kwargs):
        for idx, agent in enumerate(env.agents):
            agent.cumulative_rewards = [float(idx + 1)] * num_of_trajs

    config = {
        "num_days": 1,
        "num_trajs": 2,
        "num_epochs": 1,
        "setup": "one_segment",
        "agents": [
            {
                "name": "bidder_a",
                "label": "FixedBid|Bid={agents.0.params.bid_per_item}",
                "policy": "FixedBidPolicy",
                "params": {"bid_per_item": 5, "total_limit": 5},
            },
            {
                "name": "bidder_b",
                "label": "FixedBid|Bid={agents.1.params.bid_per_item}",
                "policy": "FixedBidPolicy",
                "params": {"bid_per_item": 7, "total_limit": 7},
            },
        ],
    }

    with patch.object(one_camp_runner, "train", fake_train):
        one_camp_runner.run_experiment_from_config(config, output_dir_override=str(tmp_path))

    parquet_files = list(tmp_path.glob("epoch_agg__*.parquet"))
    assert len(parquet_files) == 1
    df = pd.read_parquet(parquet_files[0])

    assert "label" not in df.columns
    assert "agent_label" in df.columns
    assert set(df["agent_label"]) == {"FixedBid|Bid=5", "FixedBid|Bid=7"}


def test_runner_writes_learning_rate_diagnostics(tmp_path) -> None:
    def fake_train(env, num_of_trajs, horizon_cutoff, **kwargs):
        for idx, agent in enumerate(env.agents):
            agent.cumulative_rewards = [float(idx + 1)] * num_of_trajs
            agent.policy.last_effective_learning_rate = 0.01 * (idx + 1)
            agent.policy.last_grad_norm = 10.0 + idx

    config = {
        "num_days": 1,
        "num_trajs": 2,
        "num_epochs": 1,
        "setup": "one_segment",
        "agents": [
            {
                "name": "bidder_a",
                "policy": "FixedBidPolicy",
                "params": {"bid_per_item": 5, "total_limit": 5},
            },
            {
                "name": "bidder_b",
                "policy": "FixedBidPolicy",
                "params": {"bid_per_item": 7, "total_limit": 7},
            },
        ],
    }

    with patch.object(one_camp_runner, "train", fake_train):
        one_camp_runner.run_experiment_from_config(config, output_dir_override=str(tmp_path))

    parquet_files = list(tmp_path.glob("epoch_agg__*.parquet"))
    assert len(parquet_files) == 1
    df = pd.read_parquet(parquet_files[0]).sort_values("agent_name").reset_index(drop=True)

    assert "effective_learning_rate" in df.columns
    assert "grad_norm" in df.columns
    np.testing.assert_allclose(df["effective_learning_rate"], [0.01, 0.02])
    np.testing.assert_allclose(df["grad_norm"], [10.0, 11.0])
