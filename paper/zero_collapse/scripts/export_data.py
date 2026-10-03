"""Export the normalized data used by the zero-collapse paper bundle.

This is a maintainer-only bridge from ignored raw result directories to the
tracked publication data.  It never trains a policy and it deliberately keeps
only fields needed to audit or redraw the paper figures and outcome table.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils.analysis import expand_policy_params, filter_epoch_rewards, load_epoch_rewards  # noqa: E402
from pipeline.make_config_grid import generate_configs  # noqa: E402


OUT = ROOT / "paper" / "zero_collapse" / "data" / "reward_traces.parquet"
RUN_OUTCOMES = ROOT / "paper" / "zero_collapse" / "data" / "run_outcomes.csv"
PARAMS = (
    "learning_rate", "use_baseline", "optimizer", "adaptive_learning_rate",
    "adaptive_lr_epsilon", "actor_hidden_activation", "actor_final_activation",
    "param_kernel_initializer", "init_action_center", "init_action_stddev",
    "min_action_stddev",
)

# Every root used by a paper figure or Table 4.  The key is retained in the
# exported data so rendering never has to infer provenance from a local path.
SOURCE_ROOTS: dict[str, Path] = {
    "control_reinforce": ROOT / "results/staging/control/reinforce/reinforce_sweep_control_2",
    "control_actor_critic": ROOT / "results/staging/control/actor_critic/actor_critic_sweep_control_2",
    "adaptive_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_adaptive_step_rate_1000ep",
    "adaptive_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_adaptive_step_rate_1000ep",
    "smooth_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_smooth_parameterization",
    "smooth_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_smooth_parameterization",
    "combined_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_combined_sgd_high_epsilon",
    "combined_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_combined_sgd_high_epsilon",
    "untreated_reinforce_123": ROOT / "results/staging/control/reinforce/reinforce_sweep_control_parallel2",
    "untreated_reinforce_456_789": ROOT / "results/staging/control/reinforce/reinforce_sweep_control_seeds_456_789",
    "untreated_actor_critic": ROOT / "results/staging/control/actor_critic/actor_critic_sweep_control",
    "bias_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_sweep_treatment_action_centering",
    "bias_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_bias_centered_glorot",
    "fixed_sgd_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_sgd_high_rate_fixed_smoke",
    "fixed_sgd_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_sgd_high_rate_fixed_smoke",
    "fixed_adam_reinforce": ROOT / "results/staging/treatment/reinforce/reinforce_adam_high_rate_fixed_full",
    "fixed_adam_actor_critic": ROOT / "results/staging/treatment/actor_critic/actor_critic_adam_high_rate_fixed_full",
}

MATCHED_SWEEP_SPECS = {
    "reinforce_baseline_off_sgd": "configs/reinforce_fixed_sgd_matched_full.yaml",
    "reinforce_baseline_off_adam": "configs/reinforce_fixed_adam_matched_full.yaml",
    "actor_critic_sgd": "configs/actor_critic_fixed_sgd_matched_full.yaml",
    "actor_critic_adam": "configs/actor_critic_fixed_adam_matched_full.yaml",
    "reinforce_baseline_on_sgd": "configs/reinforce_baseline_on_fixed_sgd_matched_full.yaml",
    "reinforce_baseline_on_adam": "configs/reinforce_baseline_on_fixed_adam_matched_full.yaml",
}
MATCHED_GROUPS = (
    ("reinforce_baseline_off_sgd", "REINFORCE", "off", "sgd", "full/reinforce_fixed_sgd_matched_full"),
    ("reinforce_baseline_off_adam", "REINFORCE", "off", "adam", "full/reinforce_fixed_adam_matched_full"),
    ("actor_critic_sgd", "ActorCritic", "state-value critic", "sgd", "full/actor_critic_fixed_sgd_matched_full"),
    ("actor_critic_adam", "ActorCritic", "state-value critic", "adam", "full/actor_critic_fixed_adam_matched_full"),
    ("reinforce_baseline_on_sgd", "REINFORCE", "on", "sgd", "baseline_on_full/reinforce_baseline_on_fixed_sgd_matched_full"),
    ("reinforce_baseline_on_adam", "REINFORCE", "on", "adam", "baseline_on_full/reinforce_baseline_on_fixed_adam_matched_full"),
)
MATCHED_ROOT = ROOT / "results/staging/paper_followup/fixed_optimizer_matched"


def load_root(source_key: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing paper source root: {path}")
    frame = load_epoch_rewards(path)
    frame = filter_epoch_rewards(expand_policy_params(frame, fields=PARAMS), agent_name="learner")
    if frame.empty:
        raise RuntimeError(f"no learner epoch records in paper source root: {path}")
    frame = frame.copy()
    frame.insert(0, "source_key", source_key)
    for field in ("epoch", "mean_reward", "global_seed", "param_learning_rate", "param_adaptive_lr_epsilon"):
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _classify_matched(rewards: pd.Series, *, complete: bool, numerical: bool) -> tuple[str, float, float]:
    finite = rewards[np.isfinite(rewards)]
    peak = float(finite.max()) if not finite.empty else np.nan
    final100 = float(rewards.tail(100).mean()) if complete and len(rewards.tail(100)) == 100 else np.nan
    if numerical:
        return "numerical_failure", peak, final100
    if finite.eq(0).all():
        return "zero_from_initialization", peak, final100
    if peak >= 5 and final100 < 1:
        return "terminal_collapse", peak, final100
    if final100 >= 5:
        return "sustained_material_reward", peak, final100
    if peak < 5:
        return "underpowered_or_low_reward", peak, final100
    return "material_peak_limited_finish", peak, final100


def extract_matched_optimizer_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read Figure 8 directly from raw results and tracked sweep specs."""
    traces: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="derby_matched_") as temporary:
        temp_root = Path(temporary)
        for group, algorithm, baseline, optimizer, result_suffix in MATCHED_GROUPS:
            spec = ROOT / MATCHED_SWEEP_SPECS[group]
            generated = temp_root / group
            generate_configs(str(spec), str(generated))
            configs = sorted(generated.glob("run_*.yaml"))
            if len(configs) != 9:
                raise RuntimeError(f"{group}: expected nine generated full-run configs, found {len(configs)}")
            result_root = MATCHED_ROOT / result_suffix
            for config_path in configs:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                params = config["agents"][0]["params"]
                run = config_path.stem
                run_dir = result_root / run
                parquet = list(run_dir.glob("epoch_agg__*.parquet"))
                seed = int(config["seed"])
                rate = float(params["learning_rate"])
                audit: dict[str, object] = {
                    "group": group, "sweep_spec": MATCHED_SWEEP_SPECS[group],
                    "algorithm": algorithm, "baseline": baseline, "optimizer": optimizer,
                    "run": run, "seed": seed, "learning_rate": rate,
                    "config_path": f"sweeps/{spec.stem}/configs/{run}.yaml",
                    "config_sha256": _config_hash(config_path), "result_dir": str(run_dir.relative_to(ROOT)),
                    "completion_record": (run_dir / "_RUN_COMPLETE.json").exists(),
                    "failure_record": (run_dir / "failure.json").exists(),
                    "parquet_count": len(parquet), "config_mismatches": "",
                }
                if len(parquet) != 1:
                    audit.update(epochs_observed=0, complete_epochs=False, finite_metrics=False, fixed_rate_identity=False,
                                 reward_min=np.nan, reward_max=np.nan, final100_mean=np.nan,
                                 peak_reward=np.nan, outcome="numerical_failure")
                    audits.append(audit)
                    continue
                learner = pd.read_parquet(parquet[0])
                learner = learner[learner.policy_class.eq(algorithm)].sort_values("epoch").copy()
                rewards = pd.to_numeric(learner["mean_reward"], errors="coerce")
                epochs = pd.to_numeric(learner["epoch"], errors="coerce")
                complete = bool(np.array_equal(epochs.to_numpy(dtype=int), np.arange(1000)))
                metric_columns = [c for c in ("mean_reward", "std_reward", "effective_learning_rate", "grad_norm") if c in learner]
                finite_metrics = bool(metric_columns and np.isfinite(learner[metric_columns].to_numpy(dtype=float)).all())
                fixed_rate = bool(np.allclose(pd.to_numeric(learner["effective_learning_rate"], errors="coerce"), rate, rtol=1e-6, atol=1e-12))
                numerical = (not audit["completion_record"] or not complete or not finite_metrics or not fixed_rate
                             or (not rewards.dropna().empty and rewards.abs().max() > 1e6))
                outcome, peak, final100 = _classify_matched(rewards, complete=complete, numerical=bool(numerical))
                audit.update(epochs_observed=len(learner), complete_epochs=complete, finite_metrics=finite_metrics,
                             fixed_rate_identity=fixed_rate, reward_min=float(rewards.min()), reward_max=float(rewards.max()),
                             final100_mean=final100, peak_reward=peak, outcome=outcome)
                trace = learner[["epoch", "mean_reward", "std_reward", "effective_learning_rate", "grad_norm"]].copy()
                trace["group"] = group
                trace["algorithm"] = algorithm
                trace["baseline"] = baseline
                trace["optimizer"] = optimizer
                trace["run"] = run
                trace["seed"] = seed
                trace["learning_rate"] = rate
                traces.append(trace)
                audits.append(audit)
    return pd.concat(traces, ignore_index=True), pd.DataFrame(audits)


def append_matched_optimizer_traces(frame: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Append directly extracted matched-study traces in the common schema."""
    matched = matched.rename(
        columns={"algorithm": "policy_class", "seed": "global_seed", "learning_rate": "param_learning_rate"}
    )
    matched["source_key"] = "matched_fixed_optimizer"
    matched["param_optimizer"] = matched["optimizer"]
    matched["param_adaptive_learning_rate"] = False
    matched["param_adaptive_lr_epsilon"] = pd.NA
    matched["param_actor_hidden_activation"] = "relu"
    matched["param_actor_final_activation"] = "softplus"
    matched["param_kernel_initializer"] = "zeros"
    matched["param_init_action_center"] = 10.0
    matched["param_init_action_stddev"] = 0.5
    matched["param_min_action_stddev"] = 0.1
    matched["param_use_baseline"] = matched["baseline"].map({"off": False, "on": True})
    matched.loc[matched.policy_class.eq("ActorCritic"), "param_use_baseline"] = pd.NA
    matched["agent_name"] = "learner"
    # The trace export is intentionally compact and does not repeat the result
    # directory.  Per-run source paths and config hashes live in run_outcomes.
    matched["source_dir"] = "matched_fixed_optimizer"
    return pd.concat([frame, matched], ignore_index=True, sort=False)


def main() -> None:
    frames = [load_root(key, path) for key, path in SOURCE_ROOTS.items()]
    matched_traces, matched_outcomes = extract_matched_optimizer_data()
    traces = append_matched_optimizer_traces(pd.concat(frames, ignore_index=True, sort=False), matched_traces)
    required = {"source_key", "policy_class", "global_seed", "epoch", "mean_reward"}
    missing = required - set(traces.columns)
    if missing:
        raise RuntimeError(f"normalized trace export is missing required columns: {sorted(missing)}")
    traces = traces.sort_values(["source_key", "policy_class", "global_seed", "epoch"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    traces.to_parquet(OUT, index=False)
    matched_outcomes.to_csv(RUN_OUTCOMES, index=False)
    print(f"wrote {len(traces):,} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
