from typing import Any, Dict, List
import inspect
import os
import json
import uuid
import time
import hashlib

import pandas as pd

from derby.experiments.one_camp_n_days import Experiment
from derby.core.agents import Agent
import derby.core.policies as policy_mod
from derby.core.environments import train


def _resolve_policy_class(name: str):
    """
    Resolve a policy class by exact name only.
    The YAML must provide the full class name as defined in derby.core.policies,
    for example: REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous
    """
    if hasattr(policy_mod, name):
        return getattr(policy_mod, name)
    raise ValueError(f"Unknown policy class (must be full name): {name}")


def _filter_kwargs_for_callable(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs dict to only include parameters accepted by callable_obj."""
    sig = inspect.signature(callable_obj)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in accepted or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))}


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of the config, excluding non-semantic fields."""
    # Exclude fields that shouldn't affect grouping/comparisons
    excluded = {"logging", "label"}
    cfg = {k: v for k, v in config.items() if k not in excluded}
    # Stable string with sorted keys
    cfg_str = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


def _run_simple_config(
    config: Dict[str, Any],
    output_dir_override: str | None = None,
    run_id: str | None = None,
) -> str:
    """
        Simplified YAML schema runner.
                Schema (simplified only):
            num_days: int
            num_trajs: int
            num_epochs: int
            debug: bool                    # optional
            setup: setup_1 | setup_2       # name of Experiment setup function to call
            agents:
                - name: agent1
                    policy: REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous   # full class name
                    params:                                                         # direct __init__ params for the policy
            shape_reward: true
        - name: agent2
          policy: StepPolicy | FixedBidPolicy
          params:
            # for StepPolicy: (num_days, step)
            # for FixedBidPolicy: (bid_per_item, total_limit)
            # etc.
    Notes:
            - auction_item_spec_ids and budget_per_reach are auto-injected when the policy accepts them.
            - Only simplified schema is supported in this file (no exp_* fallback).
    """
    num_days = int(config['num_days'])
    num_trajs = int(config['num_trajs'])
    num_epochs = int(config['num_epochs'])
    debug = bool(config.get('debug', False))
    label = config.get('label')  # optional; default None if missing

    # Optional logging configuration (CLI only; YAML logging keys are ignored)
    output_dir = output_dir_override

    experiment = Experiment()

    # Choose environment setup
    setup_name = config.get('setup')  # full function name like 'setup_1'
    if not isinstance(setup_name, str):
        raise ValueError("Missing or invalid 'setup' key. Provide the function name, e.g., setup_1 or setup_2.")
    if not hasattr(experiment, setup_name):
        raise ValueError(f"Unknown setup function: {setup_name}")
    setup_fn = getattr(experiment, setup_name)
    env, auction_item_spec_ids = setup_fn()

    # Get scaling/de-scaling helpers and scaled_avg_bpr
    scale_states_func, actions_scaler, scale_actions_func, descale_actions_func, scaled_avg_bpr = experiment.get_transformed(env)

    # Build agents
    agents_cfg: List[Dict[str, Any]] = config['agents']
    agents: List[Agent] = []
    # Keep per-agent metadata for logging
    agent_meta: List[Dict[str, Any]] = []
    for idx, a in enumerate(agents_cfg, start=1):
        name = a.get('name', f'agent{idx}')
        policy_name = a['policy']
        params: Dict[str, Any] = dict(a.get('params', {}))

        # Determine class and filter params
        policy_cls = _resolve_policy_class(policy_name)

        # Auto-inject common defaults if accepted and not already provided
        auto_defaults: Dict[str, Any] = {
            'auction_item_spec_ids': auction_item_spec_ids,
            'budget_per_reach': scaled_avg_bpr,
        }
        for k, v in auto_defaults.items():
            params.setdefault(k, v)

        # Normalize common param types to avoid YAML quoting issues
        if 'learning_rate' in params and isinstance(params['learning_rate'], str):
            try:
                params['learning_rate'] = float(params['learning_rate'])
            except ValueError:
                pass

        # Only pass kwargs that the policy's __init__ accepts
        init = policy_cls.__init__
        filtered_params = _filter_kwargs_for_callable(init, params)

        policy_instance = policy_cls(**filtered_params)

        agent = Agent(name, policy_instance, scale_states_func, scale_actions_func, descale_actions_func)
        agents.append(agent)
        agent_meta.append({
            "agent_name": name,
            "policy_class": policy_name,
            "policy_params": filtered_params,
        })

    # Train loop (mirrors Experiment.run but allows logging per epoch)
    num_of_days = num_days
    num_of_trajs = num_trajs
    NUM_EPOCHS = num_epochs
    horizon_cutoff = 100
    if debug:
        print(f"days per traj: {num_of_days}, trajs per epoch: {num_of_trajs}, EPOCHS: {NUM_EPOCHS}")

    env.vectorize = True
    env.init(agents, num_of_days)
    if debug:
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

    # Determine run identifier (allow caller to fix it, so failures can still be correlated)
    run_id = run_id or str(uuid.uuid4())
    config_hash = _compute_config_hash(config)
    rows: List[Dict[str, Any]] = []

    start = time.time()
    try:
        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
            # Compute per-agent epoch aggregates
            avg_and_std_rwds = []
            for agent in env.agents:
                last = agent.cumulative_rewards[-num_of_trajs:]
                mean_r = float(sum(last) / len(last)) if len(last) > 0 else float("nan")
                # Use numpy via pandas for std to avoid an extra import
                std_r = float(pd.Series(last).std(ddof=0)) if len(last) > 0 else float("nan")
                avg_and_std_rwds.append((agent.name, mean_r, std_r))
            if debug:
                print(f"epoch: {i}, avg and std rwds: {avg_and_std_rwds}")

            # Append logs
            if output_dir:
                for ainfo, stats in zip(agent_meta, avg_and_std_rwds):
                    row = {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "label": label,
                        "setup": setup_name,
                        "num_days": num_of_days,
                        "num_trajs": num_of_trajs,
                        "num_epochs": NUM_EPOCHS,
                        "epoch": i,
                        "agent_name": stats[0],
                        "policy_class": ainfo["policy_class"],
                        "policy_params_json": json.dumps(ainfo["policy_params"], sort_keys=True),
                        "mean_reward": stats[1],
                        "std_reward": stats[2],
                        "n_trajs": num_of_trajs,
                    }
                    rows.append(row)

            if ((i + 1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = []
                for agent in env.agents:
                    last50 = agent.cumulative_rewards[-50 * num_of_trajs:]
                    mean_r = float(sum(last50) / len(last50)) if len(last50) > 0 else float("nan")
                    std_r = float(pd.Series(last50).std(ddof=0)) if len(last50) > 0 else float("nan")
                    avg_and_std_rwds_last_50_epochs.append((agent.name, mean_r, std_r))
                max_last_50_epochs = [(agent.name, float(max(agent.cumulative_rewards[-50 * num_of_trajs:]))) for agent in env.agents]
                if debug:
                    print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
                    print("Max of last 50 epochs: {}".format(max_last_50_epochs))

        end = time.time()
        if debug:
            print("Took {} sec to train".format(end - start))
        return run_id
    finally:
        # Always attempt to write whatever we have so far
        if output_dir and rows:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"epoch_agg__{run_id}.parquet")
            df = pd.DataFrame(rows)
            try:
                df.to_parquet(out_path, index=False)
                print(f"Wrote epoch aggregates to {out_path}")
            except Exception as e:
                print(f"Failed to write parquet to {out_path}: {e}")


def main(yaml_path: str, output_dir: str | None = None) -> None:
    """
    Run an experiment from a YAML config file.
    YAML schema:
      exp: str (e.g., exp_1000)
      num_days: int
      num_trajs: int
      num_epochs: int
      lr: float
      debug: bool (optional)
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for --config usage. Install with 'poetry add pyyaml' or 'pip install pyyaml'."
        ) from e

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Simplified schema only
    _run_simple_config(config, output_dir_override=output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Derby experiment from YAML config.")
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default=None,
                        help='Optional directory to write epoch-level parquet logs (CLI only; YAML logging keys are ignored)')
    args = parser.parse_args()
    main(args.config, output_dir=args.output_dir)
