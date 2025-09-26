from typing import Any, Dict, List
import inspect
import os
import json
import uuid
import time
import hashlib
import logging

import pandas as pd
import numpy as np

from derby.experiments.one_camp_n_days import Experiment
from derby.core.agents import Agent
import derby.core.policies as policy_mod
from derby.core.environments import train

# Module-level logger
logger = logging.getLogger(__name__)


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


def run_experiment_from_config(
    config: Dict[str, Any],
    output_dir_override: str | None = None,
    run_id: str | None = None,
) -> str:
    """Run an experiment described by an in-memory config dict (public API).

    Supported (simplified) YAML schema:
        num_days: int                    # days per trajectory (episode length / horizon)
        num_trajs: int                   # trajectories (episodes) per epoch
        num_epochs: int                  # number of training epochs
        setup: setup_1 | setup_2         # name of Experiment.setup_* method
        seed: int (optional)             # reproducibility seed (YAML ONLY; no CLI override)
        label: str (optional)            # carried through to parquet output (excluded from hash)
        agents:                          # list in execution order
            - name: agent1                 # optional; auto-generated if omitted
              policy: FullPolicyClassName  # MUST exactly match symbol in derby.core.policies
              params:                      # kwargs passed to the policy __init__ (filtered)
                  learning_rate: 5e-6
                  shape_reward: false
            - name: baseline
              policy: FixedBidPolicy
              params:
                  bid_per_item: 5
                  total_limit: 5

    Behavior / Notes:
        - Seeds: If 'seed' present it is validated & passed to Experiment; absent => stochastic run.
        - Policy parameter filtering: only kwargs accepted by the policy __init__ are forwarded.
        - Auto-injected params (if accepted and not already provided):
              auction_item_spec_ids, budget_per_reach (scaled average budget-per-reach estimate).
        - State/action scaling applied ONLY to TensorFlow (learning) policies; baseline / static
          policies (e.g., FixedBidPolicy) receive raw state/action data to maintain legacy semantics.
        - Per-epoch metrics: mean & std (population, ddof=0) of per-trajectory rewards for each agent.
        - Logging: If an output directory is provided (via CLI -o/--output-dir), a parquet file with
          per-epoch rows is written containing config_hash, seed, label, and reward stats.
        - config_hash: SHA256 over JSON dump of config minus excluded keys (label, logging) ensuring
          distinct seeds produce distinct hashes.
        - Returns: run_id (str) used in parquet filename.
        - Unsupported: legacy exp_* experiment mappings; this runner only supports the simplified schema.
    """
    num_days = int(config['num_days'])
    num_trajs = int(config['num_trajs'])
    num_epochs = int(config['num_epochs'])
    label = config.get('label')  # optional; default None if missing

    # Optional logging configuration (CLI only; YAML logging keys are ignored)
    output_dir = output_dir_override

    yaml_seed = config.get('seed')
    seed = None
    if yaml_seed is not None:
        try:
            seed = int(yaml_seed)
        except Exception:
            raise ValueError(f"Invalid seed value in config: {yaml_seed}")
    experiment = Experiment(seed=seed)

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
    logger.debug("scaled_avg_bpr=%s", scaled_avg_bpr)

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

        # IMPORTANT:
        # In the legacy experiment scripts, only learning (TF) policies received
        # state normalization + action scaling/descaling. Baseline policies like
        # FixedBidPolicy operated directly on raw state vectors (e.g. auction_item_spec_id)
        # and produced already-descaled actions. Passing them through the scalers
        # distorts spec IDs and bids, leading to incorrect rewards (e.g. the RL
        # policy capturing almost all reward, others zero). We replicate the legacy
        # behavior here: only TensorFlow policies (policy_instance.is_tensorflow == True)
        # get the scalers.
        if getattr(policy_instance, 'is_tensorflow', False):
            agent = Agent(name, policy_instance, scale_states_func, scale_actions_func, descale_actions_func)
        else:
            agent = Agent(name, policy_instance)  # identity transforms
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
    logger.debug("days per traj: %s, trajs per epoch: %s, EPOCHS: %s", num_of_days, num_of_trajs, NUM_EPOCHS)

    env.vectorize = True
    env.init(agents, num_of_days)
    logger.debug("agent policies: %s", [agent.policy for agent in env.agents])

    # Determine run identifier (allow caller to fix it, so failures can still be correlated)
    run_id = run_id or str(uuid.uuid4())
    config_hash = _compute_config_hash(config)
    rows: List[Dict[str, Any]] = []

    start = time.time()
    try:
        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff)
            # Compute per-agent epoch aggregates
            avg_and_std_rwds = []
            for agent in env.agents:
                last = agent.cumulative_rewards[-num_of_trajs:]
                mean_r = float(sum(last) / len(last)) if len(last) > 0 else float("nan")
                # Use numpy via pandas for std to avoid an extra import
                std_r = float(pd.Series(last).std(ddof=0)) if len(last) > 0 else float("nan")
                avg_and_std_rwds.append((agent.name, mean_r, std_r))

            logger.info("epoch=%s avg_and_std_rwds=%s", i, avg_and_std_rwds)

            # Append logs
            if output_dir:
                for ainfo, stats in zip(agent_meta, avg_and_std_rwds):
                    row = {
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "label": label,
                        "setup": setup_name,
                        "seed": seed,
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
                max_last_50_epochs = []
                for agent in env.agents:
                    last50 = agent.cumulative_rewards[-50 * num_of_trajs:]
                    mean_r = float(sum(last50) / len(last50)) if len(last50) > 0 else float("nan")
                    std_r = float(pd.Series(last50).std(ddof=0)) if len(last50) > 0 else float("nan")
                    avg_and_std_rwds_last_50_epochs.append((agent.name, mean_r, std_r))

                    # Correct max: compute per-epoch mean rewards and then take their max.
                    if len(last50) == 50 * num_of_trajs and num_of_trajs > 0:
                        per_epoch_means = np.array(last50).reshape(50, num_of_trajs).mean(axis=1)
                        max_epoch_mean = float(np.max(per_epoch_means))
                    else:
                        max_epoch_mean = mean_r
                    max_last_50_epochs.append((agent.name, max_epoch_mean))

                logger.info("Avg of last 50 epochs: %s", avg_and_std_rwds_last_50_epochs)
                logger.info("Max of last 50 epochs (epoch means): %s", max_last_50_epochs)

        end = time.time()
        logger.info("Training complete in %.2f sec", end - start)
        return run_id
    finally:
        # Always attempt to write whatever we have so far
        if output_dir and rows:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"epoch_agg__{run_id}.parquet")
            df = pd.DataFrame(rows)
            try:
                df.to_parquet(out_path, index=False)
                logger.info("Wrote epoch aggregates to %s", out_path)
            except Exception as e:
                logger.error("Failed to write parquet to %s: %s", out_path, e)


def main(yaml_path: str, output_dir: str | None = None, log_level: str | None = "INFO") -> None:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for --config usage. Install with 'poetry add pyyaml' or 'pip install pyyaml'."
        ) from e

    # Configure logging here (CLI entrypoint)
    suppression_tokens = {"NONE", "OFF", "QUIET", "SILENT", "NA", "N/A"}
    if log_level is None:
        log_level = "INFO"
    lvl = str(log_level).upper()
    if lvl in suppression_tokens:
        logging.disable(logging.CRITICAL)
    else:
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=getattr(logging, lvl, logging.INFO),
                format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
            )
        else:
            root.setLevel(getattr(logging, lvl, logging.INFO))

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    run_experiment_from_config(config, output_dir_override=output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Derby experiment from YAML config.")
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default=None,
                        help='Optional directory to write epoch-level parquet logs')
    parser.add_argument('--log-level', dest='log_level', default='INFO', type=str,
                        help='Logging verbosity for this run (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE). Not propagated to inner APIs.')
    args = parser.parse_args()
    main(args.config, output_dir=args.output_dir, log_level=args.log_level)
