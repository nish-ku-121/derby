from typing import Any, Dict, List
import logging
import hashlib
import inspect
import os
import json
import uuid
import time
import math
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
 
from derby.experiments.one_camp_n_days import Experiment
from derby.core.agents import Agent
import derby.core.policies as policy_mod
from derby.core.environments import train

# Module-level logger
logger = logging.getLogger(__name__)

SUPPORTED_POLICY_NAMES = {
    "REINFORCE",
    "FixedBidPolicy",
    "BudgetPerReachPolicy",
    "StepPolicy",
}


def _resolve_policy_class(name: str):
    """
    Resolve a policy class from the modern supported surface only.
    """
    if name not in SUPPORTED_POLICY_NAMES:
        supported = ", ".join(sorted(SUPPORTED_POLICY_NAMES))
        raise ValueError(
            f"Unsupported policy for one_camp_n_days_v2: {name}. "
            f"Supported policies: {supported}"
        )
    if hasattr(policy_mod, name):
        return getattr(policy_mod, name)
    raise ValueError(f"Unknown supported policy class: {name}")


def _filter_kwargs_for_callable(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a kwargs dict to only include parameters accepted by callable_obj."""
    sig = inspect.signature(callable_obj)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in accepted or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))}


def _scale_action_value(actions_scaler, value: float) -> float:
    """Map an action-space scalar into the policy's scaled action space."""
    return float(actions_scaler.transform([[float(value)]])[0][0])


def _normalize_policy_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize user/config-level policy params without changing their semantics."""
    normalized = dict(params)
    for key in ("learning_rate", "init_action_value"):
        if key in normalized and isinstance(normalized[key], str):
            try:
                normalized[key] = float(normalized[key])
            except ValueError:
                pass
    return normalized


def _prepare_runtime_policy_params(filtered_params: Dict[str, Any], actions_scaler) -> Dict[str, Any]:
    """Apply runtime-only transformations before instantiating a policy."""
    runtime_params = dict(filtered_params)
    if runtime_params.get('init_action_value') is not None:
        runtime_params['init_action_value'] = _scale_action_value(actions_scaler, runtime_params['init_action_value'])
    return runtime_params


def _compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute a deterministic hash of the config, excluding non-semantic fields."""
    # Exclude fields that shouldn't affect grouping/comparisons
    excluded = {"logging", "label"}
    cfg = {k: v for k, v in config.items() if k not in excluded}
    # Stable string with sorted keys
    cfg_str = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


def _seed_everything(seed: int) -> None:
    try: random.seed(seed)
    except Exception: pass
    try: np.random.seed(seed)
    except Exception: pass
    try: tf.random.set_seed(seed)
    except Exception: pass


def _derive_policy_seed(global_seed: int, policy_class: str, agent_name: str) -> int:
    data = f"{global_seed}:{policy_class}:{agent_name}".encode()
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:4], "big")


def run_experiment_from_config(
    config: Dict[str, Any],
    output_dir_override: str | None = None,
    run_id: str | None = None,
    flush_every: int = 1,
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
        - Supported policies in this runner are limited to unified `REINFORCE` and core deterministic
          baselines (`FixedBidPolicy`, `BudgetPerReachPolicy`, `StepPolicy`).
        - Auto-injected params (if accepted and not already provided):
              auction_item_spec_ids.
        - `init_action_value` is interpreted in unscaled action space at the config layer; if supplied
          and accepted by the policy constructor, this runner scales it before instantiating the policy.
        - State/action scaling applied ONLY to TensorFlow (learning) policies; baseline / static
          policies (e.g., FixedBidPolicy) receive raw state/action data to maintain legacy semantics.
        - Per-epoch metrics: mean & std (population, ddof=0) of per-trajectory rewards for each agent.
        - Logging: If an output directory is provided (via CLI -o/--output-dir), a parquet file with
          per-epoch rows is written containing config_hash, seed, label, and reward stats.
        - config_hash: SHA256 over JSON dump of config minus excluded keys (label, logging) ensuring
          distinct seeds produce distinct hashes.
        - Returns: run_id (str) used in parquet filename.
        - Unsupported: legacy exp_* experiment mappings; this runner only supports the simplified schema.

    Runtime / Non-YAML Parameters:
        flush_every (int, CLI only): Number of epochs between parquet flushes when an output
            directory is provided. A value of 1 (default) writes each epoch's rows immediately
            (minimal memory, maximum durability if a long run crashes). Larger values batch
            multiple epochs in memory before a write, slightly reducing IO overhead at the
            cost of holding those rows temporarily and risking their loss if the process exits
            unexpectedly before the next flush. Values <1 are coerced to 1.
    """
    num_days = int(config['num_days'])
    num_trajs = int(config['num_trajs'])
    num_epochs = int(config['num_epochs'])
    label = config.get('label')  # optional; default None if missing

    # Streaming flush interval (epochs) is CLI-controlled only (not from YAML).
    if flush_every < 1:
        flush_every = 1

    # Optional logging configuration (CLI only; YAML logging keys are ignored)
    output_dir = output_dir_override

    yaml_seed = config.get("seed")
    global_seed = None
    if yaml_seed is not None:
        global_seed = int(yaml_seed)
        _seed_everything(global_seed)

    experiment = Experiment(seed=global_seed)

    # Choose environment setup
    setup_name = config.get('setup')  # full function name like 'setup_1'
    if not isinstance(setup_name, str):
        raise ValueError("Missing or invalid 'setup' key. Provide the function name, e.g., setup_1 or setup_2.")
    if not hasattr(experiment, setup_name):
        raise ValueError(f"Unknown setup function: {setup_name}")
    setup_fn = getattr(experiment, setup_name)
    env, auction_item_spec_ids = setup_fn()

    # Get scaling/de-scaling helpers.
    scale_states_func, actions_scaler, scale_actions_func, descale_actions_func, _ = experiment.get_transformed(env)

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

        # Auto-inject common defaults if accepted and not already provided.
        auto_defaults: Dict[str, Any] = {
            'auction_item_spec_ids': auction_item_spec_ids,
        }
        for k, v in auto_defaults.items():
            params.setdefault(k, v)

        # Normalize common param types to avoid YAML quoting issues.
        params = _normalize_policy_params(params)

        # Derive and inject per-policy seed (order-invariant) whenever a global seed is set
        # and the policy accepts a 'seed' parameter but one hasn't been provided in YAML.
        if global_seed is not None and 'seed' not in params:
            if 'seed' in inspect.signature(policy_cls.__init__).parameters:
                params['seed'] = _derive_policy_seed(global_seed, policy_name, name)

        # Only pass kwargs that the policy's __init__ accepts
        init = policy_cls.__init__
        filtered_params = _filter_kwargs_for_callable(init, params)
        runtime_params = _prepare_runtime_policy_params(filtered_params, actions_scaler)

        policy_instance = policy_cls(**runtime_params)

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

        if global_seed is not None and hasattr(policy_instance, 'seed'):
            logger.info(f"[seed] policy={policy_name} agent={name} seed={getattr(policy_instance,'seed', None)}")

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
    # Streaming parquet writer state
    writer = None  # type: ignore
    out_path = None
    # Column order (locked for stability / reproducibility)
    column_order = [
        "run_id",
        "config_hash",
        "label",
        "setup",
        "global_seed",
        "num_days",
        "num_trajs",
        "num_epochs",
        "epoch",
        "agent_name",
        "policy_class",
        "policy_params_json",
        "mean_reward",
        "std_reward",
        "n_trajs",
    ]
    batch_epoch_rows: List[Dict[str, Any]] = []  # buffered rows until flush

    # Pre-declare Arrow schema (forces mean/std to be float32) if Arrow available.
    arrow_schema = None
    try:
        arrow_schema = pa.schema([
            ("run_id", pa.string()),
            ("config_hash", pa.string()),
            ("label", pa.string()),
            ("setup", pa.string()),
            ("global_seed", pa.int64()),
            ("num_days", pa.int32()),
            ("num_trajs", pa.int32()),
            ("num_epochs", pa.int32()),
            ("epoch", pa.int32()),
            ("agent_name", pa.string()),
            ("policy_class", pa.string()),
            ("policy_params_json", pa.string()),
            ("mean_reward", pa.float32()),
            ("std_reward", pa.float32()),
            ("n_trajs", pa.int32()),
        ])
    except Exception:  # pragma: no cover
        arrow_schema = None

    # Rolling window stats removed (per-epoch parquet logging provides full fidelity for post-hoc analysis).

    def _flush_batch() -> None:
        nonlocal writer, out_path, batch_epoch_rows
        if not output_dir or not batch_epoch_rows:
            return

        # Arrow path
        if out_path is None:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"epoch_agg__{run_id}.parquet")
        # Build column-wise lists in defined order for efficient table creation.
        columns_data: Dict[str, List[Any]] = {k: [] for k in column_order}
        for row in batch_epoch_rows:
            for col in column_order:
                columns_data[col].append(row[col])
        # Infer schema first time.
        # Enforce float32 for mean/std and use explicit schema if available.
        try:
            columns_data["mean_reward"] = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(np.float32(v)) for v in columns_data["mean_reward"]]
            columns_data["std_reward"] = [None if v is None or (isinstance(v, float) and math.isnan(v)) else float(np.float32(v)) for v in columns_data["std_reward"]]
        except Exception:  # pragma: no cover
            pass
        if arrow_schema is not None:
            table = pa.Table.from_pydict(columns_data, schema=arrow_schema)
        else:
            table = pa.Table.from_pydict(columns_data)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        batch_epoch_rows = []

    # Memory profiling functionality removed (deprecated CLI flags eliminated).

    start = time.time()
    try:
        for i in range(NUM_EPOCHS):
            # Train one epoch worth of trajectories. If path-level profiling is enabled, also turn on deep
            # concatenation / per-step instrumentation without needing extra CLI flags.
            train(env, num_of_trajs, horizon_cutoff)

            # Compute per-agent epoch aggregates using ONLY the newly added rewards for this epoch.
            avg_and_std_rwds = []
            for agent, ainfo in zip(env.agents, agent_meta):
                # The training step appended exactly num_of_trajs trajectory rewards for this agent.
                # Grab the tail slice.
                current_rewards = agent.cumulative_rewards[-num_of_trajs:]
                # Compute stats on these trajectory rewards (population std to match prior logic ddof=0)
                if len(current_rewards) > 0:
                    mean_r = float(np.mean(current_rewards))
                    std_r = float(np.std(current_rewards, ddof=0))
                else:
                    mean_r = float("nan")
                    std_r = float("nan")

                # Append to batch rows for parquet.
                if output_dir:
                    batch_epoch_rows.append({
                        "run_id": run_id,
                        "config_hash": config_hash,
                        "label": label,
                        "setup": setup_name,
                        "global_seed": global_seed,
                        "num_days": num_of_days,
                        "num_trajs": num_of_trajs,
                        "num_epochs": NUM_EPOCHS,
                        "epoch": i,
                        "agent_name": agent.name,
                        "policy_class": ainfo["policy_class"],
                        "policy_params_json": json.dumps(ainfo["policy_params"], sort_keys=True),
                        "mean_reward": mean_r,
                        "std_reward": std_r,
                        "n_trajs": num_of_trajs,
                    })

                avg_and_std_rwds.append((agent.name, mean_r, std_r))

                # Discard per-trajectory rewards to prevent unbounded growth (retain nothing between epochs).
                try:
                    agent.cumulative_rewards = np.empty(0, dtype=np.float32)
                except Exception:
                    # Fallback safe reset
                    agent.cumulative_rewards = []

            logger.info("epoch=%s avg_and_std_rwds=%s", i, avg_and_std_rwds)

            # Flush parquet batch if needed
            if output_dir and ((i + 1) % flush_every == 0):
                _flush_batch()

            # (memory profiling removed)
        end = time.time()
        logger.info("Training complete in %.2f sec", end - start)
        return run_id
    finally:
        # Always attempt to write whatever we have so far
        try:
            # Final flush & close writer
            if output_dir:
                _flush_batch()
                if writer is not None:  # type: ignore
                    writer.close()  # type: ignore
                    logger.info("Wrote epoch aggregates to %s", out_path)
        except Exception as e:  # pragma: no cover
            logger.error("Failed final parquet write/close: %s", e)


def main(yaml_path: str, output_dir: str | None = None, log_level: str | None = "INFO", flush_every: int = 1) -> None:
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

    run_experiment_from_config(
        config,
        output_dir_override=output_dir,
        flush_every=flush_every,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Derby experiment from YAML config.")
    parser.add_argument('--config', required=True, type=str, help='Path to YAML config file')
    parser.add_argument('-o', '--output-dir', dest='output_dir', type=str, default=None,
                        help='Optional directory to write epoch-level parquet logs')
    parser.add_argument('--log-level', dest='log_level', default='INFO', type=str,
                        help='Logging verbosity for this run (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE). Not propagated to inner APIs.')
    parser.add_argument('--flush-every', dest='flush_every', default=1, type=int,
                    help='Number of epochs between parquet flushes (default 1).')
    # Memory profiling CLI flags removed (were: --profile-memory, --mem-interval, --profile-memory-path)
    args = parser.parse_args()
    main(args.config,
        output_dir=args.output_dir,
        log_level=args.log_level,
        flush_every=args.flush_every)
