from typing import Any, Dict, List
import inspect

from derby.experiments.one_camp_n_days import Experiment
from derby.core.agents import Agent
import derby.core.policies as policy_mod


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


def _run_simple_config(config: Dict[str, Any]) -> None:
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

    # Run the game (keep defaults consistent with existing experiments)
    print("Running config-driven experiment with simplified schema")
    experiment.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)


def main(yaml_path: str) -> None:
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
    _run_simple_config(config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Derby experiment from YAML config.")
    parser.add_argument('config', type=str, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)
