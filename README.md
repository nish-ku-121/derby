
## Project Overview

Derby is a bidding, auction, and *market* framework for creating and running auction or market games. Environments in Derby can be interfaced in a similar fashion as environments in OpenAI’s gym.

Example usage:
```python
env = ...
agents = ...
env.init(agents, num_of_days)
for i in range(num_of_trajs):
        all_agents_states = env.reset()
        for j in range(horizon_cutoff):
            actions = []
            for agent in env.agents:
                agent_states = env.get_folded_states(
                                    agent, all_agents_states
                                )
                actions.append(agent.compute_action(agent_states))
            all_agents_states, rewards, done = env.step(actions)
```

---

## Documentation quick links

- Policies overview: see docs/Policies.md for a grouped rundown of all policies in `derby/core/policies.py`.

---

## Game Description

A *market* can be thought of as a stateful, repeated auction:

-   A market is initialized with *m* bidders, each of which has a state.
-   A market lasts for *N* days.
-   Each day, auction items are put on sale. Each day, the bidders participate in an auction for the available items.
-   Each bidder’s state is updated at the end of every day. The state can track information such as auction items bought and amount spent.

---

## First Time Install

This project is designed to run using Docker. You only need Docker installed—no other dependencies are required on your host machine.

1. **Install Docker**  
    Download and install Docker from:  
    https://www.docker.com/products/docker-desktop

2. **Build the Docker image**  
    In the project root directory, run:
    ```bash
    make build
    ```

## Running Experiments (Outdated)

The Makefile's `run` target automatically executes commands via Poetry inside the container, so you can pass plain `python` commands in `ARGS`.

You can invoke an experiment script in several equivalent ways (pick whichever you find clearest):

1. Module form (preferred; resolves relative imports robustly)
    ```bash
    make run ARGS="python -u -m derby.experiments.one_camp_n_days <experiment_name> <num_days> <num_trajs> <num_epochs> <learning_rate>"
    ```
2. File path form
    ```bash
    make run ARGS="python -u derby/experiments/one_camp_n_days.py <experiment_name> <num_days> <num_trajs> <num_epochs> <learning_rate>"
    ```
3. Open an interactive shell then run repeatedly (good for quick loops)
    ```bash
    make shell
    # Now inside container
    poetry run python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7
    ```
4. With explicit environment variables (example: restrict TensorFlow threading)
    ```bash
    make run ARGS="TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7"
    ```
5. Using a short alias variable for readability (shell feature)
    ```bash
    EXP_ARGS="exp_1000 1 200 100 5e-7" ; make run ARGS="python -u -m derby.experiments.one_camp_n_days $EXP_ARGS"
    ```

Example (module form):
```bash
make run ARGS="python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7"
```

### Modern YAML-driven runner (v2)

For new work, prefer the YAML-based runner `derby.experiments.one_camp_n_days_v2`. It consumes a single config dict from a YAML file and logs per-epoch metrics to Parquet.

CLI usage:
```bash
make run ARGS="python -u -m derby.experiments.one_camp_n_days_v2 \
    --config configs/one_camp_n_days_v2_config.yaml \
    -o results/test_run \
    --log-level INFO"
```

Key points:
- Seed (if you want reproducibility) must be specified ONLY inside the YAML as `seed:`. There is no CLI override.
- Output directory (`-o/--output-dir`) is optional; if provided, a Parquet file `epoch_agg__<run_id>.parquet` is written there containing one row per (epoch, agent).
- Each row includes: `run_id`, `config_hash`, `seed`, `label` (if provided), per-agent mean/std reward, and core config fields.
- `config_hash` is a SHA256 of the config with non-semantic keys (`label`, `logging`) removed; changing the seed changes the hash.
- Baseline (non-TensorFlow) policies receive raw states/actions; only TensorFlow policies are scaled/normalized.

Minimal YAML example (`configs/one_camp_n_days_v2_config.yaml`) using the unified `REINFORCE` policy (old preset names removed):
```yaml
num_days: 1            # days per trajectory
num_trajs: 200         # trajectories per epoch
num_epochs: 100        # training epochs
setup: setup_1         # maps to Experiment.setup_1()
seed: 123              # optional; remove for stochastic run
label: tiny-demo       # optional, for easier filtering later
agents:
    - name: learner
        policy: REINFORCE            # unified class (no more REINFORCE_PRESET_v*)
        params:
            learning_rate: 5e-7
            # Common knobs (override as needed) ----------------------------------
            actor_hidden_layers: 1     # 0 => no hidden layers before param head
            actor_hidden_units: 8
            critic_hidden_layers: 1
            critic_hidden_units: 8
            actor_final_activation: softplus   # softplus | relu | other TF activations
            # Optional explicit initialization target in natural action space
            # init_action_value: 5.0
            sigma_scale: 0.5                   # multiplicative scale on activated sigma raw
            sigma_floor: 1e-5                  # added floor after scaling
            dist_type: gaussian                # gaussian | lognormal | triangular (stub)
            shape_reward: false                # if true: advantage shaping via log(1+r)
    - name: baseline
        policy: FixedBidPolicy
        params:
            bid_per_item: 5
            total_limit: 5
```

Python / notebook usage (public API):
```python
import yaml
from derby.experiments.one_camp_n_days_v2 import run_experiment_from_config

with open("configs/one_camp_n_days_v2_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

run_id = run_experiment_from_config(
    cfg,
    output_dir_override="results/notebook_demo",  # writes Parquet here if provided
    log_level="DEBUG",  # or "INFO" / "NONE" etc.
)
print("Run ID:", run_id)
# Parquet path: results/notebook_demo/epoch_agg__<run_id>.parquet
```

Tip: If you want many variants, use the parallel sweeper below instead of hand-editing multiple YAMLs.

---

## Unified REINFORCE Policy

The legacy preset classes (`REINFORCE_PRESET_v1` .. `v4`) have been removed. All variants are now expressed by explicitly setting constructor parameters (or YAML `params`). Key knobs:

| Parameter | Purpose |
|-----------|---------|
| `learning_rate` | SGD learning rate |
| `dist_type` | `gaussian` (default) or `lognormal` (triangular stub) |
| `actor_hidden_layers` / `actor_hidden_units` | Depth/width of actor MLP before param head |
| `critic_hidden_layers` / `critic_hidden_units` | Depth/width of value network |
| `actor_final_activation` | Activation applied to raw mean/sigma streams (`softplus`, `relu`, etc.) |
| `init_action_value` | Optional explicit initialization target for the primary action dimension |
| `sigma_scale` / `sigma_floor` | Unified sigma parameterization: `sigma = scale * act(raw+shift) + floor` |
| `shape_reward` | If true, applies `log(1+r)` shaping to positive rewards for variance reduction |
| `seed` | Optional deterministic seed (Python, NumPy, TF generator) |

Triangular distribution is a placeholder and raises `NotImplementedError` when sampled/log-prob requested.

To replicate an old preset, identify its architecture (depth/width), activation, and sigma parameters and specify them directly.

---

## Parameter Sweeps (modern two-step workflow)

The old `pipeline.parallel_sweep` script is deprecated. Use the new grid expansion + runner pipeline:

1. Author a sweep spec YAML with four top-level sections:
     - `base`: A single valid experiment config (same schema as `--config` for `one_camp_n_days_v2`).
     - `grid`: Mapping of dotted-key -> list of values (Cartesian product). Example dotted keys: `agents.0.params.learning_rate`.
     - `overrides` (optional): Dotted-key assignments applied to every expanded config after grid substitution.
     - `restrict` (optional): Integer indices (0-based) to keep from the full product (handy while iterating).

Example spec (`configs/reinforce_unified_sweepspec.yaml`):
```yaml
base:
    num_days: 1
    num_trajs: 200
    num_epochs: 50
    setup: setup_1
    seed: 123
    label: unify-reinforce-demo
    agents:
        - name: learner
            policy: REINFORCE
            params:
                learning_rate: 5e-7
                actor_hidden_layers: 1
                actor_hidden_units: 8
                critic_hidden_layers: 1
                critic_hidden_units: 8
                dist_type: gaussian
        - name: baseline
            policy: FixedBidPolicy
            params:
                bid_per_item: 5
                total_limit: 5
grid:
    agents.0.params.learning_rate: [5e-7, 1e-6]
    agents.0.params.dist_type: [gaussian, lognormal]
    agents.0.params.actor_hidden_units: [8, 16]
overrides: {}
restrict: null
```

2. Generate concrete configs:
```bash
make run ARGS="python -u -m pipeline.make_grid --spec configs/reinforce_unified_sweepspec.yaml --configs-dir sweeps/reinforce_unified/configs"
```
This produces `sweeps/reinforce_unified/configs/*.yaml` (one per combination) plus a summary JSON.

3. Execute configs sequentially or in parallel:
```bash
# Parallel (4 workers)
make run ARGS="python -u -m pipeline.run_configs --configs sweeps/reinforce_unified/configs --output results/reinforce_unified --parallel 4"

# Dry-run (print commands only)
make run ARGS="python -u -m pipeline.run_configs --configs sweeps/reinforce_unified/configs --output results/reinforce_unified --dry-run"
```

Behavior & features:
- Each generated config gets its own subdirectory under the chosen `--output` path; a stamp file `_RUN_COMPLETE.stamp` prevents accidental re-runs unless `--force` is set.
- Labels default to the base config's `label` plus an index; customize per-run by injecting `label` via grid if needed.
- Failures return a JSON summary (use `--json` for machine-readable output).

Migration note: existing workflows using `parallel_sweep.py` still work temporarily but will be removed; switch to the above pattern.

---

## Repository layout (updated)

- `derby/` — core library (environments, agents, auctions, markets, policies, utils)
- `pipeline/` — modern, process-based runners and tools
    - `make_grid.py` — expand a sweep spec YAML into concrete config files
    - `run_configs.py` — run a directory of generated configs (parallel or sequential)
    - `parallel_sweep.py` — (deprecated) legacy single-step Cartesian sweeper
- `utils/` — reusable helpers for analysis and plotting
    - `epoch_agg_loader.py` — list/load per-epoch Parquet files; basic policy summaries
    - `plot_utils.py` — load/filter/extract_fields/plot helpers for notebooks
- `legacy/` — original CSV-based helpers and plotting for older experiments
    - `plot_results.py`, `logs_to_csvs`, `csvs_to_plots`, `logs_to_plots`, `log_to_csv`
- `configs/` — experiment/sweep YAML configuration (e.g., `base_sweep.yaml`, `grid_sweep_1.yaml`)
- `notebooks/` — Jupyter notebooks for exploration/visualization
- `scripts/` — convenience scripts (e.g., running grid sweeps)
- `Dockerfile`, `Makefile`, `pyproject.toml`, `poetry.lock`

Notes:
- The repository does not track `results/` in git; it's reserved for run outputs (Parquet, JSONL, etc.).
- Update any local scripts to import from `utils.*` or execute from `pipeline/*` instead of `results.*`.

## Rebuilding the Poetry Lock File

If you change dependencies in `pyproject.toml`, you may want to regenerate the `poetry.lock` file. Use the following make command:
```bash
make lockfile
```
This will update `poetry.lock` to match the dependencies in `pyproject.toml` using Docker for a fully reproducible environment.

---

## Project Background

Derby was created by [Nishant Kumar](https://github.com/nish-ku-121) for use in his grad school research project (in collaboration with Prof. Amy Greenwald and fellow student [Enrique Areyan](https://github.com/eareyan)).

See [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for a brief summary.

The goal of the project was to apply (deep) reinforcement learning to the _AdX Game_. The AdX Game crudely models the digital advertising domain: advertisers buy _impression opportunities_ from websites, where the objective of each advertiser is to minimize spend and the objective of each website is to maximize revenue. This buying and selling is usually done through an _ad exchange_ (e.g. Google's AdX), which canonically holds digital auctions; the bidders are advertisers and the goods being sold are impression opportunities. In the AdX Game, each player plays the role of an _advertiser liaison_: advertisers procure _ad campaigns_ to liaisons, who are responsible for fulfilling the campaign within a certain time frame. The goal of each player is to learn what bids to place in order to maximize their profit by the end of the game.

(See pages 2 to 3 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for the game definition)


---

## RL Challenges

The AdX game is interesting to tackle from a reinforcement learning perspective because it poses several interesting properties and challenges:
- Stochasticity in the game can be a consequence of both the randomness of an impression opportunity's demographic(s) and the randomness of each player's strategy (i.e. players playing mixed strategies).
- Determining an optimal policy via _planning_ is difficult because determining a model _a priori_ is difficult or infeasible.
- The domain offers continuous control, as bids are real-valued. Furthermore, the domain can be highly dimensional, as there can be many types of demographics. Consequently exhaustive search of the space is often infeasible or intractable, thus smart exploration and/or generalization is required.
- The domain can be examined from both a single-agent perspective and a multi-agent perspective.

(See pages 8 to 11 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf) for an RL formulation)


---

## Algorithms

Algorithms derived, tested, and tuned include:
- Multi-Agent REINFORCE (with and without baseline)
- Multi-Agent Actor-Critic Q (with baseline)
- Multi-Agent Actor-Critic TD (has baseline)
- Multi-Agent Actor-Critic SARSA (with baseline)

(See pages 8 to 26 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf) for algorithms and their derivations)


For all algorithms, the policy network learns a Gaussian distribution using a neural net architecture.

For algorithms that learn Q or V, the value network has a standard setup: one or more dense layers taking state (and action for Q) as input and returning a Q or V value as output. Assume ReLU activation functions.

Challenges, tips, and tricks can be found on pages 28 to 34 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf).

---

## Results

Some results can be found:
- Pages 5 to 6 of [AdX RL Research Summary](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Summary.pdf)
- Page 35 of [AdX RL Research Full](https://github.com/nish-ku-121/derby/blob/9b693fe1aeebb2856b6408e202f7fafff28cd80f/AdX%20RL%20Research%20Full.pdf)
- Pages 4 to 11 of [AdX Research Select Results](https://github.com/nish-ku-121/derby/blob/master/AdX%20RL%20Research%20Select%20Results.pdf)
