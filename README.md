
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
    make docker-build
    ```

## Running Experiments

The Makefile's `docker-run` target automatically executes commands via Poetry inside the container, so you can pass plain `python` commands in `ARGS`.

You can invoke an experiment script in several equivalent ways (pick whichever you find clearest):

1. Module form (preferred; resolves relative imports robustly)
    ```bash
    make docker-run ARGS="python -u -m derby.experiments.one_camp_n_days <experiment_name> <num_days> <num_trajs> <num_epochs> <learning_rate>"
    ```
2. File path form
    ```bash
    make docker-run ARGS="python -u derby/experiments/one_camp_n_days.py <experiment_name> <num_days> <num_trajs> <num_epochs> <learning_rate>"
    ```
3. Open an interactive shell then run repeatedly (good for quick loops)
    ```bash
    make docker-shell
    # Now inside container
    poetry run python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7
    ```
4. With explicit environment variables (example: restrict TensorFlow threading)
    ```bash
    make docker-run ARGS="TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7"
    ```
5. Using a short alias variable for readability (shell feature)
    ```bash
    EXP_ARGS="exp_1000 1 200 100 5e-7" ; make docker-run ARGS="python -u -m derby.experiments.one_camp_n_days $EXP_ARGS"
    ```

Example (module form):
```bash
make docker-run ARGS="python -u -m derby.experiments.one_camp_n_days exp_1000 1 200 100 5e-7"
```

### Modern YAML-driven runner (v2)

For new work, prefer the YAML-based runner `derby.experiments.one_camp_n_days_v2`. It consumes a single config dict from a YAML file and logs per-epoch metrics to Parquet.

CLI usage:
```bash
make docker-run ARGS="python -u -m derby.experiments.one_camp_n_days_v2 \
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

Minimal YAML example (`configs/one_camp_n_days_v2_config.yaml`):
```yaml
num_days: 1            # days per trajectory
num_trajs: 200         # trajectories per epoch
num_epochs: 100        # training epochs
setup: setup_1         # maps to Experiment.setup_1()
seed: 123              # optional; remove for stochastic run
label: tiny-demo       # optional, for easier filtering later
agents:
    - name: learner
        policy: REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous
        params:
            learning_rate: 5e-7
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

## Parallel sweeps (recommended)

Use the process-based sweeper to run many configurations in parallel and collect results:

```bash
make docker-run ARGS='python -u -m pipeline.parallel_sweep \
    --base-yaml configs/base_sweep.yaml \
    --grid-yaml configs/grid_sweep_1.yaml \
    --output-dir results/sweep \
    --log-level INFO \
    --tf-intra 1 --tf-inter 1'
```

Key ideas:
- Base config (`configs/base_sweep.yaml`): a single config dict accepted by the simplified runner.
- Grid config (`configs/grid_sweep.yaml`): a mapping of dotted-keys to lists, expanded into the Cartesian product. Example keys: `num_epochs`, `agents.0.params.learning_rate`.
- One process per config variant (default workers = all CPU cores). TensorFlow and BLAS threads per process can be controlled with `--tf-intra` and `--tf-inter`.

Useful flags:
- `--base-yaml`, `--grid-yaml`: YAML files for the base config and parameter grid.
- `--output-dir`: base directory for outputs. A timestamped directory `sweep_<UTC_TIMESTAMP>` is created per run. Parquet files are written under `parquet/`. A stable `parallel_results.jsonl` and a timestamped copy are both produced.
- `--max-workers`: cap number of worker processes (default = all CPUs).
- `--tf-intra`, `--tf-inter`: TensorFlow intra/inter-op thread counts per process (defaults 1/1 to avoid oversubscription).
- `--start-method`: multiprocessing start method (default `spawn`, safest for TensorFlow).
- `--log-level`: controls ONLY the sweep/orchestrator logs (DEBUG/INFO/WARNING/ERROR or suppression aliases like NONE/OFF/QUIET). Inner experiments use their own defaults.

Memory advisory: After each sweep a `memory_advisory.txt` is written showing observed mean/median/p95/max end RSS and peak RSS plus recommended aggressive & conservative worker counts. Tuning constants:
- `DEFAULT_MEMORY_SAFETY_FACTOR = 0.85`
- `DEFAULT_MEMORY_RESERVE_MB = 1024`

See `docs/MemoryAdvisory.md` for full explanation of these fields and how to use them when scaling concurrency.

Behavior and logs:
- Per-run labels are formed automatically as `<base>-i<index>` where `<base>` is the first non-empty of: (variant `label` after overrides) → (base YAML `label`) → (grid filename stem). This preserves user intent while ensuring uniqueness per variant.
- The sweeper prints a concise line when each run starts, including label, policy, learning rate, epochs, and trajs.
- Inner experiments are run with their own default logging (no per-run override flag). To reduce inner noise, configure logging within the experiment config itself (or post-filter logs). The sweep `--log-level` only affects the orchestration layer.
- At the end, the sweeper reports wall time, aggregate CPU time, speedup, and parallel efficiency.

Outputs:
- Parquet files per run: `<output-dir>/parquet/epoch_agg__<uuid>.parquet` (ignored by git).
- JSONL summary: `<output-dir>/parallel_results.jsonl` plus a timestamped copy `parallel_results_YYYYMMDD-HHMMSS.jsonl`.

---

## Repository layout (updated)

- `derby/` — core library (environments, agents, auctions, markets, policies, utils)
- `pipeline/` — modern, process-based runners and tools
    - `parallel_sweep.py` — main entrypoint for running config sweeps in parallel (writes Parquet + JSONL outputs under a single output directory)
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
make docker-lockfile
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
