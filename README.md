# Derby

Derby is a research framework for reinforcement learning in repeated auction
markets. It provides auction and market primitives, multi-agent environments,
continuous-action policies, reproducible YAML run configurations, and
tools for running and analyzing parameter sweeps.

This repository also contains the reproducibility package for
[*Zero Collapse: A Failure Mode of Policy Gradient Methods in Discontinuous
Reward Environments*](https://arxiv.org/abs/2605.30896). The paper studies how
policy-gradient agents can collapse to near-zero actions in thresholded reward
landscapes and evaluates interventions using Derby's auction environment.

## How the environment works

In Derby, a market is modeled as a stateful sequence of auctions:

1. One or more bidders enter the market with a campaign, budget, and policy.
2. Each simulated day presents auction items associated with audience segments.
3. Agents submit continuous-valued bids, the auction allocates items, and the
   market computes rewards.
4. Campaign state carries across days, allowing policies to learn from repeated
   interaction rather than isolated auctions.

The `one_campaign_n_days` scenario family supports one- and two-segment
variants. Learning agents can use REINFORCE or a one-step TD actor-critic; fixed
bid, budget-per-reach, and step policies are available as deterministic
baselines.

Here, a **scenario family** defines the overall task, while a
`scenario_variant` selects a concrete environment specification within that
family, such as `one_segment` or `two_segment`. A run instantiates that variant
with the configured agents, seed, horizon, and training parameters.

## Quick start

The supported development and execution workflow uses Docker to provide Python
3.10, TensorFlow, Poetry, and the remaining dependencies. Install
[Docker](https://www.docker.com/products/docker-desktop/) and GNU Make, then run
from the repository root:

```bash
make build
make run ARGS="python -u -m derby.scenarios.one_campaign_n_days \
  --config configs/one_campaign_n_days_base.yaml \
  --output-dir results/quickstart"
```

The first command builds the `derby-app` image. Rebuild it after changing
`Dockerfile`, `pyproject.toml`, or `poetry.lock`; source and configuration files
are mounted into the container for subsequent commands.

The command above invokes the scenario package's command-line entry point. It
delegates to `derby.scenarios.one_campaign_n_days.runner`, which loads the YAML
configuration, constructs the environment and agents, performs training, and
records aggregate metrics. A successful run writes one row per epoch and agent
to `results/quickstart/epoch_agg__<run-id>.parquet`. Inspect the output with:

```bash
make run ARGS="python -m utils.analysis results/quickstart"
```

`results/` and `sweeps/` are intentionally ignored by Git.

## Configure a run

Start with one of the checked-in configs:

- [`configs/one_campaign_n_days_base.yaml`](configs/one_campaign_n_days_base.yaml) —
  REINFORCE against a fixed-bid baseline
- [`configs/actor_critic_td_base.yaml`](configs/actor_critic_td_base.yaml) —
  one-step TD actor-critic against a fixed-bid baseline

A run configuration defines:

```yaml
num_days: 1       # episode horizon
num_trajs: 100    # trajectories sampled per epoch
num_epochs: 10
scenario_variant: one_segment  # one_segment or two_segment
seed: 123           # omit for a stochastic run
agents:
  - name: learner
    label: REINFORCE
    policy: REINFORCE
    params:
      learning_rate: 1e-6
      dist_type: gaussian
      use_baseline: false
  - name: baseline
    label: FixedBid
    policy: FixedBidPolicy
    params:
      bid_per_item: 5
      total_limit: 5
```

Supported distributions for learning policies are `gaussian`, `lognormal`, and
`triangular`. Architecture, optimizer, action initialization, reward shaping,
and adaptive step-size settings are explicit policy parameters; the checked-in
configs provide complete examples. A config-level `seed` seeds Python, NumPy,
TensorFlow, the environment, and policies that expose a seed parameter. There
is no command-line seed override.

The command-line runner also accepts these options:

```text
-o, --output-dir PATH   write epoch aggregates to PATH
--log-level LEVEL       DEBUG, INFO, WARNING, ERROR, CRITICAL, or NONE
--flush-every N         flush Parquet output every N epochs (default: 1)
```

For notebooks or other Python workflows, invoke the same execution path through
its Python API:

```python
import yaml
from derby.scenarios.one_campaign_n_days.runner import run_from_config

with open("configs/one_campaign_n_days_base.yaml", encoding="utf-8") as file:
    config = yaml.safe_load(file)

run_id = run_from_config(
    config,
    output_dir_override="results/notebook_run",
)
```

## Run a parameter sweep

Sweep specifications combine a base run configuration with fixed overrides and
a grid of dotted config keys. The example
[`configs/reinforce_unified_sweep.yaml`](configs/reinforce_unified_sweep.yaml)
shows the complete schema.

Generate the concrete configs:

```bash
make run ARGS="python -u -m pipeline.make_config_grid \
  --spec configs/reinforce_unified_sweep.yaml \
  --output-dir sweeps/reinforce_demo/configs"
```

Preview or execute them:

```bash
# Print the commands without running them.
make run ARGS="python -u -m pipeline.run_sweep \
  --configs-dir sweeps/reinforce_demo/configs \
  --run-module derby.scenarios.one_campaign_n_days \
  --output-dir results/reinforce_demo \
  --dry-run"

# Run up to four configurations concurrently.
make run ARGS="python -u -m pipeline.run_sweep \
  --configs-dir sweeps/reinforce_demo/configs \
  --run-module derby.scenarios.one_campaign_n_days \
  --output-dir results/reinforce_demo \
  --parallel 4"
```

Each config receives its own result directory. Successful runs contain
`_RUN_COMPLETE.json`; rerunning the sweep skips those directories. The sweep
command also writes `run_summary.json` at the output root and `failure.json`
inside a failed run directory. It stops before execution if it finds a
non-empty run directory without a completion record, protecting partial output
from being silently overwritten.

## Reproduce the paper artifacts

The versioned publication bundle lives in
[`paper/zero_collapse/`](paper/zero_collapse/). It contains normalized data,
paper-numbered figures, the manuscript table, and the scripts used to render
them. To rebuild the empirical figures and table from the committed data:

```bash
make run ARGS="python paper/zero_collapse/scripts/build_outputs.py"
```

[`paper/zero_collapse/SOURCES.md`](paper/zero_collapse/SOURCES.md) maps every
artifact to its run configuration and documents the reproduction
boundary. In particular, the ordinary build above does not require the raw
training output. Data extraction and the TD-value critic rerun are maintainer
workflows that depend on the untracked `results/` tree.

## Repository guide

| Path | Purpose |
| --- | --- |
| `derby/core/` | Auctions, markets, environments, agents, states, and probability utilities |
| `derby/policies/` | Continuous actors, REINFORCE, TD actor-critic, and deterministic baselines |
| `derby/scenarios/` | Scenario definitions and scenario-specific training entry points |
| `configs/` | Single-run configurations and sweep specifications |
| `pipeline/` | Config-grid generation and resumable sweep execution |
| `utils/` | Parquet loading, filtering, aggregation, and plotting helpers |
| `paper/zero_collapse/` | Tracked data and generated artifacts for the paper |
| `derby/tests/` | Unit and integration tests |

## Development

Run the complete test suite in the container:

```bash
make test
```

Run a single file or test by overriding `TEST`:

```bash
make test TEST=derby/tests/test_actor_critic.py
```

Start an interactive shell or JupyterLab session with `make shell` or
`make jupyter`. JupyterLab is served at `http://localhost:8888` by default; set
`JUPYTER_PORT` to change the host port.

After changing dependencies in `pyproject.toml`, regenerate the lock file and
rebuild the image:

```bash
make lockfile
make build
```
