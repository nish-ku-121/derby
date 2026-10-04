# Policies

The supported policy surface is intentionally small.

## Modern Learners

- `derby.policies.reinforce.REINFORCE`: canonical Monte Carlo policy-gradient learner.
- `derby.policies.actor_critic.ActorCritic`: supported one-step TD actor-critic learner.

Both learners share the continuous stochastic actor implementation in
`derby.policies.continuous_actor`. They support the modern action distributions,
explicit action-space initialization, fixed SGD updates, and optional adaptive
gradient-norm learning-rate control.

Use these learners through YAML configs and the package runner:

```bash
make run ARGS="python -u -m derby.scenarios.one_campaign_n_days --config configs/one_campaign_n_days_base.yaml --output-dir results/example"
```

## Deterministic Baselines

The modern runner still supports these simple policies from `derby.policies.baselines`:

- `FixedBidPolicy`
- `BudgetPerReachPolicy`
- `StepPolicy`

They operate in raw environment units and are useful as comparison baselines for
learning policies.

## Legacy Policy Zoo

The old `REINFORCE_*`, `AC_TD_*`, `AC_Q_*`, `AC_SARSA_*`, tabu, Fourier, and CSV-era
interfaces have been removed from the supported codebase. Their replacement is
explicit YAML configuration over `REINFORCE` or `ActorCritic`,
with Parquet epoch aggregates as the canonical analysis output.
