# Policies overview

This document summarizes the policy classes defined in `derby/core/policies.py`, groups them by core idea, and highlights the main knobs (distribution, critic type, budget normalization, reward shaping, etc.) so you can quickly choose the right family.

## Big picture

- Deterministic, rule-based policies for quick baselines: simple bidding rules.
- Policy-gradient (REINFORCE) families: pure policy gradient with various action distributions; some add tabu search; some add learned baselines (value functions) for variance reduction.
- Actor–Critic families:
  - TD-critic (value function V).
  - Q-critic (Q(s, a)), including a Fourier-basis variant.
  - SARSA-style updates (on-policy Q).
- Variations across families primarily differ by:
  - Action distribution: Gaussian, Uniform, Triangular, LogNormal.
  - Add-ons: learned baseline (V), Q critic, tabu search, Fourier features.
  - Training options: budget-per-reach normalization, reward shaping flags, epsilon schedules, partial-state support, and number of per-spec distributions.

All subclasses implement the `AbstractPolicy` interface with `call(...)`, `choose_actions(...)`, `policy_loss(...)`, and `update(...)`. Shapes follow the pattern: `states: [batch, T, ...]`; `actions/rewards: [batch, T-1, ...]`.

---

## Version suffixes (v2, v3, v3_1, v4): what changes?

Across policy families that share the same base name, the version suffixes indicate incremental refinements. From the constructors and naming found in `policies.py`, the following patterns hold consistently:

- Base (no suffix): original variant. Typically does not expose `budget_per_reach` or `shape_reward` knobs.
- v2: introduces `budget_per_reach` (default 1.0) and `shape_reward` (default False). These appear across most Gaussian-based families (REINFORCE, REINFORCE+Baseline, AC_TD, AC_Q, AC_SARSA) and Tabu v2. The rest of the signature retains `is_partial`, `discount_factor`, and `learning_rate`.
- v2_1: a minor iteration on v2 (present where explicitly named, e.g., `REINFORCE_Gaussian_v2_1`). Shares the v2 constructor knobs; internal tweaks are incremental.
- v3: keeps all v2 knobs and applies further internal refinements (network/training details). A clear exception with an additional public knob is the Tabu family: `REINFORCE_Tabu_Gaussian_v3` adds an `epsilon` parameter for exploration scheduling.
- v3_1: a small follow-up to v3 with the same public knobs; intended as a stability/behavior refinement. Choose v3_1 if available; otherwise v3.
- v4: latest refinement in families that define it; keeps v2 knobs and incorporates additional internal improvements.

Family-specific notes (based on the public API):
- REINFORCE (Gaussian): v2 adds `budget_per_reach`/`shape_reward`; v3 and v4 retain them. `REINFORCE_Gaussian_v2_1` is an intermediate step between v2 and v3. Tabu v3 uniquely adds `epsilon`.
- REINFORCE Baseline (Gaussian): v2/v3/v3_1/v4 all expose `budget_per_reach`/`shape_reward`.
- REINFORCE Baseline (LogNormal): only a `v3_1` variant is present; it includes `budget_per_reach`/`shape_reward`.
- Actor–Critic TD/Q/SARSA (Gaussian): v2 introduces `budget_per_reach`/`shape_reward`; v3, v3_1, v4 continue to include them. Some families also have Triangular counterparts without version suffixes.
- AC_Q Fourier: not versioned by suffix; still includes `budget_per_reach`/`shape_reward` as knobs.

If you need a rule of thumb: prefer the highest available version in a given family unless you have a reason to compare against earlier ones. Start with v2 if you specifically want to ablate the effect of budget normalization/reward shaping vs. the base model.

---

### Deeper differences by family

Below are concrete diffs observed in the code for commonly used families. These highlight activation choices, output parameterizations, and sampling tricks.

#### REINFORCE Gaussian family

- Base (no suffix):
  - Network: 1 Dense layer (size 1, leaky_relu), then separate Dense heads for mu and sigma.
  - Output parameterization: softplus(mu), softplus(sigma); additionally adds 1.0 to the last mu column and replaces it with bid_per_item*multiplier at the mu level to bias total_limit upward.
  - Sampling: clips to [0, inf); ensures bid_per_item <= total_limit by replacing if violated.
  - Loss: vanilla REINFORCE with baseline=0.

- v2:
  - Adds knobs: budget_per_reach, shape_reward.
  - Network: 1 Dense layer, explicit leaky_relu after it.
  - Output parameterization: offset = -log(exp(budget_per_reach) - 1); then mu = softplus(mu - offset), sigma = 0.5*softplus(sigma - offset). This shifts the softplus to scale with budget_per_reach.
  - Action encoding: treats the last subaction dim as a multiplicative “multiplier”. At choose time, sets total_limit = bid_per_item * (multiplier + 1). In loss, inverts this by total_limit/bid_per_item - 1 to recover the modeled variable.
  - Reward shaping: if enabled, applies log(1 + r) for positive discounted rewards before computing advantage.

- v2_1:
  - Same public knobs as v2; internal tweaks are minimal and preserve the v2 patterns (budget offset softplus, multiplier encoding, shaping).

- v3:
  - Network: same 1 Dense layer size but switches activation to ELU (applied after dense).
  - Output parameterization: same budget-per-reach softplus offset scheme as v2; sigma still scaled by 0.5.
  - Action encoding and shaping: same as v2.

- v3_1:
  - Network deepening: four Dense layers (sizes ~6 each) with ELU activations; SGD optimizer retained.
  - Output parameterization: same offset softplus for mu/sigma.
  - Action encoding and shaping: same as v2/v3.

- v4:
  - Network: 1 Dense layer, leaky_relu after dense.
  - Output parameterization change: uses ReLU with additive offset (mu = ReLU(mu + budget_per_reach), sigma = 0.5*ReLU(sigma + budget_per_reach) + 1e-5) instead of the softplus-with-offset used in v2/v3.
  - Action encoding and shaping: same multiplier trick; shaping optional.

Key takeaway: v2 introduces budget-aware parameterization and optional reward shaping; v3/v3_1 refine activations and depth; v4 switches to a simpler ReLU+offset mapping and adds a small sigma floor.

#### REINFORCE Tabu Gaussian family

- v1 (no suffix):
  - Two nets: policy Normal distribution and a tabu Triangular distribution (low/peak/high derived from low+offset(+offset2)).
  - Resampling: uses tabu_dist.prob(samples) to compute a per-dimension resample probability. With that probability, resamples from the policy distribution. Encourages avoiding regions tagged by the tabu net.
  - Loss: standard policy loss plus an additional tabu loss term for episodes with zero cumulative discounted reward (pushes the tabu net to assign higher probability mass to those action regions for future rejection).

- v2:
  - Same high-level design as v1 (policy Normal + tabu Triangular), with the budget-per-reach softplus offset parameterization in the policy head and ReLU/softplus constraints in the tabu head.

- v3:
  - Tabu network changes distribution: switches tabu head to Normal (matching the policy distribution’s type) and introduces an epsilon schedule.
  - Epsilon scheduling: choose from the tabu distribution with probability (1 - epsilon_t), where epsilon_t decays with update_count/temperature; otherwise use the policy sample. This is a greedy-with-decay mechanism vs. the v1/v2 resampling-by-probability.
  - Retains budget-per-reach offset parameterization in both policy and tabu heads.

Practical guidance: use v3 if you want an explicit exploration→exploitation schedule with tabu. Use v2 for triangular tabu shapes and probability-based resampling.

#### Cross-family (AC_TD, AC_Q, AC_SARSA)

- v2:
  - Adds budget_per_reach and shape_reward where applicable.
  - Applies the same offset = -log(exp(budget_per_reach) - 1) in policy/value/Q heads when mapping network outputs to positive parameters via softplus.

- v3 and v3_1:
  - Adopt ELU/nonlinear tweaks and, in some families, deeper stacks (mirroring the REINFORCE path). The multiplier encoding for the last action component is preserved in continuous actors.

- v4:
  - Tends to use ReLU+offset for parameterization and introduces small positive floors for scales.

Note on “multiplier” encoding (continuous actors):
- Base REINFORCE encodes total_limit at the mu level by biasing the last column upward and taking a product with the first column. v2+ variants uniformly move to a cleaner runtime encoding where the last column is a multiplier; at choose-time they compute total_limit = bid_per_item * (multiplier + 1) and invert this in loss. This pattern carries over to actor parts of the AC families as well.

---

## Deterministic baselines (rule-based)

| Class | Idea | Key params | Action type | Notes |
|---|---|---|---|---|
| `FixedBidPolicy` | Constant bid per item with total cap | `bid_per_item`, `total_limit`, `auction_item_spec` | Continuous | Simple hard baseline; respects a total spend limit. |
| `BudgetPerReachPolicy` | Spend under a target budget-per-reach heuristic | — | Continuous | Heuristic allocation guided by “budget per reach.” |
| `StepPolicy` | Linearly increasing bid each day | `start_bid`, `step_per_day` | Continuous | Useful for ablation/sanity checks. |
| `DummyREINFORCE` | Minimal REINFORCE example | `is_partial`, `discount_factor`, `learning_rate` | Continuous/logit-based | Educational/toy end-to-end REINFORCE wiring. |

---

## REINFORCE (policy-gradient) families

Baseline REINFORCE variants learn the policy only; some add Tabu search over actions; others change the action distribution.

| Class (variants) | Distribution | Add-ons | Key params | Notes |
|---|---|---|---|---|
| `REINFORCE_Gaussian_MarketEnv_Continuous` | Gaussian | — | `auction_item_spec_ids`, `num_dist_per_spec`, `is_partial`, `discount_factor`, `learning_rate` | Core Gaussian policy-gradient for continuous bids. |
| `REINFORCE_Gaussian_v2_MarketEnv_Continuous`, `v2_1`, `v3`, `v3_1`, `v4` | Gaussian | Iterative refinements | Adds `budget_per_reach`, `shape_reward` (v2+) | v2 introduces budget normalization + optional reward shaping; v3/v4 are further architecture/training tweaks. |
| `REINFORCE_Tabu_Gaussian_MarketEnv_Continuous`, `v2`, `v3` | Gaussian | Tabu search, `epsilon` (v3) | `budget_per_reach`, `shape_reward`, `epsilon` (v3) | Combines policy sampling with a tabu mechanism to avoid revisiting recent bad actions. |
| `REINFORCE_Uniform_MarketEnv_Continuous` | Uniform | — | `auction_item_spec_ids`, `num_dist_per_spec` | Uses a uniform distribution over actions. |
| `REINFORCE_Triangular_MarketEnv_Continuous` | Triangular | — | `auction_item_spec_ids`, `num_dist_per_spec` | Triangular action distribution. |

When to use:
- Start with Gaussian v2/v3 for most continuous control needs; enable `budget_per_reach` if you want scale-invariant behavior vs. budget magnitude; use `shape_reward` if rewards are sparse/noisy.
- Try Tabu variants if you’re seeing mode collapse or local traps.

---

## REINFORCE + learned baseline (value function)

These add a learned state-value function V(s) to reduce gradient variance while staying in the REINFORCE family.

| Class (variants) | Distribution | Baseline | Key params | Notes |
|---|---|---|---|---|
| `REINFORCE_Baseline_Gaussian_MarketEnv_Continuous` | Gaussian | V(s) | — | Basic actor with learned baseline head. |
| `REINFORCE_Baseline_Gaussian_v2/v3/v3_1/v4_MarketEnv_Continuous` | Gaussian | V(s) | `budget_per_reach`, `shape_reward` (v2+) | Successive improvements with budget scaling and shaping. |
| `REINFORCE_Baseline_Triangular_MarketEnv_Continuous` | Triangular | V(s) | — | Same baseline idea with Triangular policy. |
| `REINFORCE_Baseline_LogNormal_v3_1_MarketEnv_Continuous` | LogNormal | V(s) | `budget_per_reach`, `shape_reward` | LogNormal action head (strictly-positive actions). |

When to use:
- If pure REINFORCE is too noisy or unstable, switch to a Baseline version (Gaussian v2+ is a solid default).

---

## Actor–Critic (TD critic over V)

Actor learns the policy; critic learns V(s) with a TD-style target.

| Class (variants) | Distribution | Critic | Key params | Notes |
|---|---|---|---|---|
| `AC_TD_Gaussian_MarketEnv_Continuous` | Gaussian | V(s), TD | — | Basic AC with Gaussian policy. |
| `AC_TD_Gaussian_v2/v3/v3_1/v4_MarketEnv_Continuous` | Gaussian | V(s), TD | `budget_per_reach`, `shape_reward` (v2+) | Adds budget normalization and shaping; multiple refinements. |
| `AC_TD_Triangular_MarketEnv_Continuous` | Triangular | V(s), TD | — | Triangular policy distribution. |

When to use:
- Prefer AC_TD when you want bootstrapped value-learning for better sample-efficiency vs. REINFORCE.

---

## Actor–Critic (Q critic)

Actor learns the policy; critic learns Q(s, a). Includes a Fourier-basis variant.

| Class (variants) | Distribution | Critic | Key params | Notes |
|---|---|---|---|---|
| `AC_Q_Gaussian_MarketEnv_Continuous` | Gaussian | Q(s,a) | — | Basic AC-Q with Gaussian policy. |
| `AC_Q_Gaussian_v2/v3/v4_MarketEnv_Continuous` | Gaussian | Q(s,a) | `budget_per_reach`, `shape_reward` (v2+) | Iterative improvements with budget scaling/shaping. |
| `AC_Q_Baseline_V_Gaussian_v2/v3_MarketEnv_Continuous` | Gaussian | Q(s,a) + V(s) | `budget_per_reach`, `shape_reward` | Hybrid critic: both state- and action-value structure. |
| `AC_Q_Fourier_Gaussian_MarketEnv_Continuous` | Gaussian | Q(s,a) (Fourier features) | `budget_per_reach`, `shape_reward` | Uses Fourier basis features for Q approximation. |
| `AC_Q_Triangular_MarketEnv_Continuous` | Triangular | Q(s,a) | — | Triangular policy counterpart. |

When to use:
- Prefer AC-Q if you want action-conditional critics (can capture sharper action-value landscapes).
- Fourier variant is helpful when you want lightweight, structured function approximation.

---

## Actor–Critic (SARSA-style)

On-policy Q learning (SARSA target) within an AC framework; several versions add baselines and distribution variants.

| Class (variants) | Distribution | Critic | Key params | Notes |
|---|---|---|---|---|
| `AC_SARSA_Gaussian_MarketEnv_Continuous` | Gaussian | SARSA Q(s,a) | — | On-policy Q update (uses next action actually taken). |
| `AC_SARSA_Gaussian_v2_MarketEnv_Continuous` | Gaussian | SARSA Q(s,a) | `budget_per_reach`, `shape_reward` | Adds budget scaling/shaping. |
| `AC_SARSA_Triangular_MarketEnv_Continuous` | Triangular | SARSA Q(s,a) | — | Triangular policy counterpart. |
| `AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous` | Gaussian | SARSA Q(s,a) + V(s) | — | Adds V(s) as a baseline to reduce variance. |
| `AC_SARSA_Baseline_V_Gaussian_v2/v3/v3_1/v4` | Gaussian | SARSA Q(s,a) + V(s) | `budget_per_reach`, `shape_reward` (v2+) | Iterative refinements similar to AC_TD/AC_Q. |

When to use:
- Consider SARSA versions when strict on-policy learning is desirable or you want more stability tied to the behavior policy.

---

## Common knobs across learning policies

- `auction_item_spec_ids`: choose which item specs the policy outputs for.
- `num_dist_per_spec`: number of per-spec distributions/heads (e.g., multi-component outputs).
- `is_partial`: whether to use a reduced state representation.
- `discount_factor`: gamma for returns/TD targets.
- `learning_rate`: optimizer step size.
- `budget_per_reach` (v2+): normalizes or conditions the policy/critic wrt budget magnitude.
- `shape_reward`: toggles reward shaping.
- `epsilon` (Tabu v3): exploration probability for tabu search routing.

All learning policies implement:
- `call(states)`: forward pass producing parameters or proto-actions.
- `choose_actions(call_output)`: converts parameters to actual actions (sampled or deterministic).
- `policy_loss(...)`: computes the loss (REINFORCE with/without baseline, AC with TD/Q/SARSA).
- `update(...)`: applies gradients (Keras/TensorFlow).

---

## Quick chooser

- Fast baselines: `FixedBidPolicy`, `BudgetPerReachPolicy`, `StepPolicy`.
- Simple policy-gradient: `REINFORCE_Gaussian_v2` or `v3`.
- Need variance reduction without TD: `REINFORCE_Baseline_Gaussian_v2+`.
- Bootstrapped value learning: `AC_TD_Gaussian_v2+`.
- Action-conditional critic: `AC_Q_Gaussian_v2+` (try Fourier variant if you want compact approximators).
- On-policy Q updates: `AC_SARSA_Baseline_V_Gaussian_v2+`.
- Exploration stuck: `REINFORCE_Tabu_Gaussian` (v3 if you want epsilon schedule).
- Non-Gaussian behaviors: Triangular or LogNormal Baseline variants.

---

## Appendix: Class index

A flat list of the policy classes in `policies.py` for quick search/reference.

- `AbstractPolicy`
- `FixedBidPolicy`
- `BudgetPerReachPolicy`
- `StepPolicy`
- `DummyREINFORCE`
- `REINFORCE_Gaussian_MarketEnv_Continuous`
- `REINFORCE_Gaussian_v2_MarketEnv_Continuous`
- `REINFORCE_Gaussian_v2_1_MarketEnv_Continuous`
- `REINFORCE_Gaussian_v3_MarketEnv_Continuous`
- `REINFORCE_Gaussian_v3_1_MarketEnv_Continuous`
- `REINFORCE_Gaussian_v4_MarketEnv_Continuous`
- `REINFORCE_Tabu_Gaussian_MarketEnv_Continuous`
- `REINFORCE_Tabu_Gaussian_v2_MarketEnv_Continuous`
- `REINFORCE_Tabu_Gaussian_v3_MarketEnv_Continuous`
- `REINFORCE_Uniform_MarketEnv_Continuous`
- `REINFORCE_Triangular_MarketEnv_Continuous`
- `REINFORCE_Baseline_Gaussian_MarketEnv_Continuous`
- `REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous`
- `REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous`
- `REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous`
- `REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous`
- `REINFORCE_Baseline_Triangular_MarketEnv_Continuous`
- `REINFORCE_Baseline_LogNormal_v3_1_MarketEnv_Continuous`
- `AC_TD_Gaussian_MarketEnv_Continuous`
- `AC_TD_Gaussian_v2_MarketEnv_Continuous`
- `AC_TD_Gaussian_v3_MarketEnv_Continuous`
- `AC_TD_Gaussian_v3_1_MarketEnv_Continuous`
- `AC_TD_Gaussian_v4_MarketEnv_Continuous`
- `AC_TD_Triangular_MarketEnv_Continuous`
- `AC_Q_Gaussian_MarketEnv_Continuous`
- `AC_Q_Gaussian_v2_MarketEnv_Continuous`
- `AC_Q_Gaussian_v3_MarketEnv_Continuous`
- `AC_Q_Gaussian_v4_MarketEnv_Continuous`
- `AC_Q_Baseline_V_Gaussian_v2_MarketEnv_Continuous`
- `AC_Q_Baseline_V_Gaussian_v3_MarketEnv_Continuous`
- `AC_Q_Fourier_Gaussian_MarketEnv_Continuous`
- `AC_Q_Triangular_MarketEnv_Continuous`
- `AC_SARSA_Gaussian_MarketEnv_Continuous`
- `AC_SARSA_Gaussian_v2_MarketEnv_Continuous`
- `AC_SARSA_Triangular_MarketEnv_Continuous`
- `AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous`
- `AC_SARSA_Baseline_V_Gaussian_v2_MarketEnv_Continuous`
- `AC_SARSA_Baseline_V_Gaussian_v3_MarketEnv_Continuous`
- `AC_SARSA_Baseline_V_Gaussian_v3_1_MarketEnv_Continuous`
- `AC_SARSA_Baseline_V_Gaussian_v4_MarketEnv_Continuous`

---

If you’d like, we can add minimal instantiation snippets and link this page from the root `README.md`.
