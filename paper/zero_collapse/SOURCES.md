# Zero-collapse paper artifact sources

This directory is the tracked publication bundle for *Zero Collapse: A Failure
Mode of Policy Gradient Methods in Discontinuous Reward Environments*.  Its
figures and tables are rebuilt from the normalized files in `data/`; raw
training results remain intentionally excluded from Git.

| Paper item | Artifact | Experiment specifications |
| --- | --- | --- |
| Figure 2: fixed-rate controls | `figures/fig2_fixed_rate_controls.pdf` | `configs/reinforce_sweep_control_2.yaml`; `configs/actor_critic_sweep_control_2.yaml` |
| Figure 3: isolated adaptive step size | `figures/fig3_adaptive_step_size.pdf` | `configs/reinforce_adaptive_step_rate_1000ep.yaml`; `configs/actor_critic_adaptive_step_rate_1000ep.yaml` |
| Figure 4: representative smooth parameterization | `figures/fig4_smooth_parameterization.pdf` | `configs/reinforce_smooth_parameterization.yaml`; `configs/actor_critic_smooth_parameterization.yaml` |
| Figure 5: combined treatment | `figures/fig5_combined_treatment.pdf` | `configs/reinforce_combined_sgd_high_epsilon.yaml`; `configs/actor_critic_combined_sgd_high_epsilon.yaml` |
| Figure 6: TD-$V$ diagnostic | `figures/fig6_td_value_critic_diagnostic.pdf` | `configs/actor_critic_sweep_control_2.yaml`, seed 456, learning rate `1e-8`, zero-based epoch 224 |
| Figure 7: full smooth-parameterization factorial | `figures/fig7_smooth_parameterization_full.pdf` | Figure 2 and Figure 4 specifications |
| Figure 8: matched fixed-optimizer comparison | `figures/fig8_matched_fixed_optimizer.pdf` | the fourteen `*_matched_*.yaml` specifications under `configs/` |
| Table 4: qualitative outcomes | `tables/experiment_outcomes.{csv,tex}` | the configurations above plus the untreated, bias-centered, and fixed-rate supporting controls listed in `scripts/export_data.py` |

The paper's Figure 1 is a conceptual threshold-payoff schematic authored in the
manuscript; it is not an empirical Derby output.

## Rebuilding

`scripts/export_data.py` is the maintainer-only extraction step.  It reads the
ignored raw `results/` tree and writes the normalized trace bundle.  The
ordinary rendering path reads only the tracked files in this directory.

`scripts/build_outputs.py` rebuilds the empirical figures and tables.  The
critic rerun is deliberately separate: `scripts/run_td_value_critic_diagnostic.py`
regenerates its data from the specified Control 2 cell and is not required for
ordinary plot rendering.
