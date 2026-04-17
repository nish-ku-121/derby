"""Unified REINFORCE policy implementations.

Configurable policy supporting Gaussian, LogNormal, and Triangular distributions.
Historical preset names (v1..v4) have been removed—reproduce them explicitly by
setting activation and actor/critic depth/width.

Public class exported: REINFORCE
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import tensorflow as tf

from derby.core.policies import AbstractPolicy  # reuse base class
from derby.core.environments import AbstractEnvironment


class REINFORCE(AbstractPolicy, tf.keras.Model):
    """Unified policy (float32, manual distribution math) with configurable activation.

    Args:
        actor_final_activation: str | callable ('softplus' | 'relu' supported for explicit init centering).
        init_action_value: optional target value for the primary action dimension at initialization.
            This is interpreted in the policy's working action space.
        sigma_scale: multiplicative factor for activated sigma pre-values.
        sigma_floor: added after scaling (unified default 1e-5).
        actor_hidden_layers / units / activation: MLP depth/width for policy feature extractor (0 => linear head).
        critic_hidden_layers / units / activation: same for value network.
    Notes:
        Previous preset system removed; configure via explicit kwargs or external sweep grids.
    """

    def __init__(self, auction_item_spec_ids, num_dist_per_spec: int = 2,
                 is_partial: bool = False, discount_factor: float = 1, learning_rate: float = 0.0001,
                 shape_reward: bool = False, seed: int | None = None,
                 actor_final_activation='softplus', init_action_value: float | None = None,
                 sigma_scale: float = 0.5, sigma_floor: float = 1e-5,
                 dist_type: str = 'gaussian',
                 # Network architecture knobs (actor/value)
                 actor_hidden_layers: int = 1, actor_hidden_units: int = 1,
                 actor_hidden_activation='leaky_relu',
                 critic_hidden_layers: int = 1, critic_hidden_units: int = 6,
                 critic_hidden_activation='relu',
                 use_baseline: bool = True):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.shape_reward = shape_reward
        self.seed = seed
        self.dist_type = dist_type.lower()
        self._actor_final_activation_fn, self._actor_final_activation_name = self._resolve_activation(actor_final_activation)
        self._activation_inverse_registry = {
            'relu': lambda B: float(B),
            'softplus': self._inverse_softplus,
        }
        self.init_action_value = None if init_action_value is None else float(init_action_value)
        # Unified sigma semantics: always sigma = sigma_scale * activation(raw+shift) + sigma_floor.
        self.sigma_scale = float(sigma_scale)
        self.sigma_floor = float(sigma_floor)
        # Store architecture parameters
        self.actor_hidden_layers = int(actor_hidden_layers)
        self.actor_hidden_units = int(actor_hidden_units)
        self.critic_hidden_layers = int(critic_hidden_layers)
        self.critic_hidden_units = int(critic_hidden_units)
        self._actor_hidden_activation_fn, self._actor_hidden_activation_name = self._resolve_activation(actor_hidden_activation)
        self._critic_hidden_activation_fn, self._critic_hidden_activation_name = self._resolve_activation(critic_hidden_activation)
        self.use_baseline = bool(use_baseline)


        # Subaction metadata must exist before validation that references num_dist_per_subaction
        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0.0
        self.subactions_max = 1e15
        self.num_subactions = len(self.auction_item_spec_ids)
        self.num_dist_per_subaction = num_dist_per_spec

        self._validate_config()
        if seed is not None:
            try:
                random.seed(seed)
            except Exception:
                pass
            try:
                np.random.seed(seed)
            except Exception:
                pass
            self._rng = tf.random.Generator.from_seed(seed)
        else:
            self._rng = None

        # (moved earlier above for validation ordering)

        # Distribution specification (raw parameter count per distribution value)
        if self.dist_type == 'gaussian':
            self._per_dist_param_count = 2  # (mu, sigma)
        elif self.dist_type == 'lognormal':
            self._per_dist_param_count = 2  # underlying normal (mu, sigma)
        elif self.dist_type == 'triangular':
            self._per_dist_param_count = 3  # (low, mode, high) via transformed raw params
        else:
            raise ValueError(f"Unsupported dist_type '{self.dist_type}'. Choose gaussian | lognormal | triangular")

        # Initializer: small uniform range for stable early exploration.
        self.param_ker_init = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Actor hidden stack (can be empty if actor_hidden_layers == 0)
        self.actor_hidden = []
        for i in range(self.actor_hidden_layers):
            self.actor_hidden.append(
                tf.keras.layers.Dense(
                    self.actor_hidden_units,
                    activation=self._actor_hidden_activation_fn,
                    dtype='float32',
                    name=f'actor_hidden_{i+1}'
                )
            )
        # Parameter head (fused raw params) always present
        # Bias initialization optionally centers the primary action dimension.
        param_dim = self._per_dist_param_count * self.num_subactions * self.num_dist_per_subaction
        bias_init = np.zeros(param_dim, dtype=np.float32)
        if self.init_action_value is not None:
            if self._actor_final_activation_name not in self._activation_inverse_registry:
                raise ValueError(
                    f"init_action_value requires an activation with a registered inverse; "
                    f"got '{self._actor_final_activation_name}'."
                )
            if self.dist_type == 'gaussian':
                primary_target = self.init_action_value
            elif self.dist_type == 'lognormal':
                primary_target = math.log(self.init_action_value)
            elif self.dist_type == 'triangular':
                primary_target = self.init_action_value
            else:  # pragma: no cover
                raise ValueError(f"Unknown dist_type '{self.dist_type}' for bias initialization.")
            primary_bias = self._activation_inverse_registry[self._actor_final_activation_name](primary_target)
            for i in range(self.num_subactions):
                idx = i * self.num_dist_per_subaction * self._per_dist_param_count
                bias_init[idx] = primary_bias
        self.param_layer = tf.keras.layers.Dense(
            param_dim,
            kernel_initializer=self.param_ker_init,
            bias_initializer=tf.constant_initializer(bias_init),
            activation=None,
            dtype='float32',
            name='param_head'
        )

        # Critic hidden stack
        self.critic_hidden = []
        if self.use_baseline:
            for i in range(self.critic_hidden_layers):
                self.critic_hidden.append(
                    tf.keras.layers.Dense(
                        self.critic_hidden_units,
                        activation=self._critic_hidden_activation_fn,
                        dtype='float32',
                        name=f'critic_hidden_{i+1}'
                    )
                )
            self.critic_out = tf.keras.layers.Dense(1, dtype='float32', name='critic_out')
        else:
            # placeholder so attribute exists; not used
            self.critic_out = None

        self._ais_tensor = None
        self._multiplier_add = tf.constant([0.0] * (self.num_dist_per_subaction - 1) + [1.0], dtype=tf.float32)
        self._replace_last_mask = tf.constant([True] * (self.num_dist_per_subaction - 1) + [False])
        self._last_col_minus1 = tf.constant([0.0] * (self.num_dist_per_subaction - 1) + [-1.0], dtype=tf.float32)
        self._log_2pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)
        self._neg_half_log_2pi = tf.constant(-0.5 * math.log(2.0 * math.pi), dtype=tf.float32)
        self._log_2 = tf.constant(math.log(2.0), dtype=tf.float32)

    # Activation helpers
    @staticmethod
    def _inverse_softplus(B: float) -> float:
        B = float(B)
        if B <= 0.0:
            return 0.0
        if B < 20.0:
            return math.log(math.expm1(B))
        return B + math.log1p(-math.exp(-B))

    @staticmethod
    def _resolve_activation(act):
        if isinstance(act, str):
            name = act.lower()
            if name == 'relu':
                return tf.nn.relu, 'relu'
            if name == 'softplus':
                return tf.nn.softplus, 'softplus'
            fn = tf.keras.activations.get(act)
            try:
                canon = fn.__name__.lower()
            except Exception:
                canon = name
            return fn, canon
        try:
            name = act.__name__.lower()
        except Exception:
            name = repr(act)
        return act, name

    def __repr__(self):
        return (
            "REINFORCE("  # overview
            f"is_partial={self.is_partial}, discount={self.discount_factor}, "
            f"lr={self.learning_rate}, num_actions={self.num_subactions}, optimizer={type(self.optimizer).__name__}, "
            f"shape_reward={self.shape_reward}, seed={self.seed}, dist_type={self.dist_type}, actor_final_activation={self._actor_final_activation_name}, "
            f"init_action_value={self.init_action_value}, sigma_scale={self.sigma_scale}, sigma_floor={self.sigma_floor}, "
            f"actor_depth={self.actor_hidden_layers}, actor_width={self.actor_hidden_units}, actor_act={self._actor_hidden_activation_name}, "
            f"use_baseline={self.use_baseline}, critic_depth={self.critic_hidden_layers}, critic_width={self.critic_hidden_units}, critic_act={self._critic_hidden_activation_name})"
        )

    # Provide a human-readable string; identical to __repr__ for now.
    def __str__(self):  # pragma: no cover - trivial delegation
        return self.__repr__()

    # Implement build to satisfy Keras expectations and silence warnings.
    # We rely on sublayers (Dense) to build lazily on first call; marking as built is sufficient.
    def build(self, input_shape):  # pragma: no cover - simple wiring
        # Optionally, advanced: manually build sublayers based on input_shape[-1].
        # For now, a minimal build is enough as layers carry their own build logic.
        self.built = True

    # ---- Validation ----
    def _validate_config(self):
        if self.actor_hidden_layers < 0:
            raise ValueError("actor_hidden_layers must be >= 0")
        if self.critic_hidden_layers < 0:
            raise ValueError("critic_hidden_layers must be >= 0")
        if self.actor_hidden_units < 1:
            raise ValueError("actor_hidden_units must be >= 1")
        if self.critic_hidden_units < 1:
            raise ValueError("critic_hidden_units must be >= 1")
        if self.sigma_scale <= 0:
            raise ValueError("sigma_scale must be > 0")
        if self.sigma_floor < 0:
            raise ValueError("sigma_floor must be >= 0")
        if self.num_dist_per_subaction < 1:
            raise ValueError("num_dist_per_spec must be >= 1")
        if self.init_action_value is not None and self.init_action_value < 0:
            raise ValueError("init_action_value must be >= 0")
        if self.dist_type == 'lognormal' and self.init_action_value is not None and self.init_action_value <= 1.0:
            raise ValueError(
                "lognormal init_action_value must be > 1.0 because the current positive final-activation "
                "parameterization initializes the underlying normal mean in nonnegative space."
            )

    # Fold types
    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE if self.is_partial else AbstractEnvironment.FOLD_TYPE_ALL

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    # Internal helpers
    def _ensure_ais_tensor(self, batch_time_shape):
        if self._ais_tensor is not None:
            return self._ais_tensor
        ais = tf.convert_to_tensor(self.auction_item_spec_ids, dtype=tf.float32)[None, None, :]
        self._ais_tensor = ais
        return self._ais_tensor

    @tf.function(reduce_retracing=True)
    def call(self, states) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward pass: states -> distribution params (bias already handles centering)."""
        states = tf.cast(states, tf.float32)
        x = states
        for layer in self.actor_hidden:
            x = layer(x)
        params = self.param_layer(x)
        # Reshape raw params to [..., num_subactions, num_dist_per_subaction, per_param]
        dyn_shape = tf.shape(params)
        params = tf.reshape(
            params,
            tf.concat([
                dyn_shape[:2],
                [self.num_subactions, self.num_dist_per_subaction, self._per_dist_param_count]
            ], axis=0)
        )

        # Inline logic for each distribution type
        if self.dist_type in ('gaussian', 'lognormal'):
            raw_mu = params[..., 0]
            raw_sigma = params[..., 1]
            mu = self._actor_final_activation_fn(raw_mu)
            sigma_pre = self._actor_final_activation_fn(raw_sigma)
            sigma = (self.sigma_scale * sigma_pre) + self.sigma_floor
            return (mu, sigma)
        if self.dist_type == 'triangular':
            a = params[..., 0]
            b = params[..., 1]
            c = params[..., 2]
            center = self._actor_final_activation_fn(a)
            left_span = self._actor_final_activation_fn(b)
            right_span = self._actor_final_activation_fn(c)
            low = center - left_span
            mode = center
            high = center + right_span
            # Safety rails: enforce non-negativity and ordering
            zmin = tf.cast(self.subactions_min, center.dtype)
            low = tf.maximum(low, zmin)
            mode = tf.maximum(mode, low)
            high = tf.maximum(high, mode)
            return (low, mode, high)
        raise RuntimeError("Unhandled dist_type in call()")

    def value_function(self, states):
        states = tf.cast(states, tf.float32)
        x = states
        for layer in self.critic_hidden:
            x = layer(x)
        v = self.critic_out(x)
        return v

    def choose_actions(self, call_output):
        if self.dist_type in ('gaussian', 'lognormal'):
            mus, sigmas = call_output
            if self._rng is not None:
                eps = self._rng.normal(tf.shape(mus), dtype=mus.dtype)
            else:
                eps = tf.random.normal(tf.shape(mus), dtype=mus.dtype)
            if self.dist_type == 'gaussian':
                samples = mus + sigmas * eps
            else:  # lognormal
                base = mus + sigmas * eps
                samples = tf.exp(base)
        elif self.dist_type == 'triangular':
            low, mode, high = call_output
            # Inverse-CDF sampling for triangular distribution
            shape = tf.shape(low)
            if self._rng is not None:
                u = self._rng.uniform(shape, dtype=low.dtype)
            else:
                u = tf.random.uniform(shape, dtype=low.dtype)
            eps = tf.constant(1e-12, dtype=low.dtype)
            denom = tf.maximum(high - low, eps)
            fb = tf.clip_by_value((mode - low) / denom, 0.0, 1.0)
            left = u <= fb
            # Left branch: a + sqrt(u*(b-a)*(c-a))
            left_val = low + tf.sqrt(tf.maximum(u, 0.0) * tf.maximum(mode - low, eps) * denom)
            # Right branch: c - sqrt((1-u)*(c-a)*(c-b))
            right_val = high - tf.sqrt(tf.maximum(1.0 - u, 0.0) * denom * tf.maximum(high - mode, eps))
            samples = tf.where(left, left_val, right_val)
        else:
            raise NotImplementedError("Unsupported dist_type in choose_actions().")
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples = samples + self._multiplier_add
        total_limit = tf.where(samples[:, :, :, 0:1] > 0,
                               samples[:, :, :, 0:1] * samples[:, :, :, -1:],
                               samples)
        samples = tf.where(self._replace_last_mask, samples, total_limit)
        shp = tf.shape(samples)
        ais = self._ensure_ais_tensor(shp[:2])
        ais = tf.broadcast_to(ais, [shp[0], shp[1], ais.shape[2]])
        ais = tf.reshape(ais, [shp[0], shp[1], -1, 1])
        return tf.concat([ais, samples], axis=3)

    @tf.function(reduce_retracing=True)
    def policy_loss(self, states, actions, rewards):
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        # States: [B, T+1, ...] where T+1 = episode_length (initial state + T transitions)
        # Actions: [B, T, ...] where T = episode_length - 1 (already one shorter!)
        # Rewards: [B, T] (one per action taken)
        # Use states[:, :-1] to get [B, T, ...] matching the T actions/rewards.
        call_out = self.call(states[:, :-1])  # [B, T, ...] returns params per dist
        # Use ALL actions (don't slice again - they're already the right length)
        sub_vals = actions[:, :, :, 1:]  # [B, T, num_subactions, num_dist_per_subaction]
        orig_last_col = tf.where(
            sub_vals[:, :, :, 0:1] > 0.0,
            sub_vals[:, :, :, -1:] / tf.maximum(sub_vals[:, :, :, 0:1], 1e-12),
            sub_vals[:, :, :, -1:]
        )
        orig_last_col = orig_last_col + self._last_col_minus1
        sub_vals = tf.where(self._replace_last_mask, sub_vals, orig_last_col)
        if self.dist_type == 'gaussian':
            mus, sigmas = call_out
            sigma_clipped = tf.clip_by_value(sigmas, 1e-12, 1e12)
            z = (sub_vals - mus) / sigma_clipped
            log_comp = self._neg_half_log_2pi - 0.5 * tf.square(z) - tf.math.log(sigma_clipped)
            log_action_prbs = tf.reduce_sum(log_comp, axis=(2, 3))  # [B,T-1]
        elif self.dist_type == 'lognormal':
            mus, sigmas = call_out
            # x = sub_vals > 0, log x -> normal
            x_clipped = tf.clip_by_value(sub_vals, 1e-12, 1e30)
            log_x = tf.math.log(x_clipped)
            sigma_clipped = tf.clip_by_value(sigmas, 1e-12, 1e12)
            z = (log_x - mus) / sigma_clipped
            # log P = normal logprob(log_x) - log_x
            log_normal = self._neg_half_log_2pi - 0.5 * tf.square(z) - tf.math.log(sigma_clipped)
            # subtract log_x term (Jacobian) component-wise
            log_comp = log_normal - log_x
            log_action_prbs = tf.reduce_sum(log_comp, axis=(2, 3))
        elif self.dist_type == 'triangular':
            low, mode, high = call_out
            # Compute triangular log-pdf
            eps = tf.constant(1e-12, dtype=sub_vals.dtype)
            denom_ca = tf.maximum(high - low, eps)
            left_support = (sub_vals >= low) & (sub_vals <= mode)
            right_support = (sub_vals > mode) & (sub_vals <= high)
            # Left branch: log(2) + log(x-a) - log(b-a) - log(c-a)
            left_num = tf.maximum(sub_vals - low, eps)
            left_den1 = tf.maximum(mode - low, eps)
            left_log = self._log_2 + tf.math.log(left_num) - tf.math.log(left_den1) - tf.math.log(denom_ca)
            # Right branch: log(2) + log(c-x) - log(c-a) - log(c-b)
            right_num = tf.maximum(high - sub_vals, eps)
            right_den1 = tf.maximum(high - mode, eps)
            right_log = self._log_2 + tf.math.log(right_num) - tf.math.log(denom_ca) - tf.math.log(right_den1)
            # Outside support -> very small probability
            very_neg = tf.constant(-1e30, dtype=sub_vals.dtype)
            branch_log = tf.where(left_support, left_log, tf.where(right_support, right_log, very_neg))
            log_action_prbs = tf.reduce_sum(branch_log, axis=(2, 3))
        else:
            raise NotImplementedError("Unsupported dist_type in policy_loss().")

        discounted_rewards = self.discount(rewards)
        # Optional heuristic: compress large positive returns for lower-variance updates.
        # This changes the optimized objective relative to vanilla REINFORCE.
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0,
                                          tf.math.log(discounted_rewards + 1.0),
                                          discounted_rewards)
        # Use ALL rewards (don't slice - they're already aligned with actions at length T)
        adv_core = discounted_rewards
        if self.use_baseline:
            state_values = tf.reshape(self.value_function(states), (tf.shape(states)[0], -1))
            # Slice state_values to match: evaluate on all T+1 states, use first T for baseline
            baseline = state_values[:, :-1]
            advantage = adv_core - baseline
        else:
            advantage = adv_core
        neg_logs = -log_action_prbs
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        actor_loss = tf.reduce_sum(neg_logs * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_sum(tf.square(advantage)) if self.use_baseline else 0.0
        return actor_loss + 0.5 * critic_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        grads = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
