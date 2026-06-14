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
from derby.policies.update_rules import GradientNormAdaptiveLearningRate


class REINFORCE(AbstractPolicy, tf.keras.Model):
    """Unified policy (float32, manual distribution math) with configurable activation.

    Args:
        actor_final_activation: str | callable ('softplus' | 'relu' supported for explicit init centering).
        init_action_center: optional target center for the primary action dimension at initialization.
            This is interpreted in the policy's working action space.
        init_action_stddev: optional target stddev for the primary action dimension at initialization.
            This is interpreted in the policy's working action space.
        min_action_stddev: minimum stddev added after the sigma activation.
        param_kernel_initializer: optional Keras initializer spec for the parameter-head kernel.
        actor_hidden_layers / units / activation: MLP depth/width for policy feature extractor (0 => linear head).
        critic_hidden_layers / units / activation: same for value network.
    Notes:
        Previous preset system removed; configure via explicit kwargs or external sweep grids.
    """

    def __init__(self, auction_item_spec_ids, num_dist_per_spec: int = 2,
                 is_partial: bool = False, discount_factor: float = 1, learning_rate: float = 0.0001,
                 shape_reward: bool = False, seed: int | None = None,
                 actor_final_activation='softplus', init_action_center: float | None = None,
                 init_action_stddev: float | None = None,
                 min_action_stddev: float = 1e-5,
                 param_kernel_initializer=None,
                 dist_type: str = 'gaussian',
                 # Network architecture knobs (actor/value)
                 actor_hidden_layers: int = 1, actor_hidden_units: int = 1,
                 actor_hidden_activation='leaky_relu',
                 critic_hidden_layers: int = 1, critic_hidden_units: int = 6,
                 critic_hidden_activation='relu',
                 use_baseline: bool = True,
                 adaptive_learning_rate: bool = False,
                 adaptive_lr_epsilon: float | None = None,
                 adaptive_lr_eta: float = 1e-8):
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
        self.init_action_center = None if init_action_center is None else float(init_action_center)
        self.init_action_stddev = None if init_action_stddev is None else float(init_action_stddev)
        # Unified sigma semantics: always sigma = activation(raw_sigma) + min_action_stddev.
        self.min_action_stddev = float(min_action_stddev)
        self.param_kernel_initializer = tf.keras.initializers.get(param_kernel_initializer)
        # Store architecture parameters
        self.actor_hidden_layers = int(actor_hidden_layers)
        self.actor_hidden_units = int(actor_hidden_units)
        self.critic_hidden_layers = int(critic_hidden_layers)
        self.critic_hidden_units = int(critic_hidden_units)
        self._actor_hidden_activation_fn, self._actor_hidden_activation_name = self._resolve_activation(actor_hidden_activation)
        self._critic_hidden_activation_fn, self._critic_hidden_activation_name = self._resolve_activation(critic_hidden_activation)
        self.use_baseline = bool(use_baseline)
        self.adaptive_learning_rate = bool(adaptive_learning_rate)
        self.adaptive_lr_epsilon = None if adaptive_lr_epsilon is None else float(adaptive_lr_epsilon)
        self.adaptive_lr_eta = float(adaptive_lr_eta)
        self.adaptive_lr_rule = None
        self.last_grad_norm = None
        self.last_effective_learning_rate = None


        # Subaction metadata must exist before validation that references num_dist_per_subaction
        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0.0
        self.subactions_max = 1e15
        self.num_subactions = len(self.auction_item_spec_ids)
        self.num_dist_per_subaction = num_dist_per_spec

        self._validate_config()
        if self.adaptive_learning_rate:
            self.adaptive_lr_rule = GradientNormAdaptiveLearningRate(
                epsilon=self.adaptive_lr_epsilon,
                eta=self.adaptive_lr_eta,
            )
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
        # Bias initialization optionally centers the primary action dimension and its spread.
        param_dim = self._per_dist_param_count * self.num_subactions * self.num_dist_per_subaction
        bias_init = np.zeros(param_dim, dtype=np.float32)
        if self.init_action_center is not None or self.init_action_stddev is not None:
            if self._actor_final_activation_name not in self._activation_inverse_registry:
                raise ValueError(
                    f"Explicit action initialization requires an activation with a registered inverse; "
                    f"got '{self._actor_final_activation_name}'."
                )
            inv = self._activation_inverse_registry[self._actor_final_activation_name]
            for i in range(self.num_subactions):
                block_start = i * self.num_dist_per_subaction * self._per_dist_param_count
                # Only initialize the primary action channel; leave the multiplier channel neutral.
                if self.dist_type == 'gaussian':
                    if self.init_action_center is not None:
                        bias_init[block_start] = inv(self.init_action_center)
                    if self.init_action_stddev is not None:
                        sigma_target = self.init_action_stddev - self.min_action_stddev
                        bias_init[block_start + 1] = inv(sigma_target)
                elif self.dist_type == 'lognormal':
                    if self.init_action_center is not None:
                        latent_mu, latent_sigma = self._lognormal_action_moments_to_latent(
                            self.init_action_center,
                            self.init_action_stddev,
                        )
                        bias_init[block_start] = inv(latent_mu)
                        if self.init_action_stddev is not None:
                            sigma_floor = self._lognormal_min_latent_sigma_for_action_stddev_scalar(
                                self.min_action_stddev,
                                latent_mu,
                            )
                            sigma_target = latent_sigma - sigma_floor
                            bias_init[block_start + 1] = inv(sigma_target)
                elif self.dist_type == 'triangular':
                    if self.init_action_center is not None:
                        bias_init[block_start] = inv(self.init_action_center)
                    if self.init_action_stddev is not None:
                        min_half_width = self.min_action_stddev * math.sqrt(6.0)
                        symmetric_half_width = self.init_action_stddev * math.sqrt(6.0)
                        span_target = symmetric_half_width - min_half_width
                        bias_init[block_start + 1] = inv(span_target)
                        bias_init[block_start + 2] = inv(span_target)
                else:  # pragma: no cover
                    raise ValueError(f"Unknown dist_type '{self.dist_type}' for bias initialization.")
        self.param_layer = tf.keras.layers.Dense(
            param_dim,
            kernel_initializer=self.param_kernel_initializer,
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
            # softplus never reaches 0 exactly; use a large negative raw value as a practical proxy
            return -20.0
        if B < 20.0:
            return math.log(math.expm1(B))
        return B + math.log1p(-math.exp(-B))

    @staticmethod
    def _lognormal_action_moments_to_latent(action_mean: float, action_stddev: float | None) -> tuple[float, float]:
        """Convert action-space mean/stddev to the latent normal parameters."""
        mean = float(action_mean)
        stddev = 0.0 if action_stddev is None else float(action_stddev)
        variance_ratio = (stddev * stddev) / (mean * mean)
        latent_sigma_sq = math.log1p(variance_ratio)
        latent_sigma = math.sqrt(latent_sigma_sq)
        latent_mu = math.log(mean) - 0.5 * latent_sigma_sq
        return latent_mu, latent_sigma

    @staticmethod
    def _lognormal_min_latent_sigma_for_action_stddev_scalar(action_stddev: float, latent_mu: float) -> float:
        """Map an action-space stddev floor to the corresponding latent sigma floor at a given latent mu."""
        action_stddev = float(action_stddev)
        latent_mu = float(latent_mu)
        if action_stddev <= 0.0:
            return 0.0
        z = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * (action_stddev ** 2) * math.exp(-2.0 * latent_mu)))
        return math.sqrt(max(math.log(z), 0.0))

    @staticmethod
    def _lognormal_min_latent_sigma_for_action_stddev_tensor(action_stddev: float, latent_mu: tf.Tensor) -> tf.Tensor:
        """Tensor version of the action-space stddev floor conversion for lognormal policies."""
        if action_stddev <= 0.0:
            return tf.zeros_like(latent_mu)
        action_stddev_t = tf.cast(action_stddev, latent_mu.dtype)
        z = 0.5 * (1.0 + tf.sqrt(1.0 + 4.0 * tf.square(action_stddev_t) * tf.exp(-2.0 * latent_mu)))
        return tf.sqrt(tf.maximum(tf.math.log(z), 0.0))

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
            f"init_action_center={self.init_action_center}, init_action_stddev={self.init_action_stddev}, "
            f"min_action_stddev={self.min_action_stddev}, "
            f"actor_depth={self.actor_hidden_layers}, actor_width={self.actor_hidden_units}, actor_act={self._actor_hidden_activation_name}, "
            f"use_baseline={self.use_baseline}, critic_depth={self.critic_hidden_layers}, critic_width={self.critic_hidden_units}, "
            f"critic_act={self._critic_hidden_activation_name}, adaptive_learning_rate={self.adaptive_learning_rate}, "
            f"adaptive_lr_epsilon={self.adaptive_lr_epsilon}, adaptive_lr_eta={self.adaptive_lr_eta})"
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
        if self.min_action_stddev < 0:
            raise ValueError("min_action_stddev must be >= 0")
        if self.num_dist_per_subaction < 1:
            raise ValueError("num_dist_per_spec must be >= 1")
        if self.init_action_center is not None and self.init_action_center < 0:
            raise ValueError("init_action_center must be >= 0")
        if self.init_action_stddev is not None and self.init_action_stddev < 0:
            raise ValueError("init_action_stddev must be >= 0")
        if self.init_action_stddev is not None and self.init_action_stddev < self.min_action_stddev:
            raise ValueError("init_action_stddev must be >= min_action_stddev")
        if self.dist_type == 'lognormal':
            if self.init_action_center is not None and self.init_action_center <= 0.0:
                raise ValueError("lognormal init_action_center must be > 0")
            if self.init_action_stddev is not None and self.init_action_center is None:
                raise ValueError("lognormal init_action_stddev requires init_action_center")
            if self.init_action_center is not None and self.init_action_stddev is None:
                raise ValueError(
                    "lognormal action-space initialization requires both init_action_center and init_action_stddev"
                )
        if self.dist_type == 'triangular' and self.init_action_center is None and self.init_action_stddev is not None:
            raise ValueError("triangular init_action_stddev requires init_action_center")
        if (
            self.dist_type == 'triangular'
            and self.init_action_center is not None
            and self.init_action_stddev is not None
            and self.init_action_center - (self.init_action_stddev * math.sqrt(6.0)) < 0.0
        ):
            raise ValueError(
                "triangular init_action_stddev is too large for the requested init_action_center; "
                "the implied symmetric low endpoint would be negative"
            )
        if self.adaptive_learning_rate and self.adaptive_lr_epsilon is None:
            raise ValueError("adaptive_lr_epsilon must be provided when adaptive_learning_rate is enabled")
        if self.adaptive_learning_rate and self.adaptive_lr_epsilon <= 0.0:
            raise ValueError("adaptive_lr_epsilon must be > 0")
        if self.adaptive_lr_eta <= 0.0:
            raise ValueError("adaptive_lr_eta must be > 0")

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
            if self.dist_type == 'gaussian':
                sigma = sigma_pre + self.min_action_stddev
            else:
                sigma_floor = self._lognormal_min_latent_sigma_for_action_stddev_tensor(
                    self.min_action_stddev,
                    mu,
                )
                sigma = sigma_pre + sigma_floor
            return (mu, sigma)
        if self.dist_type == 'triangular':
            a = params[..., 0]
            b = params[..., 1]
            c = params[..., 2]
            center = self._actor_final_activation_fn(a)
            min_half_width = self.min_action_stddev * math.sqrt(6.0)
            left_span = self._actor_final_activation_fn(b) + min_half_width
            right_span = self._actor_final_activation_fn(c) + min_half_width
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
        grads_and_vars = [(g, v) for g, v in zip(grads, self.trainable_variables) if g is not None]
        if not grads_and_vars:
            self.last_grad_norm = 0.0
            self.last_effective_learning_rate = None
            return
        if self.adaptive_lr_rule is None:
            self.last_grad_norm = float(tf.linalg.global_norm([g for g, _ in grads_and_vars]).numpy())
            self.last_effective_learning_rate = float(tf.convert_to_tensor(self.optimizer.learning_rate).numpy())
            self.optimizer.apply_gradients(grads_and_vars)
            return

        effective_lr, grad_norm = self.adaptive_lr_rule.learning_rate([g for g, _ in grads_and_vars])
        original_lr = self.optimizer.learning_rate
        self.optimizer.learning_rate = effective_lr
        try:
            self.optimizer.apply_gradients(grads_and_vars)
        finally:
            self.optimizer.learning_rate = original_lr
        self.last_grad_norm = float(grad_norm.numpy())
        self.last_effective_learning_rate = float(effective_lr.numpy())
