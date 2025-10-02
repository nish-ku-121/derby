"""Unified REINFORCE Gaussian policy implementations.

Currently provides a single configurable Gaussian policy capable of emulating
legacy variants (v1, v2/v3/v3_1, v4) via activation + start_near_bpr flags.

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
    """Unified Gaussian policy (float32, manual Gaussian math) with configurable activation.

    Legacy variant mapping:
        v1        -> softplus, start_near_bpr=False
        v2/v3/v3_1 -> softplus, start_near_bpr=True
        v4        -> relu,     start_near_bpr=True

    Args:
        actor_final_activation: str | callable ('softplus' | 'relu' supported for centering; others error if start_near_bpr True).
        start_near_bpr: center initial mean near budget_per_reach using inverse activation.
        sigma_scale: multiplicative factor for activated sigma pre-values.
        sigma_floor: added after scaling (unified default 1e-5).
        actor_hidden_layers / units / activation: MLP depth/width for policy feature extractor (0 => linear head).
        critic_hidden_layers / units / activation: same for value network.

    Presets:
        Use REINFORCE.preset(name) or REINFORCE.from_preset(name, **overrides)
        Names: 'v1', 'v2', 'v3', 'v3_1', 'v4'. (Approximations: v1 adds a 0.5 scale vs historical.)
    """

    def __init__(self, auction_item_spec_ids, num_dist_per_spec: int = 2, budget_per_reach: float = 1.0,
                 is_partial: bool = False, discount_factor: float = 1, learning_rate: float = 0.0001,
                 shape_reward: bool = False, seed: int | None = None,
                 actor_final_activation='softplus', start_near_bpr: bool = True,
                 sigma_scale: float = 0.5, sigma_floor: float = 1e-5,
                 # Network architecture knobs (actor/value)
                 actor_hidden_layers: int = 1, actor_hidden_units: int = 1,
                 actor_hidden_activation='leaky_relu',
                 critic_hidden_layers: int = 1, critic_hidden_units: int = 6,
                 critic_hidden_activation='relu'):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward
        self.seed = seed
        self.start_near_bpr = start_near_bpr
        self._actor_final_activation_fn, self._actor_final_activation_name = self._resolve_activation(actor_final_activation)
        self._activation_shift_registry = {
            'relu': lambda B: float(B),
            'softplus': self._inverse_softplus,
        }
        if self.start_near_bpr:
            if self._actor_final_activation_name not in self._activation_shift_registry:
                raise ValueError(
                    f"start_near_bpr=True but activation '{self._actor_final_activation_name}' has no registered inverse/shift."
                )
            self._mu_shift_scalar = self._activation_shift_registry[self._actor_final_activation_name](self.budget_per_reach)
        else:
            self._mu_shift_scalar = 0.0
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
        self._validate_config()
        if seed is not None:
            try: random.seed(seed)
            except Exception: pass
            try: np.random.seed(seed)
            except Exception: pass
            self._rng = tf.random.Generator.from_seed(seed)
        else:
            self._rng = None

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0.0
        self.subactions_max = 1e15
        self.num_subactions = len(self.auction_item_spec_ids)
        self.num_dist_per_subaction = num_dist_per_spec

        # Initializers: preserve legacy param head init; hidden layers use default Glorot.
        self.param_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
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
        # Parameter head (fused mu+sigma) always present
        self.param_layer = tf.keras.layers.Dense(
            2 * self.num_subactions * self.num_dist_per_subaction,
            kernel_initializer=self.param_ker_init,
            activation=None,
            dtype='float32',
            name='param_head'
        )

        # Critic hidden stack
        self.critic_hidden = []
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

        self._ais_tensor = None
        self._multiplier_add = tf.constant([0.0] * (self.num_dist_per_subaction - 1) + [1.0], dtype=tf.float32)
        self._replace_last_mask = tf.constant([True] * (self.num_dist_per_subaction - 1) + [False])
        self._last_col_minus1 = tf.constant([0.0] * (self.num_dist_per_subaction - 1) + [-1.0], dtype=tf.float32)
        self._mu_shift = tf.constant(self._mu_shift_scalar, dtype=tf.float32)
        self._log_2pi = tf.constant(math.log(2.0 * math.pi), dtype=tf.float32)
        self._neg_half_log_2pi = tf.constant(-0.5 * math.log(2.0 * math.pi), dtype=tf.float32)

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
        return ("REINFORCE(" \
                f"is_partial: {self.is_partial}, discount: {self.discount_factor}, "
                f"lr: {self.learning_rate}, num_actions: {self.num_subactions}, optimizer: {type(self.optimizer).__name__}, "
                f"shape_reward: {self.shape_reward}, seed: {self.seed}, actor_final_activation: {self._actor_final_activation_name}, "
                f"start_near_bpr: {self.start_near_bpr}, sigma_scale: {self.sigma_scale}, sigma_floor: {self.sigma_floor}, "
                f"actor_depth: {self.actor_hidden_layers}, actor_width: {self.actor_hidden_units}, actor_act: {self._actor_hidden_activation_name}, "
                f"critic_depth: {self.critic_hidden_layers}, critic_width: {self.critic_hidden_units}, critic_act: {self._critic_hidden_activation_name})")

    # ---- Presets & Validation ----
    LEGACY_PRESETS = {
        # v1 original: no centering; wider variance. We unify sigma (scale=0.5).
        'v1': dict(actor_final_activation='softplus', start_near_bpr=False,
                  actor_hidden_layers=1, actor_hidden_units=1, actor_hidden_activation='leaky_relu',
                  critic_hidden_layers=1, critic_hidden_units=6, critic_hidden_activation='relu'),
        # v2 & v3 differ only by hidden activation historically (v3 used ELU).
        'v2': dict(actor_final_activation='softplus', start_near_bpr=True,
                  actor_hidden_layers=1, actor_hidden_units=1, actor_hidden_activation='leaky_relu',
                  critic_hidden_layers=1, critic_hidden_units=6, critic_hidden_activation='relu'),
        'v3': dict(actor_final_activation='softplus', start_near_bpr=True,
                  actor_hidden_layers=1, actor_hidden_units=1, actor_hidden_activation='elu',
                  critic_hidden_layers=1, critic_hidden_units=6, critic_hidden_activation='relu'),
        # v3_1 deep ELU stack (4 x 6)
        'v3_1': dict(actor_final_activation='softplus', start_near_bpr=True,
                    actor_hidden_layers=4, actor_hidden_units=6, actor_hidden_activation='elu',
                    critic_hidden_layers=1, critic_hidden_units=6, critic_hidden_activation='relu'),
        # v4 relu additive shift originally; we emulate via relu + centering
        'v4': dict(actor_final_activation='relu', start_near_bpr=True,
                  actor_hidden_layers=1, actor_hidden_units=1, actor_hidden_activation='leaky_relu',
                  critic_hidden_layers=1, critic_hidden_units=6, critic_hidden_activation='relu'),
    }

    @classmethod
    def preset(cls, name: str) -> dict:
        """Return a copy of kwargs for a legacy preset."""
        try:
            cfg = cls.LEGACY_PRESETS[name]
        except KeyError as e:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls.LEGACY_PRESETS)}") from e
        return dict(cfg)  # shallow copy

    @classmethod
    def from_preset(cls, name: str, auction_item_spec_ids, **overrides):
        """Instantiate from preset plus overrides.

        Example:
            policy = REINFORCE.from_preset('v3_1', ids, actor_hidden_units=8)
        """
        base = cls.preset(name)
        base.update(overrides)
        return cls(auction_item_spec_ids=auction_item_spec_ids, **base)

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
        states = tf.cast(states, tf.float32)
        x = states
        for layer in self.actor_hidden:
            x = layer(x)
        params = self.param_layer(x)
        raw_mu, raw_sigma = tf.split(params, 2, axis=-1)
        shift = self._mu_shift if self.start_near_bpr else 0.0
        mus = self._actor_final_activation_fn(raw_mu + shift)
        sigma_pre = self._actor_final_activation_fn(raw_sigma + shift)
        sigmas = (self.sigma_scale * sigma_pre) + self.sigma_floor
        dyn_shape = tf.shape(mus)
        target_shape = tf.concat([dyn_shape[:2], [self.num_subactions, self.num_dist_per_subaction]], axis=0)
        mus = tf.reshape(mus, target_shape)
        sigmas = tf.reshape(sigmas, target_shape)
        return mus, sigmas

    def value_function(self, states):
        states = tf.cast(states, tf.float32)
        x = states
        for layer in self.critic_hidden:
            x = layer(x)
        v = self.critic_out(x)
        return v

    def choose_actions(self, call_output):
        mus, sigmas = call_output
        if self._rng is not None:
            eps = self._rng.normal(tf.shape(mus), dtype=mus.dtype)
        else:
            eps = tf.random.normal(tf.shape(mus), dtype=mus.dtype)
        samples = mus + sigmas * eps
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
        mus, sigmas = self.call(states[:, :-1])
        sub_vals = actions[:, :, :, 1:]
        orig_last_col = tf.where(
            sub_vals[:, :, :, 0:1] > 0.0,
            sub_vals[:, :, :, -1:] / tf.maximum(sub_vals[:, :, :, 0:1], 1e-12),
            sub_vals[:, :, :, -1:]
        )
        orig_last_col = orig_last_col + self._last_col_minus1
        sub_vals = tf.where(self._replace_last_mask, sub_vals, orig_last_col)

        sigma_clipped = tf.clip_by_value(sigmas, 1e-12, 1e12)
        z = (sub_vals - mus) / sigma_clipped
        log_comp = self._neg_half_log_2pi - 0.5 * tf.square(z) - tf.math.log(sigma_clipped)
        log_action_prbs = tf.reduce_sum(log_comp, axis=(2, 3))

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0,
                                          tf.math.log(discounted_rewards + 1.0),
                                          discounted_rewards)
        state_values = tf.reshape(self.value_function(states), (tf.shape(states)[0], -1))
        baseline = state_values[:, :-1]
        advantage = discounted_rewards - baseline
        neg_logs = -log_action_prbs
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        actor_loss = tf.reduce_sum(neg_logs * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_sum(tf.square(advantage))
        return actor_loss + 0.5 * critic_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        grads = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
