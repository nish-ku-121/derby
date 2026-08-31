"""Unified REINFORCE policy implementation.

Configurable policy supporting Gaussian, LogNormal, and Triangular distributions.
Historical preset names (v1..v4) have been removed; reproduce them explicitly by
setting activation and actor/critic depth/width.

Public class exported: REINFORCE
"""
from __future__ import annotations

import tensorflow as tf

from derby.policies.continuous_actor import ContinuousStochasticPolicy


class REINFORCE(ContinuousStochasticPolicy):
    """Monte Carlo policy-gradient estimator over the shared continuous actor."""

    def __repr__(self):
        return (
            "REINFORCE("
            f"is_partial={self.is_partial}, discount={self.discount_factor}, "
            f"lr={self.learning_rate}, num_actions={self.num_subactions}, optimizer={type(self.optimizer).__name__}, "
            f"shape_reward={self.shape_reward}, seed={self.seed}, dist_type={self.dist_type}, "
            f"actor_final_activation={self._actor_final_activation_name}, "
            f"init_action_center={self.init_action_center}, init_action_stddev={self.init_action_stddev}, "
            f"min_action_stddev={self.min_action_stddev}, "
            f"actor_depth={self.actor_hidden_layers}, actor_width={self.actor_hidden_units}, "
            f"actor_act={self._actor_hidden_activation_name}, use_baseline={self.use_baseline}, "
            f"critic_depth={self.critic_hidden_layers}, critic_width={self.critic_hidden_units}, "
            f"critic_act={self._critic_hidden_activation_name}, adaptive_learning_rate={self.adaptive_learning_rate}, "
            f"adaptive_lr_epsilon={self.adaptive_lr_epsilon}, adaptive_lr_eta={self.adaptive_lr_eta})"
        )

    @tf.function(reduce_retracing=True)
    def policy_loss(self, states, actions, rewards):
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        # States: [B, T+1, ...] where T+1 = initial state + T transitions.
        # Actions/rewards: [B, T, ...], already aligned with states[:, :-1].
        call_out = self.call(states[:, :-1])
        log_action_prbs = self._log_action_prob(call_out, actions)

        discounted_rewards = self.discount(rewards)
        # Optional heuristic: compress large positive returns for lower-variance updates.
        # This changes the optimized objective relative to vanilla REINFORCE.
        if self.shape_reward:
            discounted_rewards = tf.where(
                discounted_rewards > 0,
                tf.math.log(discounted_rewards + 1.0),
                discounted_rewards,
            )

        if self.use_baseline:
            state_values = tf.reshape(self.value_function(states), (tf.shape(states)[0], -1))
            advantage = discounted_rewards - state_values[:, :-1]
        else:
            advantage = discounted_rewards

        neg_logs = tf.clip_by_value(-log_action_prbs, -1e9, 1e9)
        actor_loss = tf.reduce_sum(neg_logs * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_sum(tf.square(advantage)) if self.use_baseline else 0.0
        return actor_loss + 0.5 * critic_loss
