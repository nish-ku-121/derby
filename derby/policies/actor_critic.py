"""Modern actor-critic policy implementations.

Public class exported: ActorCritic
"""
from __future__ import annotations

import tensorflow as tf

from derby.policies.continuous_actor import ContinuousStochasticPolicy


class ActorCritic(ContinuousStochasticPolicy):
    """Advantage actor-critic policy using the modern continuous actor.

    The first supported variant is one-step TD actor-critic with a state-value
    critic. Actor-critic always uses a state-only baseline; baseline ablations
    belong on REINFORCE, not this policy family.
    """

    SUPPORTED_CRITIC_TYPES = frozenset({"td"})

    def __init__(self, *args, critic_type: str = "td", critic_weight: float = 0.5, use_baseline=None, **kwargs):
        self.critic_type = str(critic_type).lower()
        self.critic_weight = float(critic_weight)
        if use_baseline is not None:
            raise ValueError("ActorCritic always uses a state-value baseline; do not pass use_baseline")
        if self.critic_type not in self.SUPPORTED_CRITIC_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_CRITIC_TYPES))
            raise ValueError(f"Unsupported critic_type '{critic_type}'. Supported critic types: {supported}")
        if self.critic_weight < 0.0:
            raise ValueError("critic_weight must be >= 0")
        super().__init__(*args, use_baseline=True, **kwargs)

    def __repr__(self):
        base = super().__repr__()
        return f"ActorCritic(critic_type={self.critic_type}, critic_weight={self.critic_weight}, actor={base})"

    @tf.function(reduce_retracing=True)
    def policy_loss(self, states, actions, rewards):
        """Compute one-step TD actor-critic loss.

        For each timestep t:
            y_t = r_t + gamma * stop_gradient(V(s_{t+1}))
            A_t = y_t - V(s_t)
            L_actor = -log pi(a_t|s_t) * stop_gradient(A_t)
            L_critic = (y_t - V(s_t))^2
        """
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        rewards = tf.cast(rewards, tf.float32)

        call_out = self.call(states[:, :-1])
        log_action_prbs = self._log_action_prob(call_out, actions)

        state_values = tf.reshape(self.value_function(states), (tf.shape(states)[0], -1))
        current_values = state_values[:, :-1]
        raw_next_values = state_values[:, 1:]
        terminal_zeros = tf.zeros_like(raw_next_values[:, -1:])
        next_values = tf.concat([raw_next_values[:, :-1], terminal_zeros], axis=1)

        gamma = tf.cast(self.discount_factor, rewards.dtype)
        td_target = rewards + gamma * tf.stop_gradient(next_values)
        advantage = td_target - current_values
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage + 1.0), advantage)

        neg_logs = tf.clip_by_value(-log_action_prbs, -1e9, 1e9)
        actor_loss = tf.reduce_sum(neg_logs * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_sum(tf.square(td_target - current_values))
        return actor_loss + self.critic_weight * critic_loss
