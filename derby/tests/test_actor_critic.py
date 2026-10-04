import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from derby.core.agents import Agent
from derby.scenarios.one_campaign_n_days import runner as one_campaign_runner
from derby.policies.actor_critic import ActorCritic


class ZeroLogProbActorCritic(ActorCritic):
    def _log_action_prob(self, call_out, actions):
        return tf.zeros(tf.shape(actions)[:2], dtype=tf.float32)


class TestActorCritic(unittest.TestCase):
    def setUp(self):
        self.auction_item_spec_ids = [10, 20]
        self.num_dist = 2
        self.batch_size = 2
        self.time_steps = 5
        self.state_dim = 4
        self.states_for_loss = tf.zeros([self.batch_size, self.time_steps, self.state_dim], dtype=tf.float32)
        self.states = self.states_for_loss[:, :-1]

    def test_td_shapes(self):
        policy = ActorCritic(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=123,
            dist_type="gaussian",
            critic_type="td",
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
        )

        mu, sigma = policy(self.states)
        self.assertEqual(mu.shape, (self.batch_size, self.time_steps - 1, 2, self.num_dist))
        self.assertEqual(sigma.shape, mu.shape)
        self.assertTrue(tf.reduce_all(sigma > 0).numpy())

        actions = policy.choose_actions((mu, sigma))
        self.assertEqual(actions.shape, (self.batch_size, self.time_steps - 1, 2, 1 + self.num_dist))

        values = policy.value_function(self.states_for_loss)
        self.assertEqual(values.shape, (self.batch_size, self.time_steps, 1))

    def test_td_loss_uses_terminal_zero_bootstrap(self):
        policy = ZeroLogProbActorCritic(
            auction_item_spec_ids=[10],
            num_dist_per_spec=self.num_dist,
            seed=7,
            dist_type="gaussian",
            critic_type="td",
            discount_factor=0.5,
            critic_weight=0.5,
            actor_hidden_layers=0,
            critic_hidden_layers=0,
        )
        states = tf.zeros([1, 4, self.state_dim], dtype=tf.float32)
        mu, sigma = policy(states[:, :-1])
        actions = policy.choose_actions((mu, sigma))
        rewards = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)

        policy.value_function(states)
        for var in policy.critic_out.trainable_variables:
            var.assign(tf.zeros_like(var))

        loss = policy.policy_loss(states, actions, rewards)

        # With V(s)=0 and zero log-prob, actor loss is 0 and the terminal step
        # does not bootstrap: 0.5 * (1^2 + 2^2 + 3^2).
        self.assertAlmostEqual(float(loss.numpy()), 7.0, places=5)

    def test_rejects_non_td_and_use_baseline_knob(self):
        with self.assertRaisesRegex(ValueError, "Unsupported critic_type"):
            ActorCritic(self.auction_item_spec_ids, critic_type="sarsa")
        with self.assertRaisesRegex(ValueError, "always uses a state-value baseline"):
            ActorCritic(self.auction_item_spec_ids, use_baseline=False)

    def test_update_records_learning_rate_diagnostics(self):
        learning_rate = 0.05
        policy = ActorCritic(
            self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            learning_rate=learning_rate,
            actor_hidden_layers=0,
            critic_hidden_layers=0,
        )
        policy(self.states)
        policy.value_function(self.states_for_loss)
        before = [v.numpy().copy() for v in policy.trainable_variables]

        with tf.GradientTape() as tape:
            loss = tf.add_n([tf.reduce_sum(v) for v in policy.trainable_variables])
        policy.update(None, None, None, loss, tf_grad_tape=tape)

        grad_norm = np.sqrt(sum(v.size for v in before))
        self.assertAlmostEqual(policy.last_grad_norm, grad_norm, places=5)
        self.assertAlmostEqual(policy.last_effective_learning_rate, learning_rate, places=7)

    def test_runner_supports_actor_critic(self):
        created_agents = []

        class RecordingAgent(Agent):
            def __init__(self, name, policy, states_scaler=None, actions_scaler=None, actions_descaler=None):
                super().__init__(name, policy, states_scaler, actions_scaler, actions_descaler)
                created_agents.append({
                    "name": name,
                    "policy_class": type(policy).__name__,
                    "has_states_scaler": states_scaler is not None,
                    "has_actions_scaler": actions_scaler is not None,
                    "has_actions_descaler": actions_descaler is not None,
                })

        def fake_train(env, num_of_trajs, horizon_cutoff, **kwargs):
            for agent in env.agents:
                agent.cumulative_rewards = np.zeros(num_of_trajs, dtype=np.float32)

        config = {
            "num_days": 1,
            "num_trajs": 2,
            "num_epochs": 1,
            "scenario_variant": "one_segment",
            "seed": 123,
            "agents": [
                {
                    "name": "td_ac",
                    "policy": "ActorCritic",
                    "params": {
                        "critic_type": "td",
                        "learning_rate": 1e-5,
                        "dist_type": "gaussian",
                        "actor_hidden_layers": 1,
                        "actor_hidden_units": 2,
                        "critic_hidden_layers": 1,
                        "critic_hidden_units": 2,
                    },
                },
            ],
        }

        with patch.object(one_campaign_runner, "Agent", RecordingAgent), patch.object(
            one_campaign_runner,
            "train",
            fake_train,
        ):
            one_campaign_runner.run_from_config(config)

        self.assertEqual(len(created_agents), 1)
        self.assertEqual(created_agents[0]["policy_class"], "ActorCritic")
        self.assertTrue(created_agents[0]["has_states_scaler"])
        self.assertTrue(created_agents[0]["has_actions_scaler"])
        self.assertTrue(created_agents[0]["has_actions_descaler"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
