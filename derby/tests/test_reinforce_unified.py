import unittest
import numpy as np
import tensorflow as tf

from derby.policies.reinforce import REINFORCE


class TestUnifiedREINFORCE(unittest.TestCase):
    def setUp(self):
        # Common test config
        self.auction_item_spec_ids = [10, 20, 30]  # 3 subactions
        self.num_subactions = len(self.auction_item_spec_ids)
        self.num_dist = 2  # default (bid, total_limit)
        self.batch_size = 2
        self.time_steps = 5
        self.state_dim = 4
        # States for policy_loss should have shape [B, T+1, ...] (initial + T transitions)
        # For tests, we use T+1 = time_steps for states
        self.states_for_loss = tf.zeros([self.batch_size, self.time_steps, self.state_dim], dtype=tf.float32)
        # For call() and choose_actions() tests, use T timesteps
        self.states = tf.zeros([self.batch_size, self.time_steps - 1, self.state_dim], dtype=tf.float32)

    def _assert_mu_sigma_shapes(self, mu, sigma):
        # call() returns params matching input states shape
        self.assertEqual(mu.shape, (self.batch_size, self.time_steps - 1, self.num_subactions, self.num_dist))
        self.assertEqual(sigma.shape, mu.shape)
        # sigma positivity
        self.assertTrue(tf.reduce_all(sigma > 0).numpy())

    def _assert_actions_shape(self, actions_tensor):
        # actions shape: [B, T, num_subactions, 1 + num_dist] (AIS id + dists)
        expected = (self.batch_size, self.time_steps - 1, self.num_subactions, 1 + self.num_dist)
        self.assertEqual(actions_tensor.shape, expected)

    def test_gaussian_shapes(self):
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=123,
            dist_type='gaussian',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
        )
        mu, sigma = policy(self.states)  # call()
        self._assert_mu_sigma_shapes(mu, sigma)
        actions = policy.choose_actions((mu, sigma))
        self._assert_actions_shape(actions)
        # value function shape (takes any state shape)
        values = policy.value_function(self.states)
        self.assertEqual(values.shape, (self.batch_size, self.time_steps - 1, 1))

    def test_gaussian_action_init_centers_primary_mean_and_stddev(self):
        init_action_center = 1.5
        init_action_stddev = 0.3
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=123,
            dist_type='gaussian',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
            init_action_center=init_action_center,
            init_action_stddev=init_action_stddev,
        )
        mu, sigma = policy(self.states)
        self._assert_mu_sigma_shapes(mu, sigma)
        mean_primary_mu = float(tf.reduce_mean(mu[..., 0]))
        mean_primary_sigma = float(tf.reduce_mean(sigma[..., 0]))
        self.assertAlmostEqual(mean_primary_mu, init_action_center, places=3)
        self.assertAlmostEqual(mean_primary_sigma, init_action_stddev, places=3)
        # The multiplier channel should remain near its neutral default, not the explicit init value.
        mean_multiplier_mu = float(tf.reduce_mean(mu[..., -1]))
        self.assertNotAlmostEqual(mean_multiplier_mu, init_action_center, places=2)

    def test_lognormal_shapes_and_mean(self):
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=321,
            dist_type='lognormal',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
        )
        mu, sigma = policy(self.states)
        self._assert_mu_sigma_shapes(mu, sigma)
        actions = policy.choose_actions((mu, sigma))
        self._assert_actions_shape(actions)
        # Lognormal consistency: mean(log(sample)) ~ mean(mu)
        act_only = actions[..., 1:]  # drop AIS id column
        bid_col = tf.math.log(tf.clip_by_value(act_only[..., 0], 1e-12, 1e30))
        mean_log_sample = float(tf.reduce_mean(bid_col))
        mean_mu = float(tf.reduce_mean(mu[..., 0]))
        self.assertTrue(abs(mean_log_sample - mean_mu) < 0.2)

    def test_lognormal_action_init_centers_primary_mean_and_stddev(self):
        init_action_center = 2.5
        init_action_stddev = 0.75
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=321,
            dist_type='lognormal',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
            init_action_center=init_action_center,
            init_action_stddev=init_action_stddev,
        )
        mu, sigma = policy(self.states)
        self._assert_mu_sigma_shapes(mu, sigma)
        implied_mean = tf.exp(mu[..., 0] + 0.5 * tf.square(sigma[..., 0]))
        implied_var = (tf.exp(tf.square(sigma[..., 0])) - 1.0) * tf.exp(2.0 * mu[..., 0] + tf.square(sigma[..., 0]))
        implied_stddev = tf.sqrt(implied_var)
        self.assertAlmostEqual(float(tf.reduce_mean(implied_mean)), init_action_center, places=3)
        self.assertAlmostEqual(float(tf.reduce_mean(implied_stddev)), init_action_stddev, places=3)

    def test_seed_reproducibility_gaussian(self):
        # Two policies with same seed & states should produce identical actions
        seed = 77
        pol1 = REINFORCE(self.auction_item_spec_ids, num_dist_per_spec=self.num_dist, seed=seed, dist_type='gaussian')
        pol2 = REINFORCE(self.auction_item_spec_ids, num_dist_per_spec=self.num_dist, seed=seed, dist_type='gaussian')
        mu1, sigma1 = pol1(self.states)
        mu2, sigma2 = pol2(self.states)
        # Means and sigmas should match exactly (same initial weights because of identical RandomUniform init range)
        self.assertTrue(np.allclose(mu1.numpy(), mu2.numpy()))
        self.assertTrue(np.allclose(sigma1.numpy(), sigma2.numpy()))
        a1 = pol1.choose_actions((mu1, sigma1))
        a2 = pol2.choose_actions((mu2, sigma2))
        self.assertTrue(np.allclose(a1.numpy(), a2.numpy()))

    def test_triangular_shapes_ordering_and_centering(self):
        init_action_center = 1.5
        init_action_stddev = 0.5
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=42,
            dist_type='triangular',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
            init_action_center=init_action_center,
            init_action_stddev=init_action_stddev,
        )
        low, mode, high = policy(self.states)
        # Shapes (now using self.states which is [B, T-1, ...])
        expected = (self.batch_size, self.time_steps - 1, self.num_subactions, self.num_dist)
        self.assertEqual(low.shape, expected)
        self.assertEqual(mode.shape, expected)
        self.assertEqual(high.shape, expected)
        # Non-negativity and ordering
        self.assertTrue(tf.reduce_all(low >= 0).numpy())
        self.assertTrue(tf.reduce_all(mode >= low).numpy())
        self.assertTrue(tf.reduce_all(high >= mode).numpy())
        # Centering applies to the primary action dimension only.
        mean_primary_mode = float(tf.reduce_mean(mode[..., 0]))
        implied_half_width = float(tf.reduce_mean(high[..., 0] - mode[..., 0]))
        self.assertAlmostEqual(mean_primary_mode, init_action_center, places=3)
        self.assertAlmostEqual(implied_half_width, init_action_stddev * np.sqrt(6.0), places=3)
        self.assertAlmostEqual(float(tf.reduce_mean(mode[..., 0] - low[..., 0])), implied_half_width, places=3)
        # Actions shape and non-negativity (values excluding AIS id)
        actions = policy.choose_actions((low, mode, high))
        self._assert_actions_shape(actions)
        self.assertTrue(tf.reduce_all(actions[..., 1:] >= 0).numpy())

    def test_triangular_policy_loss_finite(self):
        policy = REINFORCE(
            auction_item_spec_ids=self.auction_item_spec_ids,
            num_dist_per_spec=self.num_dist,
            seed=7,
            dist_type='triangular',
            actor_hidden_layers=1,
            actor_hidden_units=4,
            critic_hidden_layers=1,
            critic_hidden_units=6,
            init_action_center=1.0,
            init_action_stddev=0.2,
        )
        # For policy_loss, use states_for_loss [B, T+1, ...], actions/rewards [B, T, ...]
        low, mode, high = policy(self.states_for_loss[:, :-1])  # get T timesteps of params
        actions = policy.choose_actions((low, mode, high))
        rewards = tf.zeros([self.batch_size, self.time_steps - 1], dtype=tf.float32)
        loss = policy.policy_loss(self.states_for_loss, actions, rewards)
        self.assertTrue(tf.math.is_finite(loss).numpy())


if __name__ == '__main__':
    unittest.main(verbosity=2)
