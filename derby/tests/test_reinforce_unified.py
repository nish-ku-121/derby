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
        # Deterministic states (zeros) for reproducibility
        self.states = tf.zeros([self.batch_size, self.time_steps, self.state_dim], dtype=tf.float32)

    def _assert_mu_sigma_shapes(self, mu, sigma):
        self.assertEqual(mu.shape, (self.batch_size, self.time_steps, self.num_subactions, self.num_dist))
        self.assertEqual(sigma.shape, mu.shape)
        # sigma positivity
        self.assertTrue(tf.reduce_all(sigma > 0).numpy())

    def _assert_actions_shape(self, actions_tensor):
        # actions shape: [B, T, num_subactions, 1 + num_dist] (AIS id + dists)
        expected = (self.batch_size, self.time_steps, self.num_subactions, 1 + self.num_dist)
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
        # value function shape
        values = policy.value_function(self.states)
        self.assertEqual(values.shape, (self.batch_size, self.time_steps, 1))

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
