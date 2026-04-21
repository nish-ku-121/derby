import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from derby.core.agents import Agent
from derby.core.environments import (
    AbstractEnvironment,
    generate_trajectories,
)
from derby.core.policies import AbstractPolicy, FixedBidPolicy
from derby.experiments.one_camp_n_days.experiment import OneCampNDaysExperiment
from derby.experiments.one_camp_n_days import runner as one_camp_runner


class RecordingTensorflowPolicy(AbstractPolicy):
    def __init__(self):
        super().__init__(is_tensorflow=True)
        self.call_states = None
        self.loss_states = None
        self.loss_actions = None
        self.loss_rewards = None

    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def call(self, states):
        self.call_states = states.numpy()
        return states

    def choose_actions(self, call_output):
        batch_size, episode_length = call_output.shape[:2]
        return tf.ones((batch_size, episode_length, 1, 3), dtype=tf.float32)

    def policy_loss(self, states, actions, rewards):
        self.loss_states = states.numpy()
        self.loss_actions = actions.numpy()
        self.loss_rewards = rewards.numpy()
        return tf.constant(0.0, dtype=tf.float32)


class RecordingStaticPolicy(AbstractPolicy):
    def __init__(self):
        super().__init__(is_tensorflow=False)
        self.call_states = None
        self.loss_states = None
        self.loss_actions = None
        self.loss_rewards = None

    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def call(self, states):
        self.call_states = np.array(states, copy=True)
        return states

    def choose_actions(self, call_output):
        batch_size, episode_length = call_output.shape[:2]
        return np.ones((batch_size, episode_length, 1, 3), dtype=np.float32)

    def policy_loss(self, states, actions, rewards):
        self.loss_states = np.array(states, copy=True)
        self.loss_actions = np.array(actions, copy=True)
        self.loss_rewards = np.array(rewards, copy=True)
        return 0.0


class JointStatePolicy(AbstractPolicy):
    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_ALL

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def call(self, states):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def choose_actions(self, call_output):  # pragma: no cover - not used in these tests
        raise NotImplementedError

    def policy_loss(self, states, actions, rewards):  # pragma: no cover - not used in these tests
        raise NotImplementedError


class TestEnvironmentAgentContract(unittest.TestCase):
    def setUp(self):
        self.experiment = OneCampNDaysExperiment(seed=123)

    def test_market_env_folds_joint_and_single_state_views_for_modern_policies(self):
        """State folding should support both joint-state and per-agent policy views."""
        env, _ = self.experiment.build_two_segment_setup()
        joint_agent = Agent("joint", JointStatePolicy())
        single_agent = Agent("single", FixedBidPolicy(bid_per_item=1.0, total_limit=1.0))
        env.init([joint_agent, single_agent], horizon=1)

        states = np.arange(1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
        folded_joint = env.get_folded_states(joint_agent, states)
        folded_single = env.get_folded_states(single_agent, states)

        self.assertEqual(folded_joint.shape, (1, 2, 6))
        self.assertEqual(folded_single.shape, (1, 2, 3))
        np.testing.assert_allclose(folded_joint[0, 0], np.array([0, 1, 2, 3, 4, 5], dtype=np.float32))
        np.testing.assert_allclose(folded_single[0, 1], np.array([9, 10, 11], dtype=np.float32))

    def test_market_env_folds_actions_and_rewards_per_agent_for_modern_path(self):
        """Action and reward folding should isolate the current agent's trajectory slice."""
        env, _ = self.experiment.build_two_segment_setup()
        agent0 = Agent("a0", FixedBidPolicy(bid_per_item=1.0, total_limit=1.0))
        agent1 = Agent("a1", FixedBidPolicy(bid_per_item=2.0, total_limit=2.0))
        env.init([agent0, agent1], horizon=1)

        actions = [
            np.array([[[1.0, 5.0, 5.0], [1.0, 6.0, 6.0]]], dtype=np.float32),
            np.array([[[2.0, 7.0, 7.0], [2.0, 8.0, 8.0]]], dtype=np.float32),
        ]
        rewards = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32)

        folded_actions = env.get_folded_actions(agent1, actions)
        folded_rewards = env.get_folded_rewards(agent0, rewards)

        self.assertEqual(folded_actions.shape, (1, 2, 3))
        self.assertEqual(folded_rewards.shape, (1, 2))
        np.testing.assert_allclose(folded_actions[0, 0], np.array([2.0, 7.0, 7.0], dtype=np.float32))
        np.testing.assert_allclose(folded_rewards[0], np.array([1.5, 3.5], dtype=np.float32))

    def test_agent_applies_scaling_and_descaling_for_tensorflow_policies(self):
        """TF policies should see scaled tensors and emit descaled env-facing actions."""
        policy = RecordingTensorflowPolicy()

        def scale_states(states):
            return states * 2.0

        def scale_actions(actions):
            scaled = np.array(actions, copy=True)
            scaled[..., 1:] = scaled[..., 1:] - 10.0
            return scaled

        def descale_actions(actions):
            descaled = np.array(actions, copy=True)
            descaled[..., 1:] = descaled[..., 1:] + 10.0
            return descaled

        agent = Agent(
            "learner",
            policy,
            states_scaler=scale_states,
            actions_scaler=scale_actions,
            actions_descaler=descale_actions,
        )

        states = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        # `compute_action()` should scale state inputs, then descale the sampled action
        # before returning it back to the environment layer.
        action = agent.compute_action(states)
        np.testing.assert_allclose(policy.call_states, np.array([[[2.0, 4.0, 6.0]]], dtype=np.float32))
        np.testing.assert_allclose(action, np.array([[1.0, 11.0, 11.0]], dtype=np.float32))

        actions = np.array([[[[7.0, 14.0, 16.0]]]], dtype=np.float32)
        rewards = np.array([[3.0]], dtype=np.float32)
        # Loss/update paths see the same scaled-state view, but actions are scaled
        # back into the policy's working space before policy_loss() runs.
        loss = agent.compute_policy_loss(states, actions, rewards)

        self.assertTrue(tf.is_tensor(loss))
        np.testing.assert_allclose(policy.loss_states, np.array([[[2.0, 4.0, 6.0]]], dtype=np.float32))
        np.testing.assert_allclose(policy.loss_actions, np.array([[[[7.0, 4.0, 6.0]]]], dtype=np.float32))
        np.testing.assert_allclose(policy.loss_rewards, rewards)

    def test_agent_leaves_static_policies_in_raw_units(self):
        """Static policies should bypass scaling entirely and operate in raw env units."""
        policy = RecordingStaticPolicy()
        agent = Agent("baseline", policy)

        states = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        actions = np.array([[[[7.0, 14.0, 16.0]]]], dtype=np.float32)
        rewards = np.array([[3.0]], dtype=np.float32)

        action = agent.compute_action(states)
        loss = agent.compute_policy_loss(states, actions, rewards)

        self.assertEqual(loss, 0.0)
        np.testing.assert_allclose(policy.call_states, states)
        np.testing.assert_allclose(action, np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(policy.loss_states, states)
        np.testing.assert_allclose(policy.loss_actions, actions)
        np.testing.assert_allclose(policy.loss_rewards, rewards)

    def test_convert_from_actions_tensor_accepts_float_spec_ids(self):
        """Action conversion should accept float-valued spec IDs from numeric policies."""
        env, auction_item_spec_ids = self.experiment.build_one_segment_setup()
        agent = Agent("baseline", FixedBidPolicy(bid_per_item=1.0, total_limit=1.0))
        env.init([agent], horizon=1)

        actions = np.array([[[float(auction_item_spec_ids[0]), 5.0, 9.0]]], dtype=np.float32)
        bids = env.convert_from_actions_tensor(actions, env.agents, env._auction_item_specs_by_id)

        self.assertEqual(len(bids), 1)
        self.assertEqual(len(bids[0]), 1)
        self.assertEqual(bids[0][0].auction_item_spec.uid, auction_item_spec_ids[0])
        self.assertEqual(bids[0][0].bid_per_item, 5.0)
        self.assertEqual(bids[0][0].total_limit, 9.0)

    def test_convert_from_actions_tensor_raises_for_unknown_ids(self):
        """Unknown action spec IDs should fail fast instead of silently remapping."""
        env, _ = self.experiment.build_two_segment_setup()
        agent = Agent("baseline", FixedBidPolicy(bid_per_item=1.0, total_limit=1.0))
        env.init([agent], horizon=1)

        actions = np.array(
            [[
                [999.0, 5.0, 9.0],
                [998.0, 6.0, 10.0],
            ]],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(KeyError, "Unknown auction_item_spec_id 999"):
            env.convert_from_actions_tensor(actions, env.agents, env._auction_item_specs_by_id)

    def test_generate_trajectories_returns_aligned_batched_shapes(self):
        """Trajectory generation should align state, action, and reward batch shapes."""
        env, _ = self.experiment.build_one_segment_setup()
        agents = [
            Agent("a0", FixedBidPolicy(bid_per_item=5.0, total_limit=5.0)),
            Agent("a1", FixedBidPolicy(bid_per_item=5.0, total_limit=5.0)),
        ]
        env.vectorize = True
        env.init(agents, horizon=1)

        states, actions, rewards = generate_trajectories(
            env,
            num_of_trajs=3,
            horizon_cutoff=5,
        )

        self.assertEqual(states.shape, (3, 2, 2, 6))
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0].shape, (3, 1, 1, 3))
        self.assertEqual(actions[1].shape, (3, 1, 1, 3))
        self.assertEqual(rewards.shape, (3, 1, 2))
        self.assertEqual(states.dtype, np.float32)
        self.assertEqual(actions[0].dtype, np.float32)
        self.assertEqual(rewards.dtype, np.float32)
        # States include the initial observation, so time length is one greater than
        # the action/reward sequence length.
        self.assertEqual(states.shape[1], actions[0].shape[1] + 1)
        self.assertEqual(actions[0].shape[1], rewards.shape[1])

    def test_runner_applies_scalers_only_to_tensorflow_policies(self):
        """The modern runner should wire scalers only onto learning/TF policies."""
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
            # The runner expects cumulative_rewards to exist after each epoch;
            # we stub just enough training behavior to inspect agent wiring.
            for agent in env.agents:
                agent.cumulative_rewards = np.zeros(num_of_trajs, dtype=np.float32)

        config = {
            "num_days": 1,
            "num_trajs": 2,
            "num_epochs": 1,
            "setup": "one_segment",
            "seed": 123,
            "agents": [
                {
                    "name": "learner",
                    "policy": "REINFORCE",
                    "params": {
                        "learning_rate": 1e-5,
                        "dist_type": "gaussian",
                        "use_baseline": False,
                        "actor_hidden_layers": 1,
                        "actor_hidden_units": 2,
                        "critic_hidden_layers": 1,
                        "critic_hidden_units": 2,
                    },
                },
                {
                    "name": "baseline",
                    "policy": "FixedBidPolicy",
                    "params": {
                        "bid_per_item": 5,
                        "total_limit": 5,
                    },
                },
            ],
        }

        # Patch the runner module's Agent/train references so we can observe how
        # policies are wired without running a full training pass.
        with patch.object(one_camp_runner, "Agent", RecordingAgent), patch.object(
            one_camp_runner,
            "train",
            fake_train,
        ):
            one_camp_runner.run_experiment_from_config(config)

        self.assertEqual(len(created_agents), 2)
        learner_meta = next(meta for meta in created_agents if meta["name"] == "learner")
        baseline_meta = next(meta for meta in created_agents if meta["name"] == "baseline")

        self.assertEqual(learner_meta["policy_class"], "REINFORCE")
        self.assertTrue(learner_meta["has_states_scaler"])
        self.assertTrue(learner_meta["has_actions_scaler"])
        self.assertTrue(learner_meta["has_actions_descaler"])

        self.assertEqual(baseline_meta["policy_class"], "FixedBidPolicy")
        self.assertFalse(baseline_meta["has_states_scaler"])
        self.assertFalse(baseline_meta["has_actions_scaler"])
        self.assertFalse(baseline_meta["has_actions_descaler"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
