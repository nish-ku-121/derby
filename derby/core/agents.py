from typing import Set, List, Dict, Iterable, Callable
import itertools
import numpy as np
import os
import tensorflow as tf
from derby.core.policies import AbstractPolicy


# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Agent:
    _uid_generator = itertools.count(1)

    uid: int
    name: str
    policy: AbstractPolicy
    agent_num: int # assuming this is set by the environment/game
    cumulative_rewards: Iterable[float] # cumulative rewards for the last 1000 trajectories
    # function to scale/normalize states, where states is an array of
    # shape [batch_size, episode_length, ...].
    states_scaler: Callable
    # function to descale/denormalize actions, where actions is an array of
    # shape [batch_size, episode_length, ...]. actions of a policy may need 
    # to be de-normalized before being passed to the environment if the 
    # states passed to the policy are normalized.
    actions_descaler: Callable
    # Likewise, function to scale/normalize descaled actions.
    actions_scaler: Callable

    def __init__(self, name: str, policy: AbstractPolicy):
        self.uid = next(type(self)._uid_generator)
        self.name = name
        self.policy = policy
        self.agent_num = None
        self.cumulative_rewards = None
        self.states_scaler = lambda states : states
        self.actions_descaler = lambda actions: actions
        self.actions_scaler = lambda actions: actions
# TODO: Github issue #29
        self.policy.agent = self
#

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.uid == other.uid

    def __hash__(self):
        return hash(self.__class__.__name__ + str(self.uid))

    def __repr__(self):
        return "{}(uid: {}, name: {}, policy: {})".format(self.__class__.__name__, 
                                                          self.uid, self.name, self.policy)

    def set_scalers(self, states_scaler, actions_scaler):
        self.states_scaler = states_scaler
        self.actions_scaler = actions_scaler

    def set_descaler(self, actions_descaler):
        self.actions_descaler = actions_descaler
        
    def compute_action(self, states):
        '''
        :param states: an array of shape [batch_size, episode_length, ...],
        assumed to be folded appropriately by the environment. 
        Note that batch_size and episode_length will be 1 (i.e. a single state).
        :return: the action to take.
        '''
        action = None
        states = self.convert_to_tf_tensor_if_needed(self.states_scaler(states))
        scaled_actions = self.policy.choose_actions(self.policy.call(states))
        if tf.is_tensor(scaled_actions):
            scaled_actions = scaled_actions.numpy()
        # Descale/de-normalize the action before returning (i.e. passing to the environment).
        action = self.actions_descaler(scaled_actions)[0,0]
        return action

    def compute_policy_loss(self, states, actions, rewards):
        '''
        :param states: an array of shape [batch_size, episode_length, ...].
        :param actions: an array of shape [batch_size, episode_length, ...]. 
        Note that the actions are descaled/de-normalized because they're the ones returned 
        by compute_action.
        :param rewards: an array of shape [batch_size, episode_length, ...].
        Where "..." means the array is assumed to be folded appropriately by the environment.
        :return: The agent's policy's loss.
        '''
        states = self.convert_to_tf_tensor_if_needed(self.states_scaler(states))
        actions = self.convert_to_tf_tensor_if_needed(self.actions_scaler(actions))
        rewards = self.convert_to_tf_tensor_if_needed(rewards)
        return self.policy.loss(states, actions, rewards)

    def update_policy(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        '''
        :param states: an array of shape [batch_size, episode_length, ...].
        :param actions: an array of shape [batch_size, episode_length, ...].
        Note that the actions are descaled/de-normalized because they're the ones returned 
        by compute_action.
        :param rewards: an array of shape [batch_size, episode_length, ...].
        :param policy_loss: The loss of the policy (if tf policy, the loss computed under tf.GradientTape()).
        Where "..." means the array is assumed to be folded appropriately by the environment.
        :return:
        '''
        states = self.convert_to_tf_tensor_if_needed(self.states_scaler(states))
        actions = self.convert_to_tf_tensor_if_needed(self.actions_scaler(actions))
        rewards = self.convert_to_tf_tensor_if_needed(rewards)

        self.policy.update(states, actions, rewards, policy_loss, tf_grad_tape=tf_grad_tape)

    def update_stats(self, states, actions, rewards):
        '''
        Updates stats for the agent, e.g. the agent's cumulative reward.
        :param states: an array of shape [batch_size, episode_length, ...].
        :param actions: an array of shape [batch_size, episode_length, ...].
        Note that the actions are descaled/de-normalized because they're the ones returned 
        by compute_action.
        :param rewards: an array of shape [batch_size, episode_length, ...].
        Where "..." means the array is assumed to be folded appropriately by the environment.
        :return:
        '''
        # resulting shape should be [batch_size]
        cuml_rwd_for_each_batch = np.sum(rewards, axis=1)

        if self.cumulative_rewards is None \
           or (type(self.cumulative_rewards) is list and len(self.cumulative_rewards) == 0) \
           or (type(self.cumulative_rewards) is np.ndarray and self.cumulative_rewards.size == 0):
            self.cumulative_rewards = cuml_rwd_for_each_batch
        else:
            self.cumulative_rewards = np.concatenate((self.cumulative_rewards, cuml_rwd_for_each_batch))
        self.cumulative_rewards[-1000:]

    def convert_to_tf_tensor_if_needed(self, arr):
        if self.policy.is_tensorflow:
            return tf.convert_to_tensor(arr)
        else:
            return arr