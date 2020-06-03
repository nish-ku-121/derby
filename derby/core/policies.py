from abc import ABC, abstractmethod
import numpy as np
from derby.core.environments import AbstractEnvironment
import os
import tensorflow as tf
from derby.core.basic_structures import Bid
from derby.core.states import CampaignBidderState

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class AbstractPolicy(ABC):

    def __init__(self, agent=None, is_tensorflow=False, discount_factor=.99):
        super().__init__()
# TODO: Github issue #29.
# remove/replace this so that agents and policies don't point to each other
        self.agent = agent # assuming Agent class sets this
#
        self.is_tensorflow = is_tensorflow # used by Agent class to send input as tf.tensor
        self.discount_factor = discount_factor

    @abstractmethod
    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    @abstractmethod
    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    @abstractmethod
    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    @abstractmethod
    def call(self, states):
        '''
        :param states: an array of shape [batch_size, episode_length, ...], where 
        "..." means the array is assumed to be folded according to states_fold_type().
        :return: an array of shape [batch_size, episode_length, <other dims>] representing 
        an intermediary (e.g. logits, probs, etc.) which can be used by choose_actions() 
        to determine actual actions to take.
        '''
        pass

    @abstractmethod
    def choose_actions(self, call_output):
        '''
        :param call_output: the output of call().
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        pass

    @abstractmethod
    def loss(self, states, actions, rewards):
        '''
        Computes the policy's loss.
        :param states: an array of shape [batch_size, episode_length, ...], where 
        "..." means the array is assumed to be folded according to states_fold_type().
        :param actions: an array of shape [batch_size, episode_length-1, ...], where 
        "..." means the array is assumed to be folded according to actions_fold_type().
        :param rewards: an array of shape [batch_size, episode_length-1, ...], where 
        "..." means the array is assumed to be folded according to rewards_fold_type().
        :return: the policy's loss.
        '''
        pass

    @abstractmethod
    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        '''
        Updates the policy.
        :param states: an array of shape [batch_size, episode_length, ...], where 
        "..." means the array is assumed to be folded according to states_fold_type().
        :param actions: an array of shape [batch_size, episode_length-1, ...], where 
        "..." means the array is assumed to be folded according to actions_fold_type().
        :param rewards: an array of shape [batch_size, episode_length-1, ...], where 
        "..." means the array is assumed to be folded according to rewards_fold_type().
        :param policy_loss: The loss of the policy (if tf policy, then assuming the loss 
        is computed under tf.GradientTape()).
        '''
        pass

    def discount(self, rewards):
        '''
        :param rewards: an array of shape [batch_size, episode_length-1].
        :return: discounted_rewards: an array of shape [batch_size, episode_length-1] 
        containing the sum of discounted rewards for each timestep.
        '''
        if type(rewards) is np.ndarray or type(rewards) is list:
            rewards = tf.convert_to_tensor(rewards)
        if tf.is_tensor(rewards):
            return tf.map_fn(self.discount_helper, rewards)
        else:
            raise Exception("Don't know how to discount rewards where type(rewards) = {}".format(type(rewards)))

    def discount_helper(self, rewards):
        '''
        Takes in a list of rewards for each timestep in an episode, 
        and returns a list of the sum of discounted rewards for
        each timestep.

        :param rewards: List of rewards from an episode [r_{t0},..., r_{tN-1}]. 
        shape is [episode_length-1].
        :param discount_factor: Gamma discounting factor to use, defaults to .99.
        :return: discounted_rewards: list containing the sum of discounted 
        rewards for each timestep in the original rewards list.
        '''
        discount_factor = self.discount_factor
        timesteps = len(rewards)
        discounted_rewards = np.zeros(timesteps)
        discounted_rewards[timesteps-1] = rewards[timesteps-1]
        for i in range(timesteps-2,-1,-1):
            discounted_rewards[i] = (discounted_rewards[i+1]*discount_factor) + rewards[i]
        return discounted_rewards


class DummyPolicy1(AbstractPolicy):

    def __init__(self, auction_item_spec, bid_per_item, total_limit):
        super().__init__()
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE
    
    def call(self, states):
        actions = [] 
        for i in range(states.shape[0]):
            actions_i = []
            for j in range(states.shape[1]):
                action = [ [self.auction_item_spec.uid, self.bid_per_item, self.total_limit] ]
                actions_i.append(action)
            actions.append(actions_i)
        actions = np.array(actions)
        return actions

    def choose_actions(self, call_output):
        return call_output

    def loss(self, states, actions, rewards):
        return 0


class DummyPolicy2(AbstractPolicy):

    def __init__(self, auction_item_spec, bid_per_item, total_limit):
        super().__init__()
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE
    
    def call(self, states):
        agent = self.agent
        auction_item_spec = self.auction_item_spec
        bpi = self.bid_per_item
        lim = self.total_limit
        actions = [] 
        for i in range(states.shape[0]):
            actions_i = []
            for j in range(states.shape[1]):
                action = [ Bid(agent, auction_item_spec, bid_per_item=bpi, total_limit=lim) ]
                actions_i.append(action)
            actions.append(actions_i)
        actions = np.array(actions)
        return actions

    def choose_actions(self, call_output):
        return call_output

    def loss(self, states, actions, rewards):
        return 0


class BudgetPerReachPolicy(AbstractPolicy):

    def __init__(self):
        super().__init__()

    def states_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def call(self, states):
        '''
        :param states: an array of shape [batch_size, episode_length, state_size].
        :return: array of shape [batch_size, episode_length] representing the actions to take.
        '''
        actions = []
        for i in range(states.shape[0]):
            actions_i = []
            for j in range(states.shape[1]):
                state_i_j = states[i][j]
                if isinstance(state_i_j, CampaignBidderState):
                    bpr = state_i_j.campaign.budget / (1.0 * state_i_j.campaign.reach)
                    if state_i_j.impressions >= state_i_j.campaign.reach:
                        bpr = 0.0
                    auction_item_spec = state_i_j.campaign.target
                    lim = state_i_j.campaign.budget
                    action = [ Bid(self.agent, auction_item_spec, bid_per_item=bpr, total_limit=lim) ]
                else:
                    reach = state_i_j[0]
                    budget = state_i_j[1]
                    auction_item_spec = state_i_j[2]
                    impressions = state_i_j[4]
                    bpr = budget / (1.0 * reach)
                    if (impressions >= reach):
                        bpr = 0.0
                    total_limit = state_i_j[1]
                    action = [ [auction_item_spec, bpr, total_limit] ]
                actions_i.append(action)
            actions.append(actions_i)
        actions = np.array(actions)
        return actions

    def choose_actions(self, call_output):
        return call_output

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: an array of shape [batch_size, episode_length, state_size].
        :param actions: an array of shape [batch_size, episode_length].
        :param rewards: an array of shape [batch_size, episode_length].
        '''
        return 0


'''
Make sure tf.keras.Model comes last so that it plays nice with multiple inheritance.
See: https://stackoverflow.com/questions/9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
tf.keras.Model __init__ does have a super().__init__ call, but still plays poorly with
multiple inheritance for some reason. Tried implementing an Adapter for it, but it didn't
work.
'''
class DummyREINFORCE(AbstractPolicy, tf.keras.Model):

    def __init__(self, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.choices = [
            [ [1, 1.0, 1.0] ],
            [ [2, 1.0, 1.0] ]
        ]

        # Network parameters and optimizer
        self.num_actions = 2
        self.layer1_size = 1 #50
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0., maxval=0.25)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.num_actions, use_bias=False, kernel_initializer=self.layer2_ker_init, activation='softmax', dtype='float64')

    def states_fold_type(self):
        if self.is_partial:
            return AbstractEnvironment.FOLD_TYPE_SINGLE
        else:
            return AbstractEnvironment.FOLD_TYPE_ALL

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def call(self, states):
        '''
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length, num_actions] matrix representing the probability 
        distribution over actions of each state in the episode.
        '''

        output = self.dense1(states)
# DEBUG
#        print("interm: {}".format(output))
#
        output = self.dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param actions: an array of shape [batch_size, episode_length, num_actions] (output of call func).
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        
        # sample_action_indices = tf.argmax(call_output, axis=2)
        # chosen_actions = tf.gather(self.choices, sample_action_indices)

        sample_action_indices = [tf.random.categorical(tf.math.log(call_output[:,i]), 1) 
                                 for i in range(call_output.shape[1])]
        chosen_actions = tf.gather(self.choices, tf.concat(sample_action_indices, axis=1))

# # DEBUG
#        print(call_output)
#         print(sample_action_indices)
#         print(chosen_actions)
# #
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        action_distr = self.call(states)
        zipped = [tup for tup in zip(range(len(self.choices)), self.choices)]
        action_choices = []
        for tup in zipped:
            action_choices.append(
                tf.where(tf.reduce_all(actions == tup[1], axis=-1), tup[0], 0)
            )
        action_choices = tf.add(*action_choices)
        # Use gather_nd to get the probability of each action that was actually taken in the episode.
        action_prbs = tf.gather_nd(action_distr[:,:-1], action_choices, batch_dims=2)
        discounted_rewards = self.discount(rewards)
        neg_logs = -tf.math.log(action_prbs)
        losses = neg_logs * discounted_rewards
        total_loss = tf.reduce_sum(losses)
# DEBUG
        # # print(self.choices)
        # print(action_distr)
        # print(states)
        # print(actions)
        # # print(action_choices)
        # # print(action_prbs)
        # # print(rewards)
        # print(discounted_rewards)
        # # print(neg_logs)
        # # print(losses)
        # print(total_loss)
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))