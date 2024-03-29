from abc import ABC, abstractmethod
import math
import numpy as np
from derby.core.basic_structures import Bid
from derby.core.states import CampaignBidderState
from derby.core.environments import AbstractEnvironment
import os
import tensorflow as tf
import tensorflow_probability as tfp
# DEBUG
import matplotlib.pyplot as plt
#

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


class FixedBidPolicy(AbstractPolicy):

    def __init__(self, bid_per_item, total_limit, auction_item_spec=None):
        super().__init__()
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit
        self.auction_item_spec = auction_item_spec

    def __repr__(self):
        return "{}(bid_per_item: {}, total_limit: {})".format(self.__class__.__name__, 
                                                              self.bid_per_item, self.total_limit)

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
                state_i_j = states[i][j]
                if isinstance(state_i_j, CampaignBidderState):
                    if (self.auction_item_spec is None):
                        auction_item_spec = state_i_j.campaign.target
                    else:
                        auction_item_spec = self.auction_item_spec
                    action = [ Bid(self.agent, auction_item_spec, self.bid_per_item, self.total_limit) ]
                else:
                    if (self.auction_item_spec is None):
                        spec_id = state_i_j[2]
                    else:
                        spec_id = self.auction_item_spec.uid
                    action = [ [spec_id, self.bid_per_item, self.total_limit] ]
                actions_i.append(action)
            actions.append(actions_i)
        actions = np.array(actions)
        return actions

    def choose_actions(self, call_output):
        return call_output

    def loss(self, states, actions, rewards):
        return 0

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        pass


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

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        pass


class StepPolicy(AbstractPolicy):

    def __init__(self, start_bid, step_per_day):
        super().__init__()
        self.start_bid = start_bid
        self.step_per_day = step_per_day

    def __repr__(self):
        return "{}(start_bid: {}, step_per_day: {})".format(self.__class__.__name__, 
                                                   self.start_bid, self.step_per_day)

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
                    day = self.timestep
                    bpr = self.start_bid + day*self.step_per_day
                    if state_i_j.impressions >= state_i_j.campaign.reach:
                        bpr = 0.0
                    auction_item_spec = state_i_j.campaign.target
                    action = [ Bid(self.agent, auction_item_spec, bid_per_item=bpr, total_limit=bpr) ]
                else:
                    reach = state_i_j[0]
                    budget = state_i_j[1]
                    auction_item_spec = state_i_j[2]
                    spend = state_i_j[3]
                    impressions = state_i_j[4]
                    day = state_i_j[5]
                    bpr = self.start_bid + day*self.step_per_day
                    if (impressions >= reach):
                        bpr = 0.0
                    action = [ [auction_item_spec, bpr, bpr] ]
                actions_i.append(action)
            actions.append(actions_i)
        actions = np.array(actions)
        return actions

    def choose_actions(self, call_output):
        return call_output

    def loss(self, states, actions, rewards):
        return 0

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        pass


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
        output = self.dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        
        # sample_action_indices = tf.argmax(call_output, axis=2)
        # chosen_actions = tf.gather(self.choices, sample_action_indices)

        sample_action_indices = [tf.random.categorical(tf.math.log(call_output[:,i]), 1) 
                                 for i in range(call_output.shape[1])]
        chosen_actions = tf.gather(self.choices, tf.concat(sample_action_indices, axis=1))
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
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
        print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        

        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
        print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_v3_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.elu, dtype='float64')
        

        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
        print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Gaussian_v4_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = self.budget_per_reach
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus+offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.relu(output_sigmas+offset) + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Tabu_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
# DEBUG
        self.resample_count = 0
#
        self.tabu_layer1_size = 1
        self.tabu_layer1_ker_init = None
        self.tabu_mu_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.tabu_sigma_ker_init = None
        self.tabu_dense1 = tf.keras.layers.Dense(self.tabu_layer1_size, kernel_initializer=self.tabu_layer1_ker_init, activation=None, dtype='float64')
        self.tabu_mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.tabu_mu_ker_init, activation=None, dtype='float64')
        self.tabu_sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.tabu_sigma_ker_init, activation=None, dtype='float64')
        
    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: Two distributions: the first represents the policy network distribution
        and the second represents the tabu network distribution.
        '''
        policy_dist = self.policy_call(states)
        tabu_dist = self.tabu_call(states)
        return policy_dist, tabu_dist

    def policy_call(self, states):
        '''
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def tabu_call(self, states):     
        '''
        :param action_samples: An array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction].
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents the probability that an action should be rejected (i.e. tabued).
        '''
        # 
        # Apply dense layers
        output = self.tabu_dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.tabu_mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.tabu_sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus)
        # variance needs to be a positive number.
        output_sigmas = self.budget_per_reach*tf.nn.softplus(output_sigmas)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        policy_dist = call_output[0]
        tabu_dist = call_output[1]

        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = policy_dist.sample()
        
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction], 
        # representing the probability of tabuing (i.e. rejecting) each sample.
        tabu_prbs = tabu_dist.prob(samples)
        # For now, let's only tabu based on bid_per_item (assuming first column).
        tabu_prbs = tabu_dist.prob(samples)[:,:,:,0:1]
        # With probability tabu_prbs, resample each sample.
        rand_generator = tf.random.get_global_generator()
        rands = rand_generator.uniform(tabu_prbs.shape, dtype=tabu_prbs.dtype)
        should_resample = rands < tabu_prbs
# DEBUG
        # print("rands: {}, tabu_prbs: {}".format(rands, tabu_prbs))
        # print("should_resample: {}".format(should_resample))
        self.resample_count += tf.math.count_nonzero(should_resample)
#
        samples = tf.where(should_resample, policy_dist.sample(), samples)
        # Clip the samples as needed.
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr, tabu_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        tabu_prbs = tf.reduce_prod(tf.reduce_prod(tabu_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards) 
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        # Calculate tabu loss for batches which got 0 cumulative reward.
        # Let tabu loss = \sum_i \sum_j \grad \log tabu_prbs, for batch i and timestep j.
        cuml_disc_rwds = tf.reduce_sum(discounted_rewards, axis=1)
        tabu_neg_logs = -tf.math.log(tabu_prbs)
        tabu_neg_logs = tf.clip_by_value(tabu_neg_logs, -1e9, 1e9)
        tabu_loss = tf.reduce_sum(tf.where(cuml_disc_rwds == 0, tf.reduce_sum(tabu_neg_logs, axis=1), 0.), axis=0)
# DEBUG
        # # print("weights:\n{}".format(self.trainable_variables))
        # # print("actions:\n{}".format(actions))
        print("resample_count: {}".format(self.resample_count))
        print("avg. action_distr loc:\n{}".format(tf.reduce_mean(action_distr.loc, axis=0)))
        print("avg. tabu_distr loc:\n{}".format(tf.reduce_mean(tabu_distr.loc, axis=0)))
        print("avg. tabu_distr scale:\n{}".format(tf.reduce_mean(tabu_distr.scale, axis=0)))
        # # print("cuml_disc_rwds:\n{}".format(cuml_disc_rwds))
        # # print("tabu_neg_logs:\n{}".format(tabu_neg_logs))
        # print("total_loss: {}".format(total_loss))
        print("tabu_loss: {}".format(tabu_loss))
#
        return (1.0*total_loss) + (10.0*tabu_loss)

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Tabu_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
# DEBUG
        self.resample_count = 0
#
        self.tabu_layer1_size = 1
        self.tabu_layer1_ker_init = None
        self.tabu_dense1 = tf.keras.layers.Dense(self.tabu_layer1_size, kernel_initializer=self.tabu_layer1_ker_init, activation=None, dtype='float64')
        self.tabu_low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, activation=None, dtype='float64')
        self.tabu_offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, activation=None, dtype='float64')
        self.tabu_offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, activation=None, dtype='float64')


    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: Two distributions: the first represents the policy network distribution
        and the second represents the tabu network distribution.
        '''
        policy_dist = self.policy_call(states)
        tabu_dist = self.tabu_call(states)
        return policy_dist, tabu_dist

    def policy_call(self, states):
        '''
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def tabu_call(self, states):     
        '''
        :param action_samples: An array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction].
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents the probability that an action should be rejected (i.e. tabued).
        '''
        # 
        # Apply dense layers
        output = self.tabu_dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.tabu_low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.tabu_offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.tabu_offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.relu(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        policy_dist = call_output[0]
        tabu_dist = call_output[1]

        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = policy_dist.sample()
        
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction], 
        # representing the probability of tabuing (i.e. rejecting) each sample.
        tabu_prbs = tabu_dist.prob(samples)
        # For now, let's only tabu based on bid_per_item (assuming first column).
        tabu_prbs = tabu_dist.prob(samples)[:,:,:,0:1]
        # With probability tabu_prbs, resample each sample.
        rand_generator = tf.random.get_global_generator()
        should_resample = rand_generator.uniform(tabu_prbs.shape, dtype=tabu_prbs.dtype) < tabu_prbs
# DEBUG
        # print("tabu_prbs: {}".format(tabu_prbs))
        self.resample_count += tf.math.count_nonzero(should_resample)
#
        samples = tf.where(should_resample, policy_dist.sample(), samples)
        # Clip the samples as needed.
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr, tabu_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        tabu_prbs = tf.reduce_prod(tf.reduce_prod(tabu_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards) 
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        # Calculate tabu loss for batches which got 0 cumulative reward.
        # Let tabu loss = \sum_i \sum_j \grad \log tabu_prbs, for batch i and timestep j.
        cuml_disc_rwds = tf.reduce_sum(discounted_rewards, axis=1)
        tabu_neg_logs = -tf.math.log(tabu_prbs)
        tabu_neg_logs = tf.clip_by_value(tabu_neg_logs, -1e9, 1e9)
        tabu_loss = tf.reduce_sum(tf.where(cuml_disc_rwds == 0, tf.reduce_sum(tabu_neg_logs, axis=1), 0.), axis=0)
# DEBUG
        # # print("weights:\n{}".format(self.trainable_variables))
        # # print("actions:\n{}".format(actions))
        print("resample_count: {}".format(self.resample_count))
        print("avg. action_distr loc:\n{}".format(tf.reduce_mean(action_distr.loc, axis=0)))
        print("avg. tabu_distr low:\n{}".format(tf.reduce_mean(tabu_distr.low, axis=0)))
        print("avg. tabu_distr peak:\n{}".format(tf.reduce_mean(tabu_distr.peak, axis=0)))
        print("avg. tabu_distr high:\n{}".format(tf.reduce_mean(tabu_distr.high, axis=0)))
        # # print("cuml_disc_rwds:\n{}".format(cuml_disc_rwds))
        # # print("tabu_neg_logs:\n{}".format(tabu_neg_logs))
        # print("total_loss: {}".format(total_loss))
        print("tabu_loss: {}".format(tabu_loss))
#
        return (1.0*total_loss) + (10.0*tabu_loss)

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Tabu_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, 
                        epsilon=0.9, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward
        self.epsilon = epsilon # pick best action w/ prob 1-epsilon
        self.epsilon_min = 0.0
        self.epsilon_max = 1.0
        self.temperature = 100.0
        self.update_count = 1

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        # Policy network \pi(a|s)
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))
        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
# DEBUG
        self.sample_from_tabu_count = 0
#
        # Tabu network represents "elite list", 
        # i.e. actions to take because they lead to positive reward 
        # (as discovered via a past trajectory).
        # In other words, this tabu is the opposite of "reject".
        # However, we don't trust tabu network in the beginning, so 
        # we increase our confidence in it over time (via epsilon).
        # Tabu policy network
        self.tabu_layer1_size = 1
        self.tabu_layer1_ker_init = None
        self.tabu_dense1 = tf.keras.layers.Dense(self.tabu_layer1_size, kernel_initializer=self.tabu_layer1_ker_init, activation=None, dtype='float64')
        
        self.tabu_mu_ker_init = None
        self.tabu_sigma_ker_init = None
        self.tabu_mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.tabu_mu_ker_init, activation=None, dtype='float64')
        self.tabu_sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.tabu_sigma_ker_init, activation=None, dtype='float64')
        
    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, eps: {}, temp: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(
                                                                    self.__class__.__name__, 
                                                                    self.is_partial, self.discount_factor, 
                                                                    self.learning_rate, self.epsilon,
                                                                    self.temperature, self.num_subactions,
                                                                    type(self.optimizer).__name__, self.shape_reward)
    
    def states_fold_type(self):
        if self.is_partial:
            return AbstractEnvironment.FOLD_TYPE_SINGLE
        else:
            return AbstractEnvironment.FOLD_TYPE_ALL

    def actions_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def rewards_fold_type(self):
        return AbstractEnvironment.FOLD_TYPE_SINGLE

    def calc_iter_epsilon(self):
        return tf.clip_by_value(self.epsilon / (1.0 + (self.update_count / self.temperature)),
                                self.epsilon_min, self.epsilon_max)

    def call(self, states):     
        '''
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :return: Two distributions: the first represents the policy network distribution
        and the second represents the tabu network distribution.
        '''
        policy_dist = self.policy_call(states)
        tabu_dist = self.tabu_call(states)
        return policy_dist, tabu_dist

    def policy_call(self, states):
        '''
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def tabu_call(self, states):     
        '''
        :param action_samples: An array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction].
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents the probability that an action should be rejected (i.e. tabued).
        '''
        # 
        # Apply dense layers
        output = self.tabu_dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.tabu_mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.tabu_sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        policy_dist = call_output[0]
        tabu_dist = call_output[1]

        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = policy_dist.sample()
        
        # With probability 1-epsilon, choose tabu sample (i.e. be greedy)
        rand_generator = tf.random.get_global_generator()
        rands = rand_generator.uniform(samples.shape[0:2], dtype=tf.float64)
        # Shape [batch_size, episode_length]
        should_sample_from_tabu = rands < float(1 - self.calc_iter_epsilon())
# DEBUG
        # print("rands: {}, tabu_prbs: {}".format(rands, tabu_prbs))
        # print("should_resample: {}".format(should_resample))
        self.sample_from_tabu_count += tf.math.count_nonzero(should_sample_from_tabu)
#
        samples = tf.where(should_sample_from_tabu[:,:,None,None], tabu_dist.sample(), samples)
        # Clip the samples as needed.
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr, tabu_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        tabu_prbs = tf.reduce_prod(tf.reduce_prod(tabu_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards) 
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -10, 10)
        losses = neg_logs * advantage
        policy_loss = tf.reduce_sum(losses)
        # Calculate tabu loss for batches which got > 0 cumulative reward.
        # Let tabu loss = \sum_i \sum_j \grad \log tabu_prbs * advtg, for batch i and timestep j.
        cuml_disc_rwds = tf.reduce_sum(discounted_rewards, axis=1)
        tabu_neg_logs = -tf.math.log(tabu_prbs)
        tabu_neg_logs = tf.clip_by_value(tabu_neg_logs, -10, 10)
        tabu_losses = tabu_neg_logs * advantage
        tabu_loss = tf.reduce_sum(tf.where(cuml_disc_rwds > 0, tf.reduce_sum(tabu_losses, axis=1), 0.), axis=0)
# DEBUG
        # # print("weights:\n{}".format(self.trainable_variables))
        # # print("actions:\n{}".format(actions))
        print("iter_epsilon: {}".format(self.calc_iter_epsilon()))
        print("sample_from_tabu_count: {}".format(self.sample_from_tabu_count))
        print("avg. action_distr loc:\n{}".format(tf.reduce_mean(action_distr.loc, axis=0)))
        print("avg. action_distr scale:\n{}".format(tf.reduce_mean(action_distr.scale, axis=0)))
        print("avg. tabu_distr loc:\n{}".format(tf.reduce_mean(tabu_distr.loc, axis=0)))
        print("avg. tabu_distr scale:\n{}".format(tf.reduce_mean(tabu_distr.scale, axis=0)))
        # # print("cuml_disc_rwds:\n{}".format(cuml_disc_rwds))
        # # print("tabu_neg_logs:\n{}".format(tabu_neg_logs))
        print("policy_loss: {}".format(policy_loss))
        print("tabu_loss: {}".format(tabu_loss))
#
        return (1.0*policy_loss) + (0.8*tabu_loss)

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        self.update_count += 1
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Uniform_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Uniform(a|low(s),high(s)) 
        #                                 = \prod_j \prod_k Uniform(sub_a_j_dist_k|low(s),high(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        
    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low and high layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make highs = lows + offsets, so that they are always higher than lows.
        output_highs = output_lows + output_offsets

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Uniform(low=output_lows, high=output_highs)

        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)

        return dist

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        baseline = 0
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * advantage
        total_loss = tf.reduce_sum(losses)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        advantage = discounted_rewards - state_values[:,:-1]

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
       
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        baseline = state_values[:,:-1]
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
       
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        baseline = state_values[:,:-1]
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec
  
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.elu, dtype='float64')
        
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
       
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        baseline = state_values[:,:-1]
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = self.budget_per_reach
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus+offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.relu(output_sigmas+offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
       
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        baseline = state_values[:,:-1]
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)

        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        advantage = discounted_rewards - state_values[:,:-1]

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class REINFORCE_Baseline_LogNormal_v3_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec
  
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.elu, dtype='float64')
        
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = LogNormal(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k LogNormal(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = tf.math.sqrt(-tf.math.log(self.budget_per_reach))
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = output_mus-offset
        # variance needs to be a positive number.
        output_sigmas = tf.nn.softplus(output_sigmas)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.LogNormal. Otherwise it will throw an error.
        dist = tfp.distributions.LogNormal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
       
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        baseline = state_values[:,:-1]
        advantage = discounted_rewards - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("avg. action:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("avg. loc:\n{}".format(tf.reduce_mean(action_distr.loc, axis=0)))
        print("avg. scale:\n{}".format(tf.reduce_mean(action_distr.scale, axis=0)))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_TD_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        targets = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        advantage = targets - state_values[:,:-1]

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            
class AC_TD_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        targets = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        baseline = state_values[:,:-1]
        # shape [batch_size, episode_length-1]
        advantage = targets - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((targets - baseline)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_TD_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        targets = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        baseline = state_values[:,:-1]
        # shape [batch_size, episode_length-1]
        advantage = targets - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((targets - baseline)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_TD_Gaussian_v3_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)

        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.elu, dtype='float64')
        
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        targets = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        baseline = state_values[:,:-1]
        # shape [batch_size, episode_length-1]
        advantage = targets - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((targets - baseline)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_TD_Gaussian_v4_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = self.budget_per_reach
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus+offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.relu(output_sigmas+offset) + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # Only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):

        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1]

        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        targets = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        baseline = state_values[:,:-1]
        # shape [batch_size, episode_length-1]
        advantage = targets - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((targets - baseline)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_TD_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)

        return dist

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.critic_dense1(states)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)
      
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))
        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * state_values[:,1:])
        # shape [batch_size, episode_length-1]
        advantage = target - state_values[:,:-1]

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((advantage)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        
        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        advantage = q_state_values

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15
# DEBUG
        self.plot_count = 0
#

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.sigma_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
 
        self.critic_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(self.critic_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense3 = tf.keras.layers.Dense(self.critic_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense4 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        output = self.critic_dense3(output)
        output = self.critic_dense4(output)
        
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = q_state_values - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        critic_lr_mult = 1e2
        total_loss = (1.0*actor_loss) + (critic_lr_mult*critic_loss)
# # DEBUG
#         if debug:
#             all_bids = subaction_dists_vals[:,:,0,0]
#             avg_bid = tf.reduce_mean(all_bids, axis=0)
#             lower_bid_prbs = tf.where(all_bids <= avg_bid, -neg_logs, 0.0)
#             higher_bid_prbs = tf.where(all_bids > avg_bid, -neg_logs, 0.0)
#             lower_rwds = lower_bid_prbs * discounted_rewards
#             higher_rwds = higher_bid_prbs * discounted_rewards

#             # print("all bids:\n{}".format(all_bids))
#             # print("avg_bid:\n{}".format(avg_bid))
#             # print("prbs:\n{}".format(action_prbs))
#             # print("lower_bid_prbs:\n{}".format(lower_bid_prbs))
#             # print("higher_bid_prbs:\n{}".format(higher_bid_prbs))
#             # print("disc. rwds:\n{}".format(discounted_rewards))
# # DEBUG
#             divisor = 50
#             def plotter(fig, axs, bids, rwds, q_vals, iter, loss):
#                 plt.subplots_adjust(
#                     left  = 0.15,  # the left side of the subplots of the figure
#                     right = 0.95,    # the right side of the subplots of the figure
#                     bottom = 0.15,   # the bottom of the subplots of the figure
#                     top = 0.85,      # the top of the subplots of the figure
#                     wspace = 0.3,   # the amount of width reserved for blank space between subplots
#                     hspace = 0.5 
#                 )
#                 axs[0].scatter(bids, q_vals, c='purple')
#                 axs[0].scatter(bids, rwds, c='red')
#                 axs[0].set(
#                             title="Network Q-Values",
#                             xlabel="Bid", 
#                             ylabel="Q Value",
#                             xlim=[0.0,0.3],
#                             ylim=[-10,10],
#                             xticks=np.arange(0.0, 0.3, 0.02),
#                             yticks=np.arange(-10, 10, 2)
#                         )

#                 axs[1].scatter(iter, loss, color='blue')
#                 axs[1].set(
#                             title="Network Loss",
#                             xlabel="Iteration",
#                             ylabel="Critic Loss",
#                             ylim=[-500,5000],
#                             xticks=[iter],
#                             yticks=np.arange(-500, 5000, 500)
#                         )
#             if ((self.plot_count % divisor) == 0):
#                 fig, axs = plt.subplots(2)
#                 plotter(fig, axs, all_bids[:,0], discounted_rewards[:,0], 
#                     q_state_values[:,0], self.plot_count, critic_loss)
#                 plt.savefig('q_figs/run_{}__iter_{}__lr_{}__critic_lr_mult_{}__q_vals.png'.format(
#                                     id(self), self.plot_count, self.learning_rate, critic_lr_mult))
#                 plt.close(fig)
#             self.plot_count += 1
# #
#             # print("avg. disc. rwds:\n{}".format(tf.reduce_mean(self.discount(rewards), axis=0)))
#             # print("avg. shaped disc. rwds:\n{}".format(tf.reduce_mean(discounted_rewards, axis=0)))
#             # print("avg. q_vals:\n{}".format(tf.reduce_mean(q_state_values, axis=0)))
#             print("5 disc. rwds:\n{}".format(self.discount(rewards)[:5]))
#             print("5 q_vals:\n{}".format(q_state_values[:5]))
#             print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
#             print("exp. lower rwd (disc.), lower rwd (advtg.):\n{}, {}".format(tf.reduce_sum(lower_rwds), tf.reduce_sum(lower_bid_prbs * advantage)))
#             print("exp. higher rwd (disc.), higher rwd (advtg.):\n{}, {}".format(tf.reduce_sum(higher_rwds), tf.reduce_sum(higher_bid_prbs * advantage)))
#             # # print("subaction_dists_vals:\n{}".format(subaction_dists_vals))
#             # # print("neg_logs:\n{}".format(neg_logs))
#             print("avg. loss:\n{}".format(tf.reduce_mean(losses, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
# #
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
 
        self.critic_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(self.critic_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense3 = tf.keras.layers.Dense(self.critic_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense4 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        output = self.critic_dense3(output)
        output = self.critic_dense4(output)
        
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = q_state_values - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        critic_lr_mult = 1e2
        total_loss = (1.0*actor_loss) + (critic_lr_mult*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Gaussian_v4_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
 
        self.critic_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(self.critic_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense3 = tf.keras.layers.Dense(self.critic_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense4 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = self.budget_per_reach
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus+offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.relu(output_sigmas+offset) + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        output = self.critic_dense3(output)
        output = self.critic_dense4(output)
        
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = q_state_values - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        critic_lr_mult = 1e2
        total_loss = (1.0*actor_loss) + (critic_lr_mult*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Baseline_V_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1)) 
        baseline = state_values[:,:-1]
        advantage = q_state_values - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("baseline_loss: {}".format(baseline_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Baseline_V_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1)) 
        baseline = state_values[:,:-1]
        advantage = q_state_values - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Fourier_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15
# DEBUG
        self.plot_count = 0
#

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.sigma_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
 
        
        # self.critic_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        # self.critic_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        # self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        # self.critic_dense2 = tf.keras.layers.Dense(self.critic_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        # self.critic_dense3 = tf.keras.layers.Dense(self.critic_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        # self.critic_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # self.critic_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.c = None
        self.W1 = None
        # self.b1 = tf.Variable(tf.random.truncated_normal(shape=[self.critic_layer1_size], stddev=0.1, dtype=tf.float32))
        

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def repeated_cart_prod(self, arr, n):
        if n <= 0:
            raise Exception("n cannot be <= 0!")
        if n == 1:
            return tf.constant(arr)
        else:
            # Using code from SOURCE.
            # SOURCE: https://stackoverflow.com/questions/48204382/creating-all-possible-combinations-from-vectors-in-tensorflow
            tile_a = tf.tile(tf.expand_dims(arr, 1), [1, tf.shape(arr)[0]])
            tile_a = tf.expand_dims(tile_a, 2)
            tile_b = tf.tile(tf.expand_dims(arr, 0), [tf.shape(arr)[0], 1])
            tile_b = tf.expand_dims(tile_b, 2)
            cart = tf.concat([tile_a, tile_b], axis=2)
            cart = tf.reshape(cart,[-1,2])
            i = 2
            while(i < n):
                tile_c = tf.tile(tf.expand_dims(arr, 1), [1, tf.shape(cart)[0]])
                tile_c = tf.expand_dims(tile_c, 2)
                tile_c = tf.reshape(tile_c, [-1,1])
                cart = tf.tile(cart,[tf.shape(arr)[0],1])
                cart = tf.concat([cart, tile_c], axis=1)
                i += 1
            return cart

    # def basis(self, n, x):
    #     """
    #     :param n: The order of the approximation.
    #     :param x: An array of shape [batch_size, episode_length, x_dim].
    #     :return: An array of shape [batch_size, episode_length, (n+1)^d], where each row of (n+1)^d
    #     rows represents a basis function \phi_i(x).
    #     """
    #     # An array of shape [(n+1)^d, d], where d is x_dim.
    #     c = self.repeated_cart_prod(tf.constant(range(n+1)), x.shape[-1])
    #     # Broadcasting to array of shape [batch_size, episode_length, (n+1)^d, d].
    #     c = tf.broadcast_to(c, [*x.shape[:2]] + [*c.shape])
    #     c = tf.cast(c, x.dtype)
    #     # Reshape to [batch_size, episode_length, x_dim, 1]. Note: x_dim = d.
    #     x = x[:,:,:,None]
    #     # Matrix multiply in order to compute c^i \dot x, for each row c^i in c.
    #     # Shape is [batch_size, episode_length, (n+1)^d, 1].
    #     dot_prds = tf.matmul(c, x)
    #     # Reshape to [batch_size, episode_length, (n+1)^d].
    #     dot_prds = tf.reshape(dot_prds, dot_prds.shape[:-1])
    #     # Shape [batch_size, episode_length, (n+1)^d].
    #     bases = tf.math.cos(tf.constant(math.pi, dtype=dot_prds.dtype) * dot_prds)
    #     return bases

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length-1] matrix representing the value of each q-state.
        """
        # Array of shape [batch_size, episode_length-1, new_state_size + (num_subactions * subaction_size)].
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
        n = 1
        if (self.c is None):
            # An array of shape [(n+1)^d, d], where d = new_state_size + (num_subactions * subaction_size).
            self.c = self.repeated_cart_prod(tf.constant(range(n+1)), inputs.shape[-1])
            self.c = tf.cast(self.c, inputs.dtype)

        if (self.W1 is None):
            # Shape [(n+1)^d, 1].
            self.W1 = tf.Variable(tf.random.truncated_normal(shape=[self.c.shape[0], 1], stddev=0.1, dtype=inputs.dtype))

        # t is of shape [episode_length-1, d].
        # transposed c is of shape [d, (n+1)^d].
        # matmul of t and transposed c is of shape [episode_length-1, (n+1)^d].
        # matmul of result and W1 is of shape [episode_length-1, 1].
        # map_fn repeats this for every batch, so final shape is [batch_size, episode_length-1, 1]
        output = tf.map_fn(fn=lambda t: tf.matmul(
                                    tf.math.cos(tf.constant(math.pi, dtype=inputs.dtype) * tf.matmul(t, self.c, transpose_b=True)), 
                                    self.W1), 
                            elems=inputs)
        # Reshape to [batch_size, episode_length-1].
        output = tf.reshape(output, (*output.shape[:1],-1))
        return output

    # def q_value_function(self, states, actions):
    #     """
    #     Performs the forward pass on a batch of states to calculate the value function, to be used as the
    #     critic in the loss function.

    #     :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
    #     is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
    #     :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
    #     :return: A [batch_size, episode_length-1] matrix representing the value of each q-state.
    #     """
    #     # Array of shape [batch_size, episode_length-1, new_state_size + (num_subactions * subaction_size)].
    #     inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
    #     n = 1
    #     # Shape [batch_size, episode_length-1, (n+1)^d], where d = inputs.shape[-1].
    #     bases = self.basis(n, inputs)

    #     if (self.W1 is None):
    #         # Shape [(n+1)^d, 1].
    #         self.W1 = tf.Variable(tf.random.truncated_normal(shape=[bases.shape[-1], 1], stddev=0.1, dtype=bases.dtype))

    #     # Reshape to [batch_size, episode_length-1, (n+1)^d, 1]
    #     batch_W1 = tf.broadcast_to(self.W1, [*bases.shape[:2]] + [*self.W1.shape])
    #     # Shape [batch_size, episode_length-1, 1, (n+1)^d].
    #     bases = bases[:,:,None,:]
    #     # Shape [batch_size, episode_length-1, 1, 1].
    #     output = tf.matmul(bases, batch_W1)
    #     # Shape [batch_size, episode_length-1].
    #     output = tf.reshape(output, [*output.shape[:2]])
    #     return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        discounted_rewards = self.discount(rewards)
        if self.shape_reward:
            discounted_rewards = tf.where(discounted_rewards > 0, tf.math.log(discounted_rewards+1), discounted_rewards)
        baseline = 0
        advantage = q_state_values - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        critic_lr_mult = 0.05
        total_loss = (1.0*actor_loss) + (critic_lr_mult*critic_loss)
# # DEBUG
        debug = False
        if debug:
            all_bids = subaction_dists_vals[:,:,0,0]
            avg_bid = tf.reduce_mean(all_bids, axis=0)
            lower_bid_prbs = tf.where(all_bids <= avg_bid, -neg_logs, 0.0)
            higher_bid_prbs = tf.where(all_bids > avg_bid, -neg_logs, 0.0)
            lower_rwds = lower_bid_prbs * discounted_rewards
            higher_rwds = higher_bid_prbs * discounted_rewards

            # print("all bids:\n{}".format(all_bids))
            # print("avg_bid:\n{}".format(avg_bid))
            # print("prbs:\n{}".format(action_prbs))
            # print("lower_bid_prbs:\n{}".format(lower_bid_prbs))
            # print("higher_bid_prbs:\n{}".format(higher_bid_prbs))
            # print("disc. rwds:\n{}".format(discounted_rewards))
# # DEBUG
            divisor = 5
            def plotter(fig, axs, bids, rwds, q_vals, iter, loss):
                plt.subplots_adjust(
                    left  = 0.15,  # the left side of the subplots of the figure
                    right = 0.95,    # the right side of the subplots of the figure
                    bottom = 0.15,   # the bottom of the subplots of the figure
                    top = 0.85,      # the top of the subplots of the figure
                    wspace = 0.3,   # the amount of width reserved for blank space between subplots
                    hspace = 0.5 
                )
                axs[0].scatter(bids, q_vals, c='purple')
                axs[0].scatter(bids, rwds, c='red')
                axs[0].set(
                            title="Network Q-Values",
                            xlabel="Bid", 
                            ylabel="Q Value",
                            xlim=[0.0,0.3],
                            ylim=[-10,10],
                            xticks=np.arange(0.0, 0.3, 0.02),
                            yticks=np.arange(-10, 10, 2)
                        )

                axs[1].scatter(iter, loss, color='blue')
                axs[1].set(
                            title="Network Loss",
                            xlabel="Iteration",
                            ylabel="Critic Loss",
                            ylim=[-500,5000],
                            xticks=[iter],
                            yticks=np.arange(-500, 5000, 500)
                        )
            if ((self.plot_count % divisor) == 0):
                fig, axs = plt.subplots(2)
                plotter(fig, axs, all_bids[:,0], discounted_rewards[:,0], 
                    q_state_values[:,0], self.plot_count, critic_loss)
                plt.savefig('q_figs/run_{}__iter_{}__lr_{}__critic_lr_mult_{}__q_vals.png'.format(
                                    id(self), self.plot_count, self.learning_rate, critic_lr_mult))
                plt.close(fig)
            self.plot_count += 1
# #
#             # print("avg. disc. rwds:\n{}".format(tf.reduce_mean(self.discount(rewards), axis=0)))
#             # print("avg. shaped disc. rwds:\n{}".format(tf.reduce_mean(discounted_rewards, axis=0)))
#             # print("avg. q_vals:\n{}".format(tf.reduce_mean(q_state_values, axis=0)))
#             print("5 disc. rwds:\n{}".format(self.discount(rewards)[:5]))
#             print("5 q_vals:\n{}".format(q_state_values[:5]))
#             print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
#             print("exp. lower rwd (disc.), lower rwd (advtg.):\n{}, {}".format(tf.reduce_sum(lower_rwds), tf.reduce_sum(lower_bid_prbs * advantage)))
#             print("exp. higher rwd (disc.), higher rwd (advtg.):\n{}, {}".format(tf.reduce_sum(higher_rwds), tf.reduce_sum(higher_bid_prbs * advantage)))
#             # # print("subaction_dists_vals:\n{}".format(subaction_dists_vals))
#             # # print("neg_logs:\n{}".format(neg_logs))
#             print("avg. loss:\n{}".format(tf.reduce_mean(losses, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
# #
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_Q_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)

        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)
     
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)

        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        advantage = q_state_values

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        critic_loss = tf.reduce_sum((discounted_rewards - q_state_values)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            
class AC_SARSA_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')

        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length-1] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        
        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        advantage = target

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.sigma_bias_init = None #tf.keras.initializers.RandomUniform(minval=-2.0, maxval=-2.0)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')
 
        self.critic_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(self.critic_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense3 = tf.keras.layers.Dense(self.critic_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.critic_dense4 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        output = self.critic_dense3(output)
        output = self.critic_dense4(output)
        
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        baseline = 0
        advantage = target - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)
        
        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)
    
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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply low, offset, and peak layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_lows = self.low_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets = self.offset_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_offsets2 = self.offset2_layer(output)

        # lows need to be >= 0 because bids need to be >= 0.
        output_lows = tf.nn.softplus(output_lows)
        # offsets need to be >= 0, so pass through softplus.
        output_offsets = tf.nn.softplus(output_offsets)
        # peaks need to be >= 0, so pass through softplus.
        output_offsets2 = tf.nn.softplus(output_offsets2)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_lows = tf.reshape(output_lows, [*output_lows.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets = tf.reshape(output_offsets, [*output_offsets.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_offsets2 = tf.reshape(output_offsets2, [*output_offsets2.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_lows = output_lows + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_lows.dtype)

        # make peak = low + offset, so that they are always higher than low.
        output_peaks = output_lows + output_offsets
        # make high = peak + offset2, so that they are always higher than peak.
        output_highs = output_peaks + output_offsets2

        # A distribution which (when sampled) returns an array of shape
        # output_lows.shape, i.e. [batch_size, episode_length, low_layer_output_size].
        dist = tfp.distributions.Triangular(low=output_lows, peak=output_peaks, high=output_highs)

        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        output = self.critic_dense1(inputs)
        output = self.critic_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()

        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        total_limit = samples[:,:,:,0:1] * samples[:,:,:,-1:]

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * multiplier).
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]

        # get the multiplier by taking the total_limit column and dividing by bid_per_item column.
        # (i.e. total_limit / bid_per_item = multiplier).
        mult = subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1]

        # replace the last column with mult (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, mult)
     
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)

        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        advantage = target

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        
class AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = None
        self.mu_ker_init = None
        self.sigma_ker_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        
        # Network for learning Q(s,a)
        self.q_layer1_size = 6 * self.num_subactions
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation='relu', dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation='relu', dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus)
        # variance needs to be a positive number, so pass through softplus.
        output_sigmas = tf.nn.softplus(output_sigmas) # + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        output_mus = output_mus + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=output_mus.dtype)
        
        # create column that is 1st column multiplied by the multiplier 
        # (i.e. bid_per_item multiplied by a multiplier).
        mult = output_mus[:,:,:,0:1] * output_mus[:,:,:,-1:]
        
        # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)

        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the Q-value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length-1] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''      
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # if bid_per_item > total_limit, then replace with total_limit.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding bid_per_item dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,-1:], samples)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)
        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]
        
        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))
        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]

        # shape [batch_size, episode_length-1]
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length, 1]
        state_values = self.value_function(states)
        # reshape to [batch_size, episode_length]
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))

        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        # shape [batch_size, episode_length-1]
        baseline = state_values[:,:-1]
        # shape [batch_size, episode_length-1]
        advantage = target - baseline

        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        # print("q_state_values:\n{}".format(q_state_values))
        # print("state_values:\n{}".format(baseline))
        # print("advantage:\n{}".format(advantage))
        # print("actor_loss: {}".format(actor_loss))
        # print("critic_loss: {}".format(critic_loss))
        # print("baseline_loss: {}".format(baseline_loss))
        # print("tot. loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Baseline_V_Gaussian_v2_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))   

        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        # shape [batch_size, episode_length-1]  
        baseline = state_values[:,:-1]  
        # shape [batch_size, episode_length-1]
        advantage = target - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)
        
        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        # all_bids = subaction_dists_vals[:,:,0,0]
        # avg_bid = tf.reduce_mean(all_bids, axis=0)
        # lower_bid_prbs = tf.where(all_bids <= avg_bid, -neg_logs, 0.0)
        # higher_bid_prbs = tf.where(all_bids > avg_bid, -neg_logs, 0.0)
        # lower_rwds = lower_bid_prbs * advantage
        # higher_rwds = higher_bid_prbs * advantage

        # # print("all bids:\n{}".format(all_bids))
        # # print("avg_bid:\n{}".format(avg_bid))
        # # print("prbs:\n{}".format(action_prbs))
        # print("avg. rwds for lower action:\n{}".format(tf.reduce_sum(lower_rwds)))
        # print("avg. rwds for higher action:\n{}".format(tf.reduce_sum(higher_rwds)))
        # print("avg. advtg:\n{}".format(tf.reduce_mean(advantage, axis=0)))
#
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("baseline_loss: {}".format(baseline_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Baseline_V_Gaussian_v3_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.elu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))   

        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        # shape [batch_size, episode_length-1]  
        baseline = state_values[:,:-1]  
        # shape [batch_size, episode_length-1]
        advantage = target - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)
        
        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("baseline_loss: {}".format(baseline_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Baseline_V_Gaussian_v3_1_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.layer1_size = 6
        self.layer2_size = 6
        self.layer3_size = 6
        self.layer4_size = 6
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer2_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer3_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.layer4_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense2 = tf.keras.layers.Dense(self.layer2_size, kernel_initializer=self.layer2_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense3 = tf.keras.layers.Dense(self.layer3_size, kernel_initializer=self.layer3_ker_init, activation=tf.nn.elu, dtype='float64')
        self.dense4 = tf.keras.layers.Dense(self.layer4_size, kernel_initializer=self.layer4_ker_init, activation=tf.nn.elu, dtype='float64')
        
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = self.dense2(output)
        output = self.dense3(output)
        output = self.dense4(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = -tf.math.log(tf.math.exp(self.budget_per_reach)-1)
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.softplus(output_mus-offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.softplus(output_sigmas-offset)

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))   

        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        # shape [batch_size, episode_length-1]  
        baseline = state_values[:,:-1]  
        # shape [batch_size, episode_length-1]
        advantage = target - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)
        
        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("baseline_loss: {}".format(baseline_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


class AC_SARSA_Baseline_V_Gaussian_v4_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, num_dist_per_spec=2, budget_per_reach=1.0, 
                        is_partial=False, discount_factor=1, learning_rate=0.0001, shape_reward=False):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate
        self.budget_per_reach = budget_per_reach
        self.shape_reward = shape_reward

        self.auction_item_spec_ids = np.sort(auction_item_spec_ids)
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # Default is 2 for bid_per_item and total_limit.
        # NOTE: assuming the last dist is the dist for total_limit.
        self.num_dist_per_subaction = num_dist_per_spec

        self.layer1_size = 1
        self.layer1_ker_init = tf.keras.initializers.RandomUniform(minval=0.001, maxval=0.001)
        self.mu_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.sigma_ker_init = tf.keras.initializers.RandomUniform(minval=1, maxval=1)
        self.mu_bias_init = None
        self.sigma_bias_init = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=None, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, activation=None, dtype='float64')

        # Network for learning Q(s,a)
        self.q_layer1_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer2_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_layer3_size = 6 + (self.num_subactions * self.num_dist_per_subaction)
        self.q_dense1 = tf.keras.layers.Dense(self.q_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense2 = tf.keras.layers.Dense(self.q_layer2_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense3 = tf.keras.layers.Dense(self.q_layer3_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.q_dense4 = tf.keras.layers.Dense(1, dtype='float64')

        # Network for learning V(s)
        self.v_layer1_size = 6
        self.v_dense1 = tf.keras.layers.Dense(self.v_layer1_size, activation=tf.nn.leaky_relu, dtype='float64')
        self.v_dense2 = tf.keras.layers.Dense(1, dtype='float64')

    def __repr__(self):
        return "{}(is_partial: {}, discount: {}, lr: {}, num_actions: {}, optimizer: {}, shape_reward: {})".format(self.__class__.__name__, 
                                                                       self.is_partial, self.discount_factor, 
                                                                       self.learning_rate, self.num_subactions,
                                                                       type(self.optimizer).__name__, self.shape_reward)

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
        :return: A distribution which (when sampled) returns an array of shape 
        [batch_size, episode_length, num_subactions * num_dist_per_subaction]. 
        The distribution represents probability distributions P(a | s_i) for 
        each s_i in the episode and batch (via subactions of a, i.e. 
            P(a | s_i) = P(a_sub_1_dist_1 | s_i) * ... * P(a_sub_j_dist_k | s_i) 
        ).
        '''
        # Apply dense layers
        output = self.dense1(states)
        output = tf.nn.leaky_relu(output)

        # Apply mu and sigma layers.
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_mus = self.mu_layer(output)
        # array of size [batch_size, episode_length, num_subactions * num_dist_per_subaction]
        output_sigmas = self.sigma_layer(output)

        offset = self.budget_per_reach
        # mus need to be >= 0 because bids need to be >= 0.
        output_mus = tf.nn.relu(output_mus+offset)
        # variance needs to be a positive number.
        output_sigmas = 0.5*tf.nn.relu(output_sigmas+offset) + 1e-5

        # reshape to [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        output_mus = tf.reshape(output_mus, [*output_mus.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])
        output_sigmas = tf.reshape(output_sigmas, [*output_sigmas.shape[:2]] + [self.num_subactions, self.num_dist_per_subaction])

        # A distribution which (when sampled) returns an array of shape
        # output_mus.shape, i.e. [batch_size, episode_length, mu_layer_output_size].
        # NOTE: make sure loc and scale are float tensors so that they're compatible 
        # with tfp.distributions.Normal. Otherwise it will throw an error.
        dist = tfp.distributions.Normal(loc=output_mus, scale=output_sigmas)
        return dist

    def q_value_function(self, states, actions):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length-1, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :return: A [batch_size, episode_length] matrix representing the value of each q-state.
        """
        inputs = tf.concat([states, tf.reshape(actions, (*actions.shape[:2],-1))], axis=-1)      
        output = self.q_dense1(inputs)
        output = self.q_dense2(output)
        output = self.q_dense3(output)
        output = self.q_dense4(output)       
        return output

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An array of shape [batch_size, episode_length, new_state_size], where new_state_size 
        is single_agent_state_size if self.partial else num_of_agents * single_agent_state_size.
        :return: A [batch_size, episode_length] matrix representing the value of each state.
        """
        output = self.v_dense1(states)
        output = self.v_dense2(output)
        return output

    def choose_actions(self, call_output):
        '''
        :param call_output: output of call func.
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        # array of shape [batch_size, episode_length, num_subactions, num_dist_per_subaction]
        samples = call_output.sample()
        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)

        # adding 1.0 to last column of num_dist_per_subaction columns (i.e. total_limit column)
        # so that it can be used as a multiplier.
        samples = samples + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [1.0], dtype=samples.dtype)
        
        # create total_limit column that is bid_per_item column multiplied by the multiplier 
        # (i.e. total_limit = bid_per_item * multiplier).
        # only multiply if bid_per_item > 0. Otherwise the last column's original value will be lost.
        total_limit = tf.where(samples[:,:,:,0:1] > 0, samples[:,:,:,0:1] * samples[:,:,:,-1:], samples)

        # replace the last column (i.e. total_limit column of samples is now bid_per_item * (orig_last_col + 1)).
        # if bid_per_item is 0, then total_limit column is orig_last_col + 1.
        samples = tf.where([True]*(self.num_dist_per_subaction-1) + [False], samples, total_limit)

        samples = tf.clip_by_value(samples, self.subactions_min, self.subactions_max)
        samples_shape = tf.shape(samples)

        # Note: num_subactions = num_auction_item_spec_ids
        # array of shape [1, 1, num_auction_item_spec_ids]
        ais_reshp = tf.convert_to_tensor(self.auction_item_spec_ids)[None,None,:]
        # array of shape [batch_size, episode_length, num_auction_item_ids]
        ais_reshp = tf.broadcast_to(ais_reshp, [*samples_shape[:2]] + [ais_reshp.shape[2]])
        # array of shape [batch_size, episode_length, num_auction_item_ids, 1]
        ais_reshp = tf.reshape(ais_reshp, [*ais_reshp.shape[:2]] + [-1,1])
        # casting to same type as samples so that it can be concatenated with samples
        ais_reshp = tf.cast(ais_reshp, samples.dtype)

        # array of shape [batch_size, episode_length, num_subactions, 1 + num_dist_per_subaction]
        chosen_actions = tf.concat([ais_reshp, samples], axis=3)

        return chosen_actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: An array of shape [batch_size, episode_length, new_state_size], 
        where new_state_size is single_agent_state_size if self.partial else 
        num_of_agents * single_agent_state_size.
        :param actions: an array of shape [batch_size, episode_length-1, num_subactions, subaction_size].
        :param rewards: an array of shape [batch_size, episode_length-1].
        '''
        # states is of episode_length, but actions is of episode_length-1.
        # So delete the last state of each episode.
        action_distr = self.call(states[:,:-1])

        # if each subaction is [auction_item_spec_id, bid_per_item, total_limit],
        # then slice out the 0th index to get each [bid_per_item, total_limit].
        # Note: the length of this (i.e. 2) = self.num_dist_per_subaction.
        subaction_dists_vals = actions[:,:,:,1:]        

        # get the original last column by taking the total_limit column and dividing by bid_per_item column
        # and then subtracting 1.0 from it.
        # (i.e. (total_limit / bid_per_item) -1.0 = orig_last_col).
        orig_last_col = tf.where(subaction_dists_vals[:,:,:,0:1] > 0, 
                                 subaction_dists_vals[:,:,:,-1:] / subaction_dists_vals[:,:,:,0:1],
                                 subaction_dists_vals)
        orig_last_col = orig_last_col + tf.constant([0.0]*(self.num_dist_per_subaction-1) + [-1.0], dtype=subaction_dists_vals.dtype)

        # replace the last column with orig_last_col (i.e. total_limit column is now multiplier).
        subaction_dists_vals = tf.where([True]*(self.num_dist_per_subaction-1) + [False], subaction_dists_vals, orig_last_col)

        # shape [batch_size, episode_length-1]
        action_prbs = tf.reduce_prod(tf.reduce_prod(action_distr.prob(subaction_dists_vals), axis=3), axis=2)
        
        # shape [batch_size, episode_length-1]  
        discounted_rewards = self.discount(rewards)   
        # shape [batch_size, episode_length-1, 1]
        q_state_values = self.q_value_function(states[:,:-1], actions)
        # reshape to [batch_size, episode_length-1]
        q_state_values = tf.reshape(q_state_values, (*q_state_values.shape[:1],-1))

        # add 0.0 to the end to represent Q(s',a') for the last state of each episode,
        # then pick out Q(s',a') from 2nd state to last state.
        # shape [batch_size, episode_length-1]
        next_q_vals = tf.concat([q_state_values, 
                                tf.zeros((*q_state_values.shape[:1],1), dtype=q_state_values.dtype)], 
                                axis=1)[:,1:]
        # shape [batch_size, episode_length, 1] 
        state_values = self.value_function(states)  
        # reshape to [batch_size, episode_length]   
        state_values = tf.reshape(state_values, (*state_values.shape[:1],-1))   

        # shape [batch_size, episode_length-1]
        target = rewards + (self.discount_factor * next_q_vals)
        # shape [batch_size, episode_length-1]  
        baseline = state_values[:,:-1]  
        # shape [batch_size, episode_length-1]
        advantage = target - baseline
        if self.shape_reward:
            advantage = tf.where(advantage > 0, tf.math.log(advantage+1), advantage)
        
        neg_logs = -tf.math.log(action_prbs)
        # clip min/max values to avoid infinities.
        neg_logs = tf.clip_by_value(neg_logs, -1e9, 1e9)
        losses = neg_logs * tf.stop_gradient(advantage)
        actor_loss = tf.reduce_sum(losses)
        critic_loss = tf.reduce_sum((target - q_state_values)**2)
        baseline_loss = tf.reduce_sum((discounted_rewards - baseline)**2)
        total_loss = (1.0*actor_loss) + (1.0*critic_loss) + (1.0*baseline_loss)
# DEBUG
        print("avg. actions:\n{}".format(tf.reduce_mean(actions, axis=0)))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("baseline_loss: {}".format(baseline_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))