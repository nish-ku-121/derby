from abc import ABC, abstractmethod
import numpy as np
from derby.core.basic_structures import Bid
from derby.core.states import CampaignBidderState
from derby.core.environments import AbstractEnvironment
import os
import tensorflow as tf
import tensorflow_probability as tfp

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

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        pass


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


class REINFORCE_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = np.inf

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

        self.layer1_size = 1 #50
        self.layer1_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.mu_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.sigma_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = N(a|mu(s),sigma(s)) 
        #                                 = \prod_j \prod_k N(sub_a_j_dist_k|mu(s),sigma(s))        
        self.mu_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.mu_ker_init, use_bias=False, activation=None, dtype='float64')
        self.sigma_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, kernel_initializer=self.sigma_ker_init, use_bias=False, activation=None, dtype='float64')
        

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
        output_mus = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_mus, mult)
        # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # total_limit is significantly lower than the sampled bid_per_item.

# DEBUG
        # print("call states: {}".format(states))
        # print("dense1 weights: {}".format(self.dense1.kernel))
        # print("mus weights: {}".format(self.mu_layer.kernel))
        # print("output: {}".format(output))
        # print("mus: {}".format(output_mus))
        # print("sigmas: {}".format(output_sigmas))
#

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
        # if bid_per_item > total_limit, then replace with bid_per_item.
        # NOTE: this replacement sample won't be too far away from the 
        # corresponding total_limit dist, since call() handles making the 
        # total_limit dist higher than the bid_per_item dist. 
        samples = tf.where(samples[:,:,:,0:1] > samples[:,:,:,-1:], samples[:,:,:,0:1], samples)
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
# DEBUG
        # # # print(self.choices)
        # print("loc: {}".format(action_distr.loc))
        # # print("scale: {}".format(action_distr.scale))
        # # # # print("states 2: {}".format(states))
        # print("actions: {}".format(actions))
        # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # print(subaction_dists_vals)
        # # # print(action_prbs)
        # # # print(rewards)
        # # # print("advtge: {}".format(discounted_rewards))
        # print("neg logs: {}".format(neg_logs))
        # # print("losses: {}".format(losses))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))