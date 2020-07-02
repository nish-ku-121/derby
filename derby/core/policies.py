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
        self.subactions_max = 1e15

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
        # dist = tfp.distributions.TruncatedNormal(loc=output_mus, 
                                                 # scale=output_sigmas,
                                                 # low=output_mus - output_sigmas,
                                                 # high=output_mus + output_sigmas)
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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
        # # # # print(self.choices)
        # print("loc:\n{}".format(action_distr.loc))
        # print("scale: {}".format(action_distr.scale))
        # print("low: {}".format(action_distr.low))
        # print("high: {}".format(action_distr.high))
        # # # # # print("states 2: {}".format(states))
        # print("actions:\n{}".format(actions))
        # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # print(subaction_dists_vals)
        # # print("action prbs: {}".format(action_prbs))
        # # # # print(rewards)
        # # print("advtge:\n{}".format(discounted_rewards))
        # print("neg logs:\n{}".format(neg_logs))
        # # # print("losses: {}".format(losses))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class REINFORCE_MarketEnv_Continuous_Uniform(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

        self.layer1_size = 1 #50
        self.layer1_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Uniform(a|low(s),high(s)) 
        #                                 = \prod_j \prod_k Uniform(sub_a_j_dist_k|low(s),high(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        

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

# DEBUG
        # print("call states: {}".format(states))
        # print("dense1 weights: {}".format(self.dense1.kernel))
        # print("mus weights: {}".format(self.mu_layer.kernel))
        # print("output: {}".format(output))
        # print("lows: {}".format(output_lows))
        # print("highs: {}".format(output_highs))
#

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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
# DEBUG
        # print("low:\n{}".format(action_distr.low))
        # print("high:\n{}".format(action_distr.high))
        # # # # # print("states 2: {}".format(states))
        # print("actions:\n{}".format(actions))
        # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # print(subaction_dists_vals)
        # print("action prbs:\n{}".format(action_prbs))
        # # # # print(rewards)
        # # print("advtge:\n{}".format(discounted_rewards))
        # print("neg logs:\n{}".format(neg_logs))
        # # # print("losses: {}".format(losses))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class REINFORCE_MarketEnv_Continuous_Triangular(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

        self.layer1_size = 1 #50
        self.layer1_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        self.dense1 = tf.keras.layers.Dense(self.layer1_size, kernel_initializer=self.layer1_ker_init, activation=tf.nn.leaky_relu, dtype='float64')
        
        # Layers for calculating \pi(a|s) = Triangular(a|low(s),high(s),peak(s)) 
        #                                 = \prod_j \prod_k Triangular(sub_a_j_dist_k|low(s),high(s),peak(s))        
        self.low_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')
        self.offset2_layer = tf.keras.layers.Dense(self.num_subactions*self.num_dist_per_subaction, use_bias=False, activation=None, dtype='float64')

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
        # # create column that is 1st column multiplied by the multiplier 
        # # (i.e. bid_per_item * multiplier).
        # mult = output_lows[:,:,:,0:1] * output_lows[:,:,:,-1:]
        # # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # output_lows = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_lows, mult)
        # # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # # total_limit is significantly lower than the sampled bid_per_item.

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
# DEBUG
        # print("low:\n{}".format(action_distr.low))
        # print("high:\n{}".format(action_distr.high))
        # # print("peak:\n{}".format(action_distr.peak))
        # # # # # # print("states 2: {}".format(states))
        # # print("actions:\n{}".format(actions))
        # # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # print("subaction_dist_vals:\n{}".format(actions[:,:,:,1:]))
        # print("mult:\n{}".format(mult))
        # print("action prbs:\n{}".format(action_prbs))
        # # # # # print(rewards)
        # # # print("advtge:\n{}".format(discounted_rewards))
        # # print("neg logs:\n{}".format(neg_logs))
        # # # # print("losses: {}".format(losses))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class REINFORCE_baseline_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

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
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')


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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
        total_loss = (1.0*actor_loss) + (0.5*critic_loss)
# DEBUG
        # # # # print(self.choices)
        # print("loc:\n{}".format(action_distr.loc))
        # print("scale: {}".format(action_distr.scale))
        # print("low: {}".format(action_distr.low))
        # print("high: {}".format(action_distr.high))
        # # # # # print("states 2: {}".format(states))
        # print("actions:\n{}".format(actions))
        # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # print(subaction_dists_vals)
        # # print("action prbs: {}".format(action_prbs))
        # # # # print(rewards)
        # print("disc rwds:\n{}".format(discounted_rewards))
        # # print("neg logs:\n{}".format(neg_logs))
        # # print("advtg:\n{}".format(advantage))
        # print("state vals:\n{}".format(state_values))
        # print("actor_loss: {}".format(actor_loss))
        # print("critic_loss: {}".format(critic_loss))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class REINFORCE_Baseline_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

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
        # # create column that is 1st column multiplied by the multiplier 
        # # (i.e. bid_per_item * multiplier).
        # mult = output_lows[:,:,:,0:1] * output_lows[:,:,:,-1:]
        # # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # output_lows = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_lows, mult)
        # # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # # total_limit is significantly lower than the sampled bid_per_item.

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


class AC_MarketEnv_Continuous_v1(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

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
        
        self.critic_layer1_size = 6
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')


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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
# DEBUG
        # # # # # print(self.choices)
        # print("loc:\n{}".format(action_distr.loc))
        # # print("scale: {}".format(action_distr.scale))
        # # print("low: {}".format(action_distr.low))
        # # print("high: {}".format(action_distr.high))
        # # # # # # print("states 2: {}".format(states))
        # # print("actions:\n{}".format(actions))
        # # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # # print(subaction_dists_vals)
        # # # print("action prbs: {}".format(action_prbs))
        # # # # # print(rewards)
        # # print("neg logs:\n{}".format(neg_logs))
        # print("rwds:\n{}".format(rewards))
        # discounted_rewards = self.discount(rewards)
        # print("disc. rwds:\n{}".format(discounted_rewards))
        # # print("targets:\n{}".format(targets))
        # # print("state vals:\n{}".format(state_values))
        # # # print("r_error:\n{}".format(discounted_rewards - state_values[:,:-1]))
        # print("td_error:\n{}".format(advantage))
        # print("r_loss: {}".format(tf.reduce_sum(neg_logs * tf.stop_gradient(discounted_rewards))))
        # print("actor_loss: {}".format(actor_loss))
        # print("critic_loss: {}".format(critic_loss))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class AC_TD_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

        self.layer1_size = 1 #50
        self.layer1_ker_init = None # tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
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
        # # create column that is 1st column multiplied by the multiplier 
        # # (i.e. bid_per_item * multiplier).
        # mult = output_lows[:,:,:,0:1] * output_lows[:,:,:,-1:]
        # # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # output_lows = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_lows, mult)
        # # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # # total_limit is significantly lower than the sampled bid_per_item.

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


class AC_MarketEnv_Continuous_v2(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

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
        
        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')


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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
# DEBUG
        # # # # print(self.choices)
        # print("loc:\n{}".format(action_distr.loc))
        # print("scale: {}".format(action_distr.scale))
        # print("low: {}".format(action_distr.low))
        # print("high: {}".format(action_distr.high))
        # # # # # print("states 2: {}".format(states))
        # print("actions:\n{}".format(actions))
        # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # print(subaction_dists_vals)
        # # print("action prbs: {}".format(action_prbs))
        # # # # print(rewards)
        # # print("advtge:\n{}".format(discounted_rewards))
        # print("neg logs:\n{}".format(neg_logs))
        # print("q_state vals:\n{}".format(q_state_values))
        # print("actor_loss: {}".format(actor_loss))
        # print("critic_loss: {}".format(critic_loss))
        # print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class AC_Q_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

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
        # # create column that is 1st column multiplied by the multiplier 
        # # (i.e. bid_per_item * multiplier).
        # mult = output_lows[:,:,:,0:1] * output_lows[:,:,:,-1:]
        # # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # output_lows = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_lows, mult)
        # # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # # total_limit is significantly lower than the sampled bid_per_item.

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


class AC_MarketEnv_Continuous_sarsa(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

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
        
        self.critic_layer1_size = 6 * self.num_subactions
        self.critic_dense1 = tf.keras.layers.Dense(self.critic_layer1_size, activation='relu', dtype='float64')
        self.critic_dense2 = tf.keras.layers.Dense(1, dtype='float64')


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
# # DEBUG
#         print("chosen_actions:\n{}".format(chosen_actions))
# #
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
# DEBUG
        # # print("loc:\n{}".format(action_distr.loc))
        # # print("scale: {}".format(action_distr.scale))
        # # print("actions:\n{}".format(actions))
        # # # print("dist probs: {}".format(action_distr.prob(subaction_dists_vals)))
        # # # # # print(subaction_dists_vals)
        # # # print("action prbs: {}".format(action_prbs))
        # print("rewards:\n{}".format(rewards))
        # # discounted_rewards = self.discount(rewards)
        print("disc. rwds:\n{}".format(discounted_rewards))
        # print("q_state vals:\n{}".format(q_state_values))
        # print("target:\n{}".format(target))
        # # # print("advantage:\n{}".format(advantage))
        # # print("neg logs:\n{}".format(neg_logs))
        print("r_loss: {}".format(tf.reduce_sum(neg_logs * tf.stop_gradient(discounted_rewards))))
        print("actor_loss: {}".format(actor_loss))
        print("critic_loss: {}".format(critic_loss))
        print("tot loss: {}".format(total_loss))
#
        return total_loss

    def update(self, states, actions, rewards, policy_loss, tf_grad_tape=None):
        if tf_grad_tape is None:
            raise Exception("No tf_grad_tape has been provided!")
        else:
            gradients = tf_grad_tape.gradient(policy_loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
# DEBUG
            # print("grads: {}".format(gradients))
            # print("weights: {}".format(self.trainable_variables))
#


class AC_SARSA_Triangular_MarketEnv_Continuous(AbstractPolicy, tf.keras.Model):

    def __init__(self, auction_item_spec_ids, is_partial=False, discount_factor=1, learning_rate=0.0001):
        super().__init__()
        self.is_partial = is_partial
        self.discount_factor = discount_factor
        self.is_tensorflow = True
        self.learning_rate = learning_rate

        self.auction_item_spec_ids = auction_item_spec_ids
        self.subactions_min = 0
        self.subactions_max = 1e15

        # Network parameters and optimizer
        self.num_subactions = len(self.auction_item_spec_ids)
        # "2" because bid_per_item and total_limit
        # NOTE: assuming the "num_dist_per_subaction"th dist (i.e. last dist) 
        # is the dist for total_limit
        self.num_dist_per_subaction = 2

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
        # # create column that is 1st column multiplied by the multiplier 
        # # (i.e. bid_per_item * multiplier).
        # mult = output_lows[:,:,:,0:1] * output_lows[:,:,:,-1:]
        # # replace the last column with mult (i.e. total_limit column is now multiplier*bid_per_item).
        # output_lows = tf.where([True]*(self.num_dist_per_subaction-1) + [False], output_lows, mult)
        # # NOTE: why do this? 1) it guarantees the total_limit dist is slightly higher
        # # than the bid_per_item dist, which 2) makes it unlikely that sampled 
        # # total_limit is significantly lower than the sampled bid_per_item.

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