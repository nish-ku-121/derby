from abc import ABC, abstractmethod
import numpy as np
from derby.core.basic_structures import Bid
from derby.core.states import CampaignBidderState



class AbstractPolicy(ABC):

    def __init__(self, agent=None, is_partial=False):
        self.agent = agent # assuming Agent class sets this
        self.is_partial = is_partial

    @abstractmethod
    def call(self, states):
        '''
        :param states: an array of shape [batch_size, episode_length, num_of_agents, state_size].
        if self.is_partial, then array should be of shape [batch_size, episode_length, state_size] 
        instead.
        :return: an array of shape [batch_size, episode_length, ...] representing (e.g. as logits, 
        probs, etc.) the actions to take.      
        '''
        pass

    @abstractmethod
    def choose_actions(self, actions):
        '''
        :param actions: an array of shape [batch_size, episode_length, ...] (output of call func).
        :return: an array of shape [batch_size, episode_length] with actual actions choosen in some way.
        '''
        pass

    @abstractmethod
    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: an array of shape [batch_size, episode_length, num_of_agents, state_size].
        if self.is_partial, then array should be of shape [batch_size, episode_length, state_size] 
        instead.
        :param actions: an array of shape [batch_size, episode_length].
        :param rewards: an array of shape [batch_size, episode_length].
        '''
        pass


class DummyPolicy1(AbstractPolicy):

    def __init__(self, auction_item_spec, bid_per_item, total_limit):
        super().__init__(is_partial=False)
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

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

    def choose_actions(self, actions):
        return actions

    def loss(self, states, actions, rewards):
        return


class DummyPolicy2(AbstractPolicy):

    def __init__(self, auction_item_spec, bid_per_item, total_limit):
        super().__init__(is_partial=False)
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

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

    def choose_actions(self, actions):
        return actions

    def loss(self, states, actions, rewards):
        return


class BudgetPerReachPolicy(AbstractPolicy):

    def __init__(self):
        super().__init__(is_partial=True)

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

    def choose_actions(self, actions):
        return actions

    def loss(self, states, actions, rewards):
        '''
        Updates the policy.
        :param states: an array of shape [batch_size, episode_length, state_size].
        :param actions: an array of shape [batch_size, episode_length].
        :param rewards: an array of shape [batch_size, episode_length].
        '''
        return