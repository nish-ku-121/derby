from abc import ABC, abstractmethod
from derby.core.basic_structures import Bid



class AbstractPolicy(ABC):

    @abstractmethod
    def call(self, states):
        '''
        Computes what action to take.
        '''
        pass

    @abstractmethod
    def update_policy(self, states, actions, discounted_rewards):
        '''
        Updates the policy.
        '''
        pass


class DummyPolicy1(AbstractPolicy):

    def __init__(self, auction_item_spec, bid_per_item, total_limit):
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

    def call(self, states):
        return [ [self.auction_item_spec.uid, self.bid_per_item, self.total_limit] ]

    def update_policy(self, states, actions, discounted_rewards):
        return


class DummyPolicy2(AbstractPolicy):

    def __init__(self, agent, auction_item_spec, bid_per_item, total_limit):
        self.agent = agent
        self.auction_item_spec = auction_item_spec
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit

    def call(self, states):
        campaign = states[0]
        agent = self.agent
        auction_item_spec = self.auction_item_spec
        bpi = self.bid_per_item
        lim = self.total_limit
        return [ Bid(agent, auction_item_spec, bid_per_item=bpi, total_limit=lim) ]

    def update_policy(self, states, actions, discounted_rewards):
        return