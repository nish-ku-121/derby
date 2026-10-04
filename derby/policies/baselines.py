"""Deterministic baseline policies for Derby training runs."""
from __future__ import annotations

import numpy as np

from derby.core.basic_structures import Bid
from derby.core.environments import AbstractEnvironment
from derby.core.states import CampaignBidderState
from derby.policies.base import AbstractPolicy


class FixedBidPolicy(AbstractPolicy):
    def __init__(self, bid_per_item, total_limit, auction_item_spec=None):
        super().__init__()
        self.bid_per_item = bid_per_item
        self.total_limit = total_limit
        self.auction_item_spec = auction_item_spec

    def __repr__(self):
        return "{}(bid_per_item: {}, total_limit: {})".format(
            self.__class__.__name__,
            self.bid_per_item,
            self.total_limit,
        )

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
                    if self.auction_item_spec is None:
                        auction_item_spec = state_i_j.campaign.target
                    else:
                        auction_item_spec = self.auction_item_spec
                    action = [Bid(self.agent, auction_item_spec, self.bid_per_item, self.total_limit)]
                else:
                    if self.auction_item_spec is None:
                        spec_id = state_i_j[2]
                    else:
                        spec_id = self.auction_item_spec.uid
                    action = [[spec_id, self.bid_per_item, self.total_limit]]
                actions_i.append(action)
            actions.append(actions_i)
        return np.array(actions)

    def choose_actions(self, call_output):
        return call_output

    def policy_loss(self, states, actions, rewards):
        return 0


class BudgetPerReachPolicy(AbstractPolicy):
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
                    bpr = state_i_j.campaign.budget / float(state_i_j.campaign.reach)
                    if state_i_j.impressions >= state_i_j.campaign.reach:
                        bpr = 0.0
                    action = [Bid(self.agent, state_i_j.campaign.target, bid_per_item=bpr, total_limit=state_i_j.campaign.budget)]
                else:
                    reach = state_i_j[0]
                    budget = state_i_j[1]
                    auction_item_spec = state_i_j[2]
                    impressions = state_i_j[4]
                    bpr = budget / float(reach)
                    if impressions >= reach:
                        bpr = 0.0
                    action = [[auction_item_spec, bpr, budget]]
                actions_i.append(action)
            actions.append(actions_i)
        return np.array(actions)

    def choose_actions(self, call_output):
        return call_output

    def policy_loss(self, states, actions, rewards):
        return 0


class StepPolicy(AbstractPolicy):
    def __init__(self, start_bid, step_per_day):
        super().__init__()
        self.start_bid = start_bid
        self.step_per_day = step_per_day

    def __repr__(self):
        return "{}(start_bid: {}, step_per_day: {})".format(
            self.__class__.__name__,
            self.start_bid,
            self.step_per_day,
        )

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
                    day = self.timestep
                    bpr = self.start_bid + day * self.step_per_day
                    if state_i_j.impressions >= state_i_j.campaign.reach:
                        bpr = 0.0
                    action = [Bid(self.agent, state_i_j.campaign.target, bid_per_item=bpr, total_limit=bpr)]
                else:
                    reach = state_i_j[0]
                    auction_item_spec = state_i_j[2]
                    impressions = state_i_j[4]
                    day = state_i_j[5]
                    bpr = self.start_bid + day * self.step_per_day
                    if impressions >= reach:
                        bpr = 0.0
                    action = [[auction_item_spec, bpr, bpr]]
                actions_i.append(action)
            actions.append(actions_i)
        return np.array(actions)

    def choose_actions(self, call_output):
        return call_output

    def policy_loss(self, states, actions, rewards):
        return 0
