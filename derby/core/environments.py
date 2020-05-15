from abc import ABC, abstractmethod
from typing import Set, List, Dict, Any, TypeVar, Iterable
import numpy as np
from derby.core.basic_structures import Bid
from derby.core.pmfs import PMF, AuctionItemPMF
from derby.core.auctions import AbstractAuction
from derby.core.ad_structures import Campaign
from derby.core.states import CampaignBidderState
from derby.core.markets import OneCampaignMarket
from derby.core.utils import flatten_2d



class AbstractEnvironment(ABC):

    @abstractmethod
    def init(self, agents, horizon=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass


class CampaignEnv(AbstractEnvironment):

    @abstractmethod
    def init(self, agents, horizon=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def convert_from_actions_tensor(self, actions):
        '''
        Assume actions is a numpy array of shape [m, n, k]
        where:
            m = # of agents
            n = # of bids (per agent)
            k = # of fields of a bid vector
        Assuming bid vector is: [auction_item_id, bid_per_item, total_limit].
        Assuming first dimension is in the same order as the agents list passed to init.
        output: 2D list of bid objects. bids[i] are agent i's bids, 
                bids[i][j] is the jth bid of agent i.
        '''
        bids = []
        for i in range(actions.shape[0]):
            agent_i_bids = []
            for j in range(actions.shape[1]):
                bid_j_of_agent_i = actions[i][j]
                auction_item_id = bid_j_of_agent_i[0]
                bid_per_item = bid_j_of_agent_i[1]
                total_limit = bid_j_of_agent_i[2]
                bidder = self._agents[i]
                auction_item = self._auction_items_by_id[auction_item_id]
                agent_i_bids.append(Bid(bidder, auction_item, bid_per_item, total_limit))
            bids.append(agent_i_bids)
        return bids

    def convert_to_states_tensor(self, states):
        '''
        Assuming states is of the form:
        [
            Agent 1 state,
            ...,
            Agent m state,
            
        ]
        '''
        states_tensor = []
        for cbstate in states:
            vec = [
                    cbstate.campaign.reach, cbstate.campaign.budget, cbstate.campaign.target.item_id,
                    cbstate.spend, cbstate.impressions, cbstate.timestep
                ]
            states_tensor.append(vec)
        return np.array(states_tensor)


class OneCampaignNDaysEnv(CampaignEnv):

    def __init__(self, auction: AbstractAuction, num_items_per_timestep_min: int, num_items_per_timestep_max: int , 
                       campaign_pmf: PMF, auction_item_pmf: AuctionItemPMF):
        self._auction = auction
        self._campaign_pmf = campaign_pmf
        # let this be half-open, i.e. [num_items_per_timestep_min, num_items_per_timestep_max)
        self._num_items_per_timestep_range = (num_items_per_timestep_min, num_items_per_timestep_max)
        self._auction_item_pmf = auction_item_pmf
        self._auction_items_by_id = { item.item_id : item for item in self._auction_item_pmf.items() }
        self.done = False
        self._agents = None
        self._bidder_states_by_agent = None
        self._market = None

    def init(self, agents, horizon=None):
        self._agents = tuple(agents)
        self._bidder_states_by_agent = { agent : None for agent in agents }
        self._num_of_days = horizon

    def reset(self):
        self.done = False
        bidder_states = []
        for agent in self._agents:
            camp = self._campaign_pmf.draw_n(1)[0]
            cbstate = CampaignBidderState(agent, camp)
            bidder_states.append(cbstate)
            self._bidder_states_by_agent[agent] = cbstate
        self._market = OneCampaignMarket(self._auction, bidder_states, self._auction_item_pmf)
        bidder_states = self.convert_to_states_tensor(bidder_states)
        return bidder_states

    def step(self, actions):
        '''
        Assuming actions input is same as specified 
        in convert_from_actions_tensor func.
        output: states (as np array), rewards (as np array), done
        '''
        states = []
        rewards = []
        
        if not self.done:
            # Convert to a 2D list of bid objects, where 
            # bids[i] is agent i's bids and bids[i][j] is the jth bid of agent i.
            bids = self.convert_from_actions_tensor(actions)        
            
            # Randomize the number of auction items available (i.e. users who show up) in this step
            pre_step_agent_spends = [ self._bidder_states_by_agent[agent].spend for agent in self._agents ]
            num_items_min = self._num_items_per_timestep_range[0]
            num_items_max = self._num_items_per_timestep_range[1]
            self._market.num_of_items_per_timestep = np.random.randint(num_items_min, num_items_max)

            # Run the auction
            results = self._market.run_auction(flatten_2d(bids), items_to_bids_mapping_func=self._auction.items_to_bids_by_item_type_submatch)
            self.done = (self._num_of_days != None) and (self._market.timestep == self._num_of_days)
            
            # Calculate each agent's reward
            for i in range(len(self._agents)):
                agent = self._agents[i]
                agent_bids = bids[i]
                cbstate = self._bidder_states_by_agent[agent]
                states.append(cbstate)
                agent_expenditure = ( cbstate.spend - pre_step_agent_spends[i] ) # alternatively, use agent_bids
                agent_reward = -1 * agent_expenditure # i.e. negative reward
                if self.done:
                    # TODO: replace with actual formula
                    agent_reward += min(cbstate.campaign.budget, 
                                        (cbstate.impressions / (1.0 * cbstate.campaign.reach)) * cbstate.campaign.budget)
                rewards.append(agent_reward)
        
        # numpy array of shape [m, s]
        # where:
        # m = # of agents
        # s = # of fields of state vector
        states = self.convert_to_states_tensor(states)
        # numpy array of shape [m, ]
        rewards = np.array(rewards)
        return states, rewards, self.done