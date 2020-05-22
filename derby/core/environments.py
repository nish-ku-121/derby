from abc import ABC, abstractmethod
from typing import Set, List, Dict, Any, TypeVar, Iterable
import numpy as np
from derby.core.basic_structures import AuctionItemSpecification, AuctionItem, Bid
from derby.core.pmfs import PMF
from derby.core.auctions import AbstractAuction
from derby.core.ad_structures import Campaign
from derby.core.states import State, CampaignBidderState
from derby.core.markets import OneCampaignMarket, SequentialAuctionMarket
from derby.core.utils import flatten_2d



def generate_trajectories(env, agents, num_of_trajs, horizon_cutoff, debug=False):
    all_traj_states = None
    all_traj_rewards = None
    for i in range(num_of_trajs):
        traj_i_states = []
        traj_i_rewards = []
        agents_joint_state = env.reset()
        if env.vectorize:
            traj_i_states.append(agents_joint_state)
        if debug:
            print("=== Traj {} ===".format(i))
            print()
            print("states 0")
            print(agents_joint_state)
        for j in range(horizon_cutoff):
            actions = [ agent.compute_action(agents_joint_state) for agent in agents ] 
            agents_joint_state, rewards, done = env.step(actions)
            if env.vectorize:
                traj_i_states.append(agents_joint_state)
                traj_i_rewards.append(rewards)
            if debug:
                print("actions {}".format(j))
                print(actions)
                print("rewards {}".format(j))
                print(rewards)
                print("states {}".format(j+1))
                print(agents_joint_state)
                print("Done? {}".format(done))
            if done:
                break
        print()

        if env.vectorize:
            # shape [episode length, num of agents, state size, batch size]
            traj_i_states = np.array(traj_i_states)[:, :, :, None]
            # shape [episode length, num of agents, batch size]
            traj_i_rewards = np.array(traj_i_rewards)[:, :, None]
            if all_traj_states is None:
                all_traj_states = traj_i_states
            else:
                all_traj_states = np.concatenate((all_traj_states, traj_i_states), axis=3)
            if all_traj_rewards is None:
                all_traj_rewards = traj_i_rewards
            else:
                all_traj_rewards = np.concatenate((all_traj_rewards, traj_i_rewards), axis=2)
    return all_traj_states, all_traj_rewards


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


class MarketEnv(AbstractEnvironment):

    def __init__(self, vectorize=True):
        self._market = None
        self.vectorize = vectorize

    @abstractmethod
    def init(self, agents, horizon=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def convert_from_actions_tensor(self, actions, agents, auction_item_specs_by_id):
        '''
        (if self.vectorize is True)
        Assume actions is a numpy array of shape [m, n, k]
        where:
            m = # of agents
            n = # of bids (per agent)
            k = # of fields of a bid vector
        Assuming bid vector is: [auction_item_spec_id, bid_per_item, total_limit].
        Assuming first dimension is in the same order as the agents list passed to init.
        output: 2D list of bid objects. bids[i] are agent i's bids, 
                bids[i][j] is the jth bid of agent i.
        '''
        if not self.vectorize:
            # Assume actions is a 2D list of bid objects, where
            # actions[i] is agent i's bids and actions[i][j] is the jth bid of agent i.
            return actions
        bids = []
        for i in range(len(actions)):
            agent_i_bids = []
            for j in range(len(actions[i])):
                bid_j_of_agent_i = actions[i][j]
                bidder = agents[i]
                auction_item_spec_id = bid_j_of_agent_i[0]
                auction_item_spec = auction_item_specs_by_id[auction_item_spec_id]
                bid_obj = Bid.from_vector(bid_j_of_agent_i, bidder, auction_item_spec)
                agent_i_bids.append(bid_obj)
            bids.append(agent_i_bids)
        return bids

    def convert_to_states_tensor(self, states: Iterable[State]):
        '''
        Assuming states is of the form:
        [
            state 1,
            ...,
            state n,
            
        ]
        (The canonical use of this func is for n = m, where m is 
         the # of agents)
        '''
        if not self.vectorize:
            return states
        states_tensor = []
        for state in states:
            vec = state.to_vector()
            states_tensor.append(vec)
        # numpy array of shape [n, s]
        # where:
        #   n = # of states
        #   s = # of fields of a state vector
        return np.array(states_tensor)


class OneCampaignNDaysEnv(MarketEnv):

    def __init__(self, auction: AbstractAuction, auction_item_spec_pmf: PMF, campaign_pmf: PMF,
                 num_items_per_timestep_min: int, num_items_per_timestep_max: int, vectorize=True):
        super().__init__(vectorize)
        self._auction = auction
        self._campaign_pmf = campaign_pmf
        # let this be half-open, i.e. [num_items_per_timestep_min, num_items_per_timestep_max)
        self._num_items_per_timestep_range = (num_items_per_timestep_min, num_items_per_timestep_max)
        self._auction_item_spec_pmf = auction_item_spec_pmf
        self._auction_item_specs_by_id = { spec.uid : spec for spec in self._auction_item_spec_pmf.items() }
        self._agents = None
        self._num_of_days = 1
        self.done = False

    def init(self, agents, horizon=1):
        self._agents = tuple(agents)
        self._num_of_days = horizon

    def reset(self):
        self.done = False
        bidder_states = []
        camps = self._campaign_pmf.draw_n(len(self._agents))
        for i in range(len(self._agents)):
            agent = self._agents[i]
            camp = camps[i]
            cbstate = CampaignBidderState(agent, camp)
            bidder_states.append(cbstate)
        self._market = OneCampaignMarket(self._auction, bidder_states, self._auction_item_spec_pmf)
        bidder_states = self.convert_to_states_tensor(bidder_states)
        return bidder_states

    def step(self, actions):
        '''
        input: actions
                (actions same form as in convert_from_actions_tensor func)
        output: states, rewards (as np array), done 
                (states as np array if self.vectorize is True)
        '''
        states = []
        rewards = []
        
        if not self.done:
            # Convert actions to a 2D list of bid objects, where bids[i] is agent i's 
            # bids and bids[i][j] is the jth bid of agent i.
            bids = self.convert_from_actions_tensor(actions, self._agents, self._auction_item_specs_by_id)

            # Randomize the number of auction items available (i.e. users who show up) in this step
            pre_step_agent_spends = [ self._market.get_bidder_state(agent).spend for agent in self._agents ]
            num_items_min = self._num_items_per_timestep_range[0]
            num_items_max = self._num_items_per_timestep_range[1]
            self._market.num_of_items_per_timestep = np.random.randint(num_items_min, num_items_max)

            # Run the auction
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
            results = self._market.run_auction(flatten_2d(bids), item_matches_bid_spec_func)
            self.done = (self._num_of_days != None) and (self._market.timestep == self._num_of_days)
            
            # Calculate each agent's reward
            for i in range(len(self._agents)):
                agent = self._agents[i]
                agent_bids = bids[i]
                cbstate = self._market.get_bidder_state(agent)
                states.append(cbstate)
                agent_expenditure = ( cbstate.spend - pre_step_agent_spends[i] ) # alternatively, use agent_bids
                agent_reward = -1 * agent_expenditure # i.e. negative reward
                if self.done:
                    # TODO: replace with actual formula
                    agent_reward += min(cbstate.campaign.budget, 
                                        (cbstate.impressions / (1.0 * cbstate.campaign.reach)) * cbstate.campaign.budget)
                rewards.append(agent_reward)
        
        states = self.convert_to_states_tensor(states)
        # numpy array of shape [m, ]
        rewards = np.array(rewards)
        return states, rewards, self.done


class SequentialAuctionEnv(MarketEnv):

    def __init__(self, auction: AbstractAuction, all_auction_items: List[AuctionItem], 
                 campaign_pmf: PMF, num_of_items_per_timestep: int = 1, vectorize=True):
        super().__init__(vectorize)
        self._auction = auction
        self._all_auction_items = all_auction_items
        self._campaign_pmf = campaign_pmf
        self._num_items_per_timestep = num_of_items_per_timestep
        self._auction_item_specs_by_id = { item.auction_item_spec.uid : item.auction_item_spec for item in self._all_auction_items }
        self._agents = None
        self._num_of_days = 1
        self.auction_items = []
        self.done = False

    def init(self, agents, horizon=None):
        self._agents = tuple(agents)
        self._num_of_days = horizon

    def reset(self):
        self.done = False
        bidder_states = []
        camps = self._campaign_pmf.draw_n(len(self._agents))
        for i in range(len(self._agents)):
            agent = self._agents[i]
            camp = camps[i]
            cbstate = CampaignBidderState(agent, camp)
            bidder_states.append(cbstate)

        self.auction_items = self._all_auction_items[:]
        np.random.shuffle(self.auction_items)

        item_satisfies_campaign_func = lambda item, campaign: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, campaign.target)
        self._market = SequentialAuctionMarket(self._auction, bidder_states, self.auction_items, 
                                   item_satisfies_campaign_func=item_satisfies_campaign_func, 
                                   num_of_items_per_timestep=self._num_items_per_timestep)

        bidder_states = self.convert_to_states_tensor(bidder_states)
        return bidder_states

    def step(self, actions):
        '''
        input: actions
                (actions same form as in convert_from_actions_tensor func)
        output: states, rewards (as np array), done 
                (states as np array if self.vectorize is True)
        '''
        states = []
        rewards = []
        
        if not self.done:
            # Convert actions to a 2D list of bid objects, where bids[i] is agent i's 
            # bids and bids[i][j] is the jth bid of agent i.
            bids = self.convert_from_actions_tensor(actions, self._agents, self._auction_item_specs_by_id)

            pre_step_agent_spends = [ self._market.get_bidder_state(agent).spend for agent in self._agents ]

            # Run the auction
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_item_type_match(item.auction_item_spec, bid.auction_item_spec)
            results = self._market.run_auction(flatten_2d(bids), item_matches_bid_spec_func)
            self.done = (self._num_of_days != None) and (self._market.timestep == self._num_of_days)
            
            # Calculate each agent's reward
            for i in range(len(self._agents)):
                agent = self._agents[i]
                agent_bids = bids[i]
                cbstate = self._market.get_bidder_state(agent)
                states.append(cbstate)
                agent_expenditure = ( cbstate.spend - pre_step_agent_spends[i] ) # alternatively, use agent_bids
                agent_reward = -1 * agent_expenditure # i.e. negative reward
                if self.done:
                    # TODO: replace with actual formula
                    agent_reward += min(cbstate.campaign.budget, 
                                        (cbstate.impressions / (1.0 * cbstate.campaign.reach)) * cbstate.campaign.budget)
                rewards.append(agent_reward)
        
        states = self.convert_to_states_tensor(states)
        # numpy array of shape [m, ]
        rewards = np.array(rewards)
        return states, rewards, self.done