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
from derby.core.agents import Agent



class AbstractEnvironment(ABC):

    @abstractmethod
    def init(self, agents: Iterable[Agent], horizon=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass


class MarketEnv(AbstractEnvironment):

    def __init__(self, vectorize=True):
        self.vectorize = vectorize
        self._market = None
        self.agents = None
        self.horizon = None
        self.done = False

    @abstractmethod
    def init(self, agents: Iterable[Agent], horizon=None):
        self.agents = tuple(agents)
        for i in range(len(self.agents)):
            self.agents[i].agent_num = i
        self.horizon = horizon

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
            # Assume actions is a 2D array of bid objects, where
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

    @staticmethod
    def generate_trajectories(env, num_of_trajs, horizon_cutoff, 
                              debug=False, update_policies_after_every_step=False):
        all_traj_states = None
        all_traj_actions = None
        all_traj_rewards = None
        for i in range(num_of_trajs):
            traj_i_states = None
            traj_i_actions = None
            traj_i_rewards = None
            agents_joint_state = env.reset()
            # agents_joint_state is array of shape [num_of_agents, state_size]
            # so reshape to [batch_size, episode_length, num_of_agents, state_size]
            # note that state_size would be () (i.e. OOP objects instead of vectors)
            # if vectorize is off. the below code appropriately handles both cases 
            # of vectorize on/off.
            agents_joint_state = np.array(agents_joint_state)[None, None, :]
            if traj_i_states is None:
                traj_i_states = agents_joint_state
           
            if debug:
                print("=== Traj {} ===".format(i))
                print()
                print("states {}, shape {}".format(0, agents_joint_state.shape))
                print(agents_joint_state)

            for j in range(horizon_cutoff):
                actions = [ agent.compute_action(agents_joint_state[0,0]) for agent in env.agents ]
                agents_joint_state, rewards, done = env.step(actions)

                agents_joint_state = np.array(agents_joint_state)[None, None, :]

                # actions is array of shape [num_of_agents]
                # so reshape to [batch_size, episode_length-1, num_of_agents]
                actions = np.array(actions)[None, None, :]

                # rewards is array of shape [num_of_agents]
                # so reshape to [batch_size, episode_length-1, num_of_agents]
                rewards = np.array(rewards)[None, None, :]         
    # TODO
                # if update_policies_after_every_step:
                #     for agent in env.agents:  
                #         states = ...
                #         actions = ...
                #         rewards = ...
                #         agent.update_policy(states, actions, rewards) for agent in env.agents
    #
                # Update trajectory
                if env.vectorize:
                    traj_i_states = np.concatenate((traj_i_states, agents_joint_state), axis=1)
                    if traj_i_rewards is None: # shortcuting check for all
                        traj_i_actions = actions
                        traj_i_rewards = rewards    
                    else:
                        traj_i_actions = np.concatenate((traj_i_actions, actions), axis=1)
                        traj_i_rewards = np.concatenate((traj_i_rewards, rewards), axis=1)   
                if debug:
                    print("actions {}, shape {}".format(j, actions.shape))
                    print(actions)
                    print("rewards {}, shape {}".format(j, rewards.shape))
                    print(rewards)
                    print("states {}, shape {}".format(j+1, agents_joint_state.shape))
                    print(agents_joint_state)
                    print("Done? {}".format(done))
                if done:
                    break
            print()

            # Update batch
            if env.vectorize:
                if all_traj_states is None: # shortcuting check for all
                    all_traj_states = traj_i_states
                    all_traj_actions = traj_i_actions
                    all_traj_rewards = traj_i_rewards
                else:
                    all_traj_states = np.concatenate((all_traj_states, traj_i_states), axis=0)
                    all_traj_actions = np.concatenate((all_traj_actions, traj_i_actions), axis=0)
                    all_traj_rewards = np.concatenate((all_traj_rewards, traj_i_rewards), axis=0)

        return all_traj_states, all_traj_actions, all_traj_rewards


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

    def init(self, agents, horizon=1):
        super().init(agents, horizon=horizon)

    def reset(self):
        self.done = False
        bidder_states = []
        camps = self._campaign_pmf.draw_n(len(self.agents))
        for i in range(len(self.agents)):
            agent = self.agents[i]
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
            bids = self.convert_from_actions_tensor(actions, self.agents, self._auction_item_specs_by_id)

            # Randomize the number of auction items available (i.e. users who show up) in this step
            pre_step_agent_spends = [ self._market.get_bidder_state(agent).spend for agent in self.agents ]
            num_items_min = self._num_items_per_timestep_range[0]
            num_items_max = self._num_items_per_timestep_range[1]
            self._market.num_of_items_per_timestep = np.random.randint(num_items_min, num_items_max)

            # Run the auction
            item_matches_bid_spec_func = lambda item, bid: AuctionItemSpecification.is_a_type_of(item.auction_item_spec, bid.auction_item_spec)
            results = self._market.run_auction(flatten_2d(bids), item_matches_bid_spec_func)
            self.done = (self.horizon != None) and (self._market.timestep == self.horizon)
            
            # Calculate each agent's reward
            for i in range(len(self.agents)):
                agent = self.agents[i]
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